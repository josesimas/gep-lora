"""
agent.py - The conversation: each step's facts, put into words by the model.

Every function here is one step of the page's conversation and returns what
the agent says -- `{"text", "by", "fallback", "note"}` -- beside the facts the
page draws on the right. The shape of each is the same:

    facts   computed here (analysis.py, planner.py, the catalogue)
    words   the model's, told prompts.system(step) and shown the facts as JSON
    else    prompts.FALLBACK_*, filled from the same facts

so a provider that is down, has no key or is set to "scripted" changes how a
step is phrased and nothing else. The note says which it was, and the page
shows it under the message.

Nothing here keeps state between calls. The page holds the conversation and
sends back what a step needs; the dataset is re-read rather than remembered,
which costs milliseconds and means a server restart loses nothing.
"""

import json
import re

from adapters import catalog as lora_catalog
from async_api_agent import planner
from async_api_agent import prompts
from async_api_agent import providers
from async_api_agent import settings


def _bullets(lines):
    return "\n".join("- " + line for line in lines)


def _sentence(text):
    return text[:1].upper() + text[1:] + "."


def _numbered(lines):
    return "\n".join("%d. %s" % (number, line) for number, line in enumerate(lines, 1))


def say(step, facts, fallback, choice=None, history=None, message=None):
    """One step's message: the model's words, or the fallback's. -> dict."""
    try:
        resolved = providers.resolve(choice)
    except providers.ProviderError as error:
        return {"text": fallback, "by": "built-in wording", "fallback": True,
                "note": prompts.FALLBACK_NOTE.format(reason=error)}
    if resolved["kind"] == "scripted":
        return {"text": fallback, "by": "built-in wording", "fallback": True, "note": None}
    ask = "FACTS:\n" + json.dumps(facts, indent=1, ensure_ascii=False, default=str)
    ask += ("\n\nThe person wrote:\n" + message) if message else \
        "\n\nWrite your message for this step now."
    turns = list(history or [])[-settings.HISTORY_TURNS:] + [{"role": "user", "content": ask}]
    try:
        text, model = providers.chat(resolved, prompts.system(step), turns)
    except providers.ProviderError as error:
        return {"text": fallback, "by": "built-in wording", "fallback": True,
                "note": prompts.FALLBACK_NOTE.format(reason=error)}
    return {"text": text, "by": "%s · %s" % (resolved["label"], model),
            "fallback": False, "note": None}


# --- the steps ---------------------------------------------------------------


def intro(choice=None, mock=False):
    options = planner.recipe()
    facts = {"steps": prompts.PROCESS, "loras": len(settings.LORA_RANKS),
             "ranks": settings.LORA_RANKS, "base_model": options["base_model"],
             "mock": bool(mock)}
    fallback = prompts.FALLBACK_INTRO.format(
        steps=_numbered(prompts.PROCESS), loras=len(settings.LORA_RANKS),
        base_model=options["base_model"], mock=prompts.FALLBACK_INTRO_MOCK if mock else "")
    return {"message": say("intro", facts, fallback, choice), "facts": facts}


def summarise(found, choice=None):
    """What the agent says about an analysed dataset."""
    stats = found["stats"]
    facts = {"name": found["name"], "format": found["format"], "usable": found["usable"],
             "stats": stats, "samples": found["samples"], "problems": found["problems"],
             "keywords": [one["word"] for one in found["keywords"]],
             "few_records": settings.FEW_RECORDS}
    if not found["usable"]:
        fallback = prompts.FALLBACK_UNUSABLE.format(problems=_bullets(found["problems"]))
    else:
        extras = []
        if stats["system_prompts"]:
            extras.append("%d distinct system prompt(s) set the scene" % stats["system_prompts"])
        if stats["multi_turn"]:
            extras.append("%d conversation(s) go back and forth more than once"
                          % stats["multi_turn"])
        if stats["duplicates"]:
            extras.append("%d record(s) repeat another exactly" % stats["duplicates"])
        if found["keywords"]:
            extras.append("common words: %s" % ", ".join(
                one["word"] for one in found["keywords"][:6]))
        small = stats["records"] < settings.FEW_RECORDS
        fallback = prompts.FALLBACK_ANALYSIS.format(
            source=" (%s)" % found["name"] if found["name"] else "",
            usable=stats["records"],
            skipped=" (%d skipped)" % stats["skipped"] if stats["skipped"] else "",
            user_words=stats["user_words"]["median"],
            assistant_words=stats["assistant_words"]["median"],
            extras=(" " + _sentence("; ".join(extras))) if extras else "",
            verdict=(prompts.FALLBACK_ANALYSIS_SMALL if small else prompts.FALLBACK_ANALYSIS_OK)
            + prompts.FALLBACK_ANALYSIS_NEXT)
    return say("analysis", facts, fallback, choice)


def recommend(options):
    """Which wait to suggest, and why: the fullest that is still short."""
    for choice in reversed(options):
        if choice["estimate"]["seconds"] <= 15 * 60:
            return {"id": choice["id"],
                    "why": "on this dataset it still finishes in %s" % choice["time"]}
    return None


def wait(catalog, records, choice=None, mock=False, history=None):
    options = planner.choices(catalog, records, mock)
    recommended = recommend(options)
    facts = {"loras": len(settings.LORA_RANKS), "records": records,
             "choices": [{"id": one["id"], "label": one["label"], "epochs": one["epochs"],
                          "time": one["time"]} for one in options],
             "recommended": recommended, "estimate_source": options[0]["estimate"]["source"]}
    fallback = prompts.FALLBACK_WAIT.format(loras=len(settings.LORA_RANKS))
    return {"message": say("wait", facts, fallback, choice, history), "choices": options,
            "recommended": recommended["id"] if recommended else None}


_JSON = re.compile(r"\{.*\}", re.DOTALL)


def interpret(catalog, records, text, choice=None, mock=False):
    """A typed wait as epochs: the model's reading, checked, else the rules'.
    -> {epochs or None, how, estimate, message}."""
    one = planner.estimate(catalog, records, 1, mock)
    per_epoch = one["seconds_per_epoch"]
    overhead = one["overhead_seconds"] * one["loras"]
    epochs, how = None, None
    try:
        resolved = providers.resolve(choice)
        if resolved["kind"] != "scripted":
            facts = {"seconds_per_epoch": per_epoch, "overhead_seconds": overhead,
                     "min_epochs": settings.MIN_EPOCHS, "max_epochs": settings.MAX_EPOCHS,
                     "choices": [{"label": c["label"], "epochs": c["epochs"]}
                                 for c in settings.WAIT_CHOICES]}
            reply, _ = providers.chat(resolved, prompts.system("wait_interpret"), [
                {"role": "user", "content": "FACTS:\n%s\n\nTheir answer:\n%s"
                 % (json.dumps(facts), text)}])
            found = _JSON.search(reply)
            parsed = json.loads(found.group(0)) if found else {}
            value = parsed.get("epochs")
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
                epochs, how = planner.clamp(value), (parsed.get("reason") or "").strip() or None
    except (providers.ProviderError, ValueError, AttributeError):
        pass
    if epochs is None:
        epochs, how = planner.read_wait(text, per_epoch, overhead)
    if epochs is None:
        return {"epochs": None, "how": how, "estimate": None,
                "message": {"text": prompts.FALLBACK_WAIT_UNCLEAR, "by": "built-in wording",
                            "fallback": True, "note": None}}
    found = planner.estimate(catalog, records, epochs, mock)
    return {"epochs": epochs, "how": how, "estimate": found,
            "message": {"text": prompts.FALLBACK_WAIT_UNDERSTOOD.format(epochs=epochs,
                                                                        time=found["time"]),
                        "by": "built-in wording", "fallback": False,
                        "note": "How I read it: %s." % how.rstrip(".") if how else None}}


def confirm(trainings, estimate, mock=False):
    """The plan read back before it starts: exact numbers, so no model."""
    lines = ["**%s** — rank %s, %d step(s)" % (one["name"], one["settings"]["rank"],
                                                estimate["steps"]) for one in trainings]
    text = prompts.FALLBACK_CONFIRM.format(
        lines=_bullets(lines), epochs=estimate["epochs"], time=estimate["time"],
        source=estimate["source"], mock=prompts.FALLBACK_CONFIRM_MOCK if mock else "")
    return {"text": text, "by": "built-in wording", "fallback": False, "note": None}


def started(trainings, epochs, estimate, mock=False, choice=None, history=None):
    """What the agent says once the page has queued the plan."""
    queue = [one.get("queue_position") for one in trainings if one.get("queue_position")]
    facts = {"trainings": [{"name": one.get("name"), "rank": one.get("rank")}
                           for one in trainings],
             "epochs": epochs, "estimate": (estimate or {}).get("time"), "mock": bool(mock),
             "queue": ("position %d in the worker's queue" % min(queue)) if queue
             else "the worker takes them one after another"}
    fallback = prompts.FALLBACK_START.format(
        names=", ".join("**%s**" % one.get("name") for one in trainings), epochs=epochs,
        estimate=(estimate or {}).get("time") or "a while",
        mock=prompts.FALLBACK_START_MOCK if mock else "")
    return say("start", facts, fallback, choice, history)


def chat(message, stage, context=None, choice=None, history=None):
    instructions = prompts.STEP_INSTRUCTIONS.get(stage, "")
    facts = {"stage": stage, "step_instructions": instructions, "context": context or {}}
    fallback = prompts.FALLBACK_CHAT.format(instructions=instructions)
    return say("chat", facts, fallback, choice, history, message)


def debrief(catalog, user, lora_ids, choice=None, history=None, mock=False):
    """How the training went, from the user's own catalogue rows."""
    loras = []
    for lora_id in lora_ids or []:
        row = catalog.get(lora_id) if isinstance(lora_id, int) else None
        if row is None or row["owner"] != user["name"]:
            continue
        one = lora_catalog.as_dict(row)
        loras.append({"name": one["name"], "rank": one["rank"], "status": one["status"],
                      "final_loss": one["final_loss"], "steps": one["steps"],
                      "seconds": one["seconds"], "sample": one["sample"],
                      "error": one["error"], "mock": one["mock"],
                      "prompt": (one["recipe"] or {}).get("prompt")})
    prompt = next((one["prompt"] for one in loras if one["prompt"]), None)
    facts = {"loras": loras, "prompt": prompt, "mock": bool(mock)}
    lines = []
    for one in loras:
        if one["status"] == lora_catalog.READY:
            lines.append("**%s** (rank %s) is ready: final loss %s after %s step(s)%s."
                         % (one["name"], one["rank"],
                            "%.3f" % one["final_loss"] if one["final_loss"] is not None else "?",
                            one["steps"] or "?",
                            ", %s" % planner.human(one["seconds"]) if one["seconds"] else ""))
        else:
            lines.append("**%s** %s%s." % (one["name"], one["status"],
                                           ": " + one["error"] if one["error"] else ""))
    fallback = prompts.FALLBACK_DEBRIEF.format(
        lines=_bullets(lines) if lines else "- (no LoRAs to report on)",
        mock=prompts.FALLBACK_DEBRIEF_MOCK if mock else "")
    return {"message": say("debrief", facts, fallback, choice, history), "loras": loras}
