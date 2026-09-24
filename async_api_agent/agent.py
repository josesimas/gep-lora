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

The chat box is the one step that *does* things: `chat()` gives the model
the tools in tools.py and runs what it calls -- a few rounds at most -- then
lets it say what happened. With no model that can take tools, commands.py
reads the plain requests people make most into the same calls, so "only the
first 20, rank 32" works either way. What the tools changed goes back to the
page as a session and a list of actions, which the page carries out through
the API.

The second half -- combining the LoRAs -- has steps of its own (blend_intro,
blend_confirm, blend_started, blend_debrief) built the same way, from
blending.py's facts: which of the user's own LoRAs, the search's size, and
what the search found. And the third -- after the search -- has test_started,
test_debrief, verify_start, verify_debrief and live, from release.py's: every
blend's tested score, the blend the person picked beside each of its LoRAs,
and the one put live.

Nothing here keeps state between calls. The page holds the conversation and
the session and sends back what a step needs; the dataset is re-read rather
than remembered, which costs milliseconds and means a server restart loses
nothing.
"""

import json

from adapters import catalog as lora_catalog
from async_api_agent import blending
from async_api_agent import commands
from async_api_agent import planner
from async_api_agent import prompts
from async_api_agent import providers
from async_api_agent import settings
from async_api_agent import tools

# The most rounds of tool calls one chat message may take before the agent
# stops asking and reports what was done.
MAX_ROUNDS = 6


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


def intro(choice=None, mock=False, ready=0):
    """The welcome. `ready` is how many ready LoRAs the user already has, so
    one who trained before can go straight to blending them."""
    options = planner.recipe()
    facts = {"steps": prompts.PROCESS, "loras": len(settings.LORA_RANKS),
             "ranks": settings.LORA_RANKS, "base_model": options["base_model"],
             "mock": bool(mock), "ready_loras": ready}
    fallback = prompts.FALLBACK_INTRO.format(
        steps=_numbered(prompts.PROCESS), loras=len(settings.LORA_RANKS),
        base_model=options["base_model"], mock=prompts.FALLBACK_INTRO_MOCK if mock else "",
        ready=prompts.FALLBACK_INTRO_READY.format(count=ready) if ready else "")
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


def wait(catalog, records, choice=None, mock=False, history=None, session=None, quiet=False):
    """The wait question and its options. `quiet` gives the options alone,
    for a page redrawing them after the plan changed."""
    options = planner.choices(catalog, records, mock, session)
    recommended = recommend(options)
    loras = options[0]["estimate"]["loras"]
    if quiet:
        return {"message": None, "choices": options,
                "recommended": recommended["id"] if recommended else None}
    facts = {"loras": loras, "records": records,
             "choices": [{"id": one["id"], "label": one["label"], "epochs": one["epochs"],
                          "time": one["time"]} for one in options],
             "recommended": recommended, "estimate_source": options[0]["estimate"]["source"]}
    fallback = prompts.FALLBACK_WAIT.format(loras=loras)
    return {"message": say("wait", facts, fallback, choice, history), "choices": options,
            "recommended": recommended["id"] if recommended else None}


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


def _summary(box):
    """What the tools did, as the fallback says it."""
    done = [step["summary"] for step in box.steps if step["ok"]]
    failed = [step["summary"] for step in box.steps if not step["ok"]]
    parts = []
    if done:
        parts.append(prompts.FALLBACK_TOOLS.format(lines=_bullets(done)))
    if failed:
        parts.append(prompts.FALLBACK_TOOLS_FAILED.format(lines=_bullets(failed)))
    return "\n\n".join(parts)


def chat(catalog, user, message, stage, session=None, dataset=None, context=None,
         choice=None, history=None, shared_datasets=None, registry=None):
    """A typed message: answered, and acted on with the tools.

    -> {message, session, actions, steps, preview, analysis} -- the reply,
    and what the page must now do (Toolbox.outcome()).
    """
    box = tools.Toolbox(catalog, user, stage, session, dataset, shared_datasets, registry)
    instructions = prompts.STEP_INSTRUCTIONS.get(stage, "")
    facts = {"stage": stage, "step_instructions": instructions, "session": box.session,
             "dataset_given": bool(dataset), "context": context or {}}
    history = list(history or [])[-settings.HISTORY_TURNS:]
    note, reason = None, None
    try:
        resolved = providers.resolve(choice)
        if resolved["kind"] == "scripted":
            raise providers.ProviderError(None)
        exchange = [{"role": "user", "content": "FACTS:\n%s\n\nThe person wrote:\n%s" % (
            json.dumps(facts, indent=1, ensure_ascii=False, default=str), message)}]
        text, model = "", None
        for _ in range(MAX_ROUNDS):
            reply = providers.converse(resolved, prompts.system("chat"), history, exchange,
                                       tools.schemas())
            text, model = reply["text"], reply["model"]
            if not reply["calls"]:
                break
            exchange.append({"role": "assistant", "content": text, "calls": reply["calls"],
                             "raw": reply["raw"]})
            for call in reply["calls"]:
                result = box.run(call["name"], call["arguments"])
                exchange.append({"role": "tool", "id": call["id"], "name": call["name"],
                                 "content": json.dumps(result, ensure_ascii=False,
                                                       default=str)[:8000]})
        else:
            text = ""                       # still calling tools: report what they did
        spoken = {"text": text or _summary(box) or prompts.FALLBACK_CHAT.format(
                      instructions=instructions),
                  "by": "%s \u00b7 %s" % (resolved["label"], model), "fallback": False,
                  "note": None}
        return dict(box.outcome(), message=spoken)
    except providers.ProviderError as error:
        reason = error.args[0] if error.args else None
    note = prompts.FALLBACK_NOTE.format(reason=reason) if reason else None
    if box.steps:
        # The model failed part way: what already ran stands, and is reported.
        spoken = {"text": _summary(box), "by": "built-in wording", "fallback": True, "note": note}
        return dict(box.outcome(), message=spoken)
    demos = [one["file"] for one in (shared_datasets() if shared_datasets else [])]
    names = [row["name"] for row in blending.mine(catalog, user)]
    for name, arguments in commands.parse(message, stage, demos, names):
        box.run(name, arguments)
    if box.steps:
        spoken = {"text": _summary(box), "by": "built-in wording", "fallback": True, "note": note}
    else:
        fallback = prompts.FALLBACK_CHAT.format(instructions=instructions)
        spoken = say("chat", facts, fallback, choice, history, message)
    return dict(box.outcome(), message=spoken)


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


# --- the second half: combining the LoRAs ------------------------------------


def _search(blend, mock):
    found = blending.estimate(blend["population"], blend["generations"], mock)
    return ({"generations": blend["generations"], "population": blend["population"],
             "questions": blend["questions"]}, found)


def blend_intro(catalog, user, session, choice=None, history=None, prefer=(), quiet=False):
    """The blend step's opening: the user's own ready LoRAs, the ones picked,
    and the search as it stands. `quiet` gives the facts alone, for a page
    that came here because the chat asked and has already been answered.

    -> {message, available, loras, blend}: `blend` is the session's blend
    part with the pick written into it, so what the page shows is what it
    will plan."""
    available = blending.mine(catalog, user)
    try:
        rows = blending.chosen(catalog, user, session["blend"], prefer)
    except blending.BlendError:
        rows = blending.default_pick(catalog, user, prefer)
    blend = dict(session["blend"], loras=[row["id"] for row in rows])
    mock = session["mock"] or any(row["mock"] for row in rows)
    search, found = _search(blend, mock)
    out = {"available": [blending.describe(row) for row in available],
           "loras": [blending.describe(row) for row in rows], "blend": blend,
           "estimate": found, "message": None}
    if quiet:
        return out
    facts = {"loras": [{"name": row["name"], "rank": row["rank"],
                        "base_model": row["base_model"]} for row in rows],
             "available": len(available), "search": search,
             "estimate": found["time"], "mock": mock}
    if not rows:
        fallback = prompts.FALLBACK_BLEND_NONE
    else:
        fallback = prompts.FALLBACK_BLEND.format(
            questions=blend["questions"],
            picked=", ".join("**%s** (rank %s)" % (row["name"], row["rank"]) for row in rows),
            repeat=prompts.FALLBACK_BLEND_REPEAT if len(rows) < len(blending.SLOTS) else "",
            generations=1 + blend["generations"], population=blend["population"],
            time=found["time"], source=found["source"])
    out["message"] = say("blend", facts, fallback, choice, history)
    return out


def blend_confirm(plan):
    """The search read back before it starts: exact numbers, so no model."""
    places = {}
    for slot, name in sorted(plan["slots"].items()):
        places.setdefault(name, []).append(slot)
    lines = ["**%s** — rank %s, in %s" % (one["name"], one["rank"],
                                          ", ".join(places.get(one["name"], [])))
             for one in plan["loras"]]
    found = plan["estimate"]
    text = prompts.FALLBACK_BLEND_CONFIRM.format(
        lines=_bullets(lines), generations=found["generations"],
        population=found["population"], questions=plan["questions"], source=plan["source"],
        testing=(prompts.FALLBACK_BLEND_CONFIRM_TESTING.format(
            testing=plan["testing"], validation=plan.get("validation", 0))
                 if plan["testing"] else ""),
        time=found["time"], estimate_source=found["source"],
        mock=prompts.FALLBACK_BLEND_CONFIRM_MOCK if plan["mock"] else "")
    return {"text": text, "by": "built-in wording", "fallback": False, "note": None}


def blend_started(job, plan, choice=None, history=None):
    """What the agent says once the page has queued the search."""
    found = plan.get("estimate") or {}
    facts = {"label": plan.get("label"), "loras": [one.get("name") for one in plan.get("loras", [])],
             "estimate": found.get("time"), "mock": bool(plan.get("mock")),
             "queue": ("position %d in the worker's queue" % job["queue_position"])
             if job.get("queue_position") else "the worker starts it next"}
    fallback = prompts.FALLBACK_BLEND_START.format(
        label=plan.get("label") or "your search", rounds=found.get("generations") or "?",
        time=found.get("time") or "a while",
        mock=prompts.FALLBACK_BLEND_START_MOCK if plan.get("mock") else "")
    return say("blend_start", facts, fallback, choice, history)


def _trend(history):
    scores = [one["best"] for one in history if isinstance(one.get("best"), (int, float))]
    if len(scores) < 2:
        return ""
    if scores[-1] > scores[0]:
        return "The best score rose from %.2f in the first round to %.2f." % (scores[0], scores[-1])
    return "The best score stayed at about %.2f across the rounds." % scores[-1]


def blend_debrief(registry, catalog, user, job, choice=None, history=None):
    """How the search went, from the user's own job."""
    facts = blending.outcome(registry, job, user, catalog)
    best = facts.get("best")
    if facts["status"] != "done":
        # Not finished: nothing to test, verify or put live until it is.
        fallback = prompts.FALLBACK_BLEND_DEBRIEF_UNFINISHED.format(
            status=facts["status"], error=": " + facts["error"] if facts.get("error") else "",
            best=("; the best blend so far is **%s**, scoring **%.3f**"
                  % (best["formula"], best["fitness"] or 0.0)) if best else "")
    elif best is None:
        fallback = prompts.FALLBACK_BLEND_DEBRIEF_NONE.format(
            status=facts["status"], error=": " + facts["error"] if facts.get("error") else "")
    else:
        tested = best.get("tested_quality")
        fallback = prompts.FALLBACK_BLEND_DEBRIEF.format(
            status=facts["status"], formula=best["formula"],
            fitness="%.3f" % (best["fitness"] or 0.0),
            tested=(", and **%.3f** on questions it never saw" % tested
                    if isinstance(tested, (int, float)) else ""),
            trend=_trend(facts.get("history") or []),
            mock=prompts.FALLBACK_BLEND_DEBRIEF_MOCK if facts.get("mock") else "")
    return {"message": say("blend_debrief", facts, fallback, choice, history), "result": facts}


# --- after the search: testing, verification, going live ----------------------


def _score(value):
    return "%.3f" % value if isinstance(value, (int, float)) else "–"


def test_started(found, job, choice=None, history=None):
    """What the agent says once the page has queued the testing step.
    `found` is release.blends() for the job, read before it ran."""
    facts = {"blends": found["testing"]["blends"], "questions": found["testing"]["records"],
             "mock": found["mock"],
             "queue": ("position %d in the worker's queue" % job["queue_position"])
             if job.get("queue_position") else "the worker starts it next"}
    fallback = prompts.FALLBACK_TEST_START.format(
        blends=facts["blends"], questions=facts["questions"],
        mock=prompts.FALLBACK_TEST_START_MOCK if found["mock"] else "")
    return say("test_start", facts, fallback, choice, history)


def test_debrief(found, choice=None, history=None):
    """How the testing went, from release.blends(): the blends ranked, and
    the one recommended -- the best on the questions the search never saw."""
    top = [{"number": one["number"], "formula": one["formula"], "search_score": one["quality"],
            "tested_score": one["tested"]} for one in found["blends"][:3]
           if one["state"] != "BAD"]
    facts = {"status": found["status"], "blends": found["testing"]["blends"],
             "tested": found["testing"]["tested"], "questions": found["testing"]["records"],
             "top": top, "recommended": found["recommended"], "mock": found["mock"]}
    if not found["testing"]["tested"]:
        fallback = prompts.FALLBACK_TEST_DEBRIEF_NONE.format(
            why=" yet" if found["testing"]["records"] else
            " — this search kept no questions back to test on")
    else:
        fallback = prompts.FALLBACK_TEST_DEBRIEF.format(
            tested=facts["tested"], blends=facts["blends"], questions=facts["questions"],
            lines=_bullets(["**#%d** %s — search %s, tested **%s**"
                            % (one["number"], one["formula"], _score(one["search_score"]),
                               _score(one["tested_score"])) for one in top]),
            recommended=found["recommended"],
            mock=prompts.FALLBACK_TEST_START_MOCK if found["mock"] else "")
    return say("test_debrief", facts, fallback, choice, history)


def verify_start(planned, mock=False):
    """The verification read back as it is queued: exact numbers, so no model."""
    one = planned["blend"]
    text = prompts.FALLBACK_VERIFY_START.format(
        individual=one["number"], formula=one["formula"],
        against=", ".join("**%s**" % name for name in planned["against"]),
        count=planned["count"], source=planned["questions_from"],
        mock=prompts.FALLBACK_VERIFY_START_MOCK if mock else "")
    return {"text": text, "by": "built-in wording", "fallback": False, "note": None}


def verify_debrief(facts, choice=None, history=None):
    """How the verification went, from release.verification_outcome()."""
    report = facts.get("report")
    if report is None:
        fallback = prompts.FALLBACK_VERIFY_FAILED.format(
            status=facts["status"], error=": " + facts["error"] if facts.get("error") else "")
    else:
        against = report["against"]
        lines = ["**%s** alone: %s — the blend won %d, tied %d, lost %d (%s)"
                 % (one["lora"], _score(one["mean"]), one["wins"], one["ties"], one["losses"],
                    one["blend_is"]) for one in against]
        fallback = prompts.FALLBACK_VERIFY_DEBRIEF.format(
            individual=facts["individual"], blend=_score(report["blend_mean"]),
            questions=report["questions"], source=report["questions_from"],
            lines=_bullets(lines),
            verdict=(prompts.FALLBACK_VERIFY_BETTER
                     if against and all(one["blend_is"] == "better" for one in against)
                     else prompts.FALLBACK_VERIFY_MIXED),
            mock=prompts.FALLBACK_TEST_START_MOCK if report["mock"] else "")
    return say("verify_debrief", facts, fallback, choice, history)


def live(deployment, formula, choice=None, history=None):
    """What the agent says once a blend is live. The token is never among the
    facts: a model is not to be handed a secret it might repeat."""
    facts = {"individual": deployment["individual"], "formula": formula,
             "base_model": deployment["base_model"], "engine": deployment["engine"]}
    fallback = prompts.FALLBACK_LIVE.format(
        individual=deployment["individual"], formula=formula,
        base_model=deployment["base_model"],
        mock=prompts.FALLBACK_LIVE_MOCK if deployment["engine"] == "mock" else "")
    return say("live", facts, fallback, choice, history)
