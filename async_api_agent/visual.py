"""
visual.py - The visual guide: a blend drawn by hand, with the guide beside it.

The page (async_api/visual_guide.html) holds a drawing -- a tree of the
person's own LoRAs folded together, async_api/drawn.py's shape -- and this
module is its guide, built on the same bargain as agent.py:

    facts   drawn.check() and the catalogue: the chromosome, the ranks, whether
            PEFT can build it, what the weights are worth; release.py's
            reading of a finished verification
    words   the model's, told prompts.system("visual_...")
    else    prompts.VISUAL_FALLBACK_*, filled from the same facts

**The chat acts on the drawing, never on the world**, as tools.py does on the
training plan: its tools edit the session the page sends (`session_of`) and
hand it back, and the one thing that does something -- testing the blend -- is
an *action* the page carries out through its own button's call, POST
/blends/test, with the body `plan()` read back first. Nothing here writes a
job.

The session:

    {"tree":      <drawing>        drawn.py's; an empty CAT to start
     "seed":      n | None         the WEIGHT_MASTER_SEED the weights come from
     "number":    n                the individual they are drawn for: 1, or the
                                   number a blend opened from a search had there
     "questions": {"file": name} | {"lora": id} | {"given": name} | None
                                   what the test asks; "given" is a dataset the
                                   page holds (pasted or uploaded) and sends
                                   itself -- its text never travels in a session
     "count":     n                how many questions
     "mock":      bool}            a practice run: nothing loaded, random scores

A place in the drawing is named by its path (drawn.walk): "" is the top, "0"
its left side, "1.0" the right side's left side. The model may say "top",
"left" and "right" too.
"""

import copy
import json

from async_api import drawn
from async_api_agent import agent
from async_api_agent import blending
from async_api_agent import guide_defaults
from async_api_agent import prompts
from async_api_agent import providers
from async_api_agent import release
from async_api_agent import settings
from async_api.drawn import ROOT
from search.generate_population import BINARY_OPS, VARIABLES

STAGES = ("drawing", "testing", "tested")

# The questions a test asks when the person has not said how many.
DEFAULT_COUNT = drawn.DEFAULT_QUESTIONS

_INT = {"type": "integer"}
_PATH = {"type": "string",
         "description": 'a place: "" or "top" for the top, "0"/"left" and "1"/"right" '
                        'below it, "1.0" for the right side\'s left side'}
_LORA = {"type": ["integer", "string"], "description": "a LoRA of theirs, by id or name"}

# name -> (JSON schema properties, required, changes the drawing or the test)
SPECS = {
    "list_my_loras": ({}, [], False),
    "show_drawing": ({}, [], False),
    "draw_blend": ({"loras": {"type": "array", "items": _LORA},
                    "fold": {"type": "string", "enum": list(BINARY_OPS)}}, ["loras"], True),
    "place_lora": ({"where": _PATH, "lora": _LORA,
                    "weight": {"type": "string", "enum": list(VARIABLES)}},
                   ["where", "lora"], True),
    "place_fold": ({"where": _PATH, "op": {"type": "string", "enum": list(BINARY_OPS)}},
                   ["where", "op"], True),
    "set_weight": ({"where": _PATH, "weight": {"type": "string", "enum": list(VARIABLES)}},
                   ["where", "weight"], True),
    "clear_place": ({"where": _PATH}, ["where"], True),
    "swap_sides": ({"where": _PATH}, ["where"], True),
    "new_weights": ({"seed": dict(_INT, description="a seed of their choosing")}, [], True),
    "start_over": ({}, [], True),
    "list_demo_datasets": ({}, [], False),
    "set_test_questions": ({"file": {"type": "string", "description": "a demo dataset"},
                            "lora": _LORA,
                            "count": dict(_INT, description="how many questions")},
                           [], True),
    "set_practice_run": ({"on": {"type": "boolean"}}, ["on"], True),
    "start_test": ({}, [], False),
}


class VisualError(ValueError):
    """Why a tool or a step refused, in words the model (or the page) passes on."""


def empty_tree():
    return {"op": ROOT, "children": [None, None]}


def session_of(raw):
    """The page's session, checked, with the defaults filled. Raises VisualError."""
    raw = raw if isinstance(raw, dict) else {}
    # A null tree is a drawing whose top was emptied, not a missing one.
    tree = raw["tree"] if "tree" in raw else empty_tree()
    try:
        drawn.walk(tree)
    except drawn.DrawnError as error:
        raise VisualError("the drawing: %s" % error)
    seed = raw.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise VisualError("the seed is a whole number, 0 or more")
    try:
        number = drawn.number_of(raw.get("number"))
    except drawn.DrawnError as error:
        raise VisualError(str(error))
    questions = raw.get("questions")
    if questions is not None and not (
            isinstance(questions, dict) and len(questions) == 1 and (
                isinstance(questions.get("file"), str)
                or (isinstance(questions.get("lora"), int)
                    and not isinstance(questions.get("lora"), bool))
                or isinstance(questions.get("given"), str))):
        raise VisualError('questions are {"file": name}, {"lora": id} or {"given": name}')
    count = raw.get("count", DEFAULT_COUNT)
    if isinstance(count, bool) or not isinstance(count, int) \
            or not 1 <= count <= drawn.MAX_QUESTIONS:
        raise VisualError("count is how many questions, from 1 to %d" % drawn.MAX_QUESTIONS)
    return {"tree": copy.deepcopy(tree), "seed": seed, "number": number, "questions": questions,
            "count": count, "mock": bool(raw.get("mock"))}


# --- reading the drawing -------------------------------------------------------


def path_of(where):
    """A place as the model may name it -> drawn.walk()'s path."""
    if where is None:
        return ""
    text = str(where).strip().lower()
    if text in ("", "top", "root", "the top"):
        return ""
    for mark in ("/", "›", ">", ",", " "):     # "right › left" is how places are said
        text = text.replace(mark, ".")
    parts = [part for part in text.split(".") if part and part != "of"]
    out = []
    for part in parts:
        if part in ("left", "l", "0"):
            out.append("0")
        elif part in ("right", "r", "1"):
            out.append("1")
        else:
            raise VisualError("%r is not a place; a place is a path such as \"0\" or \"1.0\""
                              % (where,))
    return ".".join(out)


def at(tree, path):
    """-> (parent or None, index, node) at `path`. Raises VisualError."""
    parent, index, node = None, None, tree
    for part in (path.split(".") if path else []):
        if node is None or "op" not in node:
            raise VisualError("there is no place %s in the drawing" % path)
        parent, index, node = node, int(part), node["children"][int(part)]
    return parent, index, node


def _where(path):
    return drawn.place(path)


def _names(catalog, user):
    return {row["id"]: row["name"] for row in blending.mine(catalog, user)}


def places(tree, names, weights=None):
    """Every place in words, for the model. -> [{path, is}]."""
    out = []
    for path, node in drawn.walk(tree):
        if node is None:
            said = "empty"
        elif "op" in node:
            said = "%s fold" % node["op"]
        else:
            value = (weights or {}).get(node["weight"])
            said = "LoRA %s at %s%s" % (names.get(node["lora"], "#%s" % node["lora"]),
                                        node["weight"],
                                        " (%.2f)" % value if value is not None else "")
        out.append({"path": path, "is": said})
    return out


def check_of(catalog, user, session):
    """drawn.check() for the session's drawing, or None when it cannot be read."""
    try:
        return drawn.check(catalog, user, session["tree"], session["seed"],
                           session.get("number"))
    except drawn.DrawnError:
        return None


def _brief(found):
    """What the model is told of a check: the verdict, not every number."""
    if found is None:
        return None
    return {"state": found["state"], "problems": found["problems"],
            "empty_places": found["empty"], "formula": found["formula"],
            "rank": found["rank"], "seed": found["seed"], "weights": found["weights"],
            "ranks": {path: one["rank"] for path, one in found["nodes"].items()}}


# --- the tools ---------------------------------------------------------------


def schemas():
    """Every tool as (name, description, JSON schema), for providers.converse()."""
    return [(name, prompts.VISUAL_TOOLS[name],
             {"type": "object", "properties": spec[0], "required": spec[1]})
            for name, spec in SPECS.items()]


class Toolbox:
    """One chat message's tools: the session as they change it, and what the
    page must do once the reply is back. `specs` and `done` are the tools it
    knows and how each is said once run -- compare.py's box is this one with
    its own."""

    specs = SPECS
    done = prompts.VISUAL_TOOL_DONE

    def __init__(self, catalog, user, stage, session, shared_datasets=None):
        self.catalog = catalog
        self.user = user
        self.stage = stage if stage in STAGES else "drawing"
        self.session = session_of(session)
        self.shared_datasets = shared_datasets or (lambda: [])
        self.actions = []
        self.steps = []
        self.changed = False

    def run(self, name, arguments):
        """Run one tool call. -> its result; a refusal is {"error": ...}."""
        arguments = arguments if isinstance(arguments, dict) else {}
        try:
            if name not in self.specs:
                raise VisualError("there is no tool called %s" % name)
            result = getattr(self, "_" + name)(**arguments)
        except (VisualError, drawn.DrawnError, blending.BlendError) as error:
            result = {"error": str(error)}
        except TypeError as error:                  # an argument the tool does not take
            result = {"error": "bad arguments: %s" % error}
        if "error" in result:
            summary = prompts.TOOL_FAILED.format(tool=name, error=result["error"])
        else:
            self.changed = self.changed or self.specs[name][2]
            summary = self.done[name].format(**result)
        self.steps.append({"tool": name, "arguments": arguments,
                           "ok": "error" not in result, "summary": summary})
        return result

    def outcome(self):
        """-> {session, actions, steps, check}: the drawing and its reading."""
        return {"session": self.session, "actions": list(self.actions),
                "steps": self.steps,
                "check": check_of(self.catalog, self.user, self.session)}

    # --- helpers ---

    def _lora(self, key):
        return blending.own(self.catalog, self.user, key)

    def _next_weight(self):
        used = {node["weight"] for _, node in drawn.walk(self.session["tree"])
                if node is not None and "lora" in node}
        return next((one for one in VARIABLES if one not in used), VARIABLES[0])

    def _edit(self, tree):
        drawn.walk(tree)                    # a tool never leaves a drawing that is not one
        self.session["tree"] = tree

    # --- reading ---

    def _list_my_loras(self):
        rows = blending.mine(self.catalog, self.user)
        return {"count": len(rows), "loras": [blending.describe(row) for row in rows]}

    def _show_drawing(self):
        found = check_of(self.catalog, self.user, self.session)
        weights = (found or {}).get("weights")
        return {"places": places(self.session["tree"], _names(self.catalog, self.user), weights),
                "check": _brief(found)}

    def _list_demo_datasets(self):
        found = [{"file": one["file"], "records": one["records"]}
                 for one in self.shared_datasets() if one["usable"]]
        return {"count": len(found), "datasets": found}

    # --- the drawing ---

    def _draw_blend(self, loras, fold=ROOT):
        if fold not in BINARY_OPS:
            raise VisualError("a fold is one of %s" % ", ".join(BINARY_OPS))
        if not isinstance(loras, list) or not loras:
            raise VisualError("name one or more of your LoRAs to draw a blend of")
        if len(loras) > drawn.MAX_LEAVES:
            raise VisualError("a drawing holds at most %d LoRAs" % drawn.MAX_LEAVES)
        rows = [self._lora(one) for one in loras]
        level = [{"lora": row["id"], "weight": VARIABLES[index % len(VARIABLES)]}
                 for index, row in enumerate(rows)]
        # Folded in pairs, bottom up; an odd one out goes up a level as it is.
        # One LoRA is a blend of one: it stands at the top on its own.
        while len(level) > 1:
            paired = [{"op": fold, "children": level[index:index + 2]}
                      for index in range(0, len(level) - 1, 2)]
            level = paired + (level[-1:] if len(level) % 2 else [])
        tree = level[0]
        self._edit(tree)
        names = {row["id"]: row["name"] for row in rows}
        return {"formula_text": ", ".join(names[one["lora"]] for _, one in drawn.walk(tree)
                                          if one is not None and "lora" in one),
                "places": places(tree, names)}

    def _place_lora(self, where, lora, weight=None):
        path = path_of(where)
        row = self._lora(lora)
        if weight is not None and weight not in VARIABLES:
            raise VisualError("a weight is one of %s..%s" % (VARIABLES[0], VARIABLES[-1]))
        tree = copy.deepcopy(self.session["tree"])
        parent, index, node = at(tree, path)
        if node is not None and "op" in node:
            empty = [i for i, child in enumerate(node["children"]) if child is None]
            if not empty:
                raise VisualError("the %s at %s has no empty place; name one of its sides"
                                  % (node["op"], _where(path)))
            parent, index, node = node, empty[0], None
            path = (path + "." if path else "") + str(index)
        chosen = weight or (node["weight"] if node is not None else self._next_weight())
        if parent is None:                  # the top: a blend of this one LoRA
            tree = {"lora": row["id"], "weight": chosen}
        else:
            parent["children"][index] = {"lora": row["id"], "weight": chosen}
        self._edit(tree)
        return {"name": row["name"], "where": _where(path), "weight": chosen}

    def _place_fold(self, where, op):
        if op not in BINARY_OPS:
            raise VisualError("a fold is one of %s" % ", ".join(BINARY_OPS))
        path = path_of(where)
        tree = copy.deepcopy(self.session["tree"])
        parent, index, node = at(tree, path)
        if node is not None and "op" in node:
            node["op"] = op
        else:
            made = {"op": op, "children": [node, None]}
            if parent is None:
                tree = made
            else:
                parent["children"][index] = made
        self._edit(tree)
        return {"op": op, "where": _where(path)}

    def _set_weight(self, where, weight):
        if weight not in VARIABLES:
            raise VisualError("a weight is one of %s..%s" % (VARIABLES[0], VARIABLES[-1]))
        path = path_of(where)
        tree = copy.deepcopy(self.session["tree"])
        _, _, node = at(tree, path)
        if node is None or "lora" not in node:
            raise VisualError("there is no LoRA at %s to weight" % _where(path))
        node["weight"] = weight
        self._edit(tree)
        return {"name": _names(self.catalog, self.user).get(node["lora"], "the LoRA"),
                "where": _where(path), "weight": weight}

    def _clear_place(self, where):
        path = path_of(where)
        tree = copy.deepcopy(self.session["tree"])
        parent, index, _ = at(tree, path)
        if parent is None:                  # the top: nothing left at all
            tree = None
        else:
            parent["children"][index] = None
        self._edit(tree)
        return {"where": _where(path)}

    def _swap_sides(self, where):
        path = path_of(where)
        tree = copy.deepcopy(self.session["tree"])
        _, _, node = at(tree, path)
        if node is None or "op" not in node:
            raise VisualError("there is no fold at %s to swap" % _where(path))
        node["children"].reverse()
        self._edit(tree)
        return {"where": _where(path)}

    def _new_weights(self, seed=None):
        found = drawn.draw(seed, self.session.get("number"))
        self.session["seed"] = found["seed"]
        return found

    def _start_over(self):
        self.session["tree"] = empty_tree()
        self.session["number"] = drawn.NUMBER       # no longer a searched blend's
        return {}

    # --- the test ---

    def _set_test_questions(self, file=None, lora=None, count=None):
        if file is not None and lora is not None:
            raise VisualError("ask either a demo dataset or a LoRA's data, not both")
        said = None
        if file is not None:
            known = {one["file"] for one in self.shared_datasets() if one["usable"]}
            if file not in known:
                raise VisualError("there is no demo dataset %r; there are: %s"
                                  % (file, ", ".join(sorted(known)) or "none"))
            self.session["questions"], said = {"file": file}, file
        elif lora is not None:
            row = self._lora(lora)
            self.session["questions"] = {"lora": row["id"]}
            said = "%s's training data" % row["name"]
        if count is not None:
            if isinstance(count, bool) or not isinstance(count, int) \
                    or not 1 <= count <= drawn.MAX_QUESTIONS:
                raise VisualError("count is from 1 to %d" % drawn.MAX_QUESTIONS)
            self.session["count"] = count
        if said is None and count is None:
            raise VisualError("say which questions (a demo dataset or a LoRA's data), or how "
                              "many")
        text = source_of(self.catalog, self.user, self.session["questions"]) \
            or "the questions still to choose"
        return {"questions_text": "%d question(s) from %s" % (self.session["count"], text)}

    def _set_practice_run(self, on):
        self.session["mock"] = bool(on)
        return {"state": "on" if on else "off"}

    def _start_test(self):
        if self.stage == "testing":
            raise VisualError("a test is running; wait for it first")
        found = check_of(self.catalog, self.user, self.session)
        if found is None or found["state"] != "ok":
            raise VisualError("; ".join((found or {}).get("problems") or [])
                              or prompts.VISUAL_WORDS["incomplete"])
        if self.session["questions"] is None:
            raise VisualError("choose the questions to test on first")
        self.actions.append({"type": "test"})
        return {}


def source_of(catalog, user, questions):
    """The session's questions in words, or None when none are chosen."""
    if not questions:
        return None
    if questions.get("file"):
        return questions["file"]
    if questions.get("given"):
        return questions["given"]
    row = catalog.get(questions["lora"])
    if row is None or row["owner"] != user["name"]:
        return None
    return "%s's training data" % row["name"]


# --- the steps -----------------------------------------------------------------


def intro(catalog, user, session, choice=None):
    """The welcome, with the person's LoRAs."""
    rows = blending.mine(catalog, user)
    facts = {"loras": [{"name": row["name"], "rank": row["rank"],
                        "base_model": row["base_model"]} for row in rows],
             "drawing": places(session["tree"], {row["id"]: row["name"] for row in rows})}
    example = (prompts.VISUAL_FALLBACK_INTRO_EXAMPLE.format(first=rows[0]["name"],
                                                            second=rows[1]["name"])
               if len(rows) > 1 else "")
    fallback = prompts.VISUAL_FALLBACK_INTRO.format(
        example=example, none="" if rows else prompts.VISUAL_FALLBACK_INTRO_NONE)
    return {"message": agent.say("visual_intro", facts, fallback, choice),
            "loras": [blending.describe(row) for row in rows]}


def _commands(message):
    """The few plain requests read without a model -- narrow on purpose: an
    unrecognised request is no call at all, never a guess."""
    text = " ".join(message.lower().split()).strip(" .!?")
    if text in ("start over", "clear", "clear it", "clear the drawing", "reset"):
        return [("start_over", {})]
    if text in ("new weights", "reroll", "reroll the weights", "new seed"):
        return [("new_weights", {})]
    if text in ("test", "test it", "start the test", "test the blend", "run the test"):
        return [("start_test", {})]
    if text in ("practice run on", "practice run", "mock"):
        return [("set_practice_run", {"on": True})]
    if text in ("practice run off", "real run"):
        return [("set_practice_run", {"on": False})]
    return []


def chat(catalog, user, message, stage, session, context=None, choice=None, history=None,
         shared_datasets=None):
    """A typed message: answered, and acted on with the tools.
    -> {message, session, actions, steps, check}."""
    box = Toolbox(catalog, user, stage, session, shared_datasets)
    names = _names(catalog, user)
    found = check_of(catalog, user, box.session)
    facts = {"stage": box.stage,
             "step_instructions": prompts.STEP_INSTRUCTIONS.get("visual_" + box.stage, ""),
             "drawing": {"places": places(box.session["tree"], names,
                                          (found or {}).get("weights"))},
             "check": _brief(found),
             "loras": [{"id": key, "name": value} for key, value in names.items()],
             "test": {"questions": source_of(catalog, user, box.session["questions"]),
                      "count": box.session["count"], "practice_run": box.session["mock"]},
             "result": (context or {}).get("result")}
    return talk(box, "visual_chat", facts, message, schemas(), _commands,
                prompts.VISUAL_FALLBACK_CHAT, choice, history)


def talk(box, step, facts, message, tools, commands, fallback, choice=None, history=None):
    """One typed message, answered by the model with `tools` run by `box`
    (up to agent.MAX_ROUNDS of calls), or without a model: `commands(message)`
    read as the few plain requests it knows, and the tools' own summary said.
    The comparison guide (compare.py) talks through this too, with its own
    box, step and tools. -> dict(box.outcome(), message=...)."""
    history = list(history or [])[-settings.HISTORY_TURNS:]
    reason = None
    try:
        resolved = providers.resolve(choice)
        if resolved["kind"] == "scripted":
            raise providers.ProviderError(None)
        exchange = [{"role": "user", "content": "FACTS:\n%s\n\nThe person wrote:\n%s" % (
            json.dumps(facts, indent=1, ensure_ascii=False, default=str), message)}]
        text, model = "", None
        for _ in range(agent.MAX_ROUNDS):
            reply = providers.converse(resolved, prompts.system(step), history,
                                       exchange, tools)
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
            text = ""
        spoken = {"text": text or agent._summary(box) or fallback,
                  "by": "%s · %s" % (resolved["label"], model), "fallback": False,
                  "note": None}
        return dict(box.outcome(), message=spoken)
    except providers.ProviderError as error:
        reason = error.args[0] if error.args else None
    note = prompts.FALLBACK_NOTE.format(reason=reason) if reason else None
    if not box.steps:
        for name, arguments in commands(message):
            box.run(name, arguments)
    text = agent._summary(box) or fallback
    return dict(box.outcome(), message={"text": text, "by": "built-in wording",
                                        "fallback": True, "note": note})


def plan(catalog, user, session):
    """The POST /blends/test body for the session, read back. -> {test, message,
    check}. The page sends the body itself -- after putting the text of a
    dataset it holds ({"given"}) in the place of its name."""
    found = check_of(catalog, user, session)
    if found is None or found["state"] != "ok":
        raise VisualError("; ".join((found or {}).get("problems") or [])
                          or prompts.VISUAL_WORDS["incomplete"])
    questions = session["questions"]
    if questions is None:
        raise VisualError("choose the questions to test on first")
    source = source_of(catalog, user, questions)
    if source is None:
        raise VisualError("no LoRA of yours with id %s" % questions.get("lora"))
    wanted = {name: value for name, value in guide_defaults.job_settings().items()
              if name in drawn.TEST_SETTINGS}
    mock = session["mock"] or found["mock"]
    body = {"tree": session["tree"], "seed": found["seed"], "count": session["count"],
            "dataset": questions, "mock": mock}
    if found["number"] != drawn.NUMBER:
        body["number"] = found["number"]
    if wanted:
        body["settings"] = wanted
    against = sorted({one["name"] for one in found["slots"].values()})
    text = prompts.VISUAL_FALLBACK_CONFIRM.format(
        formula=found["formula"], count=session["count"], source=source,
        against=", ".join("**%s**" % name for name in against),
        mock=prompts.VISUAL_FALLBACK_CONFIRM_MOCK if mock else "")
    return {"test": body, "check": found, "source": source,
            "message": {"text": text, "by": "built-in wording", "fallback": False,
                        "note": None}}


def debrief(registry, catalog, user, row, choice=None, history=None):
    """How the test went, from release.verification_outcome()."""
    facts = release.verification_outcome(registry, catalog, user, row)
    report = facts.get("report")
    if report is None:
        fallback = prompts.VISUAL_FALLBACK_FAILED.format(
            status=facts["status"], error=": " + facts["error"] if facts.get("error") else "")
    else:
        against = report["against"]
        lines = ["**%s** alone: %s — your blend won %d, tied %d, lost %d (%s)"
                 % (one["lora"], agent._score(one["mean"]), one["wins"], one["ties"],
                    one["losses"], one["blend_is"]) for one in against]
        fallback = prompts.VISUAL_FALLBACK_DEBRIEF.format(
            blend=agent._score(report["blend_mean"]), questions=report["questions"],
            source=report["questions_from"], lines=agent._bullets(lines),
            verdict=(prompts.VISUAL_FALLBACK_BETTER
                     if against and all(one["blend_is"] == "better" for one in against)
                     else prompts.VISUAL_FALLBACK_MIXED),
            mock=prompts.FALLBACK_TEST_START_MOCK if report["mock"] else "")
    return {"message": agent.say("visual_debrief", facts, fallback, choice, history),
            "result": facts}
