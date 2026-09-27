"""
compare.py - The blend comparison: two blends side by side, with the guide.

The page (async_api/blend_comparison.html) holds two drawings, blend A and
blend B -- each opened from one of the person's jobs (a searched blend or a
drawn one) or drawn from nothing -- and tests both on the same questions.
This module is its guide, and it is the visual guide twice rather than a
third pipeline:

  * **A side is a visual guide's drawing.** Each side is visual.py's session
    (tree, seed, number) plus where it was opened from, and every edit is
    visual.Toolbox's own, run on that side (`Toolbox._on`). A searched blend
    opened here keeps its sweep's seed and its own number (drawn.opened), so
    its weights are the ones it was scored with.
  * **Testing both is two drawn tests.** `plan()` reads back one POST
    /blends/test body per side through visual.plan(), on the same questions,
    count and practice-run switch; the page sends both. Each is a sweep of one
    and a verification (drawn.create), read back with GET /verifications/{id}.
  * **The comparison is computed, never the model's.** `outcome()` is each
    side's release.verification_outcome() and `head_to_head()`, the two
    blends' own answers matched question by question, won, tied and lost, with
    the same exact sign test the verification uses against each LoRA. The
    model only phrases it (`debrief()`), with prompts.COMPARE_FALLBACK_* when
    there is no model.

The session:

    {"blends": {"A": {"tree", "seed", "number", "from"}, "B": {...}},
     "questions": ..., "count": n, "mock": bool}       visual.py's, shared

`from` is {"job", "individual", "label", "chromosome"} for a blend opened
from a job -- the chromosome it had there, so the page can say it has been
edited since -- or None for one drawn here.
"""

import copy

from async_api import drawn
from async_api import registry as reg
from async_api import verify
from async_api_agent import agent
from async_api_agent import blending
from async_api_agent import prompts
from async_api_agent import release
from async_api_agent import visual
from async_api.drawn import ROOT
from search.generate_population import BINARY_OPS, VARIABLES
from testing.evaluate_chromosome_against_loras import sign_test

SIDES = ("A", "B")
STAGES = visual.STAGES

# How many of a job's blends the model is shown by list_my_blends.
LISTED_BLENDS = 5

_INT = {"type": "integer"}
_SIDE = {"type": "string", "enum": list(SIDES), "description": 'which blend: "A" or "B"'}
_PATH = visual._PATH
_LORA = visual._LORA

# name -> (JSON schema properties, required, changes a drawing or the test)
SPECS = {
    "list_my_loras": ({}, [], False),
    "list_my_blends": ({}, [], False),
    "show_blends": ({}, [], False),
    "open_blend": ({"blend": _SIDE, "job": dict(_INT, description="one of their jobs"),
                    "individual": dict(_INT, description="the blend's number in that job; "
                                                         "its best when left out")},
                   ["blend", "job"], True),
    "copy_blend": ({"source": _SIDE, "target": _SIDE}, ["source", "target"], True),
    "swap_blends": ({}, [], True),
    "draw_blend": ({"blend": _SIDE, "loras": {"type": "array", "items": _LORA},
                    "fold": {"type": "string", "enum": list(BINARY_OPS)}},
                   ["blend", "loras"], True),
    "place_lora": ({"blend": _SIDE, "where": _PATH, "lora": _LORA,
                    "weight": {"type": "string", "enum": list(VARIABLES)}},
                   ["blend", "where", "lora"], True),
    "place_fold": ({"blend": _SIDE, "where": _PATH,
                    "op": {"type": "string", "enum": list(BINARY_OPS)}},
                   ["blend", "where", "op"], True),
    "set_weight": ({"blend": _SIDE, "where": _PATH,
                    "weight": {"type": "string", "enum": list(VARIABLES)}},
                   ["blend", "where", "weight"], True),
    "clear_place": ({"blend": _SIDE, "where": _PATH}, ["blend", "where"], True),
    "swap_sides": ({"blend": _SIDE, "where": _PATH}, ["blend", "where"], True),
    "new_weights": ({"blend": _SIDE, "seed": dict(_INT, description="a seed of their choosing")},
                    ["blend"], True),
    "start_over": ({"blend": _SIDE}, ["blend"], True),
    "list_demo_datasets": ({}, [], False),
    "set_test_questions": ({"file": {"type": "string", "description": "a demo dataset"},
                            "lora": _LORA,
                            "count": dict(_INT, description="how many questions")},
                           [], True),
    "set_practice_run": ({"on": {"type": "boolean"}}, ["on"], True),
    "start_test": ({}, [], False),
}


class CompareError(visual.VisualError):
    """Why a tool or a step refused, in words the model (or the page) passes on."""


def empty_side():
    return {"tree": visual.empty_tree(), "seed": None, "number": drawn.NUMBER, "from": None}


def _from(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise CompareError("from is where a blend was opened: {job, individual}")
    job, individual = value.get("job"), value.get("individual")
    if any(isinstance(one, bool) or not isinstance(one, int) for one in (job, individual)):
        raise CompareError("from is where a blend was opened: {job, individual}")
    label, chromosome = value.get("label"), value.get("chromosome")
    return {"job": job, "individual": individual,
            "label": label if isinstance(label, str) else None,
            "chromosome": chromosome if isinstance(chromosome, str) else None}


def session_of(raw):
    """The page's session, checked, with the defaults filled. Raises CompareError."""
    raw = raw if isinstance(raw, dict) else {}
    try:
        shared = visual.session_of({name: raw[name] for name in ("questions", "count", "mock")
                                    if name in raw})
    except visual.VisualError as error:
        raise CompareError(str(error))
    blends = raw.get("blends") if isinstance(raw.get("blends"), dict) else {}
    out = {"blends": {}, "questions": shared["questions"], "count": shared["count"],
           "mock": shared["mock"]}
    for side in SIDES:
        one = blends.get(side) if isinstance(blends.get(side), dict) else {}
        try:
            drawing = visual.session_of({name: one[name] for name in ("tree", "seed", "number")
                                         if name in one})
            out["blends"][side] = {"tree": drawing["tree"], "seed": drawing["seed"],
                                   "number": drawing["number"], "from": _from(one.get("from"))}
        except visual.VisualError as error:
            raise CompareError("blend %s: %s" % (side, error))
    return out


def view(pair, side):
    """One side as visual.py's session: its drawing and the shared test."""
    one = pair["blends"][side]
    return {"tree": copy.deepcopy(one["tree"]), "seed": one["seed"], "number": one["number"],
            "questions": pair["questions"], "count": pair["count"], "mock": pair["mock"]}


def side_of(blend):
    """"A", "a", "blend b" -> the side. Raises CompareError."""
    text = " ".join(str(blend if blend is not None else "").split()).upper()
    text = text[len("BLEND "):] if text.startswith("BLEND ") else text
    if text not in SIDES:
        raise CompareError('a blend is "A" or "B"')
    return text


def _brief_side(catalog, user, pair, side, names):
    found = visual.check_of(catalog, user, view(pair, side))
    one = pair["blends"][side]
    return {"from": one["from"],
            "edited_since_opened": bool(one["from"] and found and found["chromosome"]
                                        and found["chromosome"] != one["from"]["chromosome"]),
            "places": visual.places(one["tree"], names, (found or {}).get("weights")),
            "check": visual._brief(found)}


# --- the tools ---------------------------------------------------------------


def schemas():
    """Every tool as (name, description, JSON schema), for providers.converse()."""
    return [(name, prompts.COMPARE_TOOLS[name],
             {"type": "object", "properties": spec[0], "required": spec[1]})
            for name, spec in SPECS.items()]


class Toolbox(visual.Toolbox):
    """visual.Toolbox over two drawings: `pair` is the whole session, and a
    tool on one side runs the visual guide's own tool on that side's view."""

    specs = SPECS
    done = prompts.COMPARE_TOOL_DONE

    def __init__(self, registry, catalog, user, stage, session, shared_datasets=None):
        self.registry = registry
        self.pair = session_of(session)
        super().__init__(catalog, user, stage, view(self.pair, "A"), shared_datasets)

    def outcome(self):
        """-> {session, actions, steps, checks}: both drawings and their readings."""
        return {"session": self.pair, "actions": list(self.actions), "steps": self.steps,
                "checks": {side: visual.check_of(self.catalog, self.user, view(self.pair, side))
                           for side in SIDES}}

    def _on(self, blend, method, **arguments):
        """The visual guide's `method`, on one side. -> its result, and which."""
        side = side_of(blend)
        self.session = view(self.pair, side)
        result = getattr(visual.Toolbox, method)(self, **arguments)
        one = self.pair["blends"][side]
        one.update(tree=self.session["tree"], seed=self.session["seed"],
                   number=self.session["number"])
        return dict(result, blend=side)

    def _shared(self, method, **arguments):
        """The visual guide's `method` on the test both sides share."""
        self.session = view(self.pair, "A")
        result = getattr(visual.Toolbox, method)(self, **arguments)
        for name in ("questions", "count", "mock"):
            self.pair[name] = self.session[name]
        return result

    # --- reading ---

    def _list_my_blends(self):
        jobs = drawn.sources(self.registry, self.catalog, self.user)
        return {"count": len(jobs), "jobs": [
            {"job": one["job"], "label": one["label"],
             "kind": "drawn by hand" if one["task"] == reg.BLEND else "search",
             "status": one["status"], "blends": len(one["blends"]),
             "top": [{"individual": blend["number"], "formula": blend["formula"],
                      "score": blend["quality"], "best": blend["is_best"],
                      "can_be_built": blend["state"] != "BAD"}
                     for blend in one["blends"][:LISTED_BLENDS]]} for one in jobs]}

    def _show_blends(self):
        names = visual._names(self.catalog, self.user)
        return {side: _brief_side(self.catalog, self.user, self.pair, side, names)
                for side in SIDES}

    # --- the two drawings ---

    def _open_blend(self, blend, job, individual=None):
        side = side_of(blend)
        row = None
        if isinstance(job, int) and not isinstance(job, bool):
            row = self.registry.job(job, self.user["id"])
        if row is None or row["status"] == reg.DELETED:
            raise CompareError("there is no job %s of yours to open" % (job,))
        found = drawn.opened(self.registry, self.catalog, self.user, row, individual)
        self.pair["blends"][side] = {
            "tree": found["tree"], "seed": found["seed"], "number": found["number"],
            "from": {"job": found["job"], "individual": found["number"],
                     "label": found["label"], "chromosome": found["chromosome"]}}
        return {"blend": side, "job": found["job"], "individual": found["number"],
                "label": found["label"]}

    def _copy_blend(self, source, target):
        source, target = side_of(source), side_of(target)
        if source == target:
            raise CompareError("copy one blend onto the other, A onto B or B onto A")
        self.pair["blends"][target] = copy.deepcopy(self.pair["blends"][source])
        return {"source": source, "target": target}

    def _swap_blends(self):
        blends = self.pair["blends"]
        blends["A"], blends["B"] = blends["B"], blends["A"]
        return {}

    def _draw_blend(self, blend, loras, fold=ROOT):
        result = self._on(blend, "_draw_blend", loras=loras, fold=fold)
        self.pair["blends"][result["blend"]]["from"] = None
        return result

    def _place_lora(self, blend, where, lora, weight=None):
        return self._on(blend, "_place_lora", where=where, lora=lora, weight=weight)

    def _place_fold(self, blend, where, op):
        return self._on(blend, "_place_fold", where=where, op=op)

    def _set_weight(self, blend, where, weight):
        return self._on(blend, "_set_weight", where=where, weight=weight)

    def _clear_place(self, blend, where):
        return self._on(blend, "_clear_place", where=where)

    def _swap_sides(self, blend, where):
        return self._on(blend, "_swap_sides", where=where)

    def _new_weights(self, blend, seed=None):
        return self._on(blend, "_new_weights", seed=seed)

    def _start_over(self, blend):
        result = self._on(blend, "_start_over")
        self.pair["blends"][result["blend"]]["from"] = None
        return result

    # --- the test ---

    def _list_demo_datasets(self):
        return self._shared("_list_demo_datasets")

    def _set_test_questions(self, file=None, lora=None, count=None):
        return self._shared("_set_test_questions", file=file, lora=lora, count=count)

    def _set_practice_run(self, on):
        return self._shared("_set_practice_run", on=on)

    def _start_test(self):
        if self.stage == "testing":
            raise CompareError("the tests are running; wait for them first")
        for side in SIDES:
            found = visual.check_of(self.catalog, self.user, view(self.pair, side))
            if found is None or found["state"] != "ok":
                raise CompareError("blend %s: %s" % (side, "; ".join(
                    (found or {}).get("problems") or []) or prompts.VISUAL_WORDS["incomplete"]))
        if self.pair["questions"] is None:
            raise CompareError("choose the questions to test on first")
        self.actions.append({"type": "test"})
        return {}


# --- the steps -----------------------------------------------------------------


def intro(registry, catalog, user, session, choice=None):
    """The welcome, with the person's LoRAs and how many blends they could open."""
    rows = blending.mine(catalog, user)
    jobs = drawn.sources(registry, catalog, user)
    names = {row["id"]: row["name"] for row in rows}
    facts = {"loras": [{"name": row["name"], "rank": row["rank"],
                        "base_model": row["base_model"]} for row in rows],
             "jobs_with_blends": len(jobs),
             "blends": {side: visual.places(session["blends"][side]["tree"], names)
                        for side in SIDES}}
    fallback = prompts.COMPARE_FALLBACK_INTRO.format(
        jobs=prompts.COMPARE_FALLBACK_INTRO_JOBS.format(count=len(jobs)) if jobs
        else prompts.COMPARE_FALLBACK_INTRO_NO_JOBS,
        none="" if rows else prompts.VISUAL_FALLBACK_INTRO_NONE)
    return {"message": agent.say("compare_intro", facts, fallback, choice),
            "loras": [blending.describe(row) for row in rows]}


def _commands(message):
    """The few plain requests read without a model -- narrow on purpose, as
    visual.py's are: an unrecognised request is no call at all."""
    text = " ".join(message.lower().split()).strip(" .!?")
    if text in ("test", "test them", "test both", "compare", "compare them",
                "start the test", "run the test"):
        return [("start_test", {})]
    if text in ("swap", "swap them", "swap the blends", "swap a and b"):
        return [("swap_blends", {})]
    if text in ("copy a to b", "copy a onto b"):
        return [("copy_blend", {"source": "A", "target": "B"})]
    if text in ("copy b to a", "copy b onto a"):
        return [("copy_blend", {"source": "B", "target": "A"})]
    if text in ("practice run on", "practice run", "mock"):
        return [("set_practice_run", {"on": True})]
    if text in ("practice run off", "real run"):
        return [("set_practice_run", {"on": False})]
    return []


def chat(registry, catalog, user, message, stage, session, context=None, choice=None,
         history=None, shared_datasets=None):
    """A typed message: answered, and acted on with the tools.
    -> {message, session, actions, steps, checks}."""
    box = Toolbox(registry, catalog, user, stage, session, shared_datasets)
    names = visual._names(catalog, user)
    facts = {"stage": box.stage,
             "step_instructions": prompts.STEP_INSTRUCTIONS.get("compare_" + box.stage, ""),
             "blends": {side: _brief_side(catalog, user, box.pair, side, names)
                        for side in SIDES},
             "loras": [{"id": key, "name": value} for key, value in names.items()],
             "test": {"questions": visual.source_of(catalog, user, box.pair["questions"]),
                      "count": box.pair["count"], "practice_run": box.pair["mock"]},
             "result": (context or {}).get("result")}
    return visual.talk(box, "compare_chat", facts, message, schemas(), _commands,
                       prompts.COMPARE_FALLBACK_CHAT, choice, history)


def plan(catalog, user, session):
    """The two POST /blends/test bodies, one per side, on the same questions,
    read back. -> {tests: {A, B}, checks: {A, B}, source, message}. The page
    sends both -- after putting the text of a dataset it holds ({"given"}) in
    the place of its name, as the visual guide's page does."""
    tests, checks, source = {}, {}, None
    for side in SIDES:
        try:
            planned = visual.plan(catalog, user, view(session, side))
        except visual.VisualError as error:
            raise CompareError("blend %s: %s" % (side, error))
        body = planned["test"]
        body["label"] = ("compare %s: %s" % (side, planned["check"]["formula"]
                                             or planned["check"]["chromosome"]))[:80]
        tests[side], checks[side], source = body, planned["check"], planned["source"]
    mock = any(body["mock"] for body in tests.values())
    text = prompts.COMPARE_FALLBACK_CONFIRM.format(
        a=checks["A"]["formula"], b=checks["B"]["formula"], count=session["count"],
        source=source, mock=prompts.VISUAL_FALLBACK_CONFIRM_MOCK if mock else "")
    return {"tests": tests, "checks": checks, "source": source,
            "message": {"text": text, "by": "built-in wording", "fallback": False,
                        "note": None}}


# --- the two tests, side by side ---------------------------------------------------


def _question(text):
    return " ".join(str(text or "").split()).lower()


def _mean(values):
    kept = [value for value in values if value is not None]
    return sum(kept) / len(kept) if kept else None


def head_to_head(report_a, report_b):
    """Blend A against blend B, question by question, from the two
    verifications' own reports. -> {questions, only_a, only_b, a_mean, b_mean,
    a_wins, ties, b_wins, delta, p, a_is, same_grader, graders, rows}.

    Answers are paired by their question (the same question asked twice is
    paired in order), so two tests on the same questions compare exactly and
    two on different ones compare only what they share. Both scores must be
    there for a question to count as won, tied or lost; `p` is the exact sign
    test over the ones that were not ties. Scores given by different graders
    are compared all the same, and `same_grader` says so."""
    answers_a = ((report_a or {}).get("answers") or {}).get("blend") or []
    answers_b = ((report_b or {}).get("answers") or {}).get("blend") or []
    waiting = {}
    for one in answers_b:
        waiting.setdefault(_question(one.get("question")), []).append(one)
    rows, pairs = [], []
    for one in answers_a:
        theirs = waiting.get(_question(one.get("question")))
        if not theirs:
            continue
        other = theirs.pop(0)
        pairs.append((one.get("quality"), other.get("quality")))
        rows.append({"question": one.get("question"),
                     "A": {key: one.get(key) for key in ("answer", "quality", "reason")},
                     "B": {key: other.get(key) for key in ("answer", "quality", "reason")}})
    a_wins = sum(1 for a, b in pairs if a is not None and b is not None and a > b)
    b_wins = sum(1 for a, b in pairs if a is not None and b is not None and a < b)
    ties = sum(1 for a, b in pairs if a is not None and b is not None and a == b)
    p = sign_test(a_wins, b_wins)
    graders = [{key: ((report or {}).get("meta") or {}).get(key)
                for key in ("evaluator", "judge")} for report in (report_a, report_b)]
    return {"questions": len(rows), "only_a": len(answers_a) - len(rows),
            "only_b": len(answers_b) - len(rows),
            "a_mean": _mean(a for a, _ in pairs), "b_mean": _mean(b for _, b in pairs),
            "a_wins": a_wins, "ties": ties, "b_wins": b_wins,
            "delta": _mean(a - b for a, b in pairs if a is not None and b is not None),
            "p": p,
            "a_is": ("better" if a_wins > b_wins and p < 0.05 else
                     "worse" if b_wins > a_wins and p < 0.05 else "no clear difference"),
            "same_grader": graders[0] == graders[1], "graders": graders, "rows": rows}


def outcome(registry, catalog, user, row_a, row_b):
    """What the two tests came to. -> {A, B, head_to_head}: each side's
    release.verification_outcome(), and head_to_head() once both have a
    report (None until then)."""
    reports = [verify.report(registry.verification_folder(registry.job(row["job_id"]), row["id"]))
               for row in (row_a, row_b)]
    return {"A": release.verification_outcome(registry, catalog, user, row_a),
            "B": release.verification_outcome(registry, catalog, user, row_b),
            "head_to_head": head_to_head(*reports) if all(reports) else None}


def debrief(registry, catalog, user, row_a, row_b, choice=None, history=None):
    """How the comparison went: outcome()'s facts, without the answers."""
    found = outcome(registry, catalog, user, row_a, row_b)
    facts = {side: found[side] for side in SIDES}
    duel = found["head_to_head"]
    facts["head_to_head"] = None if duel is None else {
        key: value for key, value in duel.items() if key != "rows"}
    if duel is None:
        failed = [side for side in SIDES if found[side]["report"] is None]
        fallback = prompts.COMPARE_FALLBACK_FAILED.format(
            which=" and ".join("blend %s (%s)" % (side, found[side]["status"])
                               for side in failed))
    else:
        verdict = {"better": prompts.COMPARE_FALLBACK_A_BETTER,
                   "worse": prompts.COMPARE_FALLBACK_B_BETTER}.get(
            duel["a_is"], prompts.COMPARE_FALLBACK_NO_DIFFERENCE)
        mock = any(found[side]["report"]["mock"] for side in SIDES)
        fallback = prompts.COMPARE_FALLBACK_DEBRIEF.format(
            a=agent._score(found["A"]["report"]["blend_mean"]),
            b=agent._score(found["B"]["report"]["blend_mean"]),
            questions=duel["questions"], source=found["A"]["report"]["questions_from"],
            a_wins=duel["a_wins"], ties=duel["ties"], b_wins=duel["b_wins"],
            verdict=verdict,
            graders="" if duel["same_grader"] else prompts.COMPARE_FALLBACK_GRADERS,
            mock=prompts.FALLBACK_TEST_START_MOCK if mock else "")
    return {"message": agent.say("compare_debrief", facts, fallback, choice, history),
            "result": found}
