"""
evaluate.py - A job's answers, graded again after the search that produced them.

The API's half of `python main.py --evaluate`: what a form needs to offer
(`choices`), what a request may ask for (`options_for`), and the gep_lora/core/pipeline/main.py flags
the worker adds for it (`arguments`). The grading itself -- which steps, and
whether fitness may follow -- is gep_lora/core/pipeline/main.py's, and is not decided again here.

An evaluation is **the job, queued again** rather than a thing of its own: it
runs gep_lora/core/pipeline/main.py against the job's database, the way the search did, so it takes
the job's place in the queue, its log and its cancel. What it can be asked of
is a job that has stopped with answers in it -- a finished search, or one that
was stopped, cancelled or failed after its first scripts ran.

What it may change is **where the judge is**, never what it grades by. A
fitness is only comparable with the others when one rubric earned it, so the
evaluator is the sweep's own EVALUATOR; JUDGE_BACKEND, JUDGE_MODEL and
JUDGE_BASE_URL are the three a stopped job most often needs moved (the endpoint
was down, or is somewhere else now), and they go into the sweep as
`--set NAME=VALUE`, so it still records what graded it. Another rubric is a
question for a verification, which leaves the sweep alone.
"""

import json

from gep_lora.core import evaluators
from gep_lora.service import results
from gep_lora.core.storage import store

# The settings an evaluation may change, and the request field for each.
JUDGE_SETTINGS = (("judge_backend", "JUDGE_BACKEND"),
                  ("judge_model", "JUDGE_MODEL"),
                  ("judge_base_url", "JUDGE_BASE_URL"))


class EvaluateError(ValueError):
    """What is wrong with an evaluation request, in words for whoever sent it."""


def choices(db_path, run_id):
    """What an evaluation form needs, out of the job's own sweep.

    The evaluator it will grade with and whether that asks a judge, where the
    judge is now, and how many answers there are -- all of them, and the ones
    still without a score -- so a form can say what "evaluate" would do before
    it is asked.
    """
    conn = results.connect(db_path)
    try:
        conf = store.get_settings(conn, run_id)
        name = conf.get("EVALUATOR")
        try:
            asks = evaluators.get(name).asks_judge(conf)
        except (KeyError, SystemExit, ValueError):
            asks = None
        tested = store.test_results(conn, run_id)
        return {
            "evaluator": name,
            "asks_judge": asks,
            "backends": list(evaluators.BACKENDS),
            "defaults": {field: conf.get(setting) for field, setting in JUDGE_SETTINGS},
            "answers": len(store.exchanges_to_score(conn, run_id, True)),
            "unscored": len(store.exchanges_to_score(conn, run_id, False)),
            "tested": len(tested),
            "tested_unscored": len(store.test_results_to_score(conn, run_id)),
        }
    finally:
        conn.close()


def _text(body, name):
    value = body.get(name)
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise EvaluateError("%s must be a string" % name)
    return value.strip() or None


def options_for(body, offered):
    """A request, checked against what this sweep offers. -> options.

    {"force": bool, "settings": {NAME: value}} -- only the judge settings that
    differ from what the sweep already holds, so an evaluation that moves
    nothing writes nothing into it.
    """
    body = body or {}
    if not isinstance(body, dict):
        raise EvaluateError("the body must be an object")
    unknown = sorted(set(body) - {"force"} - {field for field, _ in JUDGE_SETTINGS})
    if unknown:
        raise EvaluateError("unknown field(s): %s; an evaluation takes force, %s"
                            % (", ".join(unknown),
                               ", ".join(field for field, _ in JUDGE_SETTINGS)))
    if not offered["answers"] and not offered["tested"]:
        raise EvaluateError("this job holds no answers yet -- its search stopped "
                            "before any script ran. Resume it instead.")
    force = body.get("force", False)
    if not isinstance(force, bool):
        raise EvaluateError("force must be true or false")
    changed = {}
    for field, setting in JUDGE_SETTINGS:
        value = _text(body, field)
        if value is None or value == offered["defaults"].get(field):
            continue
        if field == "judge_backend" and value not in offered["backends"]:
            raise EvaluateError("unknown judge backend %r; there are: %s"
                                % (value, ", ".join(offered["backends"])))
        changed[setting] = value
    return {"force": force, "settings": changed}


def arguments(options):
    """The gep_lora/core/pipeline/main.py flags one evaluation adds to the job's command line."""
    argv = ["--evaluate"]
    if options.get("force"):
        argv.append("--force")
    for name, value in sorted((options.get("settings") or {}).items()):
        argv += ["--set", "%s=%s" % (name, json.dumps(value))]
    return argv
