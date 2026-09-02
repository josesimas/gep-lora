"""
evaluators/llm_judge.py - A judge model, grading the answer on its own merits.

The default, and what a sweep created before EVALUATOR existed is read back
under. A *different* model from the blended one that produced the answers is
shown the question and the answer, and grades it against
JUDGE_SYSTEM_PROMPT -- which is therefore the criterion the whole search
optimises toward.

The other two judging evaluators beside it are this one plus a bigger prompt:
llm_judge_reference and llm_judge_baseline both call prepare() and, when they
have nothing extra to show the judge, score() from here.
"""

from evaluators import common


def prepare(conf, pending, context=None):
    settings = common.endpoint_settings(conf)
    if common.needs_grading(pending) and not settings["model"]:
        settings["model"] = common.discover_model(
            settings["base_url"], settings["api_key"], settings["timeout"])
    label = settings["model"] or "llm_judge"
    note = ("judge: %s at %s" % (settings["model"], settings["base_url"])
            if common.needs_grading(pending) else
            "judge: not contacted -- no answer needs grading")
    return common.Prepared(conf, label, settings=settings, notes=[note])


def score(item, prepared):
    prompt = prepared.conf.get("JUDGE_SYSTEM_PROMPT")
    return common.ask_judge(
        prompt, "QUESTION:\n%s\n\nANSWER:\n%s" % (item["question"], item["answer"]),
        prepared.settings)


common.register(common.Evaluator(
    "llm_judge",
    "a judge model grades each answer on its own merits (JUDGE_SYSTEM_PROMPT)",
    prepare, score, needs_endpoint=True,
))
