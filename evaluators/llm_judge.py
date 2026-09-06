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


# How the judge is told to grade, with no reference to compare against. This is
# the rubric the whole search selects on, so it is worth tuning deliberately.
#
# It lives here rather than in settings.py because it is this evaluator's own
# text: nothing else reads it, and the module that sends it is the one place a
# reader looks to find out what "graded on its own merits" actually means.
# llm_judge_reference falls back to score() below for an item with no reference,
# so it grades against this too; panel keeps its own copy.
#
# The cost of the move: settings.py is snapshotted into every sweep, and this is
# not, so a sweep no longer records the rubric it was judged under. score()
# therefore still lets a sweep's stored value win -- a sweep created while this
# was a setting is re-scored against the prompt it actually ran with.
JUDGE_SYSTEM_PROMPT = """\
You are grading the quality of a single answer given by an AI planning coach.

You will be shown the QUESTION a user asked and the ANSWER the coach gave.
Judge the answer only, on how well it serves the person who asked.

Consider:
- Relevance: does it address what was actually asked?
- Usefulness: could the person act on it, or are they left stuck?
- Specificity: concrete and grounded rather than vague filler.
- Coherence: well formed and consistent, free of contradictions, repetition,
  broken grammar or nonsense.
- Appropriateness: sensible length and tone. Asking one focused clarifying
  question is fine when the request genuinely needs it; deflecting every
  request without helping is not.

Score from 0.0 to 1.0:
  1.0  excellent - directly useful, specific, clear
  0.7  good - helpful, minor weaknesses
  0.5  mixed - partly useful, vague or padded
  0.3  poor - barely addresses the question
  0.0  useless - incoherent, empty, or entirely off topic

Reply with JSON and nothing else, with the score FIRST:
{"quality": <number between 0 and 1>, "reason": "<at most 12 words>"}
"""


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
    prompt = prepared.conf.get("JUDGE_SYSTEM_PROMPT") or JUDGE_SYSTEM_PROMPT
    return common.ask_judge(
        prompt, "QUESTION:\n%s\n\nANSWER:\n%s" % (item["question"], item["answer"]),
        prepared.settings)


common.register(common.Evaluator(
    "llm_judge",
    "a judge model grades each answer on its own merits (JUDGE_SYSTEM_PROMPT)",
    prepare, score, needs_endpoint=True,
))
