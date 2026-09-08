"""
evaluators/llm_judge_answers.py - The judge, shown two answers and nothing else.

llm_judge_reference shows the judge three things: the question, the answer the
dataset carries for it, and what the blend replied. This one drops the first.
The judge is handed the **reference answer** and the **blend's answer**, and is
asked how well the second stands in for the first -- so the only thing it can
grade is how the reply compares with the reply the training data gives, and it
has no way to reward an answer for serving a question the reference answers
differently.

That is the whole difference, and it is the point. With the question in front of
it a judge quietly grades merit as well as manner: a blend that answers helpfully
but nothing like its training data still reads as a good answer to the question,
and picks up score for it. Without the question there is no such credit to give
-- the reference is the only standard in the prompt, so the score is a
measurement of agreement with it and of nothing else. Dropping the question also
shortens the prompt by a question per call, which on a local judge with
JUDGE_LOCAL_MAX_SEQ_LENGTH to spend is not nothing.

The cost is symmetrical and worth knowing before selecting on it: the judge
cannot tell a blend that misread the question from one that answered a
neighbouring question well, because it never sees which question either answer
is of. It is a *match* score, not a quality score. llm_judge is the one that
grades an answer on its own merits, and llm_judge_reference the one that does
both at once.

Everything else -- backend, model, timeouts, retries, the abandon rule -- is
llm_judge's, whose prepare() this one extends, so this rubric is graded by an
endpoint or by a locally loaded judge exactly as that one is.
"""

from blends import generate_runs

from evaluators import common, llm_judge


# How the judge is told to grade when the question is deliberately withheld --
# the rubric this evaluator selects on.
#
# It says up front that the question is missing, because a judge that has not
# been told will look for it and complain about its absence instead of grading.
# And it asks the same question the reference rubric asks -- did the reply come
# out in the manner the training data replies in -- without the escape hatch of
# judging the answer against the question directly.
#
# It lives here rather than in settings.py for the reason every other rubric
# does: it is this evaluator's own text, and the module that sends it is where a
# reader looks to find out what "graded on the answers alone" means. There is no
# conf lookup beside it the way llm_judge and llm_judge_reference have one --
# those exist so a sweep created while the prompt was still a setting is
# re-scored against the prompt it actually ran with, and no sweep has ever run
# under a setting for this one.
JUDGE_ANSWERS_SYSTEM_PROMPT = """\
You are comparing two answers to the same question. You are NOT shown the
question, and you do not need it. Do not ask for it and do not speculate about
it -- judge only the two answers in front of you.

You will be shown:
  REFERENCE ANSWER  ground truth
  ANSWER            the answer to be evaluated against ground truth (REFERENCE ANSWER)

Judge how well the ANSWER stands in for the REFERENCE ANSWER:
- Manner: the same style, voice, form, length and register? This matters most
- Substance: does it say the same kind of thing, so that someone expecting the
  reference would be served by this instead?
- Coherence: well formed and consistent, free of contradictions, repetition,
  broken grammar or nonsense.

Score from 0.0 to 1.0:
  1.0  excellent - same manner, and would serve in place of the reference
  0.7  good - recognisably the same manner, minor slips
  0.5  mixed - part of the manner, or the manner without the substance
  0.3  poor - an answer, but in nothing like the intended manner
  0.0  useless - incoherent, empty, or plainly about something else

Reply with JSON and nothing else, with the score FIRST:
{"quality": <number between 0 and 1>, "reason": "<at most 12 words>"}
"""


def prepare(conf, pending, context=None):
    prepared = llm_judge.prepare(conf, pending, context)
    prepared.references = common.prepare_references(conf, "llm_judge_answers")
    prepared.notes.append(
        "reference answers: %d, from %s (the question is not sent)"
        % (len(prepared.references["by_position"]),
           generate_runs.training_set_path(conf.get("TRAINING_SET"))))
    return prepared


def score(item, prepared):
    """Grade the answer against the dataset's answer, question withheld.

    The exchange is still matched to its reference *by* its question -- that is
    what common.reference_for() keys on, and it is what stops an answer being
    paired with somebody else's reference. The question is used to find the
    reference and then dropped; only the two answers are sent.

    An exchange with no reference fails here rather than falling back to
    llm_judge.score() the way llm_judge_reference does. The fallback grades on
    merit, which means sending the question -- the one thing this evaluator
    exists not to do -- and scoring a different quantity from the answers around
    it. Failing one exchange is the contract: start_run.py counts it and moves
    on, and the individual is judged on the answers that did have a reference.
    prepare() has already refused an eval set that carries no references at all,
    so this is a gap in the file rather than the wrong dataset.
    """
    reference = common.reference_for(item, prepared)
    if not reference:
        raise ValueError(
            "no reference answer for prompt %s, and llm_judge_answers has "
            "nothing else to compare against -- it is never shown the question"
            % item["position"])
    content = ("REFERENCE ANSWER:\n%s\n\nANSWER:\n%s"
               % (reference, item["answer"]))
    return common.ask_judge(JUDGE_ANSWERS_SYSTEM_PROMPT, content,
                            prepared.settings)


common.register(common.Evaluator(
    "llm_judge_answers",
    "a judge model compares each answer with the dataset's own answer and is "
    "never shown the question (JUDGE_ANSWERS_SYSTEM_PROMPT)",
    prepare, score, wants_reference=True, needs_judge=True,
))
