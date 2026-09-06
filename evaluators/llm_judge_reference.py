"""
evaluators/llm_judge_reference.py - The judge, shown the dataset's own answer.

The eval file carries both turns: the generated scripts only ever ask the user
turn -- handing a model the answer and then scoring its reply would be marking
its own homework -- but the judge is allowed to see the assistant turn.

This is the evaluator that can see *style*. A judge grading on merit alone
happily rewards a helpful prose answer from a blend that was supposed to rhyme;
shown the answer the training data gives to the same question, it grades on
whether the blend answered in the manner it was fine-tuned to.
JUDGE_REFERENCE_SYSTEM_PROMPT is the rubric, and it deliberately does not
reward copying.

Everything else -- endpoint, model, timeouts, retries -- is llm_judge's, whose
prepare() this one extends and whose score() it falls back to.
"""

from blends import generate_runs

from evaluators import common, llm_judge


# How the judge is told to grade when it is shown the dataset's own answer --
# the rubric this evaluator selects on.
#
# The reference is what the adapters were fine-tuned to produce, so this prompt
# asks about the thing merit-only grading cannot see: did the blend answer in
# the manner the training data answers in. It deliberately does not ask for a
# copy -- a blend that reproduced the reference word for word would score well
# here and have learned nothing but that one answer.
#
# It lives here rather than in settings.py for the reason llm_judge's does: it
# is this evaluator's own text, and the module that sends it is where a reader
# looks to find out what "graded against the reference" means. panel keeps its
# own copy. An item with no reference falls through to llm_judge.score() below,
# which grades it on merit against llm_judge's prompt instead.
#
# As in llm_judge, a sweep's stored value still wins when it has one, so a sweep
# created while this was a setting is re-scored against the prompt it ran with.
JUDGE_REFERENCE_SYSTEM_PROMPT = """\
You are grading a single answer produced by a fine-tuned AI model.

You will be shown:
  QUESTION          what the user asked
  REFERENCE ANSWER  how the model's training data answers that question
  ANSWER            what the model actually replied

The REFERENCE ANSWER is an example of the intended behaviour, not the only
correct answer. Do not reward copying it, and do not punish different wording,
different examples or different details.

Judge the ANSWER on:
- Manner: does it answer in the same style, voice, form and register as the
  reference? This matters most -- it is what the model was trained for.
- Substance: does it actually answer the QUESTION, as the reference does?
- Coherence: well formed and consistent, free of contradictions, repetition,
  broken grammar or nonsense.

Score from 0.0 to 1.0:
  1.0  excellent - same manner as the reference, and a sound answer
  0.7  good - recognisably the same manner, minor slips
  0.5  mixed - part of the manner, or the manner without the substance
  0.3  poor - answers, but in nothing like the intended manner
  0.0  useless - incoherent, empty, or entirely off topic

Reply with JSON and nothing else, with the score FIRST:
{"quality": <number between 0 and 1>, "reason": "<at most 12 words>"}
"""


def prepare(conf, pending, context=None):
    prepared = llm_judge.prepare(conf, pending, context)
    prepared.references = common.prepare_references(conf, "llm_judge_reference")
    prepared.notes.append(
        "reference answers: %d, from %s"
        % (len(prepared.references["by_position"]),
           generate_runs.training_set_path(conf.get("TRAINING_SET"))))
    return prepared


def score(item, prepared):
    """Grade against the dataset's answer to the same question.

    The reference is what the LoRAs were trained to produce, so this is the
    evaluator that can see *style*: a judge grading on merit alone will happily
    reward a helpful prose answer from a blend that was supposed to rhyme.

    A prompt with no reference is graded on merit instead rather than failed --
    the answer is still an answer, and dropping it would quietly shrink the
    eval set for that individual and make its fitness incomparable with the
    rest.
    """
    reference = common.reference_for(item, prepared)
    if not reference:
        return llm_judge.score(item, prepared)
    prompt = (prepared.conf.get("JUDGE_REFERENCE_SYSTEM_PROMPT")
              or JUDGE_REFERENCE_SYSTEM_PROMPT)
    content = ("QUESTION:\n%s\n\nREFERENCE ANSWER:\n%s\n\nANSWER:\n%s"
               % (item["question"], reference, item["answer"]))
    return common.ask_judge(prompt, content, prepared.settings)


common.register(common.Evaluator(
    "llm_judge_reference",
    "a judge model compares each answer with the dataset's own answer to the "
    "same question (JUDGE_REFERENCE_SYSTEM_PROMPT)",
    prepare, score, wants_reference=True, needs_endpoint=True,
))
