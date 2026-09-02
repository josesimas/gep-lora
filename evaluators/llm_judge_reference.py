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

import generate_runs

from evaluators import common, llm_judge


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
    prompt = prepared.conf.get("JUDGE_REFERENCE_SYSTEM_PROMPT")
    content = ("QUESTION:\n%s\n\nREFERENCE ANSWER:\n%s\n\nANSWER:\n%s"
               % (item["question"], reference, item["answer"]))
    return common.ask_judge(prompt, content, prepared.settings)


common.register(common.Evaluator(
    "llm_judge_reference",
    "a judge model compares each answer with the dataset's own answer to the "
    "same question (JUDGE_REFERENCE_SYSTEM_PROMPT)",
    prepare, score, wants_reference=True, needs_endpoint=True,
))
