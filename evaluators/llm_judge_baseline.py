"""
evaluators/llm_judge_baseline.py - The judge, shown what the base model said.

The others ask whether an answer is good. This one asks the question the search
is actually for: **did folding these adapters in make the model better than it
was without them.** The judge sees the question, the bare base model's reply and
the blend's reply, and scores the difference on the centred scale
JUDGE_BASELINE_SYSTEM_PROMPT lays out -- where 0.5 means the blend changed
nothing worth having, above it means it earned its keep, and below it means it
did harm.

The control comes from baseline_run.py, which produces it once per base model
and caches it in the database; this file only reads that cache and writes the
prompt. Everything about the judge itself is llm_judge's.
"""

import baseline_run

from evaluators import common, llm_judge


def prepare(conf, pending, context=None):
    """The judge, plus the base model's own answer to every pending question.

    The control comes out of the `baselines` table, and is produced -- once --
    only if something is missing from it. That is the whole economy of this
    evaluator: the first sweep on a base model pays a model load for it, every
    sweep after it pays nothing, and adding prompts to the eval set costs only
    the new ones.

    Producing it needs the database and the run folder, which is why this is
    the one evaluator handed the step's Context.
    """
    prepared = llm_judge.prepare(conf, pending, context)
    if not common.needs_grading(pending):
        # Nothing to grade means nothing to compare, so neither the judge nor
        # the base model is troubled -- the same bargain llm_judge strikes.
        prepared.notes.append("baseline: not needed -- no answer needs grading")
        return prepared

    if context is None or getattr(context, "conn", None) is None:
        raise SystemExit(
            "EVALUATOR = 'llm_judge_baseline' needs the sweep's database to read "
            "and fill its cache of base-model answers, and this evaluate step was "
            "not given one."
        )

    questions = [row["question"] for row in pending
                 if (row["answer"] or "").strip()]
    prepared.baselines = baseline_run.ensure(
        context.conn, conf, questions, context.run_dir,
        keep_script=getattr(context.options, "keep_scripts", False))

    model = baseline_run.model_key(conf)
    covered = sum(1 for question in questions
                  if baseline_run.answer_for(question, prepared.baselines))
    prepared.notes.append(
        "baseline: %d of %d answer(s) have the base model's own reply to compare "
        "against, cached under %r" % (covered, len(questions), model))
    return prepared


def score(item, prepared):
    """Grade the blend against the model it was built on.

    Not "is this answer good" but "did the adapters earn their place", which is
    the question the search is actually asking. 0.5 is the middle of the rubric
    and means the blend changed nothing worth having, so an individual whose
    fitness lands there is the base model with extra steps.

    A missing control fails this one exchange rather than falling back to
    grading on merit: a merit score and an improvement score are not the same
    number, and averaging the two into one fitness would quietly reward
    whichever individuals happened to lose their baseline.
    """
    base = baseline_run.answer_for(item["question"], prepared.baselines)
    if not base:
        raise ValueError("no cached base-model answer for this question")
    prompt = prepared.conf.get("JUDGE_BASELINE_SYSTEM_PROMPT")
    content = ("QUESTION:\n%s\n\nBASE ANSWER:\n%s\n\nTUNED ANSWER:\n%s"
               % (item["question"], base, item["answer"]))
    return common.ask_judge(prompt, content, prepared.settings)


common.register(common.Evaluator(
    "llm_judge_baseline",
    "a judge model compares each answer with what the base model itself said, "
    "and scores the improvement (JUDGE_BASELINE_SYSTEM_PROMPT) -- 0.5 is no "
    "change; the base answers are cached in the database",
    prepare, score, needs_endpoint=True, wants_baseline=True,
))
