"""
evaluators - Score an answer. Which way is a setting.

The process step stores the question/answer pairs a blended model produced.
This package grades them, and returns the number main.py writes back onto that
exchange:

    question   "Help me organize my desktop."
    answer     "Before we lay anything out, ..."
    quality    0.65
    reason     "asks a useful clarifying question but gives no concrete step"

Quality runs 0.0 to 1.0, where 1.0 is the best answer and 0.0 the worst. That is
the number a fitness function selects on, whichever way it was arrived at.

There is more than one sensible way to arrive at it, so this is a **registry of
evaluators** rather than one hardcoded judge -- one module per evaluator, plus
common.py for what they share. `EVALUATOR` in settings.py names the one a sweep
uses, and -- like every other setting -- the name is frozen into the sweep when
it starts, so a sweep is always scored the way it was created to be scored, and
a stored sweep can say how:

    llm_judge.py             a judge model grades the answer on its own merits
    llm_judge_reference.py   the same, but shown the dataset's own answer too
    llm_judge_baseline.py    the same, but shown what the base model itself
                             said, and asked how much the blend improved on it
    similarity.py            token overlap with the dataset's answer, no model
    heuristic.py             local checkable properties, no model
    panel.py                 several judge models, aggregated
    common.py                the registry, the judge transport, the reference
                             answers, the tokeniser -- everything two of the
                             six would otherwise both own

Each module ends in a register() call, and importing this package is what runs
them: the imports at the foot of this file are the registration, which is why
they are not unused. **Adding an evaluator is adding a file here and a line
there** -- nothing else in the pipeline changes, because every step reaches an
evaluator through get(EVALUATOR).

Each is an Evaluator: a name, a description, `prepare()` and `score()`.
`prepare(conf, pending, context=None)` is called once per evaluate step and
returns the Prepared bundle every call then works from -- the endpoint it
discovered, the references it loaded, the label to record. `context` is the
step's own Context, for the one evaluator that needs more than the settings and
the pending rows: llm_judge_baseline reads and fills the base-answer cache, so
it needs the database and the run folder. Everything else ignores it.
`score(item, prepared)` grades one exchange and returns `(quality, reason)`.
Raising ValueError or RuntimeError fails that one exchange and no more: main.py
counts it and moves on, which is what makes a half-scored sweep resumable.

Every module here keeps the same two names, `prepare` and `score`, because the
file it lives in already says which evaluator they belong to. That is also what
lets one build on another: llm_judge_reference and llm_judge_baseline are
llm_judge's prepare() and score() plus a bigger prompt.

**No knob lives in this package.** They are all in settings.py, prefixed by the
evaluator that reads them (JUDGE_*, BASELINE_*, SIMILARITY_*, HEURISTIC_*,
PANEL_*) -- see common.py for the one exception, the API key.

Scoring is resumable: an exchange that already has a quality is left alone
unless the evaluate step is run with --force, so an interrupted sweep can simply
be re-run. A sweep generated from template_code_mocked.py arrives already scored
and never reaches an evaluator at all.

Usage, which is all main.py does with it:

    evaluator = evaluators.get(conf.get("EVALUATOR"))
    prepared = evaluator.prepare(conf, pending, context)
    quality, reason = evaluator.score(item, prepared)
"""

from evaluators.common import (API_KEY, DEFAULT, Evaluator, Prepared,
                               abandon_after, available, get, register)

# Importing each module is what registers its evaluator, so these are the
# registry itself rather than unused imports. Listed in the order --evaluators
# is happiest to read them in: the judges, then the local ones.
from evaluators import llm_judge                # noqa: F401,E402
from evaluators import llm_judge_reference      # noqa: F401,E402
from evaluators import llm_judge_baseline       # noqa: F401,E402
from evaluators import similarity               # noqa: F401,E402
from evaluators import heuristic                # noqa: F401,E402
from evaluators import panel                    # noqa: F401,E402

__all__ = ["API_KEY", "DEFAULT", "Evaluator", "Prepared", "abandon_after",
           "available", "get", "register"]
