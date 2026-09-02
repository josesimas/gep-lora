"""
evaluators/heuristic.py - Local checkable properties. No model, no reference.

Four equally weighted checks, averaged over the ones that apply: a length band,
a distinct-word ratio that catches the collapsed blend saying the same thing
over and over, and optionally a pattern the answer must match and one it must
not.

This measures whether an answer is **malformed**, not whether it is good --
which is the half of quality a judge charges the most to notice. Use it to
smoke-test the pipeline, to pair with the mocked template, or to score a task
whose rule really is checkable.
"""

import re

from evaluators import common


def _repetition(words):
    """0..1, how little this answer repeats itself.

    A blend that has collapsed says the same thing over and over -- the failure
    mode a judge is expensive to detect and a distinct-word ratio is free to.
    Short answers are exempted: five words cannot help repeating.
    """
    if len(words) < 10:
        return 1.0
    return len(set(words)) / len(words)


def prepare(conf, pending, context=None):
    for name in ("HEURISTIC_REQUIRE", "HEURISTIC_FORBID"):
        pattern = conf.get(name)
        if pattern:
            try:
                re.compile(pattern)
            except re.error as error:
                raise SystemExit("%s is not a valid regular expression: %s" % (name, error))
    return common.Prepared(conf, "heuristic",
                           notes=["heuristic: local checks only, no judge contacted"])


def score(item, prepared):
    """Score from properties that can be checked without asking anyone.

    Four equally weighted checks -- length, repetition, a pattern the answer
    must match, a pattern it must not -- averaged over the ones that apply.
    This is not a measure of whether an answer is *good*; it is a measure of
    whether it is malformed, which is the half of quality a judge charges the
    most to notice. Use it to smoke-test the pipeline, to pair with the mocked
    template, or to score a task whose rule really is checkable -- an
    all-uppercase adapter, say, with HEURISTIC_REQUIRE.
    """
    conf = prepared.conf
    answer = item["answer"]
    words = common.WORD.findall(answer.lower())
    parts, faults = [], []

    low = conf.get("HEURISTIC_MIN_WORDS", 8)
    high = conf.get("HEURISTIC_MAX_WORDS", 400)
    if len(words) < low:
        length = len(words) / low if low else 1.0
        faults.append("short (%d words)" % len(words))
    elif high and len(words) > high:
        length = max(0.0, 1.0 - (len(words) - high) / float(high))
        faults.append("long (%d words)" % len(words))
    else:
        length = 1.0
    parts.append(length)

    repetition = _repetition(words)
    parts.append(repetition)
    if repetition < 0.5:
        faults.append("repetitive (%.0f%% distinct)" % (repetition * 100))

    required = conf.get("HEURISTIC_REQUIRE")
    if required:
        hit = bool(re.search(required, answer, re.MULTILINE))
        parts.append(1.0 if hit else 0.0)
        if not hit:
            faults.append("missing required pattern")

    forbidden = conf.get("HEURISTIC_FORBID")
    if forbidden:
        hit = bool(re.search(forbidden, answer, re.MULTILINE))
        parts.append(0.0 if hit else 1.0)
        if hit:
            faults.append("matched forbidden pattern")

    value = sum(parts) / len(parts)
    return value, ", ".join(faults) if faults else "well formed"


common.register(common.Evaluator(
    "heuristic",
    "local checks only -- length, repetition, required/forbidden patterns "
    "(HEURISTIC_*); no judge, deterministic",
    prepare, score,
))
