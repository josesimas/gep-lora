"""
evaluators/similarity.py - Overlap with the dataset's own answer. No model.

Cheap, deterministic and offline: the whole eval half of a sweep costs nothing
and repeats exactly, which is what makes it worth having next to a judge whose
scores wobble. The trade is that it measures agreement with one particular
answer rather than quality -- an answer better than the dataset's scores badly.

SIMILARITY_METRIC picks how the overlap is counted; SIMILARITY_CASE_SENSITIVE
whether case counts.
"""

import difflib

from evaluators import common


def _token_f1(answer, reference, case_sensitive):
    """Bag-of-words F1, counting repeats -- the SQuAD-style overlap score."""
    got = common.tokens(answer, case_sensitive)
    want = common.tokens(reference, case_sensitive)
    if not got or not want:
        return 0.0, 0.0, 0.0
    shared = 0
    pool = list(want)
    for token in got:
        if token in pool:
            pool.remove(token)
            shared += 1
    if not shared:
        return 0.0, 0.0, 0.0
    precision, recall = shared / len(got), shared / len(want)
    return 2 * precision * recall / (precision + recall), precision, recall


def prepare(conf, pending, context=None):
    metric = conf.get("SIMILARITY_METRIC", "token_f1")
    if metric not in ("token_f1", "sequence", "containment"):
        raise SystemExit("SIMILARITY_METRIC must be 'token_f1', 'sequence' or "
                         "'containment', not %r" % metric)
    references = common.prepare_references(conf, "similarity")
    return common.Prepared(
        conf, "similarity:" + metric, references=references,
        notes=["similarity: %s against %d reference answer(s), no judge "
               "contacted" % (metric, len(references["by_position"]))])


def score(item, prepared):
    """How much of the dataset's answer this answer reproduces, 0..1.

    Cheap, deterministic and offline -- the whole eval half of a sweep costs
    nothing and repeats exactly, which is what makes it worth having next to a
    judge whose scores wobble. It measures agreement with one particular
    answer, though, not quality: a better answer than the dataset's scores
    badly, and that is the trade being made.
    """
    reference = common.reference_for(item, prepared)
    if not reference:
        raise ValueError("no reference answer for this prompt")
    case_sensitive = bool(prepared.conf.get("SIMILARITY_CASE_SENSITIVE", False))
    metric = prepared.conf.get("SIMILARITY_METRIC", "token_f1")
    answer = item["answer"]

    if metric == "sequence":
        left = answer if case_sensitive else answer.lower()
        right = reference if case_sensitive else reference.lower()
        value = difflib.SequenceMatcher(None, left, right).ratio()
        return value, "character overlap with the reference %.2f" % value
    if metric == "containment":
        got = set(common.tokens(answer, case_sensitive))
        want = set(common.tokens(reference, case_sensitive))
        value = (len(got & want) / len(want)) if want else 0.0
        return value, "%d/%d reference words present" % (len(got & want), len(want))

    value, precision, recall = _token_f1(answer, reference, case_sensitive)
    return value, "token f1 %.2f (p %.2f, r %.2f)" % (value, precision, recall)


common.register(common.Evaluator(
    "similarity",
    "token or character overlap with the dataset's own answer -- no judge, "
    "deterministic (SIMILARITY_METRIC)",
    prepare, score, wants_reference=True,
))
