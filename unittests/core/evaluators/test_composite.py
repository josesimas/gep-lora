"""
test_composite.py - Several evaluators, their scores combined into one.

What is composite's own, and so what is tested here:

  * the arithmetic of every COMPOSITE_AGGREGATE, weights included, and the
    ordering min <= harmonic <= geometric <= mean <= max it promises;
  * COMPOSITE_EVALUATORS being read in all three shapes, and refused -- once,
    in prepare() -- when it is empty, names itself, names a member twice or
    gives a weight that is not a positive number;
  * each member preparing and scoring with its own bundle, handed the step's
    context;
  * the floor vetoing to 0, and COMPOSITE_ON_FAILURE deciding whether one
    failed member fails the answer;
  * asks_judge() following the members, so a composite of local scorers is
    not held to the abandon rule.

The members are stand-in evaluators patched into the registry, so nothing here
contacts a judge; one test uses the real heuristic to show a composite of a
registered evaluator runs end to end.
"""

import itertools
import statistics
import unittest
from unittest import mock

from gep_lora.core import evaluators
from gep_lora.core.evaluators import common, composite


class Member:
    """A stand-in evaluator that returns fixed scores and records its calls."""

    def __init__(self, name, value=0.5, reason="fine", needs_judge=False,
                 fails=False):
        self.value = value
        self.fails = fails
        self.prepared_with = []
        self.scored = []
        self.evaluator = common.Evaluator(name, "stand-in", self.prepare,
                                          self.score, needs_judge=needs_judge)
        self.reason = reason
        self.name = name

    def prepare(self, conf, pending, context=None):
        self.prepared_with.append(context)
        return common.Prepared(conf, "label-" + self.name, notes=["%s ready" % self.name])

    def score(self, item, prepared):
        self.scored.append(prepared.label)
        if self.fails:
            raise RuntimeError("%s is down" % self.name)
        return self.value, self.reason


class Composite(unittest.TestCase):

    def setUp(self):
        self.members = {}
        patcher = mock.patch.dict(common._REGISTRY)
        patcher.start()
        self.addCleanup(patcher.stop)

    def add(self, name, **kwargs):
        member = Member(name, **kwargs)
        common.register(member.evaluator)
        self.members[name] = member
        return member

    def run_one(self, conf, item=None):
        conf = dict(conf)
        prepared = composite.prepare(conf, [{"answer": "x"}], context="ctx")
        return composite.score(item or {"question": "q", "answer": "a",
                                        "position": 1}, prepared)


class Arithmetic(unittest.TestCase):

    def test_mean_is_weighted(self):
        self.assertAlmostEqual(composite.aggregate([(1.0, 3), (0.0, 1)], "mean"), 0.75)

    def test_median_with_equal_weights_is_statistics_median(self):
        for scores in ([0.1, 0.9, 0.4], [0.2, 0.8, 0.4, 0.6], [0.3], [0.5, 0.7]):
            pairs = [(value, 1.0) for value in scores]
            self.assertAlmostEqual(composite.aggregate(pairs, "median"),
                                   statistics.median(scores))

    def test_median_follows_weight(self):
        self.assertEqual(composite.aggregate([(0.2, 1), (0.9, 5), (0.4, 1)], "median"), 0.9)

    def test_min_and_max_ignore_weights(self):
        pairs = [(0.2, 10), (0.9, 0.1)]
        self.assertEqual(composite.aggregate(pairs, "min"), 0.2)
        self.assertEqual(composite.aggregate(pairs, "max"), 0.9)

    def test_geometric_and_harmonic(self):
        pairs = [(0.25, 1), (1.0, 1)]
        self.assertAlmostEqual(composite.aggregate(pairs, "geometric"), 0.5)
        self.assertAlmostEqual(composite.aggregate(pairs, "harmonic"), 0.4)

    def test_a_zero_is_zero_for_geometric_and_harmonic(self):
        pairs = [(0.0, 1), (1.0, 5)]
        self.assertEqual(composite.aggregate(pairs, "geometric"), 0.0)
        self.assertEqual(composite.aggregate(pairs, "harmonic"), 0.0)

    def test_trimmed_mean_drops_the_ends(self):
        pairs = [(0.0, 1), (0.5, 1), (0.7, 1), (1.0, 1)]
        self.assertAlmostEqual(composite.aggregate(pairs, "trimmed_mean"), 0.6)
        two = [(0.2, 1), (0.6, 1)]
        self.assertAlmostEqual(composite.aggregate(two, "trimmed_mean"), 0.4)

    def test_the_promised_ordering(self):
        values = [0.05, 0.3, 0.6, 1.0]
        for scores in itertools.combinations_with_replacement(values, 3):
            for weights in ([1, 1, 1], [3, 1, 0.5]):
                pairs = list(zip(scores, weights))
                got = [composite.aggregate(pairs, how)
                       for how in ("min", "harmonic", "geometric", "mean", "max")]
                for low, high in zip(got, got[1:]):
                    self.assertLessEqual(low, high + 1e-12, (pairs, got))


class Members(Composite):

    def test_three_shapes(self):
        for name in ("a", "b", "c"):
            self.add(name)
        chosen = composite.members({"COMPOSITE_EVALUATORS": [
            "a", ["b", 2], {"name": "c", "weight": 0.5}]})
        self.assertEqual([(evaluator.name, weight) for evaluator, weight in chosen],
                         [("a", 1.0), ("b", 2.0), ("c", 0.5)])

    def test_refusals(self):
        self.add("a")
        for bad in ([], ["composite"], ["a", "a"], [["a", 0]], [["a", -1]],
                    [["a", "2"]], [["a", True]], [{"name": "a", "wait": 2}],
                    ["nobody"], [42]):
            with self.assertRaises(SystemExit, msg=bad):
                composite.members({"COMPOSITE_EVALUATORS": bad})

    def test_bad_settings_refused_in_prepare(self):
        self.add("a")
        for conf in ({"COMPOSITE_AGGREGATE": "mode"}, {"COMPOSITE_FLOOR": 2},
                     {"COMPOSITE_ON_FAILURE": "retry"}):
            conf["COMPOSITE_EVALUATORS"] = ["a"]
            with self.assertRaises(SystemExit, msg=conf):
                composite.prepare(conf, [])


class Scoring(Composite):

    def test_each_member_scores_with_its_own_bundle(self):
        a = self.add("a", value=0.8, reason="good")
        b = self.add("b", value=0.4, reason="thin")
        quality, reason = self.run_one({"COMPOSITE_EVALUATORS": ["a", ["b", 3]]})
        self.assertAlmostEqual(quality, (0.8 + 0.4 * 3) / 4)
        self.assertEqual(a.prepared_with, ["ctx"])
        self.assertEqual(b.scored, ["label-b"])
        self.assertIn("a 0.80 (good)", reason)
        self.assertIn("b 0.40 (thin)", reason)
        self.assertTrue(reason.startswith("mean of"))

    def test_prepare_label_and_notes(self):
        self.add("a")
        prepared = composite.prepare({"COMPOSITE_EVALUATORS": ["a"],
                                      "COMPOSITE_AGGREGATE": "median"}, [])
        self.assertEqual(prepared.label, "composite:median(a=label-a)")
        self.assertIn("  a ready", prepared.notes)

    def test_floor_vetoes(self):
        self.add("a", value=0.95)
        self.add("b", value=0.1)
        quality, reason = self.run_one({"COMPOSITE_EVALUATORS": ["a", "b"],
                                        "COMPOSITE_FLOOR": 0.2})
        self.assertEqual(quality, 0.0)
        self.assertTrue(reason.startswith("vetoed: b 0.10 below 0.2"))

    def test_a_failed_member_fails_the_answer_by_default(self):
        self.add("a", value=0.9)
        self.add("b", fails=True)
        with self.assertRaises(RuntimeError):
            self.run_one({"COMPOSITE_EVALUATORS": ["a", "b"]})

    def test_skip_combines_the_rest(self):
        self.add("a", value=0.9)
        self.add("b", fails=True)
        quality, reason = self.run_one({"COMPOSITE_EVALUATORS": ["a", "b"],
                                        "COMPOSITE_ON_FAILURE": "skip"})
        self.assertAlmostEqual(quality, 0.9)
        self.assertIn("skipped b", reason)

    def test_skip_with_nobody_left_fails(self):
        self.add("a", fails=True)
        with self.assertRaises(RuntimeError):
            self.run_one({"COMPOSITE_EVALUATORS": ["a"],
                          "COMPOSITE_ON_FAILURE": "skip"})

    def test_a_real_member(self):
        common.register(evaluators.get("heuristic"))
        quality, reason = self.run_one(
            {"COMPOSITE_EVALUATORS": ["heuristic"]},
            {"question": "q", "position": 1,
             "answer": "Start by clearing the desk, then sort what is left into three piles."})
        self.assertEqual(quality, 1.0)
        self.assertIn("heuristic 1.00 (well formed)", reason)


class AsksJudge(Composite):

    def test_follows_the_members(self):
        self.add("local")
        self.add("judge", needs_judge=True)
        registered = evaluators.get("composite")
        self.assertTrue(registered.needs_judge)
        self.assertFalse(registered.asks_judge({"COMPOSITE_EVALUATORS": ["local"]}))
        self.assertTrue(registered.asks_judge({"COMPOSITE_EVALUATORS": ["local", "judge"]}))
        # Unreadable settings: assume a judge, the way wants_the_card() does.
        self.assertTrue(registered.asks_judge({"COMPOSITE_EVALUATORS": []}))

    def test_everyone_else_answers_needs_judge(self):
        for name, _description in evaluators.available():
            if name != "composite":
                evaluator = evaluators.get(name)
                self.assertEqual(evaluator.asks_judge({}), evaluator.needs_judge)


if __name__ == "__main__":
    unittest.main()
