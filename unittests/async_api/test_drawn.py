"""
test_drawn.py - A blend drawn by hand: read as the pipeline reads a chromosome,
and tested as a verification of a sweep of one.

  * a drawing is a chromosome: slots in reading order, one per LoRA however
    often it is used, and back again;
  * the rank rule is the pipeline's: LIN over two ranks is BAD, and says where;
  * the weights shown are the ones the sweep's scripts will draw;
  * a test is a job that is done the moment it exists and is never queued as a
    search, and a verification the worker runs -- the blend and each of its
    LoRAs alone, on the dataset the person chose.
"""

import json
import os

from async_api import drawn
from async_api import registry as reg
from async_api import worker
from storage import store

from unittests.async_api.support import RECORDS, JobsTestCase
from unittests.async_api.test_server import ServerTestCase


def leaf(lora, weight="w1"):
    return {"lora": lora, "weight": weight}


def fold(op, left, right):
    return {"op": op, "children": [left, right]}


class DrawingTests(JobsTestCase):

    def setUp(self):
        super().setUp()
        self.own_slots()
        # slot-L1 16, slot-L2 16, slot-L3 8, slot-L4 4, slot-L5 32
        self.ids = {row["name"][5:]: row["id"]
                    for row in self.registry.catalog.all(owner=self.user["name"])}

    def test_a_drawing_is_a_chromosome_and_back(self):
        a, b, c = self.ids["L3"], self.ids["L1"], self.ids["L5"]
        tree = fold("CAT", fold("SVD", leaf(a, "w2"), leaf(b, "w5")), leaf(a, "w1"))
        chromosome, slots = drawn.encode(tree)
        # Level order: CAT, then SVD and the right-hand leaf, then SVD's leaves.
        # The LoRA read first is L1, and keeps it the second time it appears.
        self.assertEqual(chromosome, "CAT.SVD.L1.L1.L2.w1.w2.w5")
        self.assertEqual(slots, {"L1": a, "L2": b})
        self.assertEqual(drawn.decode_tree(chromosome, slots), tree)
        self.assertEqual(drawn.loras_in(fold("CAT", leaf(c), leaf(a))), [c, a])

    def test_the_top_may_be_any_fold_or_one_lora(self):
        a, b = self.ids["L3"], self.ids["L1"]
        for tree, chromosome in ((fold("SVD", leaf(a, "w2"), leaf(b, "w5")), "SVD.L1.L2.w2.w5"),
                                 (fold("LIN", leaf(a), leaf(a, "w3")), "LIN.L1.L1.w1.w3"),
                                 (leaf(b, "w4"), "L1.w4")):
            encoded, slots = drawn.encode(tree)
            self.assertEqual(encoded, chromosome)
            self.assertEqual(drawn.decode_tree(encoded, slots), tree)
        # Nothing drawn at all is a drawing with one empty place: the top.
        self.assertEqual(drawn.walk(None), [("", None)])
        with self.assertRaises(drawn.DrawnError):
            drawn.encode(None)

    def test_one_lora_alone_can_be_built(self):
        found = drawn.check(self.registry.catalog, self.user, leaf(self.ids["L5"], "w2"))
        self.assertEqual((found["state"], found["chromosome"]), ("ok", "L1.w2"))
        self.assertEqual(found["rank"], found["nodes"][""]["rank"])
        empty = drawn.check(self.registry.catalog, self.user, None)
        self.assertEqual((empty["state"], empty["empty"]), ("incomplete", [""]))

    def test_what_is_not_a_drawing(self):
        for tree, fragment in (
                ("CAT", "not a node"),
                ({"op": "CAT", "children": [leaf(1)]}, "exactly two"),
                (fold("CAT", {"op": "XOR", "children": [None, None]}, leaf(1)), "not a fold"),
                (fold("CAT", leaf(1, "w11"), leaf(2)), "weight"),
                (fold("CAT", {"lora": "one", "weight": "w1"}, leaf(2)), "by its id"),
                (fold("CAT", {"what": 1}, leaf(2)), "neither")):
            with self.assertRaises(drawn.DrawnError) as caught:
                drawn.walk(tree)
            self.assertIn(fragment, str(caught.exception))
        with self.assertRaises(drawn.DrawnError):
            drawn.encode(fold("CAT", None, leaf(1)))

    def test_an_unfinished_drawing_is_incomplete_and_says_where(self):
        found = drawn.check(self.registry.catalog, self.user,
                            fold("CAT", leaf(self.ids["L5"]), fold("LIN", None, None)))
        self.assertEqual(found["state"], "incomplete")
        self.assertEqual(found["empty"], ["1.0", "1.1"])
        self.assertIsNone(found["chromosome"])
        self.assertEqual(found["nodes"]["0"]["rank"], 32)   # a leaf's rank is known already

    def test_the_rank_rule_is_the_pipelines(self):
        catalog = self.registry.catalog
        # 16 and 16 mix; 16 and 8 do not.
        ok = drawn.check(catalog, self.user, fold("CAT", leaf(self.ids["L4"]),
                         fold("LIN", leaf(self.ids["L1"]), leaf(self.ids["L2"]))))
        self.assertEqual(ok["state"], "ok", ok["problems"])
        self.assertEqual(ok["nodes"]["1"]["rank"], 16)
        self.assertEqual(ok["nodes"][""]["rank"], 20)       # cat sums 4 + 16
        self.assertEqual(ok["rank"], 20)
        bad = drawn.check(catalog, self.user, fold("CAT", leaf(self.ids["L4"]),
                          fold("LIN", leaf(self.ids["L1"]), leaf(self.ids["L3"]))))
        self.assertEqual(bad["state"], "BAD")
        self.assertTrue(bad["nodes"]["1"]["broken"])
        self.assertIn("LIN at right mixes rank 16 and rank 8", bad["problems"][0])
        self.assertEqual(drawn.place("1.0"), "right › left")

    def test_the_weights_are_the_sweeps(self):
        first = drawn.draw(12345)
        self.assertEqual(first, drawn.draw(12345))
        self.assertNotEqual(first["weights"], drawn.draw(12346)["weights"])
        self.assertEqual(sorted(first["weights"]), sorted("w%d" % n for n in range(1, 11)))
        self.assertTrue(all(0 < value < 1 for value in first["weights"].values()))
        fresh = drawn.draw()
        self.assertIsInstance(fresh["seed"], int)
        with self.assertRaises(drawn.DrawnError):
            drawn.draw(-1)

    def test_the_weights_are_an_individuals(self):
        # Individual 1 unless said; another number is another draw, the one a
        # search makes for that individual under the same master seed.
        self.assertEqual(drawn.draw(12345)["number"], drawn.NUMBER)
        seventh = drawn.draw(12345, 7)
        self.assertEqual((seventh["seed"], seventh["number"]), (12345, 7))
        self.assertEqual(seventh["weight_seed"], drawn.weight_seed(12345, 7))
        self.assertNotEqual(seventh["weights"], drawn.draw(12345)["weights"])
        for wrong in (0, -3, True, "7"):
            with self.assertRaises(drawn.DrawnError):
                drawn.draw(12345, wrong)

    def test_only_their_own_ready_loras(self):
        other = self.registry.user_for_key(self.registry.add_user("bob"))
        with self.assertRaises(drawn.DrawnError):
            drawn.check(self.registry.catalog, other,
                        fold("CAT", leaf(self.ids["L1"]), leaf(self.ids["L2"])))


class TestingTests(ServerTestCase):

    def setUp(self):
        super().setUp()
        self.own_slots()
        self.ids = {row["name"][5:]: row["id"]
                    for row in self.registry.catalog.all(owner=self.user["name"])}
        self.tree = fold("CAT", fold("SVD", leaf(self.ids["L3"], "w2"),
                                     leaf(self.ids["L5"], "w4")), leaf(self.ids["L1"], "w1"))

    def body(self, **change):
        body = {"tree": self.tree, "seed": 99, "mock": True, "count": 3,
                "dataset": {"text": "\n".join(json.dumps(one) for one in RECORDS)},
                "settings": {"EVALUATOR": "similarity"}}
        body.update(change)
        return body

    def test_check_over_http(self):
        status, found = self.call("POST", "/blends/check", {"tree": self.tree, "seed": 7})
        self.assertEqual(status, 200, found)
        self.assertEqual(found["state"], "ok")
        self.assertEqual(found["chromosome"], "CAT.SVD.L1.L2.L3.w1.w2.w4")
        self.assertEqual(found["seed"], 7)
        status, found = self.call("POST", "/blends/check", {"tree": {"op": "SVD"}})
        self.assertEqual(status, 400)

    def test_a_drawn_blend_tested_end_to_end(self):
        status, reply = self.call("POST", "/blends/test", self.body())
        self.assertEqual(status, 201, reply)
        job, row = reply["job"], reply["verification"]
        # Done the moment it exists, and never a search.
        self.assertEqual(job["status"], reg.DONE)
        self.assertEqual(job["task"], reg.BLEND)
        self.assertFalse(job["can_evaluate"] or job["can_test"] or job["resumable"])
        self.assertEqual(row["status"], reg.QUEUED)
        self.assertEqual(row["number"], drawn.NUMBER)
        self.assertEqual(row["options"]["slots"], ["L1", "L2", "L3"])
        self.assertEqual(row["options"]["split"], "training")

        # The sweep holds the one individual, built as a search builds one: its
        # weight seed is the one the page was shown the weights of.
        conn = store.connect(self.registry.database(self.registry.job(job["id"])))
        try:
            one, = store.individuals(conn, job["run_id"])
            self.assertEqual(one["chromosome"], "CAT.SVD.L1.L2.L3.w1.w2.w4")
            self.assertEqual(one["state"], "ok")
            self.assertEqual(one["weight_seed"], drawn.draw(99)["weight_seed"])
            self.assertIn("SVD", one["tree"])
            run_dir = os.path.join(os.path.dirname(conn.path),
                                   os.path.basename(conn.path).rsplit(".", 1)[0]
                                   + "_run%d" % job["run_id"])
            self.assertFalse(os.path.exists(os.path.join(run_dir, one["script_name"])))
        finally:
            conn.close()

        # Nothing to resume, evaluate or test: it was never a search.
        for path in ("resume", "evaluate", "test"):
            status, reply = self.call("POST", "/jobs/%d/%s" % (job["id"], path), {})
            self.assertEqual(status, 409, (path, reply))
            self.assertIn("drawn by hand", reply["error"])

        worker.serve(self.registry, once=True)
        status, reply = self.call("GET", "/verifications/%d" % row["id"])
        one = reply["verification"]
        self.assertEqual(one["status"], reg.DONE, one)
        report = one["report"]
        self.assertEqual(report["meta"]["chromosome"], "CAT.SVD.L1.L2.L3.w1.w2.w4")
        self.assertEqual(report["meta"]["questions"], 3)
        self.assertEqual(sorted(report["answers"]), ["L1", "L2", "L3", "blend"])

        # It is one of their runs, like any other.
        status, runs = self.call("GET", "/runs")
        self.assertEqual(status, 200, runs)
        self.assertIn(job["id"], [one["id"] for one in runs["runs"]])

    def test_a_searched_blend_opened_keeps_its_weights(self):
        status, reply = self.call("POST", "/jobs", self.submission())
        self.assertEqual(status, 201, reply)
        search = reply["job"]["id"]
        worker.serve(self.registry, once=True)

        # Every blend that can be opened, the search's best first.
        status, listed = self.call("GET", "/blends")
        self.assertEqual(status, 200, listed)
        found = next(one for one in listed["jobs"] if one["job"] == search)
        self.assertEqual(found["task"], reg.SEARCH)
        self.assertTrue(found["blends"][0]["is_best"])
        self.assertIn("formula", found["blends"][0])

        # Opened with no number: its best, as a drawing of the user's LoRAs,
        # under the search's seed and its own number.
        status, best = self.call("GET", "/blends/%d" % search)
        self.assertEqual(status, 200, best)
        number = best["number"]
        self.assertEqual(number, found["blends"][0]["number"])
        status, stored = self.call("GET", "/jobs/%d/individuals/%d" % (search, number))
        stored = stored["individual"]
        self.assertEqual(status, 200, stored)
        check = self.call("POST", "/blends/check", {"tree": best["tree"], "seed": best["seed"],
                                                    "number": number})[1]
        self.assertEqual(check["weight_seed"], stored["weight_seed"])

        # Tested, it is the same individual: its number, so its weights.
        status, reply = self.call("POST", "/blends/test", self.body(
            tree=best["tree"], seed=best["seed"], number=number))
        self.assertEqual(status, 201, reply)
        self.assertEqual(reply["verification"]["number"], number)
        conn = store.connect(self.registry.database(self.registry.job(reply["job"]["id"])))
        try:
            one, = store.individuals(conn, reply["job"]["run_id"])
            self.assertEqual((one["number"], one["weight_seed"]), (number, stored["weight_seed"]))
        finally:
            conn.close()
        status, again = self.call("GET", "/blends/%d" % reply["job"]["id"])
        self.assertEqual((again["number"], again["verification"]),
                         (number, reply["verification"]["id"]))

        # Any blend of it by number; one it does not hold, and another's job, are not.
        other = next(one for one in found["blends"] if one["number"] != number)
        status, opened = self.call("GET", "/blends/%d?individual=%d" % (search, other["number"]))
        self.assertEqual((status, opened["number"]), (200, other["number"]))
        self.assertEqual(self.call("GET", "/blends/%d?individual=999" % search)[0], 409)
        self.assertEqual(self.call("GET", "/blends/%d?individual=x" % search)[0], 400)
        bob = self.registry.add_user("bob")
        self.assertEqual(self.call("GET", "/blends/%d" % search, key=bob)[0], 404)
        self.assertEqual(self.call("GET", "/blends", key=bob)[1], {"jobs": []})

    def test_a_lin_is_a_fold_not_a_slot(self):
        # CAT(A, LIN(B, A)): two LoRAs of one rank, the second used twice.
        tree = fold("CAT", leaf(self.ids["L1"], "w1"),
                    fold("LIN", leaf(self.ids["L2"], "w2"), leaf(self.ids["L1"], "w3")))
        status, reply = self.call("POST", "/blends/test", self.body(tree=tree))
        self.assertEqual(status, 201, reply)
        self.assertEqual(reply["verification"]["options"]["slots"], ["L1", "L2"])

    def test_the_refusals(self):
        for change, fragment in (
                ({"tree": fold("CAT", None, leaf(self.ids["L1"]))}, "not finished"),
                ({"tree": fold("CAT", leaf(self.ids["L4"]),
                               fold("LIN", leaf(self.ids["L1"]), leaf(self.ids["L3"])))},
                 "LIN"),
                ({"count": 0}, "count"),
                ({"dataset": {"file": "no-such-file.json"}}, "no shared dataset"),
                ({"dataset": {"lora": self.ids["L1"]}}, "training data"),
                ({"settings": {"COUNT": 9}}, "settings may name only"),
                ({"extra": 1}, "unknown field")):
            status, reply = self.call("POST", "/blends/test", self.body(**change))
            self.assertEqual(status, 400, (change, reply))
            self.assertIn(fragment, reply["error"], change)
        # Nothing was left behind by any of them.
        self.assertEqual(self.registry.jobs(self.user["id"]), [])
