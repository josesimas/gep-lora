"""
test_compare.py - The blend comparison: two drawings, their tools, the two
tests and blend A against blend B.

  * every tool is described, summarised and in the schema the model is sent,
    and the page's words and question-mark blocks are there;
  * a tool on one blend is the visual guide's tool on that side and leaves the
    other alone; copying, swapping and opening a searched blend (with its own
    number, so its weights) work on the pair;
  * testing is an action for the page only once both blends can be built and
    the questions are chosen, and the plan is one POST /blends/test body per
    side on the same questions;
  * head_to_head pairs the two blends' answers by question and counts what
    each won;
  * the endpoints: config, intro, chat without a model, plan, the two tests
    run by the worker, the outcome and the debrief.
"""

from async_api import registry as reg
from async_api import worker
from async_api_agent import compare
from async_api_agent import prompts
from async_api_agent import ui_help

from unittests.async_api.support import JobsTestCase
from unittests.async_api.test_server import ServerTestCase

SCRIPTED = {"provider": "scripted"}
LINES = "\n".join('{"messages": [{"role": "user", "content": "Q%d?"}, '
                  '{"role": "assistant", "content": "A%d."}]}' % (n, n) for n in range(1, 5))


class Mixin:

    def ids(self):
        self.own_slots()
        # slot-L1 16, slot-L2 16, slot-L3 8, slot-L4 4, slot-L5 32
        return {row["name"][5:]: row["id"]
                for row in self.registry.catalog.all(owner=self.user["name"])}

    def box(self, session=None, stage="drawing"):
        return compare.Toolbox(self.registry, self.registry.catalog, self.user, stage,
                               session or {},
                               lambda: [{"file": "poems.json", "records": 9, "usable": True}])


class WordsTests(JobsTestCase):

    def test_every_tool_is_described_and_summarised(self):
        self.assertEqual(set(compare.SPECS), set(prompts.COMPARE_TOOLS))
        self.assertEqual(set(compare.SPECS), set(prompts.COMPARE_TOOL_DONE))
        self.assertEqual([one[0] for one in compare.schemas()], list(compare.SPECS))
        for step in prompts.COMPARE_STEPS:
            self.assertIn(prompts.COMPARE_PERSONA, prompts.system(step))
        for stage in compare.STAGES:
            self.assertIn("compare_" + stage, prompts.STEP_INSTRUCTIONS)

    def test_its_blocks_are_explained_by_its_own_guide(self):
        with open("async_api/blend_comparison.html", encoding="utf-8") as handle:
            page = handle.read()
        blocks = [key for key in prompts.UI_BLOCKS if key.startswith("compare_")]
        for key in blocks:
            self.assertIn('"%s"' % key, page, key)
        told = ui_help.explain("compare_duel", choice=SCRIPTED)
        self.assertIn("A against B", told["message"]["text"])


class ToolTests(Mixin, JobsTestCase):

    def setUp(self):
        super().setUp()
        self.lora = self.ids()

    def run_ok(self, box, name, **arguments):
        result = box.run(name, arguments)
        self.assertNotIn("error", result, (name, result))
        return result

    def test_a_tool_on_one_blend_leaves_the_other(self):
        box = self.box()
        self.run_ok(box, "draw_blend", blend="A", loras=["slot-L1", "slot-L2"], fold="SVD")
        self.run_ok(box, "place_lora", blend="b", where="left", lora="slot-L3")
        self.run_ok(box, "place_lora", blend="blend B", where="right", lora="slot-L5",
                    weight="w4")
        pair = box.outcome()["session"]
        self.assertEqual(pair["blends"]["A"]["tree"]["op"], "SVD")
        self.assertEqual(pair["blends"]["B"]["tree"]["op"], "CAT")
        self.assertEqual(pair["blends"]["B"]["tree"]["children"][1]["weight"], "w4")
        checks = box.outcome()["checks"]
        self.assertEqual((checks["A"]["state"], checks["B"]["state"]), ("ok", "ok"))
        self.assertEqual(box.steps[1]["summary"], "B: slot-L3 at left")
        self.assertIn("\"A\" or \"B\"", box.run("start_over", {"blend": "C"})["error"])

        seed_b = self.run_ok(box, "new_weights", blend="B", seed=7)["seed"]
        self.run_ok(box, "new_weights", blend="A", seed=8)
        self.assertEqual(box.pair["blends"]["B"]["seed"], seed_b)
        self.assertEqual(box.pair["blends"]["A"]["seed"], 8)

    def test_a_random_blend_on_one_side(self):
        box = self.box()
        self.run_ok(box, "draw_blend", blend="A", loras=["slot-L1"])
        before = dict(box.pair["blends"]["A"])
        self.assertEqual(self.run_ok(box, "random_blend", blend="B")["blend"], "B")
        self.assertEqual(box.pair["blends"]["A"], before)
        self.assertEqual(box.outcome()["checks"]["B"]["state"], "ok")
        self.assertIsNone(box.pair["blends"]["B"]["from"])

    def test_a_weight_value_on_one_side(self):
        box = self.box()
        self.run_ok(box, "draw_blend", blend="A", loras=["slot-L1", "slot-L2"])
        self.run_ok(box, "draw_blend", blend="B", loras=["slot-L1", "slot-L2"])
        self.run_ok(box, "set_weight_value", blend="B", weight="w1", value=0.2)
        self.assertEqual((box.pair["blends"]["A"]["values"], box.pair["blends"]["B"]["values"]),
                         ({}, {"w1": 0.2}))
        self.assertEqual(box.outcome()["checks"]["B"]["weights"]["w1"], 0.2)

    def test_copying_and_swapping(self):
        box = self.box()
        self.run_ok(box, "draw_blend", blend="A", loras=["slot-L1"])
        self.run_ok(box, "copy_blend", source="A", target="B")
        self.assertEqual(box.pair["blends"]["A"], box.pair["blends"]["B"])
        self.run_ok(box, "place_fold", blend="B", where="top", op="CAT")
        self.assertNotEqual(box.pair["blends"]["A"]["tree"], box.pair["blends"]["B"]["tree"])
        before = dict(box.pair["blends"])
        self.run_ok(box, "swap_blends")
        self.assertEqual((box.pair["blends"]["A"], box.pair["blends"]["B"]),
                         (before["B"], before["A"]))
        self.assertIn("onto the other", box.run("copy_blend", {"source": "A", "target": "a"})["error"])

    def test_testing_needs_both_and_the_questions(self):
        box = self.box()
        self.run_ok(box, "draw_blend", blend="A", loras=["slot-L1", "slot-L2"])
        self.assertIn("blend B", box.run("start_test", {})["error"])
        self.run_ok(box, "draw_blend", blend="B", loras=["slot-L3", "slot-L5"])
        self.assertIn("questions", box.run("start_test", {})["error"])
        self.run_ok(box, "set_test_questions", file="poems.json", count=5)
        self.assertEqual((box.pair["questions"], box.pair["count"]), ({"file": "poems.json"}, 5))
        self.run_ok(box, "start_test")
        self.assertEqual(box.outcome()["actions"], [{"type": "test"}])
        busy = self.box(box.pair, stage="testing")
        self.assertIn("running", busy.run("start_test", {})["error"])

        planned = compare.plan(self.registry.catalog, self.user, box.pair)
        self.assertEqual(set(planned["tests"]), {"A", "B"})
        for side, body in planned["tests"].items():
            self.assertEqual((body["dataset"], body["count"]), ({"file": "poems.json"}, 5))
            self.assertTrue(body["label"].startswith("compare %s:" % side))
        self.assertIn("slot-L3", planned["message"]["text"])

    def test_the_session_is_checked(self):
        with self.assertRaises(compare.CompareError) as caught:
            compare.session_of({"blends": {"B": {"tree": {"op": "XOR", "children": [None, None]}}}})
        self.assertIn("blend B", str(caught.exception))
        with self.assertRaises(compare.CompareError):
            compare.session_of({"count": 0})
        with self.assertRaises(compare.CompareError):
            compare.session_of({"blends": {"A": {"from": {"job": "one"}}}})
        empty = compare.session_of({})
        self.assertEqual(empty["blends"]["A"], compare.empty_side())
        self.assertEqual(empty["blends"]["B"], compare.empty_side())

    def test_plain_requests_without_a_model(self):
        box = self.box()
        self.run_ok(box, "draw_blend", blend="A", loras=["slot-L1", "slot-L2"])
        out = compare.chat(self.registry, self.registry.catalog, self.user, "copy A to B",
                           "drawing", box.pair, choice=SCRIPTED)
        self.assertEqual(out["session"]["blends"]["B"]["tree"],
                         out["session"]["blends"]["A"]["tree"])
        self.assertTrue(out["message"]["fallback"])
        out = compare.chat(self.registry, self.registry.catalog, self.user, "make it better",
                           "drawing", box.pair, choice=SCRIPTED)
        self.assertEqual(out["steps"], [])
        self.assertEqual(out["message"]["text"], prompts.COMPARE_FALLBACK_CHAT)


class HeadToHeadTests(JobsTestCase):

    @staticmethod
    def report(scores, evaluator="similarity", questions=None):
        questions = questions or ["Q%d?" % n for n in range(1, len(scores) + 1)]
        return {"meta": {"evaluator": evaluator, "judge": "local"},
                "answers": {"blend": [{"question": q, "answer": "a", "quality": s, "reason": ""}
                                      for q, s in zip(questions, scores)]}}

    def test_paired_by_question_and_counted(self):
        a = self.report([0.9, 0.5, 0.2, None, 0.7])
        # B answered in another order, and one question A never saw.
        b = self.report([0.5, 0.1, 0.4, 0.3, 0.6, 0.9],
                        questions=["Q2?", "Q1?", "Q3?", "Q4?", "Q5?", "Q9?"])
        duel = compare.head_to_head(a, b)
        self.assertEqual((duel["questions"], duel["only_a"], duel["only_b"]), (5, 0, 1))
        self.assertEqual((duel["a_wins"], duel["ties"], duel["b_wins"]), (2, 1, 1))
        self.assertEqual(duel["rows"][0]["B"]["quality"], 0.1)      # Q1 found where B put it
        self.assertAlmostEqual(duel["a_mean"], (0.9 + 0.5 + 0.2 + 0.7) / 4)
        self.assertEqual(duel["a_is"], "no clear difference")
        self.assertTrue(duel["same_grader"])

    def test_clear_wins_and_other_graders(self):
        duel = compare.head_to_head(self.report([0.9] * 10), self.report([0.1] * 10, "heuristic"))
        self.assertEqual((duel["a_wins"], duel["b_wins"], duel["a_is"]), (10, 0, "better"))
        self.assertLess(duel["p"], 0.05)
        self.assertFalse(duel["same_grader"])
        self.assertEqual(compare.head_to_head(self.report([0.1] * 10),
                                              self.report([0.9] * 10))["a_is"], "worse")


class EndpointTests(Mixin, ServerTestCase):

    def setUp(self):
        super().setUp()
        self.lora = self.ids()
        leaf = lambda name, weight: {"lora": self.lora[name], "weight": weight}
        self.session = {"blends": {
            "A": {"tree": {"op": "CAT", "children": [leaf("L1", "w1"), leaf("L3", "w2")]},
                  "seed": 5},
            "B": {"tree": {"op": "SVD", "children": [leaf("L3", "w1"), leaf("L5", "w2")]},
                  "seed": 6}},
            "questions": {"given": "mine.jsonl"}, "count": 3, "mock": True}

    def test_config_intro_and_chat(self):
        status, config = self.call("GET", "/agent/compare/config")
        self.assertEqual(status, 200, config)
        self.assertEqual(config["sides"], ["A", "B"])
        self.assertEqual(len(config["loras"]), 5)
        self.assertIn("tested", config["words"]["instructions"])
        status, reply = self.call("POST", "/agent/compare/intro",
                                  {"agent": SCRIPTED, "session": {}})
        self.assertEqual(status, 200, reply)
        self.assertIn("**A**", reply["message"]["text"])
        status, reply = self.call("POST", "/agent/compare/chat",
                                  {"agent": SCRIPTED, "message": "swap them",
                                   "stage": "drawing", "session": self.session})
        self.assertEqual(status, 200, reply)
        self.assertEqual(reply["session"]["blends"]["A"]["seed"], 6)
        self.assertEqual(reply["checks"]["A"]["chromosome"], "SVD.L1.L2.w1.w2")
        self.assertEqual(self.call("POST", "/agent/compare/chat",
                                   {"message": "hi", "session": {"count": -1}})[0], 400)

    def test_open_a_searched_blend_into_a_side(self):
        status, reply = self.call("POST", "/jobs", self.submission())
        self.assertEqual(status, 201, reply)
        search = reply["job"]["id"]
        worker.serve(self.registry, once=True)
        box = self.box(self.session)
        listed = box.run("list_my_blends", {})
        self.assertEqual(listed["jobs"][0]["job"], search)
        self.assertEqual(listed["jobs"][0]["kind"], "search")
        done = box.run("open_blend", {"blend": "B", "job": search})
        self.assertNotIn("error", done, done)
        best = self.call("GET", "/blends/%d" % search)[1]
        side = box.pair["blends"]["B"]
        self.assertEqual((side["tree"], side["seed"], side["number"]),
                         (best["tree"], best["seed"], best["number"]))
        self.assertEqual(side["from"]["job"], search)
        # Just opened it is not edited, whatever slot numbers the search used;
        # other weights are an edit.
        self.assertFalse(box.run("show_blends", {})["B"]["edited_since_opened"])
        box.run("new_weights", {"blend": "B", "seed": best["seed"] + 1})
        self.assertTrue(box.run("show_blends", {})["B"]["edited_since_opened"])
        box.run("open_blend", {"blend": "B", "job": search})
        self.assertEqual(box.pair["blends"]["A"], compare.session_of(self.session)["blends"]["A"])
        self.assertIn("no job", box.run("open_blend", {"blend": "A", "job": 999})["error"])

    def test_both_tested_compared_and_debriefed(self):
        status, planned = self.call("POST", "/agent/compare/plan", {"session": self.session})
        self.assertEqual(status, 200, planned)
        ids = {}
        for side in ("A", "B"):
            body = dict(planned["tests"][side], dataset={"text": LINES, "name": "mine.jsonl"},
                        settings={"EVALUATOR": "similarity"})
            status, reply = self.call("POST", "/blends/test", body)
            self.assertEqual(status, 201, reply)
            ids[side.lower()] = reply["verification"]["id"]
        for _ in range(2):
            worker.serve(self.registry, once=True)
        for one in ids.values():
            self.assertEqual(self.call("GET", "/verifications/%d" % one)[1]
                             ["verification"]["status"], reg.DONE)

        status, found = self.call("POST", "/agent/compare/outcome", ids)
        self.assertEqual(status, 200, found)
        duel = found["head_to_head"]
        self.assertEqual(duel["questions"], 3)
        self.assertEqual(duel["a_wins"] + duel["ties"] + duel["b_wins"], 3)
        self.assertEqual(len(duel["rows"]), 3)
        self.assertIn("slot-L5", found["B"]["report"]["formula"])

        status, reply = self.call("POST", "/agent/compare/debrief", dict(ids, agent=SCRIPTED))
        self.assertEqual(status, 200, reply)
        self.assertIn("Blend A scored", reply["message"]["text"])
        self.assertEqual(self.call("POST", "/agent/compare/outcome", {"a": ids["a"], "b": 999})[0],
                         404)
        bob = self.registry.add_user("bob")
        self.assertEqual(self.call("POST", "/agent/compare/outcome", ids, key=bob)[0], 404)
