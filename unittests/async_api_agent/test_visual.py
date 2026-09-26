"""
test_visual.py - The visual guide: the chat's tools on a drawing, and its steps.

  * every tool is described, summarised and in the schema the model is sent;
  * the tools edit the session's drawing the way the page's clicks do -- a LoRA
    at a place, a fold at a place, weights, swaps, clearing -- and refuse what
    would not be a drawing (the top is always a CAT);
  * a place may be named as a path or in words ("right › left");
  * starting a test is an action for the page, and only for a drawing that can
    be built with questions chosen;
  * the endpoints: config, intro, chat without a model, the POST /blends/test
    body read back, and a debrief of a real verification.
"""

from async_api import drawn
from async_api import registry as reg
from async_api import worker
from async_api_agent import prompts
from async_api_agent import visual

from unittests.async_api.support import JobsTestCase
from unittests.async_api.test_server import ServerTestCase

SCRIPTED = {"provider": "scripted"}


class Mixin:

    def ids(self):
        self.own_slots()
        # slot-L1 16, slot-L2 16, slot-L3 8, slot-L4 4, slot-L5 32
        return {row["name"][5:]: row["id"]
                for row in self.registry.catalog.all(owner=self.user["name"])}


class ToolTests(Mixin, JobsTestCase):

    def setUp(self):
        super().setUp()
        self.lora = self.ids()
        self.box = visual.Toolbox(self.registry.catalog, self.user, "drawing", {},
                                  lambda: [{"file": "poems.json", "records": 9, "usable": True}])

    def run_ok(self, name, **arguments):
        result = self.box.run(name, arguments)
        self.assertNotIn("error", result, (name, result))
        return result

    def refused(self, name, fragment, **arguments):
        result = self.box.run(name, arguments)
        self.assertIn("error", result, name)
        self.assertIn(fragment, result["error"])

    def test_every_tool_is_described_and_summarised(self):
        self.assertEqual(set(visual.SPECS), set(prompts.VISUAL_TOOLS))
        self.assertEqual(set(visual.SPECS), set(prompts.VISUAL_TOOL_DONE))
        self.assertEqual([one[0] for one in visual.schemas()], list(visual.SPECS))
        for step in prompts.VISUAL_STEPS:
            self.assertIn(prompts.VISUAL_PERSONA, prompts.system(step))

    def test_places_in_words_or_paths(self):
        self.assertEqual(visual.path_of("top"), "")
        self.assertEqual(visual.path_of("1.0"), "1.0")
        self.assertEqual(visual.path_of("right › left"), "1.0")
        self.assertEqual(visual.path_of("left right"), "0.1")
        with self.assertRaises(visual.VisualError):
            visual.path_of("up")

    def test_drawing_with_the_tools(self):
        self.run_ok("place_lora", where="left", lora="slot-L1")
        self.run_ok("place_fold", where="right", op="SVD")
        # A LoRA given a fold fills its first empty side, with the next unused weight.
        done = self.run_ok("place_lora", where="right", lora=self.lora["L3"])
        self.assertEqual(done["where"], "right › left")
        self.run_ok("place_lora", where="1.1", lora="slot-L5", weight="w7")
        tree = self.box.session["tree"]
        self.assertEqual(drawn.encode(tree)[0], "CAT.L1.SVD.w1.L2.L3.w2.w7")
        self.run_ok("swap_sides", where="right")
        self.assertEqual(self.box.session["tree"]["children"][1]["children"],
                         list(reversed(tree["children"][1]["children"])))
        self.run_ok("set_weight", where="left", weight="w4")
        self.run_ok("place_fold", where="left", op="LIN")        # a LoRA is folded, not lost
        left = self.box.session["tree"]["children"][0]
        self.assertEqual((left["op"], left["children"][0]["weight"], left["children"][1]),
                         ("LIN", "w4", None))
        self.run_ok("clear_place", where="left")
        self.assertIsNone(self.box.session["tree"]["children"][0])
        found = self.box.outcome()["check"]
        self.assertEqual(found["state"], "incomplete")
        self.assertTrue(self.box.changed)

    def test_the_refusals(self):
        self.refused("place_fold", "always a CAT", where="top", op="SVD")
        self.refused("clear_place", "cannot be emptied", where="top")
        self.refused("place_lora", "no LoRA of yours", where="left", lora="nobody's")
        self.refused("set_weight", "no LoRA at", where="left", weight="w2")
        self.refused("swap_sides", "no fold", where="left")
        self.refused("place_lora", "not a place", where="up", lora="slot-L1")
        self.refused("draw_blend", "two or more", loras=["slot-L1"])
        self.refused("set_test_questions", "no demo dataset", file="nope.json")
        self.refused("start_test", "Fill every empty place")
        self.refused("nothing", "no tool")

    def test_draw_blend_folds_in_pairs_under_a_cat(self):
        done = self.run_ok("draw_blend", loras=["slot-L1", "slot-L2", "slot-L3"], fold="SVD")
        self.assertEqual(done["formula_text"], "slot-L3, slot-L1, slot-L2")
        chromosome, _ = drawn.encode(self.box.session["tree"])
        self.assertEqual(chromosome, "CAT.SVD.L1.L2.L3.w3.w1.w2")

    def test_a_test_is_an_action_once_it_can_run(self):
        self.run_ok("draw_blend", loras=["slot-L1", "slot-L2"])
        self.refused("start_test", "choose the questions")
        self.run_ok("set_test_questions", file="poems.json", count=7)
        self.assertEqual(self.box.session["questions"], {"file": "poems.json"})
        self.assertEqual(self.box.session["count"], 7)
        self.run_ok("start_test")
        self.assertEqual(self.box.outcome()["actions"], [{"type": "test"}])
        busy = visual.Toolbox(self.registry.catalog, self.user, "testing", self.box.session)
        self.assertIn("running", busy.run("start_test", {})["error"])

    def test_the_session_is_checked(self):
        with self.assertRaises(visual.VisualError):
            visual.session_of({"tree": {"op": "SVD", "children": [None, None]}})
        with self.assertRaises(visual.VisualError):
            visual.session_of({"count": 0})
        with self.assertRaises(visual.VisualError):
            visual.session_of({"questions": {"text": "a whole dataset"}})
        self.assertEqual(visual.session_of({})["tree"], visual.empty_tree())

    def test_plain_requests_without_a_model(self):
        self.run_ok("draw_blend", loras=["slot-L1", "slot-L2"])
        out = visual.chat(self.registry.catalog, self.user, "start over", "drawing",
                          self.box.session, choice=SCRIPTED)
        self.assertEqual(out["session"]["tree"], visual.empty_tree())
        self.assertTrue(out["message"]["fallback"])
        # Anything else is no call at all, never a guess.
        out = visual.chat(self.registry.catalog, self.user, "make it better", "drawing",
                          self.box.session, choice=SCRIPTED)
        self.assertEqual(out["steps"], [])
        self.assertEqual(out["message"]["text"], prompts.VISUAL_FALLBACK_CHAT)


class EndpointTests(Mixin, ServerTestCase):

    def setUp(self):
        super().setUp()
        self.lora = self.ids()
        self.session = {"tree": {"op": "CAT", "children": [
            {"lora": self.lora["L1"], "weight": "w1"},
            {"op": "SVD", "children": [{"lora": self.lora["L3"], "weight": "w2"},
                                       {"lora": self.lora["L5"], "weight": "w3"}]}]},
            "seed": 5, "questions": None, "count": 3, "mock": True}

    def test_config_intro_and_chat(self):
        status, config = self.call("GET", "/agent/visual/config")
        self.assertEqual(status, 200, config)
        self.assertEqual(len(config["loras"]), 5)
        self.assertEqual(set(config["folds"]), {"CAT", "SVD", "LIN"})
        self.assertIn("drawing", config["words"]["instructions"])
        status, reply = self.call("POST", "/agent/visual/intro",
                                  {"agent": SCRIPTED, "session": {}})
        self.assertEqual(status, 200, reply)
        self.assertIn("CAT", reply["message"]["text"])
        status, reply = self.call("POST", "/agent/visual/chat",
                                  {"agent": SCRIPTED, "message": "new weights",
                                   "stage": "drawing", "session": self.session})
        self.assertEqual(status, 200, reply)
        self.assertNotEqual(reply["session"]["seed"], 5)
        self.assertEqual(reply["check"]["state"], "ok")
        self.assertEqual(self.call("POST", "/agent/visual/chat",
                                   {"message": "hi", "session": {"count": -1}})[0], 400)

    def test_the_test_planned_sent_and_debriefed(self):
        status, reply = self.call("POST", "/agent/visual/plan", {"session": self.session})
        self.assertEqual(status, 400)                       # no questions yet
        self.assertIn("questions", reply["error"])
        lines = "\n".join('{"messages": [{"role": "user", "content": "Q%d?"}, '
                          '{"role": "assistant", "content": "A%d."}]}' % (n, n)
                          for n in range(1, 5))
        session = dict(self.session, questions={"given": "mine.jsonl"})
        status, planned = self.call("POST", "/agent/visual/plan", {"session": session})
        self.assertEqual(status, 200, planned)
        self.assertEqual(planned["source"], "mine.jsonl")
        self.assertIn("slot-L3", planned["message"]["text"])
        body = dict(planned["test"], dataset={"text": lines, "name": "mine.jsonl"})
        status, reply = self.call("POST", "/blends/test", body)
        self.assertEqual(status, 201, reply)
        job_id, verification_id = reply["job"]["id"], reply["verification"]["id"]

        # Opened again, it is the same drawing under the same seed.
        status, found = self.call("GET", "/blends/%d" % job_id)
        self.assertEqual(status, 200, found)
        self.assertEqual(found["tree"], self.session["tree"])
        self.assertEqual((found["seed"], found["verification"]), (5, verification_id))

        worker.serve(self.registry, once=True)
        status, reply = self.call("POST", "/agent/visual/debrief",
                                  {"agent": SCRIPTED, "verification": verification_id})
        self.assertEqual(status, 200, reply)
        self.assertEqual(reply["result"]["status"], reg.DONE)
        self.assertIn("slot-L3", reply["message"]["text"])
        self.assertEqual(self.call("POST", "/agent/visual/debrief",
                                   {"verification": 999})[0], 404)
