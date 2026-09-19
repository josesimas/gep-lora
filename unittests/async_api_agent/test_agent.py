"""
test_agent.py - The LoRA agent: what it reads, what it plans, how it talks.

What async_api_agent owns, and so what is tested here:

  * reading a dataset in any shape a person has to hand -- JSON Lines, a JSON
    array, prompt/answer pairs, CSV with or without a header -- into the
    conversations a LoRA trains on, and the facts about it;
  * turning "how long can you wait" into epochs without a model, and the
    plan that trains them: names free in the user's catalogue, the dataset's
    own first question as the smoke test;
  * the two wire formats, against a stand-in server: what is sent, what is
    read back, and that a key never goes anywhere a page names;
  * the fallback: no key, no endpoint or no model still answers every step;
  * the chat's tools: part of a dataset chosen (selection.py), plain
    requests read without a model (commands.py), what each tool may do at
    each step and what it hands the page (tools.py), and the tool-call round
    trip in both wire formats;
  * the endpoints, with a real worker behind them: the page's whole path from
    a shared dataset to trained LoRAs and a debrief of them, and a chat that
    narrows the dataset and changes the plan on the way.

No test here asks a real model: every provider is the scripted one, or a
stand-in on localhost.
"""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

from async_api import worker
from async_api_agent import agent
from async_api_agent import analysis
from async_api_agent import commands
from async_api_agent import planner
from async_api_agent import prompts
from async_api_agent import providers
from async_api_agent import selection
from async_api_agent import settings
from async_api_agent import tools

from unittests.async_api.support import RECORDS, JobsTestCase
from unittests.async_api.test_server import ServerTestCase

SCRIPTED = {"provider": "scripted"}


def jsonl(records):
    return "\n".join(json.dumps(one) for one in records)


class AnalysisTests(JobsTestCase):

    def test_json_lines_of_conversations_are_read_as_they_are(self):
        found = analysis.analyse(jsonl(RECORDS), "x.jsonl")
        self.assertEqual((found["format"], found["records"], found["usable"]), ("jsonl", 6, True))
        self.assertEqual(json.loads(found["lines"][0]), RECORDS[0])
        self.assertEqual(found["stats"]["assistant_words"]["median"], 4)
        self.assertEqual(found["samples"][0]["user"], "What is question 1?")

    def test_every_other_shape_becomes_conversations(self):
        pairs = [{"prompt": "Hi", "response": "Hello"}, {"question": "Why?", "answer": "Because."}]
        for text, fmt in ((json.dumps(RECORDS), "json-array"),
                          (jsonl(pairs), "jsonl"),
                          ("question,answer\nWhat is it?,It is this.\nAnd that?,That too.\n", "csv"),
                          ("conversation\tsummary\nA long talk.\tShort.\n", "csv")):
            found = analysis.analyse(text)
            self.assertEqual(found["format"], fmt, text)
            self.assertTrue(found["usable"], text)
            first = json.loads(found["lines"][0])["messages"]
            self.assertEqual([turn["role"] for turn in first][-2:], ["user", "assistant"])

    def test_a_csv_with_no_header_uses_its_first_two_columns(self):
        found = analysis.analyse("What is a cat?,A small animal that purrs.\n"
                                 "What is a dog?,A loyal animal that barks.\n")
        self.assertEqual(found["records"], 2)

    def test_what_cannot_be_learned_is_skipped_and_said(self):
        text = jsonl(RECORDS[:2] + [RECORDS[0]]) + "\n" + json.dumps({"messages": [
            {"role": "user", "content": "no answer"}]})
        found = analysis.analyse(text)
        self.assertEqual((found["records"], found["stats"]["skipped"], found["stats"]["duplicates"]),
                         (3, 1, 1))
        self.assertTrue(any("skipped" in one for one in found["problems"]))
        self.assertTrue(any("repeat" in one for one in found["problems"]))

    def test_bare_prompts_are_not_a_dataset_to_train_on(self):
        found = analysis.analyse('{"messages": [{"role": "user", "content": "only a question"}]}')
        self.assertFalse(found["usable"])
        with self.assertRaises(analysis.DatasetError):
            analysis.analyse("   ")


class PlannerTests(JobsTestCase):

    def test_a_typed_wait_is_read_without_a_model(self):
        per_epoch, overhead = 60.0, 120.0
        for said, epochs in (("5 epochs", 5), ("about an hour", 58), ("half an hour", 28),
                             ("90 minutes", 88), ("2h", settings.MAX_EPOCHS),
                             ("the quick one", settings.WAIT_CHOICES[0]["epochs"]),
                             ("10", 10), ("whenever you like", None)):
            self.assertEqual(planner.read_wait(said, per_epoch, overhead)[0], epochs, said)
        self.assertEqual(planner.read_wait("1000 epochs", per_epoch, overhead)[0],
                         settings.MAX_EPOCHS)

    def test_the_estimate_counts_steps_the_way_create_lora_does(self):
        recipe = planner.recipe()
        batch = recipe["batch_size"] * recipe["grad_accum"]
        found = planner.estimate(self.registry.catalog, 50, 3)
        self.assertEqual(found["steps_per_epoch"], -(-50 // batch))
        self.assertEqual(found["steps"], 3 * -(-50 // batch))
        self.assertEqual(found["loras"], len(settings.LORA_RANKS))
        self.assertIn("rough guess", found["source"])

    def test_the_estimate_learns_its_pace_from_loras_trained_here(self):
        catalog = self.registry.catalog
        base = planner.recipe()["base_model"]
        for number, (steps, seconds) in enumerate(((10, 70), (110, 170))):
            catalog.add("old%d" % number, os.path.join(self.folder, "old%d" % number), "ready",
                        "trained", base_model=base, steps=steps, seconds=seconds)
        per_step, overhead, source = planner.speed(catalog, base)
        self.assertAlmostEqual(per_step, 1.0)
        self.assertAlmostEqual(overhead, 60.0)
        self.assertIn("measured from 2", source)

    def test_the_plan_is_one_lora_per_rank_with_free_names(self):
        catalog = self.registry.catalog
        catalog.add("poem-r%d" % settings.LORA_RANKS[0], os.path.join(self.folder, "taken"),
                    "ready", "trained", owner=self.user["name"])
        found = analysis.analyse(jsonl(RECORDS), "poem_lora_dataset.json")
        bodies = planner.plan(catalog, self.user, found, 4, mock=True)
        self.assertEqual([body["settings"]["rank"] for body in bodies], settings.LORA_RANKS)
        self.assertEqual(bodies[0]["name"], "poem-r%d-2" % settings.LORA_RANKS[0])
        self.assertEqual(bodies[0]["settings"]["prompt"], "What is question 1?")
        self.assertEqual(bodies[0]["dataset"], jsonl(RECORDS))
        self.assertTrue(all(body["mock"] for body in bodies))
        # A shared file sent as it is keeps its name, so the record says where it came from.
        shared = planner.plan(catalog, self.user, found, 4, shared_file="poem_lora_dataset.json")
        self.assertEqual(shared[0]["dataset"], {"file": "poem_lora_dataset.json"})


class Stand_in(BaseHTTPRequestHandler):
    """A provider on localhost: records what it was sent, answers as told."""

    replies = []
    seen = []

    def log_message(self, *args):
        pass

    def _answer(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.seen.append(("GET", self.path, {k.lower(): v for k, v in self.headers.items()}, None))
        self._answer(200, {"data": [{"id": "chatty"}, {"id": "text-embed"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.seen.append(("POST", self.path, {k.lower(): v for k, v in self.headers.items()}, body))
        status, payload = self.replies.pop(0)
        self._answer(status, payload)


class ProviderTests(JobsTestCase):

    def setUp(self):
        super().setUp()
        Stand_in.replies, Stand_in.seen = [], []
        self.server = HTTPServer(("127.0.0.1", 0), Stand_in)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = "http://127.0.0.1:%d/v1" % self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        super().tearDown()

    def test_a_hosted_provider_needs_its_key_and_keeps_its_url(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            with self.assertRaisesRegex(providers.ProviderError, "OPENAI_API_KEY"):
                providers.resolve({"provider": "openai"})
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-secret"}):
            with self.assertRaisesRegex(providers.ProviderError, "cannot be pointed"):
                providers.resolve({"provider": "openai", "base_url": self.url})
            described = json.dumps(providers.describe())
        self.assertNotIn("sk-secret", described)

    def test_a_local_provider_pointed_elsewhere_goes_without_a_key(self):
        with mock.patch.dict(os.environ, {"JUDGE_API_KEY": "judge-secret"}):
            resolved = providers.resolve({"provider": "lmstudio", "base_url": self.url})
        self.assertEqual((resolved["base_url"], resolved["api_key"]), (self.url, ""))

    def test_an_openai_compatible_reply_and_what_was_sent(self):
        Stand_in.replies = [(200, {"choices": [{"message": {
            "content": "<think>hmm</think>Hello there!"}}]})]
        resolved = providers.resolve({"provider": "lmstudio", "base_url": self.url})
        text, model = providers.chat(resolved, "SYSTEM", [
            {"role": "assistant", "content": "a welcome no API takes first"},
            {"role": "user", "content": "one"}, {"role": "user", "content": "two"}])
        self.assertEqual((text, model), ("Hello there!", "chatty"))   # thinking gone; first chat model
        body = Stand_in.seen[-1][3]
        self.assertEqual(body["messages"], [{"role": "system", "content": "SYSTEM"},
                                            {"role": "user", "content": "one\n\ntwo"}])
        self.assertEqual(body["max_tokens"], settings.MAX_TOKENS)

    def test_an_anthropic_reply_is_its_text_blocks(self):
        Stand_in.replies = [
            (400, {"error": {"message": "output_config.effort is not supported"}}),
            (200, {"stop_reason": "end_turn", "content": [
                {"type": "thinking", "thinking": ""}, {"type": "text", "text": "Hi!"}]})]
        with mock.patch.dict(settings.PROVIDERS["anthropic"], base_url=self.url), \
                mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "ak"}):
            resolved = providers.resolve({"provider": "anthropic", "model": "claude-haiku-4-5"})
            text, _ = providers.chat(resolved, "SYSTEM", [{"role": "user", "content": "hey"}])
        self.assertEqual(text, "Hi!")
        (_, path, headers, first), (_, _, _, second) = Stand_in.seen
        self.assertEqual(path, "/v1/messages")
        self.assertEqual((headers["x-api-key"], headers["anthropic-version"]),
                         ("ak", providers.ANTHROPIC_VERSION))
        self.assertEqual(first["system"], "SYSTEM")
        self.assertNotIn("temperature", first)
        self.assertIn("output_config", first)
        self.assertNotIn("output_config", second)     # retried without what it refused

    def test_every_step_answers_without_a_model(self):
        dead = {"provider": "lmstudio", "base_url": "http://127.0.0.1:9/v1"}
        for choice, noted in ((SCRIPTED, False), (dead, True)):
            message = agent.intro(choice, mock=True)["message"]
            self.assertTrue(message["fallback"])
            self.assertIn("practice run", message["text"])
            self.assertEqual(bool(message["note"]), noted)
        found = analysis.analyse(jsonl(RECORDS), "x")
        self.assertIn("6 usable", agent.summarise(found, SCRIPTED)["text"])
        self.assertIn("I can't train", agent.summarise(
            analysis.analyse('{"messages": [{"role": "user", "content": "q"}]}'), SCRIPTED)["text"])

    def test_every_step_has_a_prompt_and_it_carries_the_persona(self):
        for step in prompts.STEPS:
            self.assertIn(prompts.PERSONA, prompts.system(step), step)

    def test_every_tool_is_described_and_summarised(self):
        self.assertEqual(set(tools.SPECS), set(prompts.TOOLS))
        self.assertEqual(set(tools.SPECS), set(prompts.TOOL_DONE))

    def test_a_tool_round_trip_in_the_openai_shape(self):
        Stand_in.replies = [
            (200, {"choices": [{"message": {"content": "", "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "select_records", "arguments": '{"first": 2}'}}]}}]}),
            (200, {"choices": [{"message": {"content": "Now training on the first 2."}}]})]
        reply = agent.chat(self.registry.catalog, self.user, "only the first 2", "analysis",
                           {}, (jsonl(RECORDS), "x", None), None,
                           {"provider": "lmstudio", "base_url": self.url, "model": "chatty"}, [])
        self.assertEqual(reply["message"]["text"], "Now training on the first 2.")
        self.assertEqual(reply["session"]["selection"], {"first": 2})
        self.assertEqual(reply["analysis"]["records"], 2)
        self.assertEqual([step["tool"] for step in reply["steps"]], ["select_records"])
        first, second = Stand_in.seen[0][3], Stand_in.seen[1][3]
        self.assertIn("select_records", [tool["function"]["name"] for tool in first["tools"]])
        assistant, result = second["messages"][-2:]
        self.assertEqual(assistant["tool_calls"][0]["id"], "c1")
        self.assertEqual((result["role"], result["tool_call_id"]), ("tool", "c1"))
        self.assertEqual(json.loads(result["content"])["kept"], 2)

    def test_a_tool_round_trip_in_the_anthropic_shape(self):
        blocks = [{"type": "thinking", "thinking": "", "signature": "sig"},
                  {"type": "tool_use", "id": "t1", "name": "set_loras", "input": {"ranks": [32]}}]
        Stand_in.replies = [
            (200, {"stop_reason": "tool_use", "content": blocks}),
            (200, {"stop_reason": "end_turn", "content": [{"type": "text", "text": "Rank 32."}]})]
        with mock.patch.dict(settings.PROVIDERS["anthropic"], base_url=self.url), \
                mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "ak"}):
            reply = agent.chat(self.registry.catalog, self.user, "one lora at rank 32", "wait",
                               {}, (jsonl(RECORDS), "x", None), None, {"provider": "anthropic"}, [])
        self.assertEqual(reply["session"]["ranks"], [32])
        self.assertEqual(reply["actions"], [{"type": "choices"}])
        second = Stand_in.seen[1][3]
        # The assistant's blocks go back exactly as they came, thinking included.
        self.assertEqual(second["messages"][-2], {"role": "assistant", "content": blocks})
        self.assertEqual(second["messages"][-1]["content"][0]["tool_use_id"], "t1")
        self.assertEqual(second["tools"][0].keys(), {"name", "description", "input_schema"})


class SelectionTests(JobsTestCase):

    def test_filters_then_position_then_sample(self):
        records = RECORDS + [RECORDS[0]]
        kept = selection.indexed(records, {"drop_duplicates": True, "last": 2})
        self.assertEqual([number for number, _ in kept], [5, 6])
        kept = selection.indexed(records, {"contains": ["answer 3", "answer 5"], "first": 1})
        self.assertEqual([number for number, _ in kept], [3])
        kept = selection.indexed(records, {"start": 2, "end": 4, "sample": 2, "seed": 1})
        self.assertEqual(len(kept), 2)
        self.assertTrue(all(2 <= number <= 4 for number, _ in kept))
        self.assertEqual(selection.indexed(records, {"sample": 2, "seed": 1}),
                         selection.indexed(records, {"sample": 2, "seed": 1}))

    def test_a_new_position_replaces_the_old_and_filters_add_up(self):
        chosen = selection.merge({"first": 3, "excludes": ["x"]}, {"last": 2})
        self.assertEqual(chosen, {"last": 2, "excludes": ["x"]})
        self.assertEqual(selection.merge(chosen, {"excludes": None}), {"last": 2})

    def test_nonsense_is_refused_in_words(self):
        for bad in ({"first": 0}, {"percent": 150}, {"first": 2, "last": 2},
                    {"start": 5, "end": 2}, {"colour": "red"}):
            with self.assertRaises(selection.SelectionError):
                selection.clean(bad)

    def test_it_is_described_with_what_it_keeps(self):
        self.assertIn("first 20", selection.describe({"first": 20}, 20, 60))
        self.assertIn("20 of 60", selection.describe({"first": 20}, 20, 60))
        self.assertEqual(selection.describe({}, 60, 60), "all 60 record(s)")


class CommandTests(JobsTestCase):

    def test_plain_requests_become_the_calls_a_model_would_make(self):
        for said, stage, calls in (
                ("only use the first 20 and drop duplicates", "analysis",
                 [("select_records", {"first": 20, "drop_duplicates": True})]),
                ("records 5 to 30 please", "wait", [("select_records", {"start": 5, "end": 30})]),
                ("leave out the ones about fever", "wait",
                 [("select_records", {"excludes": ["fever"]})]),
                ("one lora at rank 32 with a learning rate of 1e-4", "wait",
                 [("set_loras", {"ranks": [32]}),
                  ("set_training_options", {"learning_rate": 1e-4})]),
                ("ranks 4, 8 and 16", "confirm", [("set_loras", {"ranks": [4, 8, 16]})]),
                ("I can wait half an hour", "wait", [("set_epochs", {"minutes": 30})]),
                ("make it a practice run", "wait", [("set_practice_run", {"on": True})]),
                ("show me records 3 to 4", "confirm", [("show_records", {"start": 3, "count": 2})]),
                ("go ahead", "confirm", [("start_training", {})]),
                ("stop!", "training", [("stop_training", {})]),
                ("use the poem dataset", "dataset",
                 [("use_demo_dataset", {"file": "poem_lora_dataset.json"})])):
            self.assertEqual(commands.parse(said, stage, ["poem_lora_dataset.json",
                                                          "uppercase_lora_dataset.json"]),
                             calls, said)

    def test_what_only_looks_like_a_request_is_left_alone(self):
        for said in ("start from record 5", "what is a rank?", "what is a real lora?",
                     "tell me about epochs"):
            self.assertEqual(commands.parse(said, "wait"), [], said)


class ToolboxTests(JobsTestCase):

    def box(self, stage="wait", session=None, dataset=True):
        return tools.Toolbox(self.registry.catalog, self.user, stage, session or {},
                             (jsonl(RECORDS), "x.jsonl", None) if dataset else None,
                             lambda: [{"file": "poem_lora_dataset.json", "records": 50,
                                       "usable": True, "keywords": []}])

    def test_the_stage_and_the_dataset_decide_what_may_be_asked(self):
        self.assertIn("training is running",
                      self.box("training").run("select_records", {"first": 2})["error"])
        self.assertIn("no dataset", self.box("dataset", dataset=False)
                      .run("select_records", {"first": 2})["error"])
        self.assertIn("error", self.box("wait").run("stop_training", {}))
        self.assertIn("epochs first", self.box("wait").run("start_training", {})["error"])
        self.assertIn("no tool", self.box().run("format_disk", {})["error"])

    def test_a_selection_that_keeps_nothing_changes_nothing(self):
        box = self.box(session={"selection": {"first": 3}})
        self.assertIn("no records", box.run("select_records", {"contains": ["zebra"]})["error"])
        self.assertEqual(box.session["selection"], {"first": 3})
        self.assertFalse(box.steps[0]["ok"])

    def test_what_the_page_is_told_to_do(self):
        box = self.box("confirm", {"epochs": 2})
        box.run("set_loras", {"ranks": [4]})
        self.assertEqual(box.outcome()["actions"], [{"type": "plan", "epochs": 2}])
        box = self.box("wait")
        box.run("set_epochs", {"minutes": 30})
        out = box.outcome()
        self.assertEqual(out["actions"][0]["type"], "plan")
        self.assertGreaterEqual(out["session"]["epochs"], settings.MIN_EPOCHS)
        box = self.box("confirm", {"epochs": 2})
        box.run("start_training", {})
        self.assertEqual([one["type"] for one in box.outcome()["actions"]], ["plan", "start"])
        box = self.box("wait")
        box.run("use_demo_dataset", {"file": "poem"})
        self.assertEqual(box.outcome()["actions"],
                         [{"type": "dataset", "file": "poem_lora_dataset.json"}])

    def test_options_are_checked_by_train_pys_rules(self):
        box = self.box()
        self.assertIn("learning_rate", box.run("set_training_options",
                                               {"learning_rate": 5})["error"])
        box.run("set_training_options", {"learning_rate": 1e-4, "name": "My Poems!"})
        self.assertEqual((box.session["options"], box.session["name"]),
                         ({"learning_rate": 1e-4}, "my-poems"))
        too_many = list(range(1, settings.MAX_LORAS + 2))
        self.assertIn("at most", box.run("set_loras", {"ranks": too_many})["error"])
        self.assertNotIn("error", box.run("set_loras", {"ranks": too_many[:-1]}))

    def test_a_plan_follows_the_session(self):
        found = analysis.analyse(jsonl(RECORDS), "poem_lora_dataset.json", {"first": 3})
        session = planner.session_of({"ranks": [4], "options": {"learning_rate": 1e-4},
                                      "name": "verse", "prompt": "Say hi"})
        bodies = planner.plan(self.registry.catalog, self.user, found, 2, True,
                              "poem_lora_dataset.json", session)
        self.assertEqual([body["name"] for body in bodies], ["verse-r4"])
        self.assertEqual(bodies[0]["settings"], {"learning_rate": 1e-4, "epochs": 2.0,
                                                 "rank": 4, "prompt": "Say hi"})
        # Part of a shared file goes as its lines: the whole file is not what it trains on.
        self.assertEqual(len(bodies[0]["dataset"].splitlines()), 3)


class EndpointTests(ServerTestCase):

    def test_the_page_is_served_and_the_agent_needs_a_key(self):
        status, _, body = self.download("/agent")
        self.assertEqual(status, 200)
        self.assertIn(b"<title>LoRA guide</title>", body)
        self.assertEqual(self.call("GET", "/agent/config", key="nope")[0], 401)
        status, config = self.call("GET", "/agent/config")
        self.assertEqual(status, 200, config)
        self.assertEqual(config["ranks"], settings.LORA_RANKS)
        self.assertIn("scripted", [one["name"] for one in config["providers"]])
        self.assertTrue(any(one["usable"] for one in config["datasets"]))
        self.assertIn("dataset", config["words"]["instructions"])

    def test_a_shared_dataset_from_analysis_to_trained_loras(self):
        status, reply = self.call("POST", "/agent/analyse", {
            "agent": SCRIPTED, "dataset": {"file": "poem_lora_dataset.json"}})
        self.assertEqual(status, 200, reply)
        self.assertEqual(reply["shared_file"], "poem_lora_dataset.json")
        self.assertNotIn("lines", reply["analysis"])
        records = reply["analysis"]["records"]

        status, reply = self.call("POST", "/agent/wait", {"agent": SCRIPTED, "records": records,
                                                          "mock": True})
        self.assertEqual([one["id"] for one in reply["choices"]],
                         [one["id"] for one in settings.WAIT_CHOICES])

        # A wait typed into the chat, read without a model.
        status, reply = self.call("POST", "/agent/chat", {
            "agent": SCRIPTED, "message": "2 epochs", "stage": "wait", "session": {"mock": True},
            "dataset": {"file": "poem_lora_dataset.json"}})
        self.assertEqual(status, 200, reply)
        self.assertEqual(reply["session"]["epochs"], 2)
        self.assertEqual(reply["actions"], [{"type": "plan", "epochs": 2}])

        status, plan = self.call("POST", "/agent/plan", {
            "dataset": {"file": "poem_lora_dataset.json"}, "epochs": 1, "mock": True})
        self.assertEqual(status, 200, plan)
        self.assertIn("Shall I start", plan["message"]["text"])

        # The page's half: the plan goes to the API's own POST /loras as it is.
        ids = []
        for body in plan["trainings"]:
            status, queued = self.call("POST", "/loras", body)
            self.assertEqual(status, 201, queued)
            ids.append(queued["lora"]["id"])
        for _ in ids:
            worker.serve(self.registry, once=True, poll=0.05, catalog=self.app.catalog)
        for lora_id in ids:
            lora = self.call("GET", "/loras/%d" % lora_id)[1]["lora"]
            self.assertEqual(lora["status"], "ready", lora)
            self.assertEqual(lora["dataset"], "datasets/poem_lora_dataset.json")

        # The debrief reads the user's own rows, and nobody else's.
        bob = self.registry.add_user("bob")
        status, reply = self.call("POST", "/agent/debrief", {"agent": SCRIPTED, "loras": ids},
                                  key=bob)
        self.assertEqual(reply["loras"], [])
        status, reply = self.call("POST", "/agent/debrief", {"agent": SCRIPTED, "loras": ids,
                                                             "mock": True})
        self.assertEqual([one["status"] for one in reply["loras"]], ["ready"] * len(ids))
        self.assertIn("practice run", reply["message"]["text"])

    def test_a_chat_narrows_the_dataset_and_the_plan_follows(self):
        status, reply = self.call("POST", "/agent/chat", {
            "agent": SCRIPTED, "message": "only use the first 12, rank 4", "stage": "analysis",
            "session": {}, "dataset": {"file": "poem_lora_dataset.json"}})
        self.assertEqual(status, 200, reply)
        self.assertEqual(reply["analysis"]["records"], 12)
        self.assertEqual(reply["session"]["ranks"], [4])
        self.assertTrue(reply["message"]["fallback"])
        status, plan = self.call("POST", "/agent/plan", {
            "dataset": {"file": "poem_lora_dataset.json"}, "epochs": 1, "mock": True,
            "session": reply["session"]})
        self.assertEqual(status, 200, plan)
        self.assertEqual([body["settings"]["rank"] for body in plan["trainings"]], [4])
        self.assertEqual(len(plan["trainings"][0]["dataset"].splitlines()), 12)
        status, lora = self.call("POST", "/loras", plan["trainings"][0])
        self.assertEqual(status, 201, lora)
        self.assertEqual(lora["lora"]["records"], 12)
        # A question with no tool behind it, and no model, is answered anyway.
        status, reply = self.call("POST", "/agent/chat", {
            "agent": SCRIPTED, "message": "what is a rank?", "stage": "wait", "session": {}})
        self.assertEqual((status, reply["steps"], reply["actions"]), (200, [], []))

    def test_bad_requests_are_400s(self):
        for path, body in (("/agent/analyse", {"dataset": {"file": "../config/settings.py"}}),
                           ("/agent/analyse", {"dataset": {"text": "  "}}),
                           ("/agent/analyse", {"dataset": {"text": "[not json"}}),
                           ("/agent/wait", {"records": 0}),
                           ("/agent/plan", {"dataset": {"text": jsonl(RECORDS)}, "epochs": 0}),
                           ("/agent/chat", {"message": ""}),
                           ("/agent/chat", {"message": "hi", "session": {"ranks": [0]}}),
                           ("/agent/plan", {"dataset": {"text": jsonl(RECORDS)}, "epochs": 1,
                                            "session": {"selection": {"first": -1}}})):
            status, reply = self.call("POST", path, body)
            self.assertEqual(status, 400, (path, reply))
        status, reply = self.call("GET", "/agent/models?provider=openai&base_url=http://x/v1")
        self.assertEqual(status, 502, reply)
