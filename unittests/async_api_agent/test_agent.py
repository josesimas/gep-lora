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
  * the endpoints, with a real worker behind them: the page's whole path from
    a shared dataset to trained LoRAs and a debrief of them.

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
from async_api_agent import planner
from async_api_agent import prompts
from async_api_agent import providers
from async_api_agent import settings

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
            text = prompts.system(step)
            self.assertEqual(step == "wait_interpret", prompts.PERSONA not in text, step)


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

        status, reply = self.call("POST", "/agent/interpret", {"agent": SCRIPTED, "records": records,
                                                               "text": "2 epochs", "mock": True})
        self.assertEqual(reply["epochs"], 2)

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

    def test_bad_requests_are_400s(self):
        for path, body in (("/agent/analyse", {"dataset": {"file": "../config/settings.py"}}),
                           ("/agent/analyse", {"dataset": {"text": "  "}}),
                           ("/agent/analyse", {"dataset": {"text": "[not json"}}),
                           ("/agent/wait", {"records": 0}),
                           ("/agent/plan", {"dataset": {"text": jsonl(RECORDS)}, "epochs": 0}),
                           ("/agent/chat", {"message": ""})):
            status, reply = self.call("POST", path, body)
            self.assertEqual(status, 400, (path, reply))
        status, reply = self.call("GET", "/agent/models?provider=openai&base_url=http://x/v1")
        self.assertEqual(status, 502, reply)
