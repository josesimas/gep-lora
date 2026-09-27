"""
test_agent.py - The LoRA agent: what it reads, what it plans, how it talks.

What async_api_agent owns, and so what is tested here:

  * reading a dataset in any shape a person has to hand -- JSON Lines, a JSON
    array, prompt/answer pairs, CSV with or without a header -- into the
    conversations a LoRA trains on, and the facts about it;
  * turning "how long can you wait" into epochs without a model, and the
    plan that trains them: names free in the user's catalogue, the dataset's
    own first question as the smoke test;
  * the three wire formats, against a stand-in server: what is sent, what is
    read back, which one a gateway's model is asked in, and that a key never
    goes anywhere a page names;
  * the fallback: no key, no endpoint or no model still answers every step;
  * the chat's tools: part of a dataset chosen (selection.py), plain
    requests read without a model (commands.py), what each tool may do at
    each step and what it hands the page (tools.py), and the tool-call round
    trip in every wire format;
  * the endpoints, with a real worker behind them: the page's whole path from
    a shared dataset to trained LoRAs and a debrief of them, and a chat that
    narrows the dataset and changes the plan on the way;
  * the second half -- blending: only the user's own ready LoRAs, on one
    base model, round the five places; the POST /jobs body the plan becomes;
    the chat's blend tools and their no-model commands; and trained LoRAs
    blended by a real search and read back, which another user can neither
    plan with, submit nor read;
  * after the search: every blend tested, one picked and verified against
    each of its LoRAs on the validation split, and a model put live and
    asked -- through the API's own endpoints, with the chat's tools and
    their no-model commands for each.

No test here asks a real model: every provider is the scripted one, or a
stand-in on localhost.
"""

import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

from async_api import worker
from async_api_agent import agent
from async_api_agent import analysis
from async_api_agent import blending
from async_api_agent import commands
from async_api_agent import guide_defaults
from async_api_agent import planner
from async_api_agent import prompts
from async_api_agent import providers
from async_api_agent import selection
from async_api_agent import settings
from async_api_agent import tools
from async_api_agent import ui_help

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
        # Named for the dataset, the base model, the day and the rank.
        first = planner.lora_name("poem", planner.recipe()["base_model"], settings.LORA_RANKS[0])
        self.assertRegex(first, r"^poem-[a-z0-9._-]+-\d{8}-r%d$" % settings.LORA_RANKS[0])
        self.assertIn(planner.model_tag(planner.recipe()["base_model"]), first)
        catalog.add(first, os.path.join(self.folder, "taken"),
                    "ready", "trained", owner=self.user["name"])
        found = analysis.analyse(jsonl(RECORDS), "poem_lora_dataset.json")
        bodies = planner.plan(catalog, self.user, found, 4, mock=True)
        self.assertEqual([body["settings"]["rank"] for body in bodies], settings.LORA_RANKS)
        self.assertEqual(bodies[0]["name"], first + "-2")
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

    def test_thinking_is_switched_in_each_wires_own_words(self):
        self.assertEqual(providers.thinking_fields("openai", None), {})
        self.assertEqual(providers.thinking_fields("openai", False), {"reasoning_effort": "none"})
        self.assertEqual(providers.thinking_fields("openai", True), {"reasoning_effort": "medium"})
        self.assertEqual(providers.thinking_fields("anthropic", False),
                         {"thinking": {"type": "disabled"}})
        self.assertEqual(providers.thinking_fields("responses", False),
                         {"reasoning": {"effort": "none"}})
        # The page's choice wins over the default; unset leaves the request alone.
        Stand_in.replies = [(200, {"choices": [{"message": {"content": "a"}}]}),
                            (422, {"error": {"message": "unknown field reasoning_effort"}}),
                            (200, {"choices": [{"message": {"content": "b"}}]}),
                            (200, {"choices": [{"message": {"content": "c"}}]})]
        off = providers.resolve({"provider": "lmstudio", "base_url": self.url, "model": "m",
                                 "thinking": False})
        self.assertEqual(providers.chat(off, "S", [{"role": "user", "content": "q"}])[0], "a")
        self.assertEqual(Stand_in.seen[-1][3]["reasoning_effort"], "none")
        # An endpoint that refuses the switch is asked again without it.
        self.assertEqual(providers.chat(off, "S", [{"role": "user", "content": "q"}])[0], "b")
        self.assertNotIn("reasoning_effort", Stand_in.seen[-1][3])
        with guide_defaults.applied({"thinking": True}):
            left = providers.resolve({"provider": "lmstudio", "base_url": self.url, "model": "m"})
        providers.chat(left, "S", [{"role": "user", "content": "q"}])
        self.assertEqual(Stand_in.seen[-1][3]["reasoning_effort"], "medium")

    def test_the_judge_is_asked_to_think_or_not(self):
        from evaluators import common
        base = {"backend": "endpoint", "base_url": self.url, "api_key": "", "model": "j",
                "temperature": 0.0, "max_tokens": 50, "timeout": 10, "retries": 0,
                "retry_wait": 0, "response_format": None}
        good = (200, {"choices": [{"message": {"content": '{"quality": 0.5, "reason": "ok"}'}}]})
        Stand_in.replies = [good, (400, {"error": {"message": "no reasoning_effort"}}), good, good]
        self.assertEqual(common.ask_judge("S", "U", dict(base, thinking=False))[0], 0.5)
        self.assertEqual(Stand_in.seen[-1][3]["reasoning_effort"], "none")
        self.assertEqual(common.ask_judge("S", "U", dict(base, thinking=True))[0], 0.5)
        self.assertNotIn("reasoning_effort", Stand_in.seen[-1][3])    # refused, so dropped
        common.ask_judge("S", "U", dict(base, thinking=None))
        self.assertNotIn("reasoning_effort", Stand_in.seen[-1][3])
        self.assertIsNone(common.judge_settings({})["thinking"])
        self.assertFalse(common.judge_settings({"JUDGE_THINKING": False})["thinking"])

    def test_opencode_go_picks_the_wire_by_the_model(self):
        Stand_in.replies = [
            (200, {"choices": [{"message": {"content": "From Kimi."}}]}),
            (200, {"stop_reason": "end_turn", "content": [{"type": "text", "text": "From MiniMax."}]}),
            (200, {"output": [{"type": "reasoning", "id": "rs_1", "summary": []},
                              {"type": "message", "role": "assistant", "content": [
                                  {"type": "output_text", "text": "From Grok."}]}]})]
        texts = []
        with mock.patch.dict(settings.PROVIDERS["opencode-go"], base_url=self.url), \
                mock.patch.dict(os.environ, {"OPENCODE_API_KEY": "oc"}):
            for model in ("kimi-k3", "minimax-m3", "grok-4.6"):
                resolved = providers.resolve({"provider": "opencode-go", "model": model,
                                              "conversation": "conv-1234abcd"})
                texts.append(providers.chat(resolved, "SYSTEM",
                                            [{"role": "user", "content": "hey"}])[0])
        self.assertEqual(texts, ["From Kimi.", "From MiniMax.", "From Grok."])
        (_, kimi, kimi_headers, _), (_, minimax, minimax_headers, minimax_body), \
            (_, grok, grok_headers, grok_body) = Stand_in.seen
        # Every wire carries the page's conversation id, which OpenCode Go routes by.
        self.assertEqual({headers["x-opencode-session"]
                          for headers in (kimi_headers, minimax_headers, grok_headers)},
                         {"conv-1234abcd"})
        self.assertEqual((kimi, minimax, grok),
                         ("/v1/chat/completions", "/v1/messages", "/v1/responses"))
        self.assertEqual(kimi_headers["authorization"], "Bearer oc")
        self.assertEqual(kimi_headers["user-agent"], providers.USER_AGENT)   # not urllib's
        self.assertEqual((minimax_headers["x-api-key"], minimax_headers["authorization"]),
                         ("oc", "Bearer oc"))
        self.assertNotIn("output_config", minimax_body)    # Claude's own knob
        self.assertEqual((grok_body["instructions"], grok_body["input"]),
                         ("SYSTEM", [{"role": "user", "content": "hey"}]))
        self.assertNotIn("temperature", grok_body)

    def test_a_conversation_id_is_only_sent_as_an_id_and_only_where_wanted(self):
        with mock.patch.dict(os.environ, {"OPENCODE_API_KEY": "oc"}):
            unsafe = providers.resolve({"provider": "opencode-go",
                                        "conversation": "abc\r\nX-Evil: 1"})["session"]
            missing = providers.resolve({"provider": "opencode-go"})["session"]
        for header, value in (unsafe, missing):
            self.assertEqual(header, "x-opencode-session")
            self.assertRegex(value, "^[0-9a-f]{32}$")       # made up, not the page's text
        self.assertIsNone(providers.resolve({"provider": "lmstudio",
                                             "conversation": "conv-1234abcd"})["session"])

    def test_a_tool_round_trip_in_the_responses_shape(self):
        Stand_in.replies = [
            (200, {"output": [{"type": "reasoning", "id": "rs_1", "summary": []},
                              {"type": "function_call", "id": "fc_1", "call_id": "c1",
                               "name": "select_records", "arguments": '{"first": 2}'}]}),
            (200, {"output": [{"type": "message", "content": [
                {"type": "output_text", "text": "The first 2 it is."}]}]})]
        with mock.patch.dict(settings.PROVIDERS["opencode-go"], base_url=self.url), \
                mock.patch.dict(os.environ, {"OPENCODE_API_KEY": "oc"}):
            reply = agent.chat(self.registry.catalog, self.user, "only the first 2", "analysis",
                               {}, (jsonl(RECORDS), "x", None), None,
                               {"provider": "opencode-go", "model": "gpt-5.6-luna",
                                "conversation": "conv-1234abcd"}, [])
        self.assertEqual(reply["message"]["text"], "The first 2 it is.")
        self.assertEqual(reply["session"]["selection"], {"first": 2})
        first, second = Stand_in.seen[0][3], Stand_in.seen[1][3]
        self.assertEqual([seen[2]["x-opencode-session"] for seen in Stand_in.seen],
                         ["conv-1234abcd"] * 2)                # one id for every round
        self.assertEqual(first["tools"][0].keys(), {"type", "name", "description", "parameters"})
        call, result = second["input"][-2:]
        # The call goes back without its item id, and the reasoning not at all.
        self.assertEqual(call, {"type": "function_call", "call_id": "c1",
                                "name": "select_records", "arguments": '{"first": 2}'})
        self.assertEqual((result["type"], result["call_id"]), ("function_call_output", "c1"))
        self.assertEqual(json.loads(result["output"])["kept"], 2)

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
        self.assertEqual([body["name"] for body in bodies],
                         [planner.lora_name("verse", planner.recipe()["base_model"], 4)])
        self.assertEqual(bodies[0]["settings"], {"learning_rate": 1e-4, "epochs": 2.0,
                                                 "rank": 4, "prompt": "Say hi"})
        # Part of a shared file goes as its lines: the whole file is not what it trains on.
        self.assertEqual(len(bodies[0]["dataset"].splitlines()), 3)


class EndpointTests(ServerTestCase):

    def test_the_page_is_served_and_the_agent_needs_a_key(self):
        status, _, body = self.download("/guide.html")
        self.assertEqual(status, 200)
        self.assertIn("<title>Guide · GEP LoRA</title>".encode("utf-8"), body)
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

    def test_a_block_is_explained_through_the_api(self):
        status, reply = self.call("GET", "/agent/help")
        self.assertEqual(status, 200, reply)
        self.assertEqual(reply["blocks"]["search.fitness"]["where"], "right")
        status, reply = self.call("POST", "/agent/help", {
            "agent": SCRIPTED, "block": "dataset", "title": "Your dataset",
            "shown": "Conversations 6", "stage": "analysis", "history": []})
        self.assertEqual(status, 200, reply)
        self.assertEqual(reply["block"], "dataset")
        self.assertIn("**Your dataset**", reply["message"]["text"])
        self.assertEqual(self.call("POST", "/agent/help", {"block": "../x"})[0], 400)
        self.assertEqual(self.call("GET", "/agent/help", key="nope")[0], 401)


# --- "what is this?": the page's blocks explained ---------------------------------


# Both guides draw question marks; between them they name every block.
PAGES = [os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                      "async_api", name) for name in ("guide.html", "visual_guide.html",
                                                       "blend_comparison.html")]


class UiHelpTests(JobsTestCase):

    def test_every_question_mark_on_the_page_has_a_block_behind_it(self):
        named = set()
        for path in PAGES:
            with open(path, encoding="utf-8") as handle:
                found = set(re.findall(r'\b(?:qm|sub|helped|addCard)\("([a-z_.]+)"', handle.read()))
            # The visual guide's and the comparison's blocks are their own, and
            # the guide's are the guide's.
            own = ("visual_" if path.endswith("visual_guide.html") else
                   "compare_" if path.endswith("blend_comparison.html") else None)
            self.assertTrue(all((key.split("_")[0] + "_" if key.startswith(("visual_", "compare_"))
                                 else None) == own for key in found), path)
            named |= found
        self.assertEqual(named, set(prompts.UI_BLOCKS))
        for key, (where, title, about) in prompts.UI_BLOCKS.items():
            self.assertIn(where, ("left", "right"), key)
            self.assertTrue(title and about, key)
            if "." in key:
                self.assertIn(key.split(".")[0], prompts.UI_BLOCKS, key)

    def test_a_block_is_explained_without_a_model(self):
        found = ui_help.explain("training.loss", "Loss — how wrong", "step 10 loss 1.2",
                                "training", SCRIPTED)
        self.assertEqual(found["block"], "training.loss")
        self.assertTrue(found["message"]["fallback"])
        self.assertIn("lower is better", found["message"]["text"])
        self.assertIn("Loss — how wrong", found["message"]["text"])
        # No title from the page: the catalogue's.
        self.assertEqual(ui_help.explain("activity", choice=SCRIPTED)["title"], "Activity")
        with self.assertRaises(ui_help.HelpError):
            ui_help.explain("nowhere", choice=SCRIPTED)

    def test_a_model_is_told_what_the_block_is_and_what_it_shows(self):
        sent = {}

        def say(step, facts, fallback, choice=None, history=None, message=None):
            sent.update(step=step, facts=facts, message=message)
            return {"text": "It is the loss.", "by": "x", "fallback": False, "note": None}

        with mock.patch.object(agent, "say", say):
            ui_help.explain("training.log", "Log", "a\n\n  b   c\n" + "x" * 5000, "nowhere",
                            SCRIPTED)
        facts = sent["facts"]
        self.assertEqual(sent["step"], "help")
        self.assertEqual((facts["where"], facts["part_of"]), ("right", "Training"))
        self.assertEqual(facts["about"], prompts.UI_BLOCKS["training.log"][2])
        self.assertTrue(facts["shown"].startswith("a\nb c\n"))
        self.assertLessEqual(len(facts["shown"]), ui_help.SHOWN_CHARS)
        # A stage the page does not have is not passed on.
        self.assertEqual((facts["stage"], facts["step_instructions"]), (None, None))
        self.assertIn("Log", sent["message"])


# --- the second half: combining the LoRAs ---------------------------------------


def lora(catalog, name, owner="alice", rank=8, base_model="unsloth/base", mock=True,
         status="ready", folder=None):
    """A catalogue row of `owner`'s, standing in for a trained LoRA."""
    return catalog.add(name, folder or os.path.join("C:/nowhere", owner or "nobody", name),
                       status, "trained", owner=owner, rank=rank, base_model=base_model,
                       mock=int(mock))


class BlendingTests(JobsTestCase):

    def setUp(self):
        super().setUp()
        self.catalog = self.registry.catalog
        self.a = lora(self.catalog, "poem-r8")
        self.b = lora(self.catalog, "poem-r16", rank=16)

    def test_the_blend_part_of_a_session_is_checked_and_filled(self):
        session = planner.session_of({})
        self.assertEqual(session["blend"]["generations"], settings.BLEND_GENERATIONS)
        self.assertEqual(session["blend"]["loras"], [])
        for bad in ({"population": 1}, {"generations": 0}, {"loras": ["x"]},
                    {"loras": list(range(1, 12))}, {"source": {"url": "http://x"}}):
            with self.assertRaises(planner.SessionError, msg=bad):
                planner.session_of({"blend": bad})

    def test_only_the_users_own_ready_loras_can_be_blended(self):
        theirs = lora(self.catalog, "secret", owner=None)
        self.registry.add_user("bob")
        bobs = lora(self.catalog, "bobs", owner="bob")
        failed = lora(self.catalog, "broken", status="failed")
        for key in (theirs["id"], bobs["id"], "bobs", "secret"):
            with self.assertRaises(blending.BlendError) as caught:
                blending.own(self.catalog, self.user, key)
            self.assertIn("no LoRA of yours", str(caught.exception))
        with self.assertRaises(blending.BlendError):
            blending.own(self.catalog, self.user, failed["id"])
        self.assertEqual({row["name"] for row in blending.mine(self.catalog, self.user)},
                         {"poem-r16", "poem-r8"})

    def test_more_than_five_loras_get_a_place_each_up_to_ten(self):
        seven = [lora(self.catalog, "set-%d" % n) for n in range(7)]
        self.assertEqual(blending.slots(seven),
                         {"L%d" % (n + 1): row["id"] for n, row in enumerate(seven)})
        blending.check_together([lora(self.catalog, "ten-%d" % n) for n in range(10)])
        with self.assertRaises(blending.BlendError):
            blending.check_together([lora(self.catalog, "eleven-%d" % n) for n in range(11)])

    def test_one_base_model_and_five_places(self):
        other = lora(self.catalog, "other", base_model="someone/else")
        with self.assertRaises(blending.BlendError):
            blending.check_together([self.a, other])
        self.assertEqual(blending.slots([self.a, self.b]),
                         {"L1": self.a["id"], "L2": self.b["id"], "L3": self.a["id"],
                          "L4": self.b["id"], "L5": self.a["id"]})
        # Nothing chosen: the ones just trained, else the biggest set on one model.
        self.assertEqual([row["id"] for row in blending.default_pick(self.catalog, self.user,
                                                                     [self.b["id"]])],
                         [self.b["id"]])
        self.assertEqual({row["id"] for row in blending.default_pick(self.catalog, self.user)},
                         {self.a["id"], self.b["id"]})

    def test_the_label_keeps_the_names_dots(self):
        named = lora(self.catalog, "poem-qwen3.5-0.8b-20260923-r8-2")
        session = planner.session_of({"blend": {"loras": [named["id"]], "questions": 3}})
        found = blending.plan(self.catalog, self.registry, self.user, session,
                              dataset=(jsonl(RECORDS), "x.jsonl", None))
        self.assertEqual(found["label"], "poem-qwen3.5-0.8b-20260923-blend")

    def test_an_unfinished_search_is_not_offered_for_testing(self):
        class Gone:
            def database(self, job):
                return os.path.join("C:/nowhere", "job.sqlite3")
        for status in ("stopped", "cancelled", "failed"):
            said = agent.blend_debrief(Gone(), self.catalog, self.user,
                                       {"status": status, "error": None, "run_id": 1},
                                       SCRIPTED)
            self.assertIn("Resume the search", said["message"]["text"])
            self.assertNotIn("Test all blends", said["message"]["text"])

    def test_a_base_model_in_a_name(self):
        self.assertEqual(planner.model_tag("unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit"),
                         "qwen2.5-1.5b-instruct")
        self.assertEqual(planner.model_tag("unsloth/Qwen3.5-0.8B"), "qwen3.5-0.8b")
        name = planner.lora_name("x" * 40, "org/" + "m" * 60, 256, "20260923")
        self.assertLessEqual(len(name + "-99"), 64)
        self.assertTrue(name.endswith("-20260923-r256"))

    def test_a_chromosome_in_words(self):
        said = blending.formula("CAT.SVD.L3.L1.L2.w1.w2.w3", {"L1": "a", "L2": "b", "L3": "c"},
                                {"w1": 0.5, "w2": 1.0})
        self.assertEqual(said, "stack(merge(a ×1.00, b ×w3), c ×0.50)")
        # A lone adapter has no fold to apply its weight, so none is claimed.
        self.assertEqual(blending.formula("L2.w1", {"L2": "b"}, {"w1": 0.5}),
                         "b alone, at full strength")

    def test_the_plan_is_a_job_naming_the_loras_by_id(self):
        session = planner.session_of({"blend": {"loras": [self.a["id"], self.b["id"]],
                                                "generations": 2, "population": 5,
                                                "questions": 4}})
        found = blending.plan(self.catalog, self.registry, self.user, session,
                              dataset=(jsonl(RECORDS), "x.jsonl", None))
        job = found["job"]
        self.assertEqual(job["settings"]["LORA_SLOTS"], blending.slots([self.a, self.b]))
        self.assertEqual((job["settings"]["GENERATIONS"], job["settings"]["COUNT"],
                          job["settings"]["TRAINING_COUNT"]), (2, 5, 4))
        self.assertEqual(job["settings"]["BASE_MODEL"], "unsloth/base")
        # A practice LoRA has no weights: only the mocked template runs one.
        self.assertEqual(job["settings"]["TEMPLATE"], blending.MOCKED_TEMPLATE)
        # The two left over are shared out: one to test the blends on, one to
        # verify the picked blend on -- and no testing pass at the search's end,
        # since testing is a step of its own.
        self.assertEqual((found["questions"], found["testing"], found["validation"]), (4, 1, 1))
        self.assertEqual(len(job["datasets"]["testing"].splitlines()), 1)
        self.assertEqual(len(job["datasets"]["validation"].splitlines()), 1)
        self.assertTrue(job["options"]["no_test"])
        # With no dataset and no copy of the LoRAs' own data, it says so.
        with self.assertRaises(blending.BlendError):
            blending.plan(self.catalog, self.registry, self.user, session)


class BlendToolboxTests(JobsTestCase):

    def setUp(self):
        super().setUp()
        self.catalog = self.registry.catalog
        self.a = lora(self.catalog, "poem-r8")
        self.b = lora(self.catalog, "poem-r16", rank=16)
        self.registry.add_user("bob")
        self.bobs = lora(self.catalog, "bobs", owner="bob")

    def box(self, stage="blend", session=None):
        return tools.Toolbox(self.catalog, self.user, stage, session or {}, None,
                             lambda: [{"file": "poem_lora_dataset.json", "records": 50,
                                       "usable": True, "keywords": []}])

    def test_the_chat_blends_the_users_loras_and_no_others(self):
        box = self.box()
        self.assertIn("no LoRA of yours",
                      box.run("choose_blend_loras", {"loras": [str(self.bobs["id"])]})["error"])
        result = box.run("choose_blend_loras", {"loras": ["poem-r8", str(self.b["id"])]})
        self.assertEqual(box.session["blend"]["loras"], [self.a["id"], self.b["id"]])
        self.assertEqual(result["names_text"], "poem-r8, poem-r16")
        box.run("set_blend_search", {"generations": 5, "questions": 20})
        self.assertEqual((box.session["blend"]["generations"], box.session["blend"]["questions"]),
                         (5, 20))
        self.assertIn("error", box.run("set_blend_search", {"population": 1}))
        box.run("set_blend_questions", {"file": "poem"})
        self.assertEqual(box.session["blend"]["source"], {"file": "poem_lora_dataset.json"})
        self.assertEqual(box.outcome()["actions"], [{"type": "blend_plan"}])

    def test_what_the_page_is_told_to_do(self):
        box = self.box("done")
        box.run("open_blending", {})
        self.assertEqual(box.outcome()["actions"], [{"type": "blend"}])
        box = self.box("intro")
        box.run("choose_blend_loras", {"loras": ["poem-r8"]})
        self.assertEqual([one["type"] for one in box.outcome()["actions"]], ["blend", "blend_plan"])
        box = self.box("blend_confirm")
        box.run("start_blend", {})
        self.assertEqual([one["type"] for one in box.outcome()["actions"]],
                         ["blend_plan", "start_blend"])
        self.assertIn("search is running", self.box("blending").run(
            "choose_blend_loras", {"loras": ["poem-r8"]})["error"])
        self.assertIn("error", self.box("blend").run("stop_blend", {}))
        box = self.box("blending")
        box.run("stop_blend", {})
        self.assertEqual(box.outcome()["actions"], [{"type": "stop_blend"}])
        # Training's own tools have no say over a search.
        self.assertIn("error", self.box("blend").run("set_loras", {"ranks": [4]}))

    def test_plain_blend_requests_without_a_model(self):
        names = ["poem-r8", "poem-r16"]
        for said, stage, calls in (
                ("blend them", "done", [("open_blending", {})]),
                ("only poem-r16 and poem-r8, 4 rounds", "blend",
                 [("choose_blend_loras", {"loras": ["poem-r16", "poem-r8"]}),
                  ("set_blend_search", {"generations": 4})]),
                ("12 blends each, judged on 20 questions", "blend",
                 [("set_blend_search", {"population": 12, "questions": 20})]),
                ("go ahead", "blend_confirm", [("start_blend", {})]),
                ("stop", "blending", [("stop_blend", {})]),
                ("5 epochs", "blend", [])):
            self.assertEqual(commands.parse(said, stage, ["poem_lora_dataset.json"], names),
                             calls, said)


class BlendEndpointTests(ServerTestCase):

    def test_trained_loras_blended_by_a_search_and_read_back(self):
        # Two practice LoRAs, trained through the API as the page does it.
        status, plan = self.call("POST", "/agent/plan", {
            "dataset": {"text": jsonl(RECORDS), "name": "tiny.jsonl"}, "epochs": 1, "mock": True})
        self.assertEqual(status, 200, plan)
        ids = [self.call("POST", "/loras", body)[1]["lora"]["id"] for body in plan["trainings"]]
        for _ in ids:
            worker.serve(self.registry, once=True, poll=0.05, catalog=self.app.catalog)

        status, intro = self.call("POST", "/agent/intro", {"agent": SCRIPTED})
        self.assertEqual(intro["ready_loras"], len(ids))
        status, reply = self.call("POST", "/agent/blend/intro", {
            "agent": SCRIPTED, "session": {}, "prefer": ids})
        self.assertEqual(status, 200, reply)
        self.assertEqual(sorted(reply["blend"]["loras"]), sorted(ids))
        self.assertIn("Plan the search", reply["message"]["text"])

        # No dataset in the page: the questions are the LoRAs' own training data.
        session = {"blend": dict(reply["blend"], generations=1, population=4, questions=3)}
        status, found = self.call("POST", "/agent/blend/plan", {"session": session})
        self.assertEqual(status, 200, found)
        self.assertIn("Shall I start", found["message"]["text"])
        self.assertIn("training data", found["source"])
        self.assertEqual(found["job"]["settings"]["LORA_SLOTS"]["L1"], reply["blend"]["loras"][0])

        # Another user can neither plan with these LoRAs nor submit them.
        bob = self.registry.add_user("bob")
        status, refused = self.call("POST", "/agent/blend/plan", {"session": session}, key=bob)
        self.assertEqual(status, 400)
        self.assertIn("no LoRA of yours", refused["error"])
        status, refused = self.call("POST", "/jobs", found["job"], key=bob)
        self.assertEqual(status, 400)
        self.assertIn("no LoRA of yours", refused["error"])

        # The page's half: the body goes to POST /jobs as it is.
        status, queued = self.call("POST", "/jobs", found["job"])
        self.assertEqual(status, 201, queued)
        job_id = queued["job"]["id"]
        status, said = self.call("POST", "/agent/blend/started", {
            "agent": SCRIPTED, "job": {"id": job_id, "queue_position": 1}, "plan": found})
        self.assertIn(found["label"], said["message"]["text"])
        self.assertEqual(worker.serve(self.registry, once=True, poll=0.2), 0)
        self.assertEqual(self.call("GET", "/jobs/%d/status" % job_id)[1]["job"]["status"], "done",
                         self.call("GET", "/jobs/%d/log" % job_id)[1])

        status, debrief = self.call("POST", "/agent/blend/debrief", {"agent": SCRIPTED,
                                                                    "job": job_id})
        self.assertEqual(status, 200, debrief)
        result = debrief["result"]
        self.assertTrue(result["mock"])
        self.assertEqual(sorted(set(result["slots"].values())),
                         sorted(one["name"] for one in found["loras"]))
        self.assertIsNotNone(result["best"])
        self.assertTrue(any(one["name"] in result["best"]["formula"] for one in found["loras"]))
        self.assertIn("practice run", debrief["message"]["text"])
        self.assertEqual(self.call("POST", "/agent/blend/debrief", {"job": job_id}, key=bob)[0],
                         404)


class ReleaseEndpointTests(ServerTestCase):
    """After the search, as the page drives it: test every blend, pick one and
    verify it against its LoRAs, put a model live and ask it."""

    def searched(self):
        """Two practice LoRAs trained and blended by a finished search.
        -> (job id, the blend plan)."""
        status, plan = self.call("POST", "/agent/plan", {
            "dataset": {"text": jsonl(RECORDS), "name": "tiny.jsonl"}, "epochs": 1, "mock": True})
        ids = [self.call("POST", "/loras", body)[1]["lora"]["id"] for body in plan["trainings"]]
        for _ in ids:
            worker.serve(self.registry, once=True, poll=0.05, catalog=self.app.catalog)
        session = {"blend": {"loras": ids, "generations": 1, "population": 4, "questions": 3}}
        found = self.call("POST", "/agent/blend/plan", {"session": session})[1]
        self.assertEqual((found["testing"], found["validation"]), (2, 1), found)
        job_id = self.call("POST", "/jobs", found["job"])[1]["job"]["id"]
        self.assertEqual(worker.serve(self.registry, once=True, poll=0.2), 0)
        return job_id, found

    def test_every_blend_tested_one_verified_and_a_model_put_live(self):
        job_id, found = self.searched()
        job = self.call("GET", "/jobs/%d" % job_id)[1]
        self.assertTrue(job["job"]["can_test"])
        # The search ran no testing pass of its own: that is the next step.
        self.assertFalse(job["results"]["testing"]["summary"])

        status, offered = self.call("GET", "/jobs/%d/test" % job_id)
        self.assertEqual(status, 200, offered)
        self.assertEqual((offered["records"], offered["tested"]), (2, 0))
        self.assertEqual(self.call("POST", "/jobs/%d/test" % job_id, {"min_quality": 2})[0], 400)
        status, queued = self.call("POST", "/jobs/%d/test" % job_id, {})
        self.assertEqual(status, 200, queued)
        self.assertEqual((queued["job"]["status"], queued["job"]["task"]), ("queued", "test"))
        status, said = self.call("POST", "/agent/test/started", {
            "agent": SCRIPTED, "job": {"id": job_id, "queue_position": 1}})
        self.assertIn("2 question(s)", said["message"]["text"])
        self.assertEqual(worker.serve(self.registry, once=True, poll=0.2), 0)
        status = self.call("GET", "/jobs/%d/status" % job_id)[1]["job"]
        self.assertEqual((status["status"], status["error"]), ("done", None),
                         self.call("GET", "/jobs/%d/log" % job_id)[1])
        # Every blend that ran, not only those above TESTING_MIN_QUALITY.
        offered = self.call("GET", "/jobs/%d/test" % job_id)[1]
        self.assertEqual(offered["tested"], offered["blends"])

        status, tested = self.call("POST", "/agent/test/debrief", {"agent": SCRIPTED,
                                                                   "job": job_id})
        self.assertEqual(status, 200, tested)
        result = tested["result"]
        self.assertEqual(result["testing"]["tested"], offered["blends"])
        self.assertEqual(sorted(result["splits"]), ["testing", "training", "validation"])
        self.assertEqual(tested["release"]["individual"], result["recommended"])
        best = next(one for one in result["blends"] if one["number"] == result["recommended"])
        self.assertIsNotNone(best["tested"])
        self.assertIn("Verify it", tested["message"]["text"])

        # The verification: the picked blend against each distinct LoRA it uses,
        # on the validation split, through POST /jobs/{id}/verify as it is.
        session = {"release": tested["release"]}
        status, planned = self.call("POST", "/agent/verify/plan", {"session": session})
        self.assertEqual(status, 200, planned)
        body = planned["body"]
        self.assertEqual((body["individual"], body["split"], body["count"]),
                         (best["number"], "validation", 1))
        self.assertEqual(body["slots"], best["slots"])
        self.assertEqual(len(set(planned["against"])), len(planned["against"]))
        status, verification = self.call("POST", "/jobs/%d/verify" % job_id, body)
        self.assertEqual(status, 201, verification)
        verification_id = verification["verification"]["id"]
        self.assertEqual(worker.serve(self.registry, once=True, poll=0.2), 0)
        row = self.call("GET", "/verifications/%d" % verification_id)[1]["verification"]
        self.assertEqual(row["status"], "done", row)
        status, verified = self.call("POST", "/agent/verify/debrief", {
            "agent": SCRIPTED, "verification": verification_id})
        self.assertEqual(status, 200, verified)
        report = verified["result"]["report"]
        self.assertEqual(sorted(one["lora"] for one in report["against"]),
                         sorted(planned["against"]))
        self.assertIn("Go live", verified["message"]["text"])

        # Another dataset instead of a split: a LoRA's own training data -- the
        # user's, and nobody else's -- or a shared file, and nothing outside them.
        mine = found["loras"][0]
        other = dict(tested["release"], questions={"lora": mine["id"]}, count=2)
        planned = self.call("POST", "/agent/verify/plan", {"session": {"release": other}})[1]
        self.assertEqual(planned["body"]["dataset"], {"lora": mine["id"]})
        self.assertIn("training data", planned["questions_from"])
        status, queued = self.call("POST", "/jobs/%d/verify" % job_id, planned["body"])
        self.assertEqual(status, 201, queued)
        self.assertEqual(queued["verification"]["options"]["dataset_label"],
                         "%s's training data" % mine["name"])
        self.assertNotIn("dataset", queued["verification"]["options"])
        # Or a file the page read: stored in the job's folder, the path never shown.
        status, queued = self.call("POST", "/jobs/%d/verify" % job_id, {
            "dataset": {"text": json.dumps(RECORDS), "name": "mine.json"}, "count": 2})
        self.assertEqual(status, 201, queued)
        self.assertIn("mine.json (uploaded", queued["verification"]["options"]["dataset_label"])
        self.assertNotIn("dataset", queued["verification"]["options"])
        for dataset in ({"text": "  "}, {"text": '{"no": "messages"}'}):
            status, refused = self.call("POST", "/jobs/%d/verify" % job_id, {"dataset": dataset})
            self.assertEqual(status, 400, (dataset, refused))
        self.registry.add_user("bob")
        bobs = lora(self.registry.catalog, "bobs", owner="bob")
        for dataset in ({"lora": bobs["id"]}, {"file": "../secrets.txt"}, {"url": "x"}):
            status, refused = self.call("POST", "/jobs/%d/verify" % job_id, {"dataset": dataset})
            self.assertEqual(status, 400, (dataset, refused))
        bob = self.registry.add_user("carol")
        self.assertEqual(self.call("POST", "/agent/verify/plan", {"session": session},
                                   key=bob)[0], 404)

        # Going live: the model the person picks, and the guide never sees its key.
        status, live = self.call("POST", "/jobs/%d/live" % job_id, {"individual": best["number"]})
        self.assertEqual(status, 201, live)
        status, said = self.call("POST", "/agent/live/started", {
            "agent": SCRIPTED, "deployment": live["deployment"]["id"]})
        self.assertEqual(status, 200, said)
        self.assertNotIn(live["token"], json.dumps(said))
        self.assertIn("#%d" % best["number"], said["message"]["text"])
        status, text = self.call("POST", "/infer", {"token": live["token"], "prompt": "hello"},
                                 raw=True)
        self.assertEqual(status, 200)
        self.assertIn("hello", text)
        self.assertEqual(self.call("POST", "/agent/live/started", {
            "deployment": live["deployment"]["id"]}, key=bob)[0], 404)

    def test_the_chat_picks_verifies_and_puts_live(self):
        job_id, found = self.searched()
        base = {"agent": SCRIPTED, "session": {"release": {"job": job_id}}}
        blends = self.call("POST", "/agent/test/debrief", dict(base, job=job_id))[1]["result"]
        number = blends["recommended"]

        reply = self.call("POST", "/agent/chat", dict(
            base, stage="tested", message="verify #%d on the testing questions" % number))[1]
        self.assertEqual([one["type"] for one in reply["actions"]], ["start_verification"])
        self.assertEqual(reply["session"]["release"]["individual"], number)
        self.assertEqual(reply["session"]["release"]["questions"], {"split": "testing"})
        # A blend the search does not hold is refused.
        reply = self.call("POST", "/agent/chat", dict(base, stage="tested",
                                                      message="#999 is best"))[1]
        self.assertFalse(reply["steps"][0]["ok"])
        self.assertIsNone(reply["session"]["release"]["individual"])
        reply = self.call("POST", "/agent/chat", dict(base, stage="tested",
                                                      message="put #%d live" % number))[1]
        self.assertEqual([one["type"] for one in reply["actions"]], ["go_live"])
        # Stopping the testing is an action; nothing else may be asked meanwhile.
        reply = self.call("POST", "/agent/chat", dict(base, stage="testing", message="stop"))[1]
        self.assertEqual([one["type"] for one in reply["actions"]], ["stop_testing"])
        # Another user's search is nobody else's to pick from.
        bob = self.registry.add_user("bob")
        reply = self.call("POST", "/agent/chat", dict(base, stage="tested",
                                                      message="#%d is best" % number), key=bob)[1]
        self.assertIn("no search", reply["steps"][0]["summary"])


class ReleaseToolboxTests(JobsTestCase):

    def test_the_release_part_of_a_session_is_checked(self):
        session = planner.session_of({})
        self.assertEqual(session["release"], {"job": None, "individual": None,
                                              "questions": None, "count": None})
        for bad in ({"job": "x"}, {"questions": {"split": "nope"}}, {"count": 0},
                    {"questions": {"file": "a", "lora": 1}}):
            with self.assertRaises(planner.SessionError, msg=bad):
                planner.session_of({"release": bad})

    def test_no_search_no_release_tools(self):
        box = tools.Toolbox(self.registry.catalog, self.user, "tested", {}, None, None,
                            self.registry)
        self.assertIn("no finished search", box.run("start_verification", {})["error"])
        self.assertIn("error", tools.Toolbox(self.registry.catalog, self.user, "blend", {})
                      .run("start_testing", {}))

    def test_plain_release_requests_without_a_model(self):
        demo = ["poem_lora_dataset.json"]
        for said, stage, calls in (
                ("test all blends", "blended", [("start_testing", {})]),
                ("verify #7 on the testing questions", "tested",
                 [("choose_best_blend", {"individual": 7}),
                  ("set_verify_questions", {"split": "testing"}), ("start_verification", {})]),
                ("verify it on the poem dataset, 20 questions", "tested",
                 [("set_verify_questions", {"file": "poem_lora_dataset.json", "count": 20}),
                  ("start_verification", {})]),
                ("put #3 live", "verified", [("go_live", {"individual": 3})]),
                ("stop", "testing", [("stop_testing", {})]),
                ("5 rounds", "blended", [("set_blend_search", {"generations": 5})]),
                ("5 epochs", "tested", [])):
            self.assertEqual(commands.parse(said, stage, demo, ["poem-r8"]), calls, said)


class SummaryTests(ReleaseEndpointTests):
    """The journey so far (create_summary.py): told from the user's own rows,
    with charts as data, in built-in words or a model's."""

    # The release flow is ReleaseEndpointTests' to test; these only reuse
    # its searched().
    test_every_blend_tested_one_verified_and_a_model_put_live = None
    test_the_chat_picks_verifies_and_puts_live = None

    def searched_and_tested(self):
        """A finished search whose blends were all tested. -> (job id, plan)."""
        job_id, found = self.searched()
        self.assertEqual(self.call("POST", "/jobs/%d/test" % job_id, {})[0], 200)
        self.assertEqual(worker.serve(self.registry, once=True, poll=0.2), 0)
        return job_id, found

    def test_nothing_yet_is_said_so(self):
        status, out = self.call("POST", "/agent/summary", {"agent": SCRIPTED})
        self.assertEqual(status, 200, out)
        self.assertFalse(any(one["done"] for one in out["journey"]))
        self.assertEqual(out["charts"], [])
        self.assertIn(prompts.SUMMARY_NOTHING, out["markdown"])

    def test_the_journey_in_built_in_words_charts_and_tables(self):
        job_id, found = self.searched_and_tested()
        ids = [one["id"] for one in found["loras"]]
        analysis = self.call("POST", "/agent/analyse", {
            "agent": SCRIPTED, "dataset": {"text": jsonl(RECORDS), "name": "tiny.jsonl"}})[1]
        body = {"agent": SCRIPTED, "stage": "tested", "mock": True, "loras": ids, "job": job_id,
                "dataset": analysis["analysis"],
                "transcript": [{"who": "me", "text": "only the first 20"}]}
        status, out = self.call("POST", "/agent/summary", body)
        self.assertEqual(status, 200, out)
        done = {one["key"]: one["done"] for one in out["journey"]}
        self.assertEqual(done, {"dataset": True, "training": True, "search": True,
                                "testing": True, "verification": False, "live": False})
        self.assertTrue(out["fallback"])
        self.assertEqual(out["title"], "Your LoRA journey with tiny")
        markdown = out["markdown"]
        for chapter in ("The data", "Training the LoRAs", "Searching for a blend",
                        "Testing the blends"):
            self.assertIn("## " + chapter, markdown)
        self.assertIn(prompts.SUMMARY_MOCK, markdown)
        self.assertIn(prompts.SUMMARY_NEXT_STEPS["verification"], markdown)
        self.assertIn("| LoRA | Rank | Final loss | Time |", markdown)
        # Every chart is data, placed in the story by id.
        charts = {one["id"]: one for one in out["charts"]}
        self.assertIn("tested_blends", charts)
        for key, chart in charts.items():
            self.assertTrue(chart["placed"], key)
            self.assertIn("[[chart:%s]]" % key, markdown)
        rows = charts["tested_blends"]["rows"]
        self.assertTrue(all("value2" in one for one in rows))
        self.assertEqual(sum(one["highlight"] for one in rows), 1)
        # The conversation is read, never handed back.
        self.assertNotIn("conversation", out["facts"])

        # Another user's ids are nobody else's story.
        bob = self.registry.add_user("bob")
        status, theirs = self.call("POST", "/agent/summary", dict(body, dataset=None), key=bob)
        self.assertEqual(status, 200, theirs)
        self.assertFalse(any(one["done"] for one in theirs["journey"]))

    def test_a_models_story_places_only_the_charts_there_are(self):
        job_id, found = self.searched_and_tested()
        told = ("```markdown\n# A tiny journey\n\nYou trained two LoRAs.\n\n"
                "[[chart:training_loss]]\n\n[[chart:training_loss]]\n\n[[chart:made_up]]\n"
                "\n## Next\n\nVerify it.\n```")
        with mock.patch.object(providers, "chat", return_value=(told, "stand-in")) as chat:
            status, out = self.call("POST", "/agent/summary", {
                "agent": {"provider": "lmstudio"}, "loras": [one["id"] for one in found["loras"]],
                "job": job_id, "mock": True})
        self.assertEqual(status, 200, out)
        system, turns = chat.call_args[0][1], chat.call_args[0][2]
        self.assertEqual(system, prompts.SUMMARY)
        facts = json.loads(turns[0]["content"].split("FACTS:\n", 1)[1].rsplit("\n\n", 1)[0])
        self.assertEqual({one["id"] for one in facts["charts"]},
                         {one["id"] for one in out["charts"]})
        self.assertFalse(out["fallback"])
        self.assertEqual(out["by"].split(" · ")[-1], "stand-in")
        self.assertEqual(out["title"], "A tiny journey")
        self.assertFalse(out["markdown"].startswith("```"))
        self.assertEqual(out["markdown"].count("[[chart:training_loss]]"), 1)
        self.assertNotIn("made_up", out["markdown"])
        placed = {one["id"]: one["placed"] for one in out["charts"]}
        self.assertTrue(placed["training_loss"])
        self.assertFalse(placed["tested_blends"])

    def test_a_chart_inside_a_sentence_is_lifted_onto_its_own_line(self):
        from async_api_agent import create_summary
        pictures = [{"id": "search_progress", "title": "The search, round by round"},
                    {"id": "tested_blends", "title": "The top blends"}]
        told = ("You tested all eight blends on ten unseen questions. [[chart:search_progress]] "
                "shows how the best score improved each round. [[chart:tested_blends]]\n\n"
                "As [[chart:search_progress]] shows, it rose.\n\n## Next\n\nVerify it.")
        lines = create_summary.clean(told, pictures).split("\n")
        self.assertEqual(lines[:5], [
            "You tested all eight blends on ten unseen questions. *The search, round by "
            "round* (below) shows how the best score improved each round.",
            "", "[[chart:search_progress]]", "[[chart:tested_blends]]", ""])
        # Each chart is drawn once: a second mention is only words.
        self.assertIn("As *the search, round by round* (above) shows, it rose.", lines)
        self.assertEqual(sum(line.startswith("[[chart:") for line in lines), 2)
        # One alone on its line that was drawn already, or does not exist, just goes.
        self.assertEqual(create_summary.clean("A.\n\n[[chart:nope]]\n\nB [[chart:nope]] c.",
                                              pictures), "A.\n\nB the chart c.")

    def test_a_model_that_fails_leaves_the_built_in_story(self):
        with mock.patch.object(providers, "chat", side_effect=providers.ProviderError("down")):
            out = self.call("POST", "/agent/summary", {
                "agent": {"provider": "lmstudio"},
                "dataset": {"name": "x.jsonl", "records": 12}})[1]
        self.assertTrue(out["fallback"])
        self.assertIn("down", out["note"])
        self.assertIn("**12** conversation(s)", out["markdown"])

    def test_asked_for_in_the_chat(self):
        for said in ("give me a summary", "recap please", "what have we done so far?",
                     "summarise what we learned"):
            for stage in ("training", "blended", "live"):
                self.assertEqual(commands.parse(said, stage), [("show_summary", {})],
                                 (said, stage))
        reply = self.call("POST", "/agent/chat", {"agent": SCRIPTED, "message": "a recap please",
                                                  "stage": "dataset", "session": {}})[1]
        self.assertEqual(reply["actions"], [{"type": "summary"}])


# --- the user's own defaults ------------------------------------------------------------


class GuideDefaultsTests(JobsTestCase):

    def test_what_a_page_sends_is_checked_and_only_what_is_set_is_kept(self):
        kept = guide_defaults.check({"ranks": [4, 32], "learning_rate": 1e-4, "mock": True,
                                     "blend_population": 12, "model": None, "chat_template": ""})
        self.assertEqual(kept, {"ranks": [4, 32], "learning_rate": 1e-4, "mock": True,
                                "blend_population": 12, "chat_template": ""})
        for bad in ({"ranks": []}, {"ranks": list(range(1, 12))}, {"learning_rate": "fast"},
                    {"blend_population": 1}, {"scheduler": "sideways"}, {"evaluator": "nope"},
                    {"nonsense": 1}, {"model": "m"}, {"mock": "yes"}, []):
            with self.assertRaises(guide_defaults.DefaultsError, msg=bad):
                guide_defaults.check(bad)
        # Thinking is on, off, or "" -- the model's own -- and nothing else.
        self.assertEqual(guide_defaults.check({"thinking": False, "judge_thinking": ""}),
                         {"thinking": False, "judge_thinking": ""})
        with self.assertRaises(guide_defaults.DefaultsError):
            guide_defaults.check({"thinking": "maybe"})
        with guide_defaults.applied({"judge_thinking": False, "thinking": ""}):
            self.assertEqual(guide_defaults.job_settings(), {"JUDGE_THINKING": False})
            self.assertIsNone(guide_defaults.value("THINKING"))
        # A saved value that no longer checks goes back to the server's.
        self.assertEqual(guide_defaults.stored({"ranks": [8], "evaluator": "gone"}), {"ranks": [8]})
        # A chat model is read back with its provider, even beside a value that is dropped.
        self.assertEqual(guide_defaults.stored({"provider": "lmstudio", "model": "m", "ranks": []}),
                         {"provider": "lmstudio", "model": "m"})
        self.assertEqual(guide_defaults.stored({"provider": "lmstudio", "model": "m"}),
                         {"provider": "lmstudio", "model": "m"})

    def test_saved_defaults_are_what_the_guide_reads_and_only_inside(self):
        saved = {"ranks": [4], "epochs_quick": 1.5, "blend_generations": 7, "provider": "scripted",
                 "chat_template": "", "evaluator": "heuristic", "batch_size": 4}
        with guide_defaults.applied(saved):
            session = planner.session_of({})
            self.assertEqual(session["ranks"], [4])
            self.assertEqual(session["blend"]["generations"], 7)
            self.assertEqual(guide_defaults.wait_choices()[0]["epochs"], 1.5)
            self.assertEqual(planner.read_wait("the quick one", 10, 0)[0], 1.5)
            self.assertIsNone(planner.recipe()["chat_template"])
            self.assertEqual(planner.recipe()["batch_size"], 4)
            self.assertEqual(providers.resolve({})["name"], "scripted")
            self.assertEqual(guide_defaults.job_settings(), {"EVALUATOR": "heuristic"})
        self.assertEqual(planner.session_of({})["ranks"], settings.LORA_RANKS)
        self.assertEqual(guide_defaults.wait_choices(), settings.WAIT_CHOICES)


    def test_the_number_of_loras_takes_the_ranks_in_turn_each_repeat_its_own_seed(self):
        self.assertEqual(planner.default_ranks(), settings.LORA_RANKS)     # one per rank
        with guide_defaults.applied({"ranks": [8, 16], "lora_count": 5}):
            self.assertEqual(planner.default_ranks(), [8, 16, 8, 16, 8])
            session = planner.session_of({})
            self.assertEqual(session["ranks"], [8, 16, 8, 16, 8])
            found = analysis.analyse(jsonl(RECORDS), "tiny.jsonl")
            bodies = planner.plan(self.registry.catalog, self.user, found, 1, True, None, session)
            self.assertEqual(planner.estimate(self.registry.catalog, 10, 1, True,
                                              session=session)["loras"], 5)
            # A chat that names its ranks names the LoRAs too.
            self.assertEqual(planner.session_of({"ranks": [4]})["ranks"], [4])
        seed = planner.recipe()["seed"]
        self.assertEqual([body["settings"].get("seed") for body in bodies],
                         [None, None, seed + 1, seed + 1, seed + 2])
        self.assertEqual(len({body["name"] for body in bodies}), 5)
        with guide_defaults.applied({"ranks": [8], "lora_count": 1}):
            self.assertEqual(planner.default_ranks(), [8])
        for bad in (0, settings.MAX_LORAS + 1, 2.5):
            with self.assertRaises(guide_defaults.DefaultsError, msg=bad):
                guide_defaults.check({"lora_count": bad})


    def test_the_searchs_seeds_are_a_number_or_random(self):
        from config import settings as config
        self.assertIsNone(config.SEED)                   # random unless someone pins one
        kept = guide_defaults.check({"seed": 42, "weight_seed": "", "selection_seed": None})
        self.assertEqual(kept, {"seed": 42, "weight_seed": ""})
        for bad in (-1, 2 ** 31, 1.5, "42", True):
            with self.assertRaises(guide_defaults.DefaultsError, msg=bad):
                guide_defaults.check({"mutation_seed": bad})
        with guide_defaults.applied({"seed": 42, "weight_seed": ""}):
            self.assertEqual(guide_defaults.job_settings(),
                             {"SEED": 42, "WEIGHT_MASTER_SEED": None})
        # A random seed is drawn when the search is made, and recorded.
        from async_api import submit
        with mock.patch.object(config, "WEIGHT_MASTER_SEED", 5):
            conf = submit.settings_for({"SEED": 42, "WEIGHT_MASTER_SEED": None,
                                        "TEMPLATE": "template_code_mocked.py"})
        self.assertEqual(conf["SEED"], 42)
        self.assertIsInstance(conf["WEIGHT_MASTER_SEED"], int)


class GuideDefaultsEndpointTests(ServerTestCase):

    def test_the_page_is_served(self):
        status, _, body = self.download("/settings.html")
        self.assertEqual(status, 200)
        self.assertIn("<title>Settings · GEP LoRA</title>".encode("utf-8"), body)

    def test_defaults_saved_read_back_used_and_forgotten(self):
        status, form = self.call("GET", "/agent/defaults")
        self.assertEqual(status, 200, form)
        self.assertEqual(form["saved"], {})
        self.assertEqual(form["effective"]["ranks"], settings.LORA_RANKS)
        self.assertEqual({one["id"] for one in form["groups"]},
                         {one["group"] for one in form["fields"]})

        status, refused = self.call("PUT", "/agent/defaults", {"values": {"blend_population": 1}})
        self.assertEqual(status, 400)
        self.assertIn("Blends per generation", refused["error"])

        wanted = {"ranks": [4, 16], "learning_rate": 1e-4, "base_model": "unsloth/other",
                  "blend_generations": 2, "blend_population": 6, "blend_questions": 3,
                  "evaluator": "heuristic", "mock": True, "epochs_balanced": 5}
        status, form = self.call("PUT", "/agent/defaults", {"values": wanted})
        self.assertEqual(status, 200, form)
        self.assertEqual(form["saved"], wanted)
        self.assertIsNotNone(form["updated_at"])
        self.assertEqual(self.call("GET", "/agent/defaults")[1]["saved"], wanted)

        # The guide starts from them...
        config = self.call("GET", "/agent/config")[1]
        self.assertEqual(config["ranks"], [4, 16])
        self.assertEqual(config["base_model"], "unsloth/other")
        self.assertTrue(config["mock"])
        self.assertEqual((config["blend"]["generations"], config["blend"]["population"]), (2, 6))
        self.assertEqual(config["wait_choices"][1]["epochs"], 5)
        self.assertEqual(sorted(config["own_defaults"]), sorted(wanted))

        # ...its plan trains under them, the chat's changes over them...
        status, plan = self.call("POST", "/agent/plan", {
            "dataset": {"text": jsonl(RECORDS), "name": "tiny.jsonl"}, "epochs": 1, "mock": True,
            "session": {"options": {"alpha": 8}}})
        self.assertEqual(status, 200, plan)
        self.assertEqual([body["settings"]["rank"] for body in plan["trainings"]], [4, 16])
        first = plan["trainings"][0]["settings"]
        self.assertEqual((first["learning_rate"], first["base_model"], first["alpha"]),
                         (1e-4, "unsloth/other", 8))
        self.assertIn("-other-", plan["trainings"][0]["name"])

        # ...and its search is judged by the evaluator chosen.
        lora(self.app.catalog, "mine-r8", owner=self.user["name"], base_model="unsloth/other")
        status, found = self.call("POST", "/agent/blend/plan", {
            "session": {}, "dataset": {"file": "poem_lora_dataset.json"}})
        self.assertEqual(status, 200, found)
        self.assertEqual(found["job"]["settings"]["EVALUATOR"], "heuristic")
        self.assertEqual(found["job"]["settings"]["COUNT"], 6)
        self.assertEqual(found["questions"], 3)
        # A job may name them (the stand-in LoRA has no folder to submit for real).
        from async_api import submit
        submit.check_names(found["job"]["settings"])

        # Another user's defaults are their own.
        bob = self.registry.add_user("bob")
        self.assertEqual(self.call("GET", "/agent/defaults", key=bob)[1]["saved"], {})
        self.assertEqual(self.call("GET", "/agent/config", key=bob)[1]["ranks"],
                         settings.LORA_RANKS)

        status, form = self.call("DELETE", "/agent/defaults")
        self.assertEqual((status, form["saved"]), (200, {}))
        self.assertEqual(self.call("GET", "/agent/config")[1]["ranks"], settings.LORA_RANKS)
