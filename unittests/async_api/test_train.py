"""
test_train.py - A LoRA asked for through the API, trained by the worker.

What this half owns, and so what is tested here:

  * what a request may ask: create_lora.py's own defaults, the search's base
    model, and a 400 for anything the training would only fail on later --
    a bad name, a number out of range, an unknown module, data with no
    assistant turns to learn;
  * the queue entry it becomes: a training row, its dataset beside the user's
    jobs, and a catalogue row reserved as `queued` -- all or nothing;
  * the command line the worker runs, which is the only place the API says
    anything about how a LoRA is trained;
  * the worker: a mocked training run end to end through create_lora.py,
    and the catalogue row settled when the training could not settle it;
  * the endpoints, with a real worker behind them: the form, a training over
    HTTP to `ready`, its log, loss history and dataset, a cancel, and a
    mocked LoRA refused by a real search;
  * whose a LoRA is: its owner sees it and nobody else does -- not in the
    list, not by id, not through a name clash, not as a job's slot.
"""

import json
import os
import sys
from unittest import mock

from adapters import catalog as lora_catalog
from async_api import registry as reg
from async_api import settings as api_settings
from async_api import train
from async_api import worker

from unittests.async_api.support import RECORDS, JobsTestCase
from unittests.async_api.test_server import ServerTestCase


def request(name="demo", **settings):
    return {"name": name, "mock": True, "dataset": RECORDS,
            "settings": dict({"max_steps": 4, "rank": 8, "base_model": "unsloth/base"},
                             **settings)}


class OptionsTests(JobsTestCase):

    def setUp(self):
        super().setUp()
        self.catalog = self.registry.catalog

    def test_the_defaults_are_create_loras_with_the_searchs_base_model(self):
        from config import settings as config
        name, options, mock = train.options_for({"name": "x"}, self.catalog, self.user)
        self.assertEqual((name, mock), ("x", False))
        self.assertEqual(options["rank"], 16)
        self.assertEqual(options["base_model"], config.BASE_MODEL)
        self.assertIsNone(options["max_steps"])

    def test_bad_requests_are_refused_before_anything_is_queued(self):
        for body, words in (({"name": "../up"}, "name must be"),
                            ({"name": "x", "settings": {"rank": 0}}, "rank must be between"),
                            ({"name": "x", "settings": {"rank": 2.5}}, "whole number"),
                            ({"name": "x", "settings": {"epochs": "ten"}}, "must be a number"),
                            ({"name": "x", "settings": {"optim": "magic"}}, "optim must be"),
                            ({"name": "x", "settings": {"target_modules": ["lm_head"]}},
                             "unknown target module"),
                            ({"name": "x", "settings": {"colour": 1}}, "unknown setting"),
                            ({"name": "x", "extra": 1}, "unknown field")):
            with self.assertRaises(train.TrainError) as caught:
                train.options_for(body, self.catalog, self.user)
            self.assertIn(words, str(caught.exception))

    def test_target_modules_are_kept_in_create_loras_order(self):
        _, options, _ = train.options_for(
            {"name": "x", "settings": {"target_modules": "v_proj, q_proj"}}, self.catalog,
            self.user)
        self.assertEqual(options["target_modules"], ["q_proj", "v_proj"])

    def test_a_dataset_must_hold_conversations_with_answers(self):
        with self.assertRaises(train.TrainError) as caught:
            train.dataset_lines(["just a prompt"])
        self.assertIn("conversations", str(caught.exception))
        asked_only = [{"messages": [{"role": "user", "content": "hi"}]}]
        with self.assertRaises(train.TrainError):
            train.dataset_lines(asked_only)
        self.assertEqual(len(train.dataset_lines(RECORDS)), len(RECORDS))


class SubmitTests(JobsTestCase):

    def setUp(self):
        super().setUp()
        self.catalog = self.registry.catalog

    def test_a_training_is_queued_with_its_catalogue_row(self):
        training, lora = train.submit(self.registry, self.catalog, self.user, request())
        self.assertEqual(training["status"], reg.QUEUED)
        self.assertEqual(training["lora_id"], lora["id"])
        self.assertEqual((lora["status"], lora["owner"], lora["rank"]),
                         (lora_catalog.QUEUED, "alice", 8))
        self.assertEqual(os.path.normcase(lora_catalog.absolute(lora["folder"])),
                         os.path.normcase(os.path.join(train.trained_dir(), "user%d"
                                                       % self.user["id"], "demo")))
        sent = os.path.join(self.registry.training_folder(training), "dataset.jsonl")
        with open(sent, encoding="utf-8") as handle:
            self.assertEqual(len(handle.read().splitlines()), len(RECORDS))
        # The name is taken now, before a single step has run.
        with self.assertRaises(train.TrainError) as caught:
            train.submit(self.registry, self.catalog, self.user, request())
        self.assertEqual(caught.exception.status, 409)
        # ...but only among this user's: another's name is theirs to use, in
        # a folder of their own.
        bob = self.registry.user_for_key(self.registry.add_user("bob"))
        _, theirs = train.submit(self.registry, self.catalog, bob, request())
        self.assertNotEqual(theirs["folder"], lora["folder"])
        self.assertEqual(theirs["owner"], "bob")

    def test_a_failed_request_leaves_nothing_behind(self):
        body = request()
        body["dataset"] = [{"messages": [{"role": "user", "content": "no answer"}]}]
        with self.assertRaises(train.TrainError):
            train.submit(self.registry, self.catalog, self.user, body)
        self.assertEqual(self.registry.trainings(), [])
        self.assertEqual(self.catalog.all(), [])

    def test_the_command_line_says_every_option(self):
        # A shared file, named in the record by where it is shared from.
        shared = os.path.join(self.folder, "shared")
        os.makedirs(shared)
        with open(os.path.join(shared, "poems.jsonl"), "w", encoding="utf-8") as handle:
            handle.writelines(json.dumps(one) + "\n" for one in RECORDS)
        body = request(chat_template="qwen-2.5", no_sample=True)
        body["dataset"] = {"file": "poems.jsonl"}
        with mock.patch.object(api_settings, "SHARED_DATASETS_DIR", shared):
            training, lora = train.submit(self.registry, self.catalog, self.user, body)
        self.assertEqual(lora["dataset"], "%s/poems.jsonl" % shared)
        argv = train.command("python", training, "WORK", self.catalog.path)
        self.assertEqual(argv[argv.index("--dataset-source") + 1], lora["dataset"])
        self.assertEqual(argv[:4], ["python", "-u", "-m", "adapters.create_lora"])
        joined = " ".join(argv)
        for part in ("--rank 8", "--max-steps 4", "--name demo", "--chat-template qwen-2.5",
                     "--no-sample", "--mock", "--base-model unsloth/base", "--owner alice",
                     "--catalog " + self.catalog.path):
            self.assertIn(part, joined)

    def test_trainings_are_claimed_oldest_first_and_once(self):
        first, _ = train.submit(self.registry, self.catalog, self.user, request("a"))
        second, _ = train.submit(self.registry, self.catalog, self.user, request("b"))
        self.assertEqual(self.registry.training_queue_position(second["id"]), 2)
        self.assertEqual(self.registry.claim_next_training()["id"], first["id"])
        self.assertEqual(self.registry.claim_next_training()["id"], second["id"])
        self.assertIsNone(self.registry.claim_next_training())


class WorkerTests(JobsTestCase):

    def setUp(self):
        super().setUp()
        self.catalog = self.registry.catalog

    def test_a_mocked_training_runs_through_create_lora_to_ready(self):
        train.submit(self.registry, self.catalog, self.user, request())
        row = self.registry.claim_next_training()
        self.assertEqual(worker.run_training(self.registry, row, self.catalog, poll=0.05),
                         reg.DONE)
        lora = self.catalog.get(row["lora_id"])
        self.assertEqual((lora["status"], lora["steps"], lora["owner"]),
                         (lora_catalog.READY, 4, "alice"))
        with open(os.path.join(self.registry.training_folder(row), "train.log"),
                  encoding="utf-8") as handle:
            self.assertIn("TRAIN:", handle.read())
        self.assertEqual(len(lora_catalog.history(lora["folder"])), 4)

    def test_a_training_that_dies_leaves_its_row_failed(self):
        train.submit(self.registry, self.catalog, self.user, request())
        row = self.registry.claim_next_training()
        status = worker.run_training(self.registry, row, self.catalog,
                                     argv=[sys.executable, "-c", "raise SystemExit(3)"],
                                     poll=0.05)
        self.assertEqual(status, reg.FAILED)
        self.assertEqual(self.catalog.get(row["lora_id"])["status"], lora_catalog.FAILED)
        self.assertEqual(self.registry.training(row["id"])["exit_code"], 3)

    def test_a_worker_that_died_leaves_its_trainings_failed(self):
        train.submit(self.registry, self.catalog, self.user, request())
        row = self.registry.claim_next_training()
        self.assertEqual(worker.recover_trainings(self.registry, self.catalog), 1)
        self.assertEqual(self.registry.training(row["id"])["status"], reg.FAILED)
        self.assertEqual(self.catalog.get(row["lora_id"])["status"], lora_catalog.FAILED)


class EndpointTests(ServerTestCase):

    def work(self):
        worker.serve(self.registry, once=True, poll=0.05, catalog=self.app.catalog)

    def test_a_lora_from_request_to_ready_and_gone(self):
        status, form = self.call("GET", "/loras/form")
        self.assertEqual(status, 200, form)
        self.assertEqual(form["defaults"]["rank"], 16)
        self.assertIn("adamw_8bit", form["choices"]["optim"])
        self.assertTrue(any(model["search"] for model in form["base_models"]))

        status, reply = self.call("POST", "/loras", request())
        self.assertEqual(status, 201, reply)
        lora_id = reply["lora"]["id"]
        self.assertEqual(reply["lora"]["status"], "queued")
        self.assertEqual(reply["training"]["queue_position"], 1)
        self.assertTrue(reply["lora"]["can_cancel"])

        self.work()
        status, detail = self.call("GET", "/loras/%d" % lora_id)
        lora = detail["lora"]
        self.assertEqual((lora["status"], lora["rank"], lora["mock"]), ("ready", 8, True))
        # The record names what the data was, not the work folder's copy of it.
        self.assertEqual(lora["dataset"], "uploaded by alice")
        self.assertEqual(len(lora["history"]), 4)
        self.assertEqual(lora["training"]["status"], "done")
        self.assertTrue(lora["can_delete"])

        listed = self.call("GET", "/loras")[1]["loras"]
        self.assertEqual([one["name"] for one in listed], ["demo"])
        log = self.call("GET", "/loras/%d/log" % lora_id)[1]["lines"]
        self.assertTrue(any(line.startswith("TRAIN:") for line in log))
        data = self.call("GET", "/loras/%d/dataset?limit=2" % lora_id)[1]
        self.assertEqual((len(data["records"]), data["total"]), (2, len(RECORDS)))
        self.assertEqual(data["records"][0]["answer"], "It is answer 1.")

        # Another user cannot tell it exists: every road to it is a 404.
        bob = self.registry.add_user("bob")
        self.assertEqual(self.call("GET", "/loras", key=bob)[1]["loras"], [])
        for method, path in (("GET", ""), ("GET", "/log"), ("GET", "/dataset"),
                             ("POST", "/cancel"), ("DELETE", "")):
            self.assertEqual(self.call(method, "/loras/%d%s" % (lora_id, path), key=bob)[0],
                             404, path)
        form = self.call("GET", "/loras/form", key=bob)[1]
        self.assertFalse(any(model["id"] == "unsloth/base" and model["loras"]
                             for model in form["base_models"]))

        folder = lora_catalog.absolute(lora["folder"])
        self.assertTrue(os.path.isdir(folder))
        status, reply = self.call("DELETE", "/loras/%d" % lora_id)
        self.assertEqual(status, 200, reply)
        self.assertFalse(os.path.exists(folder))
        self.assertEqual(self.call("GET", "/loras/%d" % lora_id)[0], 404)

    def test_a_queued_training_is_cancelled_on_the_spot(self):
        lora_id = self.call("POST", "/loras", request())[1]["lora"]["id"]
        status, reply = self.call("POST", "/loras/%d/cancel" % lora_id)
        self.assertEqual(status, 200, reply)
        self.assertEqual(reply["lora"]["status"], "cancelled")
        self.assertIsNone(self.registry.claim_next_training())
        self.assertEqual(self.call("POST", "/loras/%d/cancel" % lora_id)[0], 409)

    def test_a_forgotten_row_hands_its_training_to_nobody(self):
        lora_id = self.call("POST", "/loras", request())[1]["lora"]["id"]
        self.call("POST", "/loras/%d/cancel" % lora_id)
        self.app.catalog.remove(lora_id)             # `catalog forget`, say
        # A row in the same id's place, as a database without AUTOINCREMENT
        # would hand out: another folder, so not that training's.
        with self.app.catalog._connect() as conn:
            conn.execute("INSERT INTO loras (id, name, folder, status, origin, created_at,"
                         " updated_at) VALUES (?, 'other', 'elsewhere', 'ready', 'scanned',"
                         " 'now', 'now')", (lora_id,))
        # Nobody's row is nobody's to see; given to alice, still not hers to delete.
        self.assertEqual(self.call("GET", "/loras/%d" % lora_id)[0], 404)
        self.app.catalog.own([lora_id], "alice")
        theirs = self.call("GET", "/loras/%d" % lora_id)[1]["lora"]
        self.assertEqual((theirs["trained_here"], theirs["can_delete"]), (False, False))
        self.assertEqual(self.call("DELETE", "/loras/%d" % lora_id)[0], 403)

    def test_bad_requests_are_400s(self):
        status, reply = self.call("POST", "/loras", request(rank=-1))
        self.assertEqual(status, 400)
        self.assertIn("rank", reply["error"])
        body = request()
        del body["dataset"]
        self.assertEqual(self.call("POST", "/loras", body)[0], 400)

    def test_a_scan_is_asked_through_the_api(self):
        where = os.path.join(self.folder, "trained", "by-hand")
        os.makedirs(where)
        with open(os.path.join(where, "adapter_config.json"), "w") as handle:
            json.dump({"r": 4, "base_model_name_or_path": "unsloth/base", "mock": True}, handle)
        status, report = self.call("POST", "/loras/scan")
        self.assertEqual(status, 200, report)
        # Found by a scan, it belongs to nobody, so nobody sees it...
        self.app.catalog.scan(os.path.join(self.folder, "trained"))
        self.assertEqual(self.call("GET", "/loras")[1]["loras"], [])
        self.assertEqual(self.call("POST", "/loras/scan")[1]["unowned"],
                         len(self.app.catalog.all(owner=None)))
        # ...until the server's `catalog own` gives it to someone.
        self.app.catalog.own([self.app.catalog.find("by-hand")["id"]], "alice")
        listed = self.call("GET", "/loras")[1]["loras"]
        self.assertEqual([(one["name"], one["origin"], one["rank"]) for one in listed],
                         [("by-hand", "scanned", 4)])
        # Not trained through the API, so nobody may delete it from here.
        self.assertEqual(self.call("DELETE", "/loras/%d" % listed[0]["id"])[0], 403)

    def test_a_mocked_lora_is_refused_by_a_search_that_loads_weights(self):
        self.call("POST", "/loras", request())
        self.work()
        folder = lora_catalog.absolute(self.app.catalog.by_name("demo", "alice")["folder"])
        slots = dict(self.slots, L3=folder)
        # The mocked template reads only the rank, so it takes a mocked LoRA...
        status, reply = self.call("POST", "/jobs", self.submission(LORA_SLOTS=slots))
        self.assertEqual(status, 201, reply)
        # ...and a template that loads the weights does not.
        status, reply = self.call("POST", "/jobs", self.submission(
            LORA_SLOTS=slots, TEMPLATE="template_code.py"))
        self.assertEqual(status, 400)
        self.assertIn("no adapter weights", reply["error"])

    def test_a_job_cannot_name_another_users_lora_as_a_slot(self):
        self.call("POST", "/loras", request())
        self.work()
        folder = lora_catalog.absolute(self.app.catalog.by_name("demo", "alice")["folder"])
        bob = self.registry.add_user("bob")
        status, reply = self.call("POST", "/jobs", self.submission(
            LORA_SLOTS=dict(self.slots, L3=folder)), key=bob)
        self.assertEqual(status, 400)
        self.assertIn("no LoRA of yours", reply["error"])
