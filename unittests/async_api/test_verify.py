"""
test_verify.py - One blend of a finished job, queued against its own LoRAs.

What this half owns, and so what is tested here:

  * the contestants a request means: the LoRAs the chromosome names by default,
    every slot on "all", a named list, and the refusals for everything else;
  * the sweep's own evaluator, judge and split as the defaults, since a
    verification graded by another rubric is not comparable with the fitness
    the search produced;
  * the command line the worker runs, which is the only place the API says
    anything about how the comparison works;
  * the registry: a verification is queued work, claimed once, and its folder
    lives inside the job's;
  * the endpoints, with a real worker behind them: a mocked job verified over
    HTTP end to end, its report read back, and a queued one refused on a job
    that has not finished.

The comparison itself is tested in unittests/testing.
"""

import json
import os
import time

from async_api import registry as reg
from async_api import submit
from async_api import verify
from async_api import worker

from unittests.async_api.support import JobsTestCase
from unittests.async_api.test_server import ServerTestCase


class SlotTests(JobsTestCase):

    def test_the_slots_a_chromosome_names(self):
        self.assertEqual(verify.blend_slots("CAT.L5.L2.w5.w4"), ["L2", "L5"])
        self.assertEqual(verify.blend_slots("CAT.CAT.L4.L2.L3.w3.w3.w4"),
                         ["L2", "L3", "L4"])
        # The same slot twice at two weights is still one contestant.
        self.assertEqual(verify.blend_slots("CAT.L3.L3.w1.w2"), ["L3"])
        # LIN starts with an L too, and is a fold, not a slot.
        self.assertEqual(verify.blend_slots("CAT.L1.LIN.w1.L2.L1.w2.w3"), ["L1", "L2"])

    def test_a_chromosome_that_does_not_decode_is_refused(self):
        with self.assertRaises(verify.VerifyError):
            verify.blend_slots("w1.L3.w1")

    def test_a_lone_adapter_names_its_one_slot(self):
        self.assertEqual(verify.blend_slots("L3.w1"), ["L3"])


class OfferedTests(JobsTestCase):

    def setUp(self):
        super().setUp()
        self.job = submit.submit(self.registry, self.user, self.submission())
        self.offered = verify.choices(self.registry.database(self.job),
                                      self.job["run_id"])
        # A prepared job holds no individuals yet, so these tests supply the
        # population the page would be choosing from.
        self.offered["individuals"] = [
            {"number": 1, "chromosome": "CAT.L5.L2.w5.w4", "state": "ok",
             "fitness": 0.4, "is_best": False, "slots": ["L2", "L5"]},
            {"number": 2, "chromosome": "CAT.L3.L1.w1.w2", "state": "ok",
             "fitness": 0.9, "is_best": True, "slots": ["L1", "L3"]},
            {"number": 3, "chromosome": "LIN.L3.L1.w1.w2", "state": "BAD",
             "fitness": 0.0, "is_best": False, "slots": ["L1", "L3"]}]

    def test_the_sweep_is_what_the_form_offers(self):
        self.assertEqual(self.offered["slots"], ["L1", "L2", "L3", "L4", "L5"])
        self.assertEqual(self.offered["splits"], ["testing", "training"])
        self.assertEqual(self.offered["defaults"]["split"], "testing")
        names = {one["value"] for one in self.offered["evaluators"]}
        self.assertIn("similarity", names)
        self.assertIn("llm_judge_baseline", names)

    def test_the_default_is_the_elite_and_the_lor_as_in_it(self):
        number, options = verify.options_for({}, self.offered)
        self.assertEqual(number, 2)
        self.assertEqual(options["slots"], ["L1", "L3"])
        self.assertEqual(options["split"], "testing")

    def test_all_slots_and_a_named_list(self):
        _, options = verify.options_for({"slots": "all"}, self.offered)
        self.assertEqual(options["slots"], ["L1", "L2", "L3", "L4", "L5"])
        _, options = verify.options_for({"slots": ["L2", "L4"]}, self.offered)
        self.assertEqual(options["slots"], ["L2", "L4"])

    def test_the_judge_can_be_swapped(self):
        _, options = verify.options_for(
            {"individual": 1, "evaluator": "llm_judge_reference",
             "judge_model": "qwen3-32b", "judge_backend": "unsloth",
             "split": "training", "count": 5}, self.offered)
        self.assertEqual(options["evaluator"], "llm_judge_reference")
        self.assertEqual(options["judge_model"], "qwen3-32b")
        self.assertEqual(options["judge_backend"], "unsloth")
        self.assertEqual(options["split"], "training")
        self.assertEqual(options["count"], 5)

    def refused(self, body, fragment):
        with self.assertRaises(verify.VerifyError) as caught:
            verify.options_for(body, self.offered)
        self.assertIn(fragment, str(caught.exception))

    def test_the_refusals(self):
        self.refused({"individual": 99}, "no individual 99")
        self.refused({"individual": 3}, "BAD")
        self.refused({"evaluator": "nope"}, "unknown evaluator")
        self.refused({"judge_backend": "nope"}, "unknown judge backend")
        self.refused({"split": "validation"}, "holds no validation split")
        self.refused({"slots": ["L9"]}, "no slot L9")
        self.refused({"slots": 7}, "slots must be")
        self.refused({"count": 0}, "count must be")
        self.refused({"count": 2.5}, "count must be")

    def test_the_command_says_what_was_asked_for(self):
        number, options = verify.options_for(
            {"individual": 1, "evaluator": "similarity", "count": 4}, self.offered)
        argv = verify.command("python", "job.sqlite3", 1, number, options, "out")
        self.assertEqual(argv[:4], ["python", "-u", "-m",
                                    "testing.evaluate_chromosome_against_loras"])
        pairs = dict(zip(argv[4::2], argv[5::2]))
        self.assertEqual(pairs["--individual"], "1")
        self.assertEqual(pairs["--evaluator"], "similarity")
        self.assertEqual(pairs["--count"], "4")
        self.assertEqual(pairs["--slots"], "L2,L5")
        self.assertEqual(pairs["--into"], "out")
        # Not asked for, so not on the command line: the script then reads the
        # sweep's own.
        self.assertNotIn("--judge-model", argv)

    def test_no_report_until_there_is_one(self):
        self.assertIsNone(verify.report(self.folder))


class RegistryTests(JobsTestCase):

    def setUp(self):
        super().setUp()
        self.job = submit.submit(self.registry, self.user, self.submission())

    def test_queued_claimed_once_and_finished(self):
        row = self.registry.add_verification(self.job, 2, "CAT.L3.L1.w1.w2",
                                             {"evaluator": "similarity"})
        self.assertEqual(row["status"], reg.QUEUED)
        self.assertEqual(json.loads(row["options"])["evaluator"], "similarity")

        claimed = self.registry.claim_next_verification()
        self.assertEqual(claimed["id"], row["id"])
        self.assertEqual(claimed["status"], reg.RUNNING)
        self.assertIsNone(self.registry.claim_next_verification())

        self.registry.finish_verification(row["id"], reg.DONE, exit_code=0)
        self.assertEqual(self.registry.verification(row["id"])["status"], reg.DONE)

    def test_its_folder_is_inside_the_jobs(self):
        folder = self.registry.verification_folder(self.job, 3)
        self.assertEqual(os.path.dirname(folder), self.registry.folder(self.job))

    def test_a_dead_workers_verification_is_recovered(self):
        row = self.registry.add_verification(self.job, 2, "CAT.L3.L1.w1.w2", {})
        self.registry.claim_next_verification()
        self.assertEqual(worker.recover_verifications(self.registry), 1)
        found = self.registry.verification(row["id"])
        self.assertEqual(found["status"], reg.FAILED)
        self.assertIn("worker stopped", found["error"])

    def test_another_users_verification_is_not_theirs(self):
        row = self.registry.add_verification(self.job, 2, "CAT.L3.L1.w1.w2", {})
        other = self.registry.user_for_key(self.registry.add_user("bob"))
        self.assertIsNone(self.registry.verification(row["id"], other["id"]))


class EndpointTests(ServerTestCase):

    def finished_job(self):
        """One mocked job, submitted and run by a worker. -> its id."""
        status, reply = self.call("POST", "/jobs", self.submission())
        self.assertEqual(status, 201)
        job_id = reply["job"]["id"]
        worker.serve(self.registry, once=True)
        status, reply = self.call("GET", "/jobs/%d/status" % job_id)
        self.assertEqual(reply["job"]["status"], reg.DONE, reply)
        return job_id

    def test_a_verification_over_http_end_to_end(self):
        job_id = self.finished_job()

        status, form = self.call("GET", "/jobs/%d/verify" % job_id)
        self.assertEqual(status, 200)
        self.assertTrue(form["can_verify"])
        self.assertEqual(form["verifications"], [])
        blend = next(one for one in form["individuals"] if one["state"] != "BAD")

        # similarity, so nothing asks a judge: the mocked answers are graded
        # against the dataset's own, locally.
        status, reply = self.call("POST", "/jobs/%d/verify" % job_id,
                                  {"individual": blend["number"],
                                   "evaluator": "similarity"})
        self.assertEqual(status, 201, reply)
        row = reply["verification"]
        self.assertEqual(row["status"], reg.QUEUED)
        self.assertEqual(row["number"], blend["number"])
        self.assertEqual(row["options"]["slots"], blend["slots"])

        # The same worker takes it, after the jobs.
        worker.serve(self.registry, once=True)
        status, reply = self.call("GET", "/verifications/%d" % row["id"])
        self.assertEqual(status, 200)
        one = reply["verification"]
        self.assertEqual(one["status"], reg.DONE, one)

        report = one["report"]
        self.assertEqual(report["meta"]["individual"], blend["number"])
        self.assertEqual(report["meta"]["chromosome"], blend["chromosome"])
        self.assertEqual(report["meta"]["evaluator"], "similarity")
        # One row and one transcript per contestant: the blend and its LoRAs.
        expected = ["blend"] + blend["slots"]
        self.assertEqual(sorted(report["answers"]), sorted(expected))
        self.assertEqual(sorted(row["key"] for row in report["summary"]),
                         sorted(expected))
        for entry in report["summary"]:
            self.assertEqual(entry["graded"], report["meta"]["questions"])
            if entry["key"] != "blend":
                self.assertEqual(entry["wins"] + entry["ties"] + entry["losses"],
                                 report["meta"]["questions"])
        # And the page can label every key without guessing.
        self.assertEqual(set(report["meta"]["contestants"]), set(expected))

        # It is the job's now, and its log is readable.
        status, form = self.call("GET", "/jobs/%d/verify" % job_id)
        self.assertEqual([one["id"] for one in form["verifications"]], [row["id"]])
        status, reply = self.call("GET", "/verifications/%d/log" % row["id"])
        self.assertEqual(status, 200)
        self.assertTrue(any("individual" in line for line in reply["lines"]), reply)

        # Deleting the run takes the verification's folder with it.
        folder = self.registry.verification_folder(
            self.registry.job(job_id), row["id"])
        self.assertTrue(os.path.isdir(folder))
        self.call("DELETE", "/jobs/%d/run" % job_id)
        self.assertFalse(os.path.exists(folder))

    def test_an_unfinished_job_cannot_be_verified(self):
        status, reply = self.call("POST", "/jobs", self.submission())
        job_id = reply["job"]["id"]
        status, reply = self.call("POST", "/jobs/%d/verify" % job_id, {})
        self.assertEqual(status, 409)
        self.assertIn("only a finished job", reply["error"])

    def test_another_users_verification_is_a_404(self):
        job_id = self.finished_job()
        status, reply = self.call("POST", "/jobs/%d/verify" % job_id,
                                  {"evaluator": "similarity"})
        row = reply["verification"]
        other = self.registry.add_user("bob")
        status, reply = self.call("GET", "/verifications/%d" % row["id"], key=other)
        self.assertEqual(status, 404)

    def test_a_bad_request_is_a_400(self):
        job_id = self.finished_job()
        status, reply = self.call("POST", "/jobs/%d/verify" % job_id,
                                  {"individual": 999})
        self.assertEqual(status, 400)
        self.assertIn("no individual 999", reply["error"])
