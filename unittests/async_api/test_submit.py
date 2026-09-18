"""
test_submit.py - A submission becomes a prepared sweep, or nothing at all.
"""

import json
import os

import main
from async_api import registry as reg
from async_api import submit
from storage import store

from unittests.async_api.support import RECORDS, JobsTestCase


class SubmitTests(JobsTestCase):

    def test_a_submission_is_a_prepared_database(self):
        job = submit.submit(self.registry, self.user, self.submission())
        self.assertEqual(job["status"], reg.QUEUED)
        conn = store.connect(self.registry.database(job))
        try:
            conf = store.get_settings(conn, job["run_id"])
            splits = {one["split"]: one["records"]
                      for one in store.dataset_summary(conn, job["run_id"])}
            self.assertEqual(store.individuals(conn, job["run_id"]), [])
        finally:
            conn.close()
        self.assertEqual(conf["COUNT"], 4)
        self.assertEqual(splits, {"training": 4, "testing": 2})
        # Not given, so not part of the job -- not settings.py's file.
        self.assertIsNone(conf["VALIDATION_SET"])
        # Every seed is written down as a number.
        for name in ("SEED", "WEIGHT_MASTER_SEED", "SELECTION_MASTER_SEED",
                     "MUTATION_MASTER_SEED", "WEIGHT_MUTATION_MASTER_SEED"):
            self.assertIsInstance(conf[name], int)

        # And main.py agrees: a bare --db would run it rather than add to it.
        options = main.parse(["--db", self.registry.database(job)])
        self.assertEqual(main.prepared(options), job["run_id"])

    def refused(self, payload, fragment):
        before = os.listdir(self.registry.root)
        with self.assertRaises(submit.SubmissionError) as caught:
            submit.submit(self.registry, self.user, payload)
        self.assertIn(fragment, str(caught.exception))
        self.assertEqual(self.registry.jobs(self.user["id"]), [])
        self.assertEqual(sorted(os.listdir(self.registry.root)), sorted(before))

    def test_an_unknown_setting_is_refused(self):
        payload = self.submission()
        payload["settings"]["NOT_A_SETTING"] = 1
        self.refused(payload, "NOT_A_SETTING")

    def test_a_locked_setting_is_refused(self):
        payload = self.submission()
        payload["settings"]["DB_RUN_DIR"] = "/elsewhere"
        self.refused(payload, "DB_RUN_DIR")

    def test_an_unknown_evaluator_is_refused(self):
        self.refused(self.submission(EVALUATOR="nope"), "EVALUATOR")

    def test_a_composite_with_no_members_is_refused(self):
        self.refused(self.submission(EVALUATOR="composite", COMPOSITE_EVALUATORS=[]),
                     "COMPOSITE_EVALUATORS")

    def test_a_composite_with_a_bad_member_is_refused(self):
        self.refused(self.submission(EVALUATOR="composite",
                                     COMPOSITE_EVALUATORS=["heuristic", ["similarity", 0]]),
                     "similarity")

    def test_a_composite_is_accepted(self):
        job = submit.submit(self.registry, self.user, self.submission(
            EVALUATOR="composite", COMPOSITE_AGGREGATE="geometric",
            COMPOSITE_EVALUATORS=["heuristic", ["similarity", 2]]))
        conn = store.connect(self.registry.database(job))
        try:
            conf = store.get_settings(conn, job["run_id"])
        finally:
            conn.close()
        self.assertEqual(conf["COMPOSITE_EVALUATORS"], ["heuristic", ["similarity", 2]])

    def test_missing_adapters_are_refused(self):
        self.refused(self.submission(LORA_SLOTS={"L1": os.path.join(self.folder, "none")}),
                     "slot L1")

    def test_a_job_needs_training_questions(self):
        payload = self.submission()
        del payload["datasets"]["training"]
        self.refused(payload, "training")

    def test_an_unreadable_record_leaves_nothing_behind(self):
        # Refused after the job row and folder exist: both must go.
        payload = self.submission()
        payload["datasets"]["training"] = ['{"messages": "not a list"}']
        with self.assertRaises(submit.SubmissionError):
            submit.submit(self.registry, self.user, payload)
        self.assertEqual(self.registry.jobs(self.user["id"]), [])
        self.assertFalse(os.path.exists(os.path.join(self.registry.root,
                                                     "user%d" % self.user["id"], "job1")))

    def test_a_shared_dataset_cannot_escape_its_folder(self):
        payload = self.submission()
        payload["datasets"]["training"] = {"file": "../config/settings.py"}
        self.refused(payload, "no shared dataset")

    def test_unknown_options_are_refused(self):
        payload = self.submission()
        payload["options"] = {"rm_rf": True}
        self.refused(payload, "rm_rf")


class DatasetLinesTests(JobsTestCase):

    def test_the_three_shapes(self):
        self.assertEqual(submit.dataset_lines("training", ["a", "b"]), ["a", "b"])
        self.assertEqual(submit.dataset_lines("training", "a\n\n b \n"), ["a", "b"])
        self.assertEqual(submit.dataset_lines("training", [RECORDS[0]]),
                         [json.dumps(RECORDS[0])])

    def test_a_prompt_spanning_lines_is_refused(self):
        with self.assertRaises(submit.SubmissionError):
            submit.dataset_lines("training", ["one\ntwo"])
