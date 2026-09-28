"""
test_registry.py - Jobs in arrival order, keys and tokens by hash, cancels,
and a stopped job queued again.
"""

import json
import os
import sqlite3

from gep_lora.service import registry as reg

from unittests.service.support import JobsTestCase


class RegistryTests(JobsTestCase):

    def queued(self, label):
        job = self.registry.reserve_job(self.user["id"], label, {})
        return self.registry.enqueue(job["id"], 1)

    def test_a_key_is_stored_only_as_its_hash(self):
        with sqlite3.connect(self.registry.path) as conn:
            stored = conn.execute("SELECT key_hash FROM users").fetchone()[0]
        self.assertNotIn(self.key, stored)
        self.assertEqual(stored, reg.digest(self.key))
        self.assertIsNone(self.registry.user_for_key("gep_wrong"))
        self.assertIsNone(self.registry.user_for_key(None))

    def test_a_rotated_key_replaces_the_old_one(self):
        new = self.registry.rotate_key("alice")
        self.assertIsNone(self.registry.user_for_key(self.key))
        self.assertEqual(self.registry.user_for_key(new)["name"], "alice")

    def test_jobs_are_claimed_oldest_first_and_once(self):
        first, second = self.queued("a"), self.queued("b")
        self.assertEqual(self.registry.queue_position(second["id"]), 2)
        claimed = self.registry.claim_next()
        self.assertEqual((claimed["id"], claimed["status"]), (first["id"], reg.RUNNING))
        self.assertIsNone(self.registry.queue_position(first["id"]))
        self.assertEqual(self.registry.queue_position(second["id"]), 1)
        self.assertEqual(self.registry.claim_next()["id"], second["id"])
        self.assertIsNone(self.registry.claim_next())

    def test_a_job_being_prepared_is_not_in_the_queue(self):
        self.registry.reserve_job(self.user["id"], "half", {})
        self.assertIsNone(self.registry.claim_next())

    def test_cancelling_a_queued_job_takes_it_out_of_the_queue(self):
        job = self.queued("a")
        self.assertEqual(self.registry.request_cancel(job["id"]), reg.CANCELLED)
        self.assertIsNone(self.registry.claim_next())

    def test_cancelling_a_running_job_only_flags_it(self):
        job = self.queued("a")
        self.registry.claim_next()
        self.assertEqual(self.registry.request_cancel(job["id"]), reg.RUNNING)
        self.assertTrue(self.registry.cancel_requested(job["id"]))
        self.assertEqual(self.registry.job(job["id"])["status"], reg.RUNNING)

    def test_a_finished_job_has_nothing_to_cancel(self):
        job = self.queued("a")
        self.registry.finish(job["id"], reg.DONE, exit_code=0)
        self.assertEqual(self.registry.request_cancel(job["id"]), reg.DONE)

    def test_another_users_job_is_not_found(self):
        job = self.queued("a")
        other = self.registry.user_for_key(self.registry.add_user("bob"))
        self.assertIsNone(self.registry.job(job["id"], other["id"]))
        self.assertEqual(self.registry.jobs(other["id"]), [])

    def test_a_token_reaches_its_deployment_until_revoked(self):
        job = self.queued("a")
        token, row = self.registry.add_deployment(job, "local", 1, "CAT.L1.L2.w1.w2", {})
        self.assertEqual(self.registry.deployment_for_token(token)["id"], row["id"])
        self.assertEqual(self.registry.revoke([row["id"]]), 1)
        self.assertIsNone(self.registry.deployment_for_token(token))
        self.assertEqual(self.registry.revoke([row["id"]]), 0)

    def test_deleting_a_job_takes_its_deployments(self):
        job = self.queued("a")
        token, _ = self.registry.add_deployment(job, "local", 1, "CAT.L1.L2.w1.w2", {})
        self.registry.delete_job(job["id"])
        self.assertIsNone(self.registry.deployment_for_token(token))


class RequeueTests(JobsTestCase):
    """A job that stopped, going back into the queue to be resumed or evaluated."""

    def stopped(self, status=reg.STOPPED):
        job = self.registry.enqueue(
            self.registry.reserve_job(self.user["id"], "a", {})["id"], 1)
        self.registry.claim_next()
        self.registry.finish(job["id"], status, exit_code=1, error="stopped")
        return job

    def test_a_stopped_job_is_queued_again_for_the_task_asked(self):
        job = self.stopped()
        queued = self.registry.requeue(job["id"], reg.RESUME)
        self.assertEqual((queued["status"], queued["task"], queued["requeued_from"]),
                         (reg.QUEUED, reg.RESUME, reg.STOPPED))
        self.assertIsNone(queued["error"])
        self.assertEqual(self.registry.claim_next()["id"], job["id"])

    def test_a_finished_search_can_be_evaluated_but_not_resumed(self):
        job = self.stopped(reg.DONE)
        self.assertIsNone(self.registry.requeue(job["id"], reg.RESUME))
        queued = self.registry.requeue(job["id"], reg.EVALUATE, {"force": True})
        self.assertEqual(queued["task"], reg.EVALUATE)
        self.assertEqual(json.loads(queued["task_options"]), {"force": True})

    def test_a_queued_or_running_job_is_not_requeued(self):
        job = self.registry.enqueue(
            self.registry.reserve_job(self.user["id"], "a", {})["id"], 1)
        self.assertIsNone(self.registry.requeue(job["id"], reg.EVALUATE))
        self.registry.claim_next()
        self.assertIsNone(self.registry.requeue(job["id"], reg.RESUME))

    def test_cancelling_a_requeue_puts_the_job_back_as_it_was(self):
        job = self.stopped(reg.DONE)
        self.registry.requeue(job["id"], reg.EVALUATE)
        self.assertEqual(self.registry.request_cancel(job["id"]), reg.DONE)
        row = self.registry.job(job["id"])
        self.assertEqual(row["status"], reg.DONE)
        self.assertIsNone(row["requeued_from"])
        self.assertIsNone(self.registry.claim_next())

    def test_a_registry_from_before_the_tasks_is_migrated(self):
        # The jobs table as it was before task, task_options and requeued_from.
        folder = os.path.join(self.folder, "old")
        os.makedirs(folder)
        conn = sqlite3.connect(os.path.join(folder, "jobs.sqlite3"))
        conn.executescript(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE,"
            " key_hash TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL);"
            "CREATE TABLE jobs (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL,"
            " label TEXT, status TEXT NOT NULL DEFAULT 'queued', created_at TEXT NOT NULL,"
            " started_at TEXT, finished_at TEXT, folder TEXT NOT NULL, run_id INTEGER,"
            " options TEXT NOT NULL DEFAULT '{}', cancel_requested INTEGER NOT NULL"
            " DEFAULT 0, pid INTEGER, exit_code INTEGER, error TEXT);"
            "INSERT INTO users VALUES (1, 'old', 'x', 'then');"
            "INSERT INTO jobs (user_id, status, created_at, folder)"
            " VALUES (1, 'failed', 'then', 'user1/job1');")
        conn.commit()
        conn.close()
        old = reg.Registry(folder)
        # Copied into the one database, and the old file put aside.
        self.assertTrue(os.path.exists(os.path.join(folder, "api.sqlite")))
        self.assertTrue(os.path.exists(os.path.join(folder, "jobs.sqlite3.merged")))
        self.assertFalse(os.path.exists(os.path.join(folder, "jobs.sqlite3")))
        row = old.job(1)
        self.assertEqual((row["task"], row["task_options"], row["requeued_from"]),
                         (reg.SEARCH, "{}", None))
        self.assertEqual(old.requeue(1, reg.RESUME)["status"], reg.QUEUED)
