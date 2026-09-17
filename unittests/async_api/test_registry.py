"""
test_registry.py - Jobs in arrival order, keys and tokens by hash, cancels.
"""

import sqlite3

from async_api import registry as reg

from unittests.async_api.support import JobsTestCase


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
