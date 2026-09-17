"""
test_server.py - The API as a client sees it, with a real worker behind it.

One mocked job goes the whole way -- submitted over HTTP, run through main.py
by the worker, read back, put live, streamed from, unset and deleted -- because
the endpoints are only worth anything if they agree with each other about a
job. The rest are the refusals.
"""

import json
import sys
import threading
import time
import urllib.error
import urllib.request

from async_api import inference
from async_api import registry as reg
from async_api import server
from async_api import worker

from unittests.async_api.support import JobsTestCase


class ServerTestCase(JobsTestCase):

    def setUp(self):
        super().setUp()
        inference.MockEngine.delay = 0
        self.app = server.App(self.registry, inference.ModelCache(ttl=60, sweep_every=3600))
        self.server = server.make_server(self.app, "127.0.0.1", 0, quiet=True)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = "http://127.0.0.1:%d" % self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.app.cache.close()
        super().tearDown()

    def call(self, method, path, body=None, key=None, headers=None, raw=False):
        request = urllib.request.Request(
            self.base + path, method=method,
            data=None if body is None else json.dumps(body).encode("utf-8"),
            headers=dict({"Authorization": "Bearer " + (key or self.key),
                          "Content-Type": "application/json"}, **(headers or {})))
        try:
            with urllib.request.urlopen(request, timeout=60) as reply:
                text = reply.read().decode("utf-8")
                return reply.status, text if raw else json.loads(text)
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read().decode("utf-8"))


class WholeJobTests(ServerTestCase):

    def test_a_job_from_submission_to_inference_and_back(self):
        status, reply = self.call("POST", "/jobs", self.submission())
        self.assertEqual(status, 201, reply)
        job_id = reply["job"]["id"]
        self.assertEqual(reply["job"]["queue_position"], 1)

        # Not live before it has run.
        self.assertEqual(self.call("POST", "/jobs/%d/live" % job_id, {})[0], 409)

        self.assertEqual(worker.serve(self.registry, once=True, poll=0.2), 0)

        status, reply = self.call("GET", "/jobs/%d/status" % job_id)
        self.assertEqual(reply["job"]["status"], reg.DONE,
                         self.call("GET", "/jobs/%d/log" % job_id)[1])
        self.assertEqual(reply["progress"]["generations_scored"], 2)

        status, detail = self.call("GET", "/jobs/%d" % job_id)
        results = detail["results"]
        self.assertEqual(len(results["population"]), 4)
        self.assertEqual(len(results["fitness_history"]), 2)
        self.assertTrue(results["testing"]["summary"])
        best = results["best"]
        self.assertIsNotNone(best)

        status, jobs = self.call("GET", "/jobs")
        self.assertEqual([one["id"] for one in jobs["jobs"]], [job_id])
        self.assertEqual(jobs["jobs"][0]["summary"]["best"]["number"], best["number"])

        status, one = self.call("GET", "/jobs/%d/individuals/%d" % (job_id, best["number"]))
        self.assertEqual(len(one["individual"]["execution"]["exchanges"]), 3)

        status, live = self.call("POST", "/jobs/%d/live" % job_id, {})
        self.assertEqual(status, 201, live)
        self.assertEqual(live["deployment"]["individual"], best["number"])
        self.assertEqual(live["deployment"]["engine"], "mock")
        token = live["token"]

        status, text = self.call("POST", "/infer", {"token": token, "prompt": "hello"},
                                 raw=True)
        self.assertEqual(status, 200)
        self.assertIn(best["chromosome"], text)
        self.assertIn("hello", text)

        status, events = self.call("POST", "/infer", {"token": token, "prompt": "hello"},
                                   headers={"Accept": "text/event-stream"}, raw=True)
        self.assertTrue(events.startswith("data: "))
        self.assertTrue(events.rstrip().endswith("data: {}"))

        self.assertEqual(len(self.call("GET", "/live")[1]["live"]), 1)
        self.assertEqual(self.call("DELETE", "/jobs/%d/live" % job_id)[1], {"unset": 1})
        self.assertEqual(self.call("POST", "/infer", {"token": token, "prompt": "x"})[0], 401)

        status, reply = self.call("DELETE", "/jobs/%d/run" % job_id)
        self.assertEqual(reply["job"]["status"], reg.DELETED)
        self.assertIsNone(self.call("GET", "/jobs/%d" % job_id)[1]["results"])

        self.assertEqual(self.call("DELETE", "/jobs/%d" % job_id)[0], 200)
        self.assertEqual(self.call("GET", "/jobs/%d" % job_id)[0], 404)


class RefusalTests(ServerTestCase):

    def test_no_key_no_jobs(self):
        self.assertEqual(self.call("GET", "/jobs", key="gep_wrong")[0], 401)

    def test_another_users_job_is_a_404(self):
        job_id = self.call("POST", "/jobs", self.submission())[1]["job"]["id"]
        bob = self.registry.add_user("bob")
        for method, path in (("GET", "/jobs/%d"), ("POST", "/jobs/%d/cancel"),
                             ("DELETE", "/jobs/%d"), ("POST", "/jobs/%d/live")):
            self.assertEqual(self.call(method, path % job_id, key=bob)[0], 404, path)

    def test_a_bad_submission_is_a_400(self):
        status, reply = self.call("POST", "/jobs", {"settings": {"NOPE": 1}})
        self.assertEqual(status, 400)
        self.assertIn("NOPE", reply["error"])

    def test_a_running_job_cannot_be_deleted(self):
        job_id = self.call("POST", "/jobs", self.submission())[1]["job"]["id"]
        self.registry.claim_next()
        self.assertEqual(self.call("DELETE", "/jobs/%d" % job_id)[0], 409)
        self.assertEqual(self.call("DELETE", "/jobs/%d/run" % job_id)[0], 409)

    def test_unknown_routes_and_methods(self):
        self.assertEqual(self.call("GET", "/nothing")[0], 404)
        self.assertEqual(self.call("PUT", "/jobs")[0], 405)

    def test_health_needs_no_key(self):
        self.assertEqual(self.call("GET", "/health", key="none")[0], 200)


class CancelTests(ServerTestCase):

    def test_a_running_job_is_stopped_by_a_cancel(self):
        job_id = self.call("POST", "/jobs", self.submission())[1]["job"]["id"]
        job = self.registry.claim_next()
        # Stand-in for a long search: main.py would be minutes of this.
        forever = [sys.executable, "-c", "import time; time.sleep(120)"]
        finished = []
        runner = threading.Thread(target=lambda: finished.append(
            worker.run_job(self.registry, job, argv=forever, poll=0.1)))
        runner.start()
        deadline = time.time() + 10
        while self.registry.job(job_id)["pid"] is None and time.time() < deadline:
            time.sleep(0.05)
        status, reply = self.call("POST", "/jobs/%d/cancel" % job_id)
        self.assertEqual(status, 202, reply)
        runner.join(timeout=60)
        self.assertEqual(finished, [reg.CANCELLED])
        self.assertEqual(self.registry.job(job_id)["status"], reg.CANCELLED)
