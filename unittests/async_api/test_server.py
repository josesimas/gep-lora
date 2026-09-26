"""
test_server.py - The API as a client sees it, with a real worker behind it.

One mocked job goes the whole way -- submitted over HTTP, run through main.py
by the worker, read back, put live, streamed from, unset and deleted -- because
the endpoints are only worth anything if they agree with each other about a
job. The rest are the refusals.
"""

import contextlib
import io
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

from async_api import inference
import start_run
from async_api import registry as reg
from async_api import server
from async_api import worker
from storage import store

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

    def download(self, path, key=None):
        """-> (status, headers, body bytes)."""
        request = urllib.request.Request(
            self.base + path, headers={"Authorization": "Bearer " + (key or self.key)})
        try:
            with urllib.request.urlopen(request, timeout=60) as reply:
                return reply.status, reply.headers, reply.read()
        except urllib.error.HTTPError as error:
            return error.code, error.headers, error.read()

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

        # The database downloads whole, as a sweep store.py can read back.
        status, headers, body = self.download("/jobs/%d/database" % job_id)
        self.assertEqual(status, 200)
        self.assertIn('filename="job%d_test.sqlite3"' % job_id, headers["Content-Disposition"])
        self.assertTrue(body.startswith(b"SQLite format 3"))
        copy = os.path.join(self.folder, "downloaded.sqlite3")
        with open(copy, "wb") as handle:
            handle.write(body)
        conn = store.connect(copy)
        try:
            self.assertEqual(len(store.individuals(conn, reply["job"]["run_id"])), 4)
            self.assertEqual(len(store.fitness_by_generation(conn, reply["job"]["run_id"])), 2)
        finally:
            conn.close()
        self.assertEqual(self.download("/jobs/%d/database" % job_id,
                                       key=self.registry.add_user("carol"))[0], 404)

        self.assertEqual(len(self.call("GET", "/live")[1]["live"]), 1)
        self.assertEqual(self.call("DELETE", "/jobs/%d/live" % job_id)[1], {"unset": 1})
        self.assertEqual(self.call("POST", "/infer", {"token": token, "prompt": "x"})[0], 401)

        status, reply = self.call("DELETE", "/jobs/%d/run" % job_id)
        self.assertEqual(reply["job"]["status"], reg.DELETED)
        self.assertIsNone(self.call("GET", "/jobs/%d" % job_id)[1]["results"])
        self.assertEqual(self.download("/jobs/%d/database" % job_id)[0], 404)

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

    def test_a_refused_body_does_not_spill_into_the_next_request(self):
        # One kept-alive connection, as a browser uses: a request answered
        # before its body is read (no such endpoint, a bad key, a wrong
        # method) must not leave that body to be read as the next request.
        import http.client
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=30)
        try:
            # The scripted provider, so the one request that is answered asks no
            # real model -- whichever the judges' endpoint has loaded may take
            # longer than the timeout to write a welcome.
            body = json.dumps({"history": [{"content": "x" * 5000}],
                               "agent": {"provider": "scripted"}})
            for method, path, key, status in (("POST", "/agent/no-such-step", self.key, 404),
                                              ("POST", "/agent/intro", "gep_wrong", 401),
                                              ("DELETE", "/agent/intro", self.key, 405),
                                              ("POST", "/agent/intro", self.key, 200)):
                conn.request(method, path, body=body,
                             headers={"Authorization": "Bearer " + key,
                                      "Content-Type": "application/json"})
                reply = conn.getresponse()
                reply.read()
                self.assertEqual(reply.status, status, (method, path))
        finally:
            conn.close()

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

    def test_the_console_page_is_served_without_a_key(self):
        request = urllib.request.Request(self.base + "/console.html")
        with urllib.request.urlopen(request, timeout=10) as reply:
            self.assertIn("text/html", reply.headers["Content-Type"])
            self.assertIn(b"/infer", reply.read())

    def test_every_page_draws_the_shared_bar(self):
        for name, page in (("guide", "/guide.html"), ("runs", "/runs.html"),
                           ("settings", "/settings.html"), ("console", "/console.html")):
            with urllib.request.urlopen(self.base + page, timeout=10) as reply:
                body = reply.read().decode("utf-8")
            self.assertIn('id="nav" data-page="%s"' % name, body, page)
            self.assertIn('<script src="/nav.js"></script>', body, page)
        with urllib.request.urlopen(self.base + "/nav.js", timeout=10) as reply:
            self.assertIn("javascript", reply.headers["Content-Type"])
            script = reply.read().decode("utf-8")
        for page in ("/guide.html", "/runs.html", "/settings.html", "/console.html"):
            self.assertIn('"%s"' % page, script)

    def test_the_old_addresses_are_redirected_with_their_query(self):
        class Stay(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None
        opener = urllib.request.build_opener(Stay)
        for old, new in (("/", "/guide.html"), ("/agent?job=3", "/guide.html?job=3"),
                         ("/demo?job=3", "/console.html?job=3"), ("/guide_defaults", "/settings.html")):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                opener.open(self.base + old, timeout=10)
            self.assertEqual(caught.exception.code, 302, old)
            self.assertEqual(caught.exception.headers["Location"], new, old)

    def test_shared_datasets_are_listed(self):
        status, reply = self.call("GET", "/datasets")
        self.assertEqual(status, 200)
        self.assertIsInstance(reply["datasets"], list)

    def test_the_form_offers_what_a_submission_may_change(self):
        status, form = self.call("GET", "/settings")
        self.assertEqual(status, 200)
        self.assertIn("COUNT", form["defaults"])
        self.assertNotIn("DB_RUN_DIR", form["defaults"])
        templates = [one["value"] for one in form["choices"]["TEMPLATE"]]
        self.assertIn("template_code_mocked.py", templates)
        self.assertFalse([name for name in templates if "baseline" in name])
        self.assertTrue(all("needs_judge" in one for one in form["choices"]["EVALUATOR"]))
        aggregates = [one["value"] for one in form["choices"]["COMPOSITE_AGGREGATE"]]
        self.assertIn("geometric", aggregates)
        self.assertIn("COMPOSITE_EVALUATORS", form["defaults"])

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
        # Stopped, not cancelled: it had started, and can carry on from there.
        self.assertEqual(finished, [reg.STOPPED])
        job = self.call("GET", "/jobs/%d/status" % job_id)[1]["job"]
        self.assertEqual(job["status"], reg.STOPPED)
        self.assertTrue(job["resumable"])


class ResumeTests(ServerTestCase):
    """Stopping a search, carrying it on, and grading what it holds afterwards."""

    def stopped_job(self):
        """A job whose search stopped in its second generation, just after process.

        Driven a step at a time in this process rather than killed at a moment
        a test cannot choose: what matters to a resume is what the database
        holds, and this is what a stop there leaves in it.
        """
        body = self.submission(GENERATIONS=2, EVALUATOR="heuristic")
        job_id = self.call("POST", "/jobs", body)[1]["job"]["id"]
        job = self.registry.claim_next()
        where = ["--db", self.registry.database(job), "--run", str(job["run_id"]), "--from-db"]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(start_run.main(["--next-generation"] + where), 0)
            self.assertEqual(start_run.main(["trees", "runs", "process"] + where), 0)
        self.registry.finish(job_id, reg.STOPPED, error="stopped")
        return job_id, job

    def processed(self, job):
        """How many individuals each process step ran, pass by pass."""
        conn = store.connect(self.registry.database(job))
        try:
            return [row["items"] for row in store.step_timings(conn, job["run_id"])
                    if row["step"] == "process"]
        finally:
            conn.close()

    def test_the_runs_are_the_users_own(self):
        job_id, job = self.stopped_job()
        status, reply = self.call("GET", "/runs")
        self.assertEqual(status, 200, reply)
        run, = reply["runs"]
        self.assertEqual((run["id"], run["status"]), (job_id, reg.STOPPED))
        # Named by the user's own catalogue, and no folder goes out.
        self.assertEqual(run["loras"], sorted("slot-" + slot for slot in self.slots))
        self.assertNotIn("slots", run["summary"])
        self.assertEqual((run["verifications"], run["live"]), ([], []))
        bob = self.registry.add_user("bob")
        self.assertEqual(self.call("GET", "/runs", key=bob), (200, {"runs": []}))
        self.assertEqual(self.call("GET", "/runs", key="nope")[0], 401)
        # The page itself is served to anyone; what it shows needs the key.
        status, headers, body = self.download("/runs.html", key="nope")
        self.assertEqual(status, 200)
        self.assertIn(b"GET /runs", body)

    def test_only_a_finished_search_is_tested(self):
        job_id, job = self.stopped_job()
        self.assertFalse(self.call("GET", "/jobs/%d" % job_id)[1]["job"]["can_test"])
        status, reply = self.call("POST", "/jobs/%d/test" % job_id, {})
        self.assertEqual(status, 409, reply)
        self.assertIn("only a finished search", reply["error"])
        self.assertEqual(self.call("POST", "/jobs/%d/test" % job_id, {"nope": 1})[0], 409)
        self.assertEqual(self.call("GET", "/jobs/%d/test" % job_id,
                                   key=self.registry.add_user("bob"))[0], 404)

    def test_a_stopped_job_carries_on_from_where_it_stopped(self):
        job_id, job = self.stopped_job()
        status, reply = self.call("POST", "/jobs/%d/resume" % job_id)
        self.assertEqual(status, 200, reply)
        self.assertEqual((reply["job"]["status"], reply["job"]["task"]), (reg.QUEUED, "resume"))
        self.assertEqual(self.call("POST", "/jobs/%d/resume" % job_id)[0], 409)

        self.assertEqual(worker.serve(self.registry, once=True, poll=0.2), 0)
        status = self.call("GET", "/jobs/%d/status" % job_id)[1]
        self.assertEqual(status["job"]["status"], reg.DONE,
                         self.call("GET", "/jobs/%d/log" % job_id)[1])
        self.assertEqual(status["progress"]["generations_scored"], 3)
        # Generation 2 had been through process before the stop, so the pass
        # that finished it ran nobody again; the one after it ran the rest.
        runs = self.processed(job)
        self.assertEqual(len(runs), 4)
        self.assertEqual(runs[2], 0)
        self.assertTrue(self.call("GET", "/jobs/%d" % job_id)[1]["results"]["testing"]["summary"])
        # Finished: nothing left to resume.
        self.assertEqual(self.call("POST", "/jobs/%d/resume" % job_id)[0], 409)

    def test_a_job_is_evaluated_again_without_moving_its_search(self):
        job_id, job = self.stopped_job()
        status, form = self.call("GET", "/jobs/%d/evaluate" % job_id)
        self.assertEqual(status, 200, form)
        self.assertEqual((form["evaluator"], form["asks_judge"]), ("heuristic", False))
        self.assertGreater(form["answers"], 0)
        self.assertTrue(form["can_evaluate"])
        self.assertEqual(self.call("POST", "/jobs/%d/evaluate" % job_id,
                                   {"judge_backend": "nowhere"})[0], 400)
        self.assertEqual(self.call("POST", "/jobs/%d/evaluate" % job_id,
                                   {"evaluator": "llm_judge"})[0], 400)

        status, reply = self.call("POST", "/jobs/%d/evaluate" % job_id, {"force": True})
        self.assertEqual(status, 200, reply)
        self.assertEqual(self.call("POST", "/jobs/%d/evaluate" % job_id, {})[0], 409)
        self.assertEqual(worker.serve(self.registry, once=True, poll=0.2), 0)

        # Graded, and still where it stopped: generation 2 has been processed,
        # so its fitness was taken -- and nothing was bred from it.
        status = self.call("GET", "/jobs/%d/status" % job_id)[1]
        self.assertEqual(status["job"]["status"], reg.STOPPED,
                         self.call("GET", "/jobs/%d/log" % job_id)[1])
        self.assertTrue(status["job"]["resumable"])
        self.assertEqual(status["progress"]["generations_scored"], 2)
        self.assertEqual(status["progress"]["unscored"], 0)
        conn = store.connect(self.registry.database(job))
        try:
            judges = {row["judge_model"] for row in store.exchanges_to_score(
                conn, job["run_id"], True)}
        finally:
            conn.close()
        self.assertEqual(len(judges), 1)
        self.assertNotIn("mock", judges.pop())

        # And the resume after it picks up at the tail of generation 2.
        self.call("POST", "/jobs/%d/resume" % job_id)
        self.assertEqual(worker.serve(self.registry, once=True, poll=0.2), 0)
        status = self.call("GET", "/jobs/%d/status" % job_id)[1]
        self.assertEqual(status["job"]["status"], reg.DONE)
        self.assertEqual(status["progress"]["generations_scored"], 3)

    def test_a_job_cancelled_before_it_started_resumes_from_the_top(self):
        job_id = self.call("POST", "/jobs", self.submission())[1]["job"]["id"]
        self.assertEqual(self.call("POST", "/jobs/%d/cancel" % job_id)[1]["job"]["status"],
                         reg.CANCELLED)
        # No script has run, so there is nothing to grade.
        status, reply = self.call("POST", "/jobs/%d/evaluate" % job_id, {})
        self.assertEqual(status, 400)
        self.assertIn("Resume", reply["error"])
        self.assertEqual(self.call("POST", "/jobs/%d/resume" % job_id)[0], 200)
        self.assertEqual(worker.serve(self.registry, once=True, poll=0.2), 0)
        status = self.call("GET", "/jobs/%d/status" % job_id)[1]
        self.assertEqual(status["job"]["status"], reg.DONE,
                         self.call("GET", "/jobs/%d/log" % job_id)[1])
        self.assertEqual(status["progress"]["generations_scored"], 2)

    def test_a_queued_evaluation_cancelled_leaves_the_job_done(self):
        job_id = self.call("POST", "/jobs", self.submission())[1]["job"]["id"]
        self.assertEqual(worker.serve(self.registry, once=True, poll=0.2), 0)
        self.assertEqual(self.call("POST", "/jobs/%d/evaluate" % job_id, {})[0], 200)
        # Not live while it is queued again, and back to done once taken out.
        self.assertEqual(self.call("POST", "/jobs/%d/live" % job_id, {})[0], 409)
        status, reply = self.call("POST", "/jobs/%d/cancel" % job_id)
        self.assertEqual(status, 200, reply)
        self.assertEqual(reply["job"]["status"], reg.DONE)
        self.assertEqual(self.call("POST", "/jobs/%d/live" % job_id, {})[0], 201)


class JudgeModelsTests(ServerTestCase):
    """GET /judge/models: an endpoint's list of models, asked from the server."""

    def endpoint(self, reply, status=200):
        """A stand-in for LM Studio's /v1/models. -> its base URL."""
        body = json.dumps(reply).encode("utf-8")

        class Models(BaseHTTPRequestHandler):
            def do_GET(handler):
                handler.send_response(status if handler.path == "/v1/models" else 404)
                handler.send_header("Content-Type", "application/json")
                handler.send_header("Content-Length", str(len(body)))
                handler.end_headers()
                handler.wfile.write(body)

            def log_message(handler, *args):
                pass

        fake = HTTPServer(("127.0.0.1", 0), Models)
        threading.Thread(target=fake.serve_forever, daemon=True).start()
        self.addCleanup(fake.server_close)
        self.addCleanup(fake.shutdown)
        return "http://127.0.0.1:%d/v1" % fake.server_address[1]

    def test_the_chat_models_are_listed(self):
        url = self.endpoint({"data": [{"id": "qwen/qwen3-8b"},
                                      {"id": "text-embedding-nomic"},
                                      {"id": "google/gemma-3-12b"}]})
        status, reply = self.call("GET", "/judge/models?base_url=" + url)
        self.assertEqual(status, 200, reply)
        # Embedding models cannot grade, so they are not offered.
        self.assertEqual(reply, {"base_url": url,
                                 "models": ["qwen/qwen3-8b", "google/gemma-3-12b"]})

    def test_an_endpoint_that_cannot_be_reached_is_a_502(self):
        url = self.endpoint({"data": []})
        status, reply = self.call("GET", "/judge/models?base_url=" + url + "/nowhere")
        self.assertEqual(status, 502)
        self.assertIn("cannot reach the judge", reply["error"])

    def test_only_an_http_url_is_asked(self):
        self.assertEqual(self.call("GET", "/judge/models?base_url=file:///etc")[0], 400)

    def test_it_needs_a_key(self):
        self.assertEqual(self.call("GET", "/judge/models", key="gep_wrong")[0], 401)
