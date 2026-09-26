"""
test_golive.py - The blend a deployment serves is the blend the search scored.

The strongest form of that is checked directly: a generated
template_remote_code.py script is run against a stand-in lora server, and the
/build request it sends has to be exactly the plan golive.build_plan() makes
for the same chromosome and weight seed.
"""

import json
import os
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from async_api import golive
from blends import generate_runs
from search.generate_population import decode

from unittests.async_api.support import RANKS, RECORDS, JobsTestCase


class _Recorder(BaseHTTPRequestHandler):
    """Answers /build and /generate, keeping what /build was sent."""

    builds = None

    def log_message(self, *args):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/build":
            self.builds.append(payload)
            reply = {"rank": 1, "builds": 1, "timings": [], "cuda": False}
        else:
            reply = {"replies": ["ok"] * len(payload["prompts"]), "seconds": 0.0}
        body = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class BuildPlanTests(JobsTestCase):

    CHROMOSOMES = ("CAT.SVD.L1.L1.L3.w3.w5.w1",
                   "CAT.L3.SVD.w5.L4.CAT.w3.L4.L1.w1.w3",
                   "CAT.L5.L2.w5.w4")

    def script_build(self, chromosome, seed):
        """The /build payload a generated remote script sends."""
        builds = []
        server = HTTPServer(("127.0.0.1", 0), type("R", (_Recorder,), {"builds": builds}))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            dataset = os.path.join(self.folder, "training.jsonl")
            with open(dataset, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(RECORDS[0]) + "\n")
            steps, final = generate_runs.plan(decode(chromosome)[0], RANKS)
            source = generate_runs.render(
                chromosome, steps, final, "run_001.py", "test", "Individual 1",
                template_path=generate_runs.template_path("template_remote_code.py"),
                weight_seed=seed, training_set=dataset, slots=self.slots, count=1,
                base_model="base", chat_template=None)
            script = os.path.join(self.folder, "run_001.py")
            with open(script, "w", encoding="utf-8") as handle:
                handle.write(source)
            env = dict(os.environ, GEP_LORA_SERVER="http://127.0.0.1:%d"
                       % server.server_address[1])
            done = subprocess.run([sys.executable, script], env=env, capture_output=True,
                                  text=True, timeout=60)
            self.assertEqual(done.returncode, 0, done.stderr)
        finally:
            server.shutdown()
        self.assertEqual(len(builds), 1)
        return builds[0]

    def test_the_plan_is_the_one_the_script_sends(self):
        for number, chromosome in enumerate(self.CHROMOSOMES, 1):
            seed = 1000 + number
            sent = self.script_build(chromosome, seed)
            plan, final, _ = golive.build_plan(chromosome, RANKS, golive.draw_weights(seed))
            self.assertEqual(plan, sent["plan"], chromosome)
            self.assertEqual(final, sent["final"])

    def test_a_blend_peft_would_refuse_cannot_go_live(self):
        with self.assertRaises(golive.GoLiveError):
            # LIN over L3 (rank 8) and L1 (rank 16).
            golive.build_plan("CAT.L2.LIN.w2.L3.L1.w5.w1",
                              RANKS, golive.draw_weights(1))


class DrawWeightsTests(unittest.TestCase):

    def test_ten_weights_in_the_open_interval_repeatably(self):
        weights = golive.draw_weights(42)
        self.assertEqual(sorted(weights), sorted("w%d" % n for n in range(1, 11)))
        self.assertTrue(all(0.0 < value < 1.0 for value in weights.values()))
        self.assertEqual(weights, golive.draw_weights(42))


if __name__ == "__main__":
    unittest.main()
