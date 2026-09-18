"""
test_evaluate_chromosome_against_loras.py - A blend against each LoRA alone.

What is the script's own, and so what is tested here:

  * a lone adapter renders through the same render() as an individual, with no
    combine() and the leaf itself as the final adapter -- so it runs at full
    strength -- and the mocked script it makes actually runs and answers;
  * the blend is rendered from the chromosome and its own weight seed, and
    --stored-script re-points the stored script instead;
  * the pairwise counts line up by question, skip a failed grade on either
    side, and the sign test is the exact two-sided one;
  * the best individual is the elite before the fittest, and a BAD one is
    refused.

Nothing here loads a model or asks a judge.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from blends import process_run
from storage import store
from testing import evaluate_chromosome_against_loras as compare
from unittests.async_api.support import RECORDS, make_slots


class Fixture(unittest.TestCase):

    def setUp(self):
        self.folder = tempfile.mkdtemp(prefix="gep-vs-loras-")
        self.addCleanup(shutil.rmtree, self.folder, ignore_errors=True)
        self.slots = make_slots(self.folder)
        self.dataset = os.path.join(self.folder, "questions.jsonl")
        with open(self.dataset, "w", encoding="utf-8") as handle:
            handle.writelines(json.dumps(record) + "\n" for record in RECORDS)
        self.conf = {"TEMPLATE": "template_code_mocked.py", "LORA_SLOTS": self.slots,
                     "BASE_MODEL": "base", "CHAT_TEMPLATE": None}


class ContestantTests(Fixture):

    def row(self, **fields):
        row = {"number": 7, "chromosome": "CAT.L5.L2.w5.w4", "weight_seed": 1234,
               "script_source": None}
        row.update(fields)
        return row

    def test_every_slot_gets_a_lone_adapter_after_the_blend(self):
        field = compare.contestants(self.row(), self.conf, ["L1", "L3"],
                                    self.dataset, 2)
        self.assertEqual([c.key for c in field], ["blend", "L1", "L3"])
        lone = field[2].source
        self.assertIn('attach("n1_L3", "L3")', lone)
        self.assertNotIn("combine(\"", lone)
        self.assertIn('FINAL_ADAPTER = "n1_L3"', lone)
        self.assertIn("TRAINING_COUNT = 2", lone)

    def test_the_blend_keeps_its_weight_seed(self):
        blend = compare.contestants(self.row(), self.conf, [], self.dataset, None)[0]
        self.assertIn("WEIGHT_SEED = 1234", blend.source)
        self.assertIn('EXPRESSION = "CAT.L5.L2.w5.w4"', blend.source)

    def test_stored_script_is_re_pointed_not_rendered(self):
        stored = "TRAINING_SET = 'old'\nTRAINING_COUNT = 9\nEXPRESSION = \"CAT.L1.L2.w1.w2\"\n"
        blend = compare.contestants(self.row(script_source=stored), self.conf, [],
                                    self.dataset, 3, stored_script=True)[0]
        self.assertIn("TRAINING_COUNT = 3", blend.source)
        self.assertIn(repr(self.dataset), blend.source)
        self.assertEqual(blend.expression, "CAT.L1.L2.w1.w2")

    def test_a_lone_adapter_script_runs_and_answers(self):
        lone = compare.contestants(self.row(), self.conf, ["L4"], self.dataset, 2)[1]
        path = os.path.join(self.folder, lone.script_name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(lone.source)
        done = subprocess.run([sys.executable, path], capture_output=True,
                              text=True, encoding="utf-8", timeout=120)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("rank 4", done.stdout)
        self.assertEqual(len(process_run.exchanges(done.stdout)), 2)

    def test_the_digest_follows_the_source(self):
        one, two = compare.contestants(self.row(), self.conf, ["L1"], self.dataset, 2)
        again = compare.contestants(self.row(), self.conf, ["L1"], self.dataset, 3)[1]
        self.assertNotEqual(two.digest, again.digest)
        self.assertEqual(two.digest,
                         compare.contestants(self.row(), self.conf, ["L1"],
                                             self.dataset, 2)[1].digest)


class ComparisonTests(unittest.TestCase):

    def test_sign_test_is_exact_and_two_sided(self):
        self.assertAlmostEqual(compare.sign_test(5, 0), 2 / 32)
        self.assertAlmostEqual(compare.sign_test(0, 5), 2 / 32)
        self.assertEqual(compare.sign_test(3, 3), 1.0)
        self.assertEqual(compare.sign_test(0, 0), 1.0)

    def test_questions_pair_by_position_and_failures_drop_out(self):
        rows = compare.compare({"blend": [0.9, 0.5, None, 0.3],
                                "L1": [0.5, 0.5, 0.9, 0.6]}, 4)
        blend, lone = rows
        self.assertEqual((lone["wins"], lone["ties"], lone["losses"]), (1, 1, 1))
        self.assertAlmostEqual(lone["delta"], (0.4 + 0.0 - 0.3) / 3)
        self.assertEqual(blend["failed"], 1)
        self.assertAlmostEqual(blend["mean"], (0.9 + 0.5 + 0.3) / 3)

    def test_a_crashed_contestant_is_missing_its_later_questions(self):
        rows = compare.compare({"blend": [0.9, 0.9, 0.9], "L2": [0.1]}, 3)
        self.assertEqual(rows[1]["missing"], 2)
        self.assertEqual(rows[1]["wins"], 1)


class BestIndividualTests(unittest.TestCase):

    def setUp(self):
        folder = tempfile.mkdtemp(prefix="gep-vs-loras-db-")
        self.addCleanup(shutil.rmtree, folder, ignore_errors=True)
        self.conn = store.connect(os.path.join(folder, "gep.sqlite3"))
        self.addCleanup(self.conn.close)
        self.run_id = store.create_run(self.conn, "template_code_mocked.py")
        store.add_individuals(self.conn, self.run_id,
                              ["CAT.L1.L2.w1.w2", "CAT.L3.L4.w1.w2", "CAT.L5.L1.w1.w2"])
        for number, fitness in ((1, 0.4), (2, 0.8), (3, 0.6)):
            store.set_fitness(self.conn, self.run_id, number, fitness)
        self.conn.commit()

    def test_the_fittest_without_an_elite(self):
        self.assertEqual(compare.best_individual(self.conn, self.run_id)["number"], 2)

    def test_the_elite_first(self):
        store.mark_best(self.conn, self.run_id, 3)
        self.assertEqual(compare.best_individual(self.conn, self.run_id)["number"], 3)

    def test_a_named_individual_and_a_bad_one(self):
        self.assertEqual(compare.best_individual(self.conn, self.run_id, 1)["number"], 1)
        self.conn.execute("UPDATE individuals SET state = 'BAD' WHERE number = 1")
        with self.assertRaises(SystemExit):
            compare.best_individual(self.conn, self.run_id, 1)


if __name__ == "__main__":
    unittest.main()
