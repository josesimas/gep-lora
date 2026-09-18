"""
test_catalog.py - The LoRA catalogue, and create_lora.py keeping it up.

What is tested here:

  * a scan finds the adapter folders under a root and nothing else (not the
    Trainer's `_outputs/` beside each, nor its checkpoints), reads what each
    says about itself, and marks a row whose folder went away as missing;
  * a row in flight is never touched by a scan -- its row is the only record
    of the training;
  * an adapter trained before training.json existed is described from its
    newest checkpoint's trainer_state.json;
  * the base models a LoRA can be trained on are read off the hub cache's own
    layout, generative heads only;
  * create_lora.py --mock end to end: the folder it writes, the record inside
    it, the progress lines and the catalogue row it leaves `ready`.

All of it on a machine with no GPU and nothing installed.
"""

import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest

from adapters import catalog
from adapters import create_lora


def adapter(folder, rank=16, base="unsloth/base", weights=True, **extra):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "adapter_config.json"), "w", encoding="utf-8") as handle:
        json.dump(dict({"r": rank, "lora_alpha": 16, "base_model_name_or_path": base,
                        "target_modules": ["q_proj", "v_proj"]}, **extra), handle)
    if weights:
        with open(os.path.join(folder, "adapter_model.safetensors"), "wb") as handle:
            handle.write(b"weights")
    return folder


class CatalogTestCase(unittest.TestCase):

    def setUp(self):
        self.folder = tempfile.mkdtemp(prefix="gep-catalog-tests-")
        self.root = os.path.join(self.folder, "loras")
        self.catalog = catalog.Catalog(os.path.join(self.folder, "catalog.sqlite3"))

    def tearDown(self):
        shutil.rmtree(self.folder, ignore_errors=True)


class ScanTests(CatalogTestCase):

    def test_a_scan_finds_adapters_and_not_their_scratch(self):
        one = adapter(os.path.join(self.root, "Lora001", "set_a"), rank=16)
        adapter(os.path.join(self.root, "Lora002", "set_a"), rank=4)
        # The Trainer's outputs beside an adapter hold adapter configs too.
        adapter(os.path.join(one + "_outputs", "checkpoint-10"))
        report = self.catalog.scan(self.root)
        self.assertEqual(len(report["added"]), 2)
        rows = {row["rank"]: row for row in self.catalog.all()}
        self.assertEqual(sorted(rows), [4, 16])
        self.assertEqual(rows[16]["status"], catalog.READY)
        self.assertEqual(rows[16]["origin"], "scanned")
        self.assertEqual(rows[16]["base_model"], "unsloth/base")
        self.assertEqual(catalog.as_dict(rows[16])["family"], "set_a")
        # A second scan adds nothing and keeps the ids.
        again = self.catalog.scan(self.root)
        self.assertEqual(again["added"], [])
        self.assertEqual({row["id"] for row in self.catalog.all()},
                         {row["id"] for row in rows.values()})

    def test_a_rank_pattern_raises_the_rank_as_peft_does(self):
        adapter(os.path.join(self.root, "a"), rank=8, rank_pattern={"q_proj": 32})
        self.catalog.scan(self.root)
        self.assertEqual(self.catalog.all()[0]["rank"], 32)

    def test_a_folder_that_went_away_is_missing_and_comes_back(self):
        where = adapter(os.path.join(self.root, "gone"))
        self.catalog.scan(self.root)
        shutil.rmtree(where)
        self.assertEqual(self.catalog.scan(self.root)["missing"], ["gone"])
        self.assertEqual(self.catalog.all()[0]["status"], catalog.MISSING)
        adapter(where)
        self.catalog.scan(self.root)
        self.assertEqual(self.catalog.all()[0]["status"], catalog.READY)

    def test_an_adapter_without_weights_is_missing_unless_mocked(self):
        adapter(os.path.join(self.root, "empty"), weights=False)
        adapter(os.path.join(self.root, "mocked"), weights=False, mock=True)
        self.catalog.scan(self.root)
        status = {row["name"]: row["status"] for row in self.catalog.all()}
        self.assertEqual(status, {"empty": catalog.MISSING, "mocked": catalog.READY})

    def test_a_training_in_flight_is_left_alone(self):
        where = os.path.join(self.root, "busy")
        row = self.catalog.add("busy", where, catalog.TRAINING, "trained", rank=8)
        adapter(where, rank=16)                      # half written, as it were
        self.catalog.scan(self.root)
        self.assertEqual(self.catalog.get(row["id"])["status"], catalog.TRAINING)
        self.assertEqual(self.catalog.get(row["id"])["rank"], 8)
        shutil.rmtree(where)
        self.catalog.scan(self.root)
        self.assertEqual(self.catalog.get(row["id"])["status"], catalog.TRAINING)

    def test_names_and_folders_are_unique(self):
        where = adapter(os.path.join(self.root, "x"))
        self.catalog.add("x", where, catalog.READY, "scanned")
        with self.assertRaises(ValueError):
            self.catalog.add("x", os.path.join(self.root, "y"), catalog.READY, "scanned")
        with self.assertRaises(ValueError):
            self.catalog.add("z", where, catalog.READY, "scanned")
        self.assertEqual(self.catalog.free_name("x"), "x-2")

    def test_names_are_unique_per_owner(self):
        self.catalog.add("x", os.path.join(self.root, "a"), catalog.READY, "trained", owner="ann")
        self.catalog.add("x", os.path.join(self.root, "b"), catalog.READY, "trained", owner="bob")
        with self.assertRaises(ValueError):
            self.catalog.add("x", os.path.join(self.root, "c"), catalog.READY, "trained",
                             owner="bob")
        self.assertEqual(self.catalog.by_name("x", "bob")["folder"], catalog.stored_folder(
            os.path.join(self.root, "b")))
        self.assertIsNone(self.catalog.by_name("x"))
        self.assertEqual(self.catalog.free_name("x", "ann"), "x-2")
        self.assertEqual(self.catalog.free_name("x", "cy"), "x")

    def test_rows_are_given_to_a_user_and_listed_as_theirs(self):
        adapter(os.path.join(self.root, "a"))
        adapter(os.path.join(self.root, "b"))
        self.catalog.scan(self.root, owner="ann")
        adapter(os.path.join(self.root, "c"))
        self.catalog.scan(self.root)
        self.assertEqual([row["name"] for row in self.catalog.all(owner="ann")], ["a", "b"])
        self.assertEqual([row["name"] for row in self.catalog.all(owner=None)], ["c"])
        self.assertEqual(self.catalog.own([self.catalog.find("c")["id"]], "ann"), 1)
        self.assertEqual(len(self.catalog.all(owner="ann")), 3)
        # A name the new owner already holds is a clash, and nothing moves.
        clash = self.catalog.add("a", os.path.join(self.root, "d"), catalog.READY, "scanned")
        with self.assertRaises(ValueError):
            self.catalog.own([clash["id"]], "ann")
        self.assertIsNone(self.catalog.get(clash["id"])["owner"])

    def test_an_owner_must_be_a_user_when_the_file_has_users(self):
        row = self.catalog.add("x", os.path.join(self.root, "x"), catalog.READY, "scanned")
        with self.catalog._connect() as conn:
            conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
            conn.execute("INSERT INTO users (name) VALUES ('ann')")
        with self.assertRaises(ValueError):
            self.catalog.own([row["id"]], "anne")
        self.assertEqual(self.catalog.own([row["id"]], "ann"), 1)

    def test_an_id_is_never_given_out_twice(self):
        # The API's trainings point at these ids.
        first = self.catalog.add("a", os.path.join(self.root, "a"), catalog.READY, "scanned")
        self.catalog.remove(first["id"])
        second = self.catalog.add("b", os.path.join(self.root, "b"), catalog.READY, "scanned")
        self.assertGreater(second["id"], first["id"])

    def test_an_old_adapter_is_described_from_its_newest_checkpoint(self):
        where = adapter(os.path.join(self.root, "old"))
        for step, loss in ((10, 2.0), (20, 0.5)):
            checkpoint = os.path.join(where + "_outputs", "checkpoint-%d" % step)
            os.makedirs(checkpoint)
            with open(os.path.join(checkpoint, "trainer_state.json"), "w") as handle:
                json.dump({"global_step": step, "max_steps": 20, "epoch": step / 10.0,
                           "num_train_epochs": 2, "train_batch_size": 2,
                           "log_history": [{"step": s, "loss": loss, "epoch": s / 10.0}
                                           for s in range(1, step + 1)]}, handle)
        described = catalog.describe(where)
        self.assertEqual((described["steps"], described["final_loss"]), (20, 0.5))
        self.assertEqual(len(catalog.history(where)), 20)


class LocalModelTests(CatalogTestCase):

    def test_only_generative_models_in_the_hub_cache_are_offered(self):
        cache = os.path.join(self.folder, "hub")
        for repo, architectures in (("models--org--chat", ["QwenForCausalLM"]),
                                    ("models--org--vision", ["X3ForConditionalGeneration"]),
                                    ("models--org--embed", ["BertModel"]),
                                    ("models--org--gguf", None)):
            snapshot = os.path.join(cache, repo, "snapshots", "abc")
            os.makedirs(snapshot)
            if architectures:
                with open(os.path.join(snapshot, "config.json"), "w") as handle:
                    json.dump({"architectures": architectures}, handle)
        found = [model["id"] for model in catalog.local_models(cache)]
        self.assertEqual(found, ["org/chat", "org/vision"])
        self.assertEqual(catalog.local_models(os.path.join(self.folder, "nowhere")), [])


class MockTrainingTests(CatalogTestCase):

    def test_a_mocked_training_writes_a_whole_folder_and_a_ready_row(self):
        data = os.path.join(self.folder, "data.jsonl")
        with open(data, "w", encoding="utf-8") as handle:
            for number in range(6):
                handle.write(json.dumps({"messages": [
                    {"role": "user", "content": "q%d" % number},
                    {"role": "assistant", "content": "a%d" % number}]}) + "\n")
        where = os.path.join(self.root, "trained", "demo")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            create_lora.main([where, "--dataset", data, "--mock", "--mock-delay", "0",
                              "--rank", "8", "--max-steps", "5", "--name", "demo",
                              "--base-model", "unsloth/base", "--owner", "ann",
                              "--catalog", self.catalog.path])
        lines = [line for line in out.getvalue().splitlines()
                 if line.startswith(create_lora.PROGRESS)]
        self.assertEqual(len(lines), 5)
        self.assertEqual(json.loads(lines[-1][len(create_lora.PROGRESS):])["step"], 5)

        self.assertEqual(sorted(os.listdir(where)),
                         ["adapter_config.json", "dataset.jsonl", "training.json"])
        with open(os.path.join(where, "training.json"), encoding="utf-8") as handle:
            record = json.load(handle)
        self.assertEqual(record["status"], "ready")
        self.assertEqual(record["dataset"]["records"], 6)
        self.assertEqual(record["options"]["rank"], 8)
        self.assertEqual(len(record["history"]), 5)

        row = self.catalog.by_name("demo", "ann")
        self.assertEqual((row["status"], row["rank"], row["mock"], row["steps"]),
                         (catalog.READY, 8, 1, 5))
        self.assertEqual(json.loads(row["recipe"])["max_steps"], 5)

    def test_the_defaults_are_the_command_lines_own(self):
        defaults = create_lora.defaults()
        self.assertEqual(defaults["rank"], 16)
        self.assertEqual(defaults["target_modules"], create_lora.TARGET_MODULES)
        self.assertNotIn("folder", defaults)
        self.assertNotIn("mock", defaults)


if __name__ == "__main__":
    unittest.main()
