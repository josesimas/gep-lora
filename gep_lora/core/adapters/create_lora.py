"""
create_lora.py - Train one LoRA adapter and leave it in a folder the pipeline
can point a slot at.

The five adapters under loras/Lora001/..loras/Lora005/ were each produced by their
own copy of a training script. This is the one script that makes another, so a
new behaviour to blend costs a dataset and a command rather than a sixth folder
of duplicated training code:

    python -m gep_lora.core.adapters.create_lora loras/Lora006/poem_adapter --dataset poem
    python -m gep_lora.core.adapters.create_lora loras/Lora007/shout_adapter --dataset uppercase --rank 8
    python -m gep_lora.core.adapters.create_lora loras/Lora001/QWen2.5-0.5b-lora_adapter --dataset medical_training --rank 16
    python -m gep_lora.core.adapters.create_lora loras/Lora004/QWen2.5-0.5b-lora_adapter --dataset medical_training --rank 4

The folder it writes is exactly the shape the generated scripts expect -- an
adapter_config.json naming the same base model, an adapter_model.safetensors,
and the tokenizer -- so making it usable is one line in settings.py's
LORA_SLOTS, which this script prints when it is done.

Two things are held fixed on purpose, because a blend is only meaningful when
its inputs agree on them: BASE_MODEL and TARGET_MODULES. add_weighted_adapter
folds tensors module by module against one base, so an adapter trained on a
different base, or over a different set of projections, does not fail cleanly so
much as produce nonsense. The rank is the opposite case -- it is *meant* to vary
(the existing five are 16, 16, 8, 4, 32) because that spread is what makes the
rank rule bite, so --rank is the knob to reach for first.

Every adapter it trains describes itself. Beside the weights go
`training.json` -- the recipe (every option below), the dataset's source,
fingerprint and size, the loss at every logged step, and how it ended -- and
`dataset.jsonl`, a copy of the data, so the folder is the whole record of
what it is even after datasets/ has been edited. And the adapter is entered in
the LoRA catalogue (gep_lora/core/adapters/catalog.py -- the `loras` table of the async
API's database, `api_jobs/api.sqlite`) as it goes:
`training`, then `ready` or `failed`. `--no-catalog` leaves the catalogue
alone; the folder is still written in full.

Progress goes to stdout as a marker line per logged step, the way the
generated scripts print TIMING lines:

    TRAIN: {"step": 12, "of": 160, "epoch": 1.5, "loss": 1.234, "lr": 0.0002}

`--mock` trains nothing: no GPU, no unsloth, a fake loss curve and a folder
holding an adapter_config.json of the asked rank and base model but no
weights. It is the plumbing check the mocked template is for a sweep -- a
mocked adapter can go into a mocked search (which reads only the config) and
into nothing else.

Interpreter: this trains, so it needs the venv one level up, the same one the
generated scripts run under --

    D:\\sage-is\\loras\\.venv\\Scripts\\python.exe create_lora.py ...
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
from gep_lora import paths

# Match the training/inference environment: disable Xet download acceleration.
os.environ["HF_HUB_DISABLE_XET"] = "1"

# The repo folder, one above this one. Every path a setting names is
# resolved against it, so nothing here depends on the cwd a driver was
# started from, or on which sub-folder this module ended up in.
_ROOT = paths.ROOT

# What the existing five were trained on. Their adapter_config.json records the
# 4-bit variant of this name, because load_in_4bit resolves it, and that
# resolved name is what the generated scripts load. Passing --base-model
# something else produces an adapter that cannot be blended with the others.
#BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct"
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# The projections every existing adapter targets. add_weighted_adapter combines
# adapters module by module, so this list has to match theirs exactly.
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

MAX_SEQ = 2048

# Where --dataset looks when it is given a bare name rather than a path.
DATASET_DIR = os.path.join(_ROOT, "datasets")

# The choices --scheduler and --optim offer: the ones transformers takes by
# name that are worth reaching for on one card. The first of each is what
# every existing adapter was trained with.
SCHEDULERS = ("linear", "cosine", "cosine_with_restarts", "polynomial",
              "constant", "constant_with_warmup")
OPTIMIZERS = ("adamw_8bit", "paged_adamw_8bit", "adamw_torch", "adafactor", "sgd")

# The marker a progress line starts with, like TIMING: in the generated scripts.
PROGRESS = "TRAIN:"

# The copy of the training data kept beside the adapter.
DATASET_COPY = "dataset.jsonl"


def check_interpreter():
    """Fail fast if this interpreter cannot train.

    The same trap process_run.check_interpreter() guards, for the same reason:
    Python 3.13 is first on PATH here and has none of this installed. The datasets
    entry earns its own case because the failure is disguised -- this folder holds
    a `datasets/` directory, so on an interpreter without the real package the
    import resolves to that as a namespace package and raises "cannot import name
    'load_dataset' ... (unknown location)" rather than saying it is missing. (With
    the library installed the real package wins, so the folder is harmless.)
    """
    missing = []
    for module in ("unsloth", "torch", "trl", "datasets"):
        spec = importlib.util.find_spec(module)
        # origin None means a namespace package -- a directory that happens to
        # carry the name, not the library.
        if spec is None or spec.origin is None:
            missing.append(module)
    if missing:
        raise SystemExit(
            "%s cannot import %s, and training needs all of it. Re-run with the "
            "project venv's python -- the same one the generated scripts run "
            "under (see the interpreter note in README.md)."
            % (sys.executable, ", ".join(missing))
        )


def resolve_dataset(name):
    """A dataset path, from a path or from a bare name under datasets/.

    `--dataset poem` should find datasets/poem_lora_dataset.json without the
    caller spelling the whole convention out, but an explicit path always wins
    and is never guessed at.
    """
    candidates = [name,
                  os.path.join(DATASET_DIR, name),
                  os.path.join(DATASET_DIR, name + ".json"),
                  os.path.join(DATASET_DIR, name + "_lora_dataset.json")]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    available = sorted(f for f in os.listdir(DATASET_DIR)
                       if f.endswith(".json")) if os.path.isdir(DATASET_DIR) else []
    raise SystemExit(
        "no dataset called %r: tried %s%s"
        % (name, ", ".join(candidates),
           ("\navailable in datasets/: " + ", ".join(available)) if available else "")
    )


def check_output_dir(path, force):
    """Refuse to train for an hour into a folder that already holds something."""
    if os.path.isdir(path) and os.listdir(path):
        existing = ", ".join(sorted(os.listdir(path))[:6])
        if not force:
            raise SystemExit(
                "%s already exists and is not empty (%s ...). Pass --force to "
                "overwrite it, or name a different folder." % (path, existing)
            )
        print("--force: writing over the contents of %s (%s ...)" % (path, existing))
    os.makedirs(path, exist_ok=True)


def load_examples(path):
    """The training set, as a Dataset of {'messages': [...]} rows.

    Both files under datasets/ are JSON Lines; load_dataset's json builder also
    reads a plain array, so either shape works and neither needs converting.
    """
    from datasets import load_dataset

    data = load_dataset("json", data_files=path, split="train")
    if "messages" not in data.column_names:
        raise SystemExit(
            "%s has columns %s; each row needs a 'messages' list of "
            "{'role', 'content'} turns, the shape datasets/*.json use."
            % (path, data.column_names)
        )
    if not len(data):
        raise SystemExit("%s is empty" % path)
    return data


def _catalog_module():
    """gep_lora.core.adapters.catalog, imported when first needed."""
    from gep_lora.core.adapters import catalog
    return catalog


def dataset_info(path, named=None):
    """Where the data came from, its fingerprint and how many records it holds.

    `named` says where it came from when the path would not -- the API trains
    from a copy in its own work folder, and what the record should say is
    which file or upload that copy was made from.
    """
    with open(path, "rb") as handle:
        raw = handle.read()
    text = raw.decode("utf-8-sig", errors="replace").strip()
    if text.startswith("["):
        try:
            records = len(json.loads(text))
        except ValueError:
            records = None
    else:
        records = sum(1 for line in text.splitlines() if line.strip())
    try:
        source = os.path.relpath(path, _ROOT)
        source = path if source.startswith("..") else source.replace(os.sep, "/")
    except ValueError:
        source = path
    return {"source": named or source, "sha256": hashlib.sha256(raw).hexdigest(),
            "records": records, "copy": DATASET_COPY}


def recipe(options):
    """Every option that decides what gets trained, as the record keeps it."""
    return {name: getattr(options, name) for name in (
        "base_model", "rank", "alpha", "dropout", "target_modules", "epochs",
        "max_steps", "learning_rate", "scheduler", "warmup_steps", "weight_decay",
        "optim", "batch_size", "grad_accum", "max_seq", "seed", "chat_template",
        "prompt", "no_sample")}


class Journal:
    """What this training has done so far, kept in two places.

    `training.json` in the adapter folder is the record that travels with the
    weights; the catalogue row is the index a list is drawn from. The file is
    rewritten at most every WRITE_EVERY seconds while training runs, and once
    more at the end, so a folder being trained can be read for progress. A
    catalogue that cannot be written (locked, say) is warned about and never
    fails a training -- the folder is the truth, and `catalog scan` re-reads it.
    """

    WRITE_EVERY = 2.0

    def __init__(self, options, dataset):
        self.options = options
        self.path = os.path.join(options.folder, _catalog_module().RECIPE)
        self.started = time.time()
        self.written = 0.0
        self.state = {"version": 1, "mock": bool(options.mock), "status": "training",
                      "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                      "options": recipe(options), "dataset": dataset,
                      "progress": {"step": 0, "of": None}, "history": [], "result": {}}
        self.catalog = None
        if not options.no_catalog:
            try:
                self.catalog = _catalog_module().Catalog(options.catalog)
            except Exception as error:              # noqa: BLE001 - never fail a training
                print("warning: the LoRA catalogue cannot be opened (%s); training "
                      "without it" % error)

    def _catalogue(self, status, **fields):
        if self.catalog is None:
            return
        try:
            self.catalog.record(self.options.folder, status, name=self.options.name,
                                owner=self.options.owner, **fields)
        except Exception as error:                  # noqa: BLE001 - never fail a training
            print("warning: could not update the LoRA catalogue: %s" % error)

    def write(self, force=False):
        if not force and time.time() - self.written < self.WRITE_EVERY:
            return
        self.written = time.time()
        temporary = self.path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(self.state, handle, indent=1)
        os.replace(temporary, self.path)

    def start(self):
        self.write(force=True)
        dataset = self.state["dataset"]
        options = self.options
        self._catalogue("training", base_model=options.base_model, rank=options.rank,
                        alpha=options.alpha, target_modules=options.target_modules,
                        chat_template=options.chat_template, mock=int(bool(options.mock)),
                        recipe=recipe(options), dataset=dataset["source"],
                        dataset_sha256=dataset["sha256"], records=dataset["records"],
                        steps=0, max_steps=None, final_loss=None, seconds=None,
                        sample=None, error=None, weights=0)

    def progress(self, step, of, epoch, loss, lr=None):
        """One logged step: a marker line on stdout, and the record kept up."""
        entry = {"step": step, "of": of, "epoch": None if epoch is None else round(epoch, 4),
                 "loss": None if loss is None else round(float(loss), 6), "lr": lr}
        print("%s %s" % (PROGRESS, json.dumps(entry)), flush=True)
        self.state["progress"] = dict(entry, seconds=round(time.time() - self.started, 1))
        if loss is not None:
            self.state["history"].append({key: entry[key] for key in ("step", "loss", "epoch", "lr")})
        self.write()

    def finish(self, sample=None):
        losses = [entry["loss"] for entry in self.state["history"] if entry["loss"] is not None]
        progress = self.state["progress"]
        self.state["status"] = "ready"
        self.state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.state["result"] = {"steps": progress.get("step"), "max_steps": progress.get("of"),
                                "epochs": progress.get("epoch"),
                                "final_loss": losses[-1] if losses else None,
                                "seconds": round(time.time() - self.started, 1),
                                "rank": rank_of(self.options.folder), "sample": sample}
        self.write(force=True)
        # Read back off the folder, so the row says what the files say.
        described = _catalog_module().describe(self.options.folder) or {}
        self._catalogue("ready", error=None, **described)

    def fail(self, error):
        self.state["status"] = "failed"
        self.state["error"] = error
        self.state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        try:
            self.write(force=True)
        except OSError:
            pass
        self._catalogue("failed", error=error[:2000])


def mock_train(options, journal):
    """A training that trains nothing: a loss curve, and a folder of the right
    shape with no weights in it. -> (None, None), where train() gives a model."""
    records = journal.state["dataset"]["records"] or 1
    per_epoch = max(1, math.ceil(records / float(options.batch_size * options.grad_accum)))
    steps = options.max_steps if options.max_steps and options.max_steps > 0 \
        else max(1, int(math.ceil(per_epoch * options.epochs)))
    rng = random.Random(options.seed)
    print("Mock training: %d step(s), %d a epoch, nothing loaded." % (steps, per_epoch))
    for step in range(1, steps + 1):
        loss = 3.5 * math.exp(-4.0 * step / steps) + 0.05 + rng.uniform(-0.04, 0.04)
        journal.progress(step, steps, step / float(per_epoch), max(0.0, loss),
                         options.learning_rate * (1 - step / float(steps + 1)))
        if options.mock_delay:
            time.sleep(options.mock_delay)
    with open(os.path.join(options.folder, "adapter_config.json"), "w", encoding="utf-8") as handle:
        json.dump({"peft_type": "LORA", "mock": True, "r": options.rank,
                   "lora_alpha": options.alpha, "lora_dropout": options.dropout,
                   "target_modules": options.target_modules,
                   "base_model_name_or_path": options.base_model}, handle, indent=2)
    return None, None


def rank_of(adapter_dir):
    """The rank PEFT will allocate for the saved adapter.

    The same reading the pipeline does -- `_rank()` in template_code.py, and
    `slot_ranks()` at generation time -- so the number printed here is the one
    the rank rule will apply to this adapter.
    """
    with open(os.path.join(adapter_dir, "adapter_config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    return max([config["r"]] + list((config.get("rank_pattern") or {}).values()))


def slot_line(folder, slot="L?"):
    """The LORA_SLOTS entry that points at `folder`, ready to paste.

    LORA_SLOTS lives in settings.py, whose relative entries are taken from the
    repo folder, so a slot inside the repo is written relative and with forward
    slashes -- the way the five existing entries are, and readable on either
    platform. Anything outside the repo goes in absolute, which LORA_SLOTS
    accepts and which is the only honest way to write a path the repo folder
    cannot reach. `slot` is left as a placeholder for one adapter, since only
    the caller knows which of L1..L10 it is filling; create_all_loras fills it
    in because it writes the whole set at once.
    """
    try:
        relative = os.path.relpath(folder, _ROOT)
    except ValueError:
        # Windows: relpath refuses to relate paths on two different drives.
        relative = ".."
    if relative.startswith(".."):
        return '"%s": r"%s",' % (slot, folder)
    return '"%s": "%s",' % (slot, relative.replace(os.sep, "/"))


def train(options, journal):
    """Fine-tune the base model on one dataset and save the adapter."""
    # Unsloth patches transformers, peft and trl as it loads, so it is imported
    # before any of them -- the same ordering the generated scripts keep.
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    import torch
    from transformers import TrainerCallback
    from trl import SFTTrainer, SFTConfig

    class Report(TrainerCallback):
        """Every logged step into the journal: a TRAIN: line and training.json."""

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                journal.progress(state.global_step, state.max_steps,
                                 logs.get("epoch", state.epoch), logs["loss"],
                                 logs.get("learning_rate"))

    print("GPU available: %s" % torch.cuda.is_available())

    dataset = load_examples(options.dataset)
    print("Training set: %d examples from %s" % (len(dataset), options.dataset))

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=options.base_model,
        max_seq_length=options.max_seq,
        dtype=None,
        load_in_4bit=True,
    )

    # The chat template the adapter is trained under -- and so the one the
    # pipeline has to prompt it in: CHAT_TEMPLATE in settings.py must match.
    # None is the base model's own template; a name is unsloth's of that name.
    # Either way it is saved with the tokenizer below, so the adapter folder's
    # chat_template.jinja records what it learned.
    if options.chat_template:
        tokenizer = get_chat_template(tokenizer, chat_template=options.chat_template)

    def to_text(batch):
        return {"text": [tokenizer.apply_chat_template(m, tokenize=False,
                                                       add_generation_prompt=False)
                         for m in batch["messages"]]}

    dataset = dataset.map(to_text, batched=True)

    model = FastLanguageModel.get_peft_model(
        model,
        r=options.rank,
        lora_alpha=options.alpha,
        lora_dropout=options.dropout,
        bias="none",
        target_modules=options.target_modules,
        use_gradient_checkpointing="unsloth",
        random_state=options.seed,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=options.max_seq,
            per_device_train_batch_size=options.batch_size,
            gradient_accumulation_steps=options.grad_accum,
            warmup_steps=options.warmup_steps,
            num_train_epochs=options.epochs,
            # -1 is the Trainer's own "no cap": the epochs decide.
            max_steps=options.max_steps if options.max_steps and options.max_steps > 0 else -1,
            learning_rate=options.learning_rate,
            logging_steps=1,
            optim=options.optim,
            weight_decay=options.weight_decay,
            lr_scheduler_type=options.scheduler,
            seed=options.seed,
            # Checkpoints and logs are scratch: they go *beside* the adapter,
            # never into it, so the folder stays a clean slot target.
            output_dir=options.folder + "_outputs",
        ),
        callbacks=[Report()],
    )
    trainer.train()

    # The tokenizer goes in too. The existing adapter folders carry one, which
    # is what lets a folder be loaded on its own (loras/Lora00*/inference.py) as well
    # as attached to an already-loaded base.
    model.save_pretrained(options.folder)
    tokenizer.save_pretrained(options.folder)
    return model, tokenizer


def sample(model, tokenizer, prompt, max_new_tokens=120):
    """One answer from the freshly trained adapter, as a smoke test."""
    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)
    # Qwen ships max_length in generation_config.json and transformers warns
    # when both caps are set; clearing it leaves max_new_tokens in charge, the
    # same fix the generated scripts carry.
    model.generation_config.max_length = None
    # And stop at the end of the turn, also as they do: a base with no
    # generation_config.json stops only on <|endoftext|>, and the smoke test
    # would show a trained adapter running on into invented turns.
    stops = model.generation_config.eos_token_id
    stops = stops if isinstance(stops, list) else [] if stops is None else [stops]
    end_of_turn = getattr(tokenizer, "tokenizer", tokenizer).eos_token_id
    if end_of_turn is not None and end_of_turn not in stops:
        model.generation_config.eos_token_id = stops + [end_of_turn]
    # Rendered to text, then tokenised by the tokenizer inside a processor, as
    # the generated scripts do: a vision-language base (Qwen3.5) loads as a
    # processor, whose apply_chat_template hands back text whatever return_dict
    # says.
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=False)
    inputs = text_tokenizer([text], return_tensors="pt",
                            add_special_tokens=False).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return text_tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:],
                                 skip_special_tokens=True).strip()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train one LoRA adapter into a folder the pipeline can use.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="example:\n"
               "  python -m gep_lora.core.adapters.create_lora loras/Lora006/poem_adapter --dataset poem --rank 8",
    )
    parser.add_argument(
        "folder",
        help="where to write the adapter -- the path a LORA_SLOTS entry points "
             "at. Created if missing.")
    parser.add_argument(
        "--dataset", "-d", required=True,
        help="training data: a path, or a bare name looked up under datasets/ "
             "(so 'poem' finds datasets/poem_lora_dataset.json). One JSON object "
             "per line, each with a 'messages' list.")
    parser.add_argument(
        "--rank", "-r", type=int, default=16,
        help="LoRA rank (default 16). The one parameter meant to differ between "
             "slots -- the existing five are 16, 16, 8, 4, 32, and that spread is "
             "what CAT/SVD/LIN have to work around.")
    parser.add_argument(
        "--alpha", type=int, default=16,
        help="lora_alpha (default 16, as in all five existing adapters).")
    parser.add_argument(
        "--dropout", type=float, default=0.0,
        help="lora_dropout (default 0, which is also the only value unsloth's "
             "fast path takes without falling back to slower kernels).")
    parser.add_argument(
        "--target-modules", type=_modules, default=list(TARGET_MODULES),
        help="comma-separated projections to adapt (default %s). Every adapter "
             "in one blend has to cover the same set, so leave it unless a whole "
             "new set of slots is being trained." % ",".join(TARGET_MODULES))
    parser.add_argument(
        "--epochs", type=float, default=20, help="training epochs (default 20).")
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="stop after this many optimiser steps, whatever --epochs says "
             "(default: no cap). Handy for a quick look at a new recipe.")
    parser.add_argument(
        "--learning-rate", type=float, default=2e-4, help="default 2e-4.")
    parser.add_argument(
        "--scheduler", choices=SCHEDULERS, default=SCHEDULERS[0],
        help="learning-rate schedule (default %s)." % SCHEDULERS[0])
    parser.add_argument(
        "--warmup-steps", type=int, default=5, help="warmup steps (default 5).")
    parser.add_argument(
        "--weight-decay", type=float, default=0.01, help="default 0.01.")
    parser.add_argument(
        "--optim", choices=OPTIMIZERS, default=OPTIMIZERS[0],
        help="optimiser (default %s)." % OPTIMIZERS[0])
    parser.add_argument(
        "--batch-size", type=int, default=2,
        help="per-device training batch size (default 2).")
    parser.add_argument(
        "--grad-accum", type=int, default=4,
        help="gradient accumulation steps (default 4).")
    parser.add_argument(
        "--max-seq", type=int, default=MAX_SEQ,
        help="max sequence length (default %d)." % MAX_SEQ)
    parser.add_argument(
        "--seed", type=int, default=3407,
        help="training seed (default 3407). Unrelated to settings.py's seeds, "
             "which govern the search rather than any one adapter.")
    parser.add_argument(
        "--base-model", default=BASE_MODEL,
        help="base model (default %s). Change it and the adapter can no longer "
             "be blended with the existing slots." % BASE_MODEL)
    parser.add_argument(
        "--chat-template", default=None,
        help="an unsloth chat template name to train under, such as qwen-2.5 "
             "(default: the base model's own). Whatever it is, CHAT_TEMPLATE in "
             "settings.py has to say the same, or the pipeline prompts the "
             "adapter in a format it never learned.")
    parser.add_argument(
        "--prompt", default="Tell me about the ocean.",
        help="prompt to answer once after training, as a smoke test.")
    parser.add_argument(
        "--no-sample", action="store_true",
        help="skip that smoke test and stop as soon as the adapter is saved.")
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite the output folder if it already holds something.")
    parser.add_argument(
        "--name", default=None,
        help="the adapter's name in the LoRA catalogue (default: its folder "
             "under loras/, such as Lora006/poem_adapter).")
    parser.add_argument(
        "--dataset-source", default=None,
        help="what the record says the data was (default: the --dataset path). "
             "For a copy, the name of what it is a copy of.")
    parser.add_argument(
        "--catalog", default=None,
        help="the database the catalogue is in (default the async API's, "
             "api_jobs/api.sqlite, or $GEP_LORA_CATALOG).")
    parser.add_argument(
        "--owner", default=None,
        help="the async API user the adapter belongs to in the catalogue -- the "
             "only one the API shows it to (default: nobody, until "
             "`python -m gep_lora.core.adapters.catalog own` says whose it is).")
    parser.add_argument(
        "--no-catalog", action="store_true",
        help="leave the catalogue alone; the folder is still written in full.")
    parser.add_argument(
        "--mock", action="store_true",
        help="train nothing: a fake loss curve and an adapter_config.json with "
             "no weights, for checking the plumbing. Needs no GPU.")
    parser.add_argument(
        "--mock-delay", type=float, default=0.05,
        help="seconds each mocked step takes (default 0.05), so a mocked "
             "training can be watched.")
    return parser.parse_args(argv)


def _modules(text):
    modules = [part.strip() for part in text.split(",") if part.strip()]
    if not modules:
        raise argparse.ArgumentTypeError("name at least one module")
    return modules


def defaults():
    """Every option's default, as parse_args() would give it. -> dict.

    The one statement of them: the API's form starts from these, so a LoRA
    trained through it with nothing changed is the one this command trains.
    """
    options = vars(parse_args(["FOLDER", "--dataset", "DATASET"]))
    for name in ("folder", "dataset", "dataset_source", "force", "name", "owner",
                 "catalog", "no_catalog", "mock", "mock_delay"):
        options.pop(name)
    return options


def main(argv=None):
    # After parsing, so --help still works on any interpreter, but before the
    # folder is touched or a model is fetched.
    options = parse_args(argv)
    if not options.mock:
        check_interpreter()
    options.dataset = resolve_dataset(options.dataset)
    options.folder = os.path.abspath(options.folder)
    check_output_dir(options.folder, options.force)

    if options.base_model != BASE_MODEL:
        print("warning: --base-model %s is not what the existing slots were "
              "trained on (%s); the result cannot be blended with them."
              % (options.base_model, BASE_MODEL))
    if sorted(options.target_modules) != sorted(TARGET_MODULES):
        print("warning: --target-modules %s differs from every existing adapter's "
              "(%s); add_weighted_adapter folds module by module, so this one "
              "can only be blended with others trained over the same set."
              % (",".join(options.target_modules), ",".join(TARGET_MODULES)))

    # The data goes into the folder with the weights, so the adapter is
    # described by what it was actually trained on, whatever datasets/ says later.
    dataset = dataset_info(options.dataset, options.dataset_source)
    shutil.copyfile(options.dataset, os.path.join(options.folder, DATASET_COPY))
    journal = Journal(options, dataset)
    journal.start()
    try:
        if options.mock:
            model, tokenizer = mock_train(options, journal)
        else:
            model, tokenizer = train(options, journal)

        print("\nSaved a rank-%d adapter to %s" % (rank_of(options.folder), options.folder))

        answer = None
        if not options.no_sample:
            print("\nYOU: %s" % options.prompt)
            answer = ("(mocked: nothing was trained, so nothing answers)" if options.mock
                      else sample(model, tokenizer, options.prompt))
            print("LORA: %s" % answer)
    except BaseException as error:
        journal.fail("%s: %s" % (type(error).__name__, error))
        raise
    journal.finish(answer)

    # The last mile: what to paste into settings.py so a tree can reach it.
    # A slot is repointed, or added as the next of L1..L10 -- up to ten is a
    # configuration change; an eleventh would widen the grammar's own alphabet
    # (MAX_SLOTS in generate_population.py).
    print("\nTo put it in the search, point a slot at it in settings.py's "
          "LORA_SLOTS, or add it as the next slot up to L10 (mind the ranks -- "
          "they decide which LIN combinations are legal):")
    print("    " + slot_line(options.folder))
    print("then re-run `python start_run.py runs` so the generated scripts pick it up.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
