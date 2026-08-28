"""
create_lora.py - Train one LoRA adapter and leave it in a folder the pipeline
can point a slot at.

The five adapters under Lora001/..Lora005/ were each produced by their own copy
of a training script. This is the one script that makes another, so a new
behaviour to blend costs a dataset and a command rather than a sixth folder of
duplicated training code:

    python create_lora.py Lora006/poem_adapter --dataset poem
    python create_lora.py Lora007/shout_adapter --dataset uppercase --rank 8

The folder it writes is exactly the shape the generated scripts expect -- an
adapter_config.json naming the same base model, an adapter_model.safetensors,
and the tokenizer -- so making it usable is one line in template_code.py's
LORA_SLOTS, which this script prints when it is done.

Two things are held fixed on purpose, because a blend is only meaningful when
its inputs agree on them: BASE_MODEL and TARGET_MODULES. add_weighted_adapter
folds tensors module by module against one base, so an adapter trained on a
different base, or over a different set of projections, does not fail cleanly so
much as produce nonsense. The rank is the opposite case -- it is *meant* to vary
(the existing five are 16, 16, 8, 4, 32) because that spread is what makes the
rank rule bite, so --rank is the knob to reach for first.

Interpreter: this trains, so it needs the venv one level up, the same one the
generated scripts run under --

    D:\\sage-is\\loras\\.venv\\Scripts\\python.exe create_lora.py ...
"""

import argparse
import importlib.util
import json
import os
import sys

# Match the training/inference environment: disable Xet download acceleration.
os.environ["HF_HUB_DISABLE_XET"] = "1"

_HERE = os.path.dirname(os.path.abspath(__file__))

# What the existing five were trained on. Their adapter_config.json records the
# 4-bit variant of this name, because load_in_4bit resolves it, and that
# resolved name is what the generated scripts load. Passing --base-model
# something else produces an adapter that cannot be blended with the others.
BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct"

# The projections every existing adapter targets. add_weighted_adapter combines
# adapters module by module, so this list has to match theirs exactly.
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

MAX_SEQ = 2048

# Where --dataset looks when it is given a bare name rather than a path.
DATASET_DIR = os.path.join(_HERE, "datasets")


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


def rank_of(adapter_dir):
    """The rank PEFT will allocate for the saved adapter.

    The same reading the pipeline does -- `_rank()` in template_code.py, and
    `slot_ranks()` at generation time -- so the number printed here is the one
    the rank rule will apply to this adapter.
    """
    with open(os.path.join(adapter_dir, "adapter_config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    return max([config["r"]] + list((config.get("rank_pattern") or {}).values()))


def slot_line(folder):
    """The LORA_SLOTS entry that points at `folder`, ready to paste.

    A generated script sits in the run folder and sets `_PROJECT` to its parent
    -- this repo -- so a slot inside the repo is written relative to that, the
    way the five existing entries are. Anything outside it goes in absolute,
    which LORA_SLOTS accepts and which is the only honest way to write a path
    `_PROJECT` cannot reach.
    """
    try:
        relative = os.path.relpath(folder, _HERE)
    except ValueError:
        # Windows: relpath refuses to relate paths on two different drives.
        relative = ".."
    if relative.startswith(".."):
        return '"L?": r"%s",' % folder
    parts = ", ".join('"%s"' % part for part in relative.replace("\\", "/").split("/"))
    return '"L?": os.path.join(_PROJECT, %s),' % parts


def train(options):
    """Fine-tune the base model on one dataset and save the adapter."""
    # Unsloth patches transformers, peft and trl as it loads, so it is imported
    # before any of them -- the same ordering the generated scripts keep.
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    import torch
    from trl import SFTTrainer, SFTConfig

    print("GPU available: %s" % torch.cuda.is_available())

    dataset = load_examples(options.dataset)
    print("Training set: %d examples from %s" % (len(dataset), options.dataset))

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=options.base_model,
        max_seq_length=options.max_seq,
        dtype=None,
        load_in_4bit=True,
    )

    # The chat template the adapters are trained under and evaluated under.
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    def to_text(batch):
        return {"text": [tokenizer.apply_chat_template(m, tokenize=False,
                                                       add_generation_prompt=False)
                         for m in batch["messages"]]}

    dataset = dataset.map(to_text, batched=True)

    model = FastLanguageModel.get_peft_model(
        model,
        r=options.rank,
        lora_alpha=options.alpha,
        lora_dropout=0,
        bias="none",
        target_modules=TARGET_MODULES,
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
            warmup_steps=5,
            num_train_epochs=options.epochs,
            learning_rate=options.learning_rate,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=options.seed,
            # Checkpoints and logs are scratch: they go *beside* the adapter,
            # never into it, so the folder stays a clean slot target.
            output_dir=options.folder + "_outputs",
        ),
    )
    trainer.train()

    # The tokenizer goes in too. The existing adapter folders carry one, which
    # is what lets a folder be loaded on its own (Lora00*/inference.py) as well
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
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:],
                            skip_special_tokens=True).strip()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train one LoRA adapter into a folder the pipeline can use.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="example:\n"
               "  python create_lora.py Lora006/poem_adapter --dataset poem --rank 8",
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
        "--epochs", type=float, default=20, help="training epochs (default 20).")
    parser.add_argument(
        "--learning-rate", type=float, default=2e-4, help="default 2e-4.")
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
        "--prompt", default="Tell me about the ocean.",
        help="prompt to answer once after training, as a smoke test.")
    parser.add_argument(
        "--no-sample", action="store_true",
        help="skip that smoke test and stop as soon as the adapter is saved.")
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite the output folder if it already holds something.")
    return parser.parse_args(argv)


def main(argv=None):
    # After parsing, so --help still works on any interpreter, but before the
    # folder is touched or a model is fetched.
    options = parse_args(argv)
    check_interpreter()
    options.dataset = resolve_dataset(options.dataset)
    options.folder = os.path.abspath(options.folder)
    check_output_dir(options.folder, options.force)

    if options.base_model != BASE_MODEL:
        print("warning: --base-model %s is not what the existing slots were "
              "trained on (%s); the result cannot be blended with them."
              % (options.base_model, BASE_MODEL))

    model, tokenizer = train(options)

    print("\nSaved a rank-%d adapter to %s" % (rank_of(options.folder), options.folder))

    if not options.no_sample:
        print("\nYOU: %s" % options.prompt)
        print("LORA: %s" % sample(model, tokenizer, options.prompt))

    # The last mile: what to paste into template_code.py so a tree can reach it.
    # A slot is *repointed* rather than added -- L1..L5 is the grammar's own
    # alphabet (UNARY_OPS in generate_population.py), so a sixth slot is a
    # grammar change, not a configuration one.
    print("\nTo put it in the search, point a slot at it in template_code.py's "
          "LORA_SLOTS (mind which rank you displace -- it decides which LIN "
          "combinations are legal):")
    print("    " + slot_line(options.folder))
    print("then re-run `python main.py runs` so the generated scripts pick it up.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
