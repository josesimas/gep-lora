#~ TEMPLATE, not a script to run. baseline_run.py reads this file and fills in
#~ the @@MARKERS@@ to produce one baseline.py -- the base model on its own,
#~ answering the eval prompts with nothing attached.
#~
#~ Why it exists: the "llm_judge_baseline" evaluator grades an individual's
#~ answer against what the model said without any adapter, and something has to
#~ produce that. It is the same shape as template_code.py -- same base model,
#~ same chat template, same generation settings, same YOU:/COACH: transcript --
#~ minus every line that mentions an adapter, so the two answers a judge is
#~ shown differ in the blend and in nothing else.
#~
#~ It runs once. Its answers go into the baselines table, keyed by model and
#~ question, and every later sweep reads them from there.
#~
#~ Markers work as in template_code.py:
#~   @@NAME@@            inline, replaced inside the line it sits on
#~   a line that is just @@NAME@@ (or "# @@NAME@@") is replaced by a block
#~ Lines starting with "#~" never reach the output, and this file is kept as
#~ valid Python so editors, linters and `python -m compileall` still work on it.
"""
@@SCRIPT_NAME@@ - the base model, with no adapter, on the eval prompts.

@@PROVENANCE@@
This is the control the blends are measured against: the same base model the
five LoRAs were trained on, the same chat template and the same generation
settings the generated run_NNN.py scripts use, with nothing attached.

Nothing here is per-individual: there is one baseline per base model, and its
answers are cached in the database the first time they are produced.

Usage
    python @@SCRIPT_NAME@@                          # answer the eval prompts
    python @@SCRIPT_NAME@@ "Help me plan my week."   # your own question
"""

import json
import os
import sys

# Match the training/inference environment: disable Xet download acceleration.
os.environ["HF_HUB_DISABLE_XET"] = "1"

# Unsloth patches transformers and peft as it loads, so it has to be
# imported before them or the optimizations are silently skipped.
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

import torch

#~ Whole-line marker: baseline_run.py replaces it with the BASE_MODEL
#~ assignment itself, filled from BASE_MODEL in settings.py -- the same value
#~ every generated run_NNN.py is stamped with, so the control really is the
#~ model the blends were built on. One of the three names a linter calls
#~ undefined in this template and finds defined in every file generated from it.
# @@BASE_MODEL@@

MAX_SEQ = 2048

#~ Whole-line marker, exactly as in template_code.py: the eval prompts file,
#~ resolved and stamped in as a literal.
# @@TRAINING_SET@@

#~ And the cap on it. Same file, same order, same cap the individuals answer
#~ under -- a baseline answering a different set of questions would not be a
#~ control.
# @@TRAINING_COUNT@@


def _prompt_of(line, path, number):
    """One eval prompt, from one non-blank line of the eval file.

    Kept identical to template_code.py's: this must ask exactly the questions
    the individuals were asked, or the comparison is between two different
    conversations.
    """
    if line[0] == "{":
        try:
            record = json.loads(line)
        except ValueError as error:
            raise SystemExit(
                f"{path} line {number}: starts like a JSON record but will not "
                f"parse ({error})."
            )
        messages = record.get("messages") if isinstance(record, dict) else None
        if not isinstance(messages, list):
            raise SystemExit(
                f"{path} line {number}: a JSON eval record needs a 'messages' "
                f"list, the shape datasets/*.json use."
            )
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
        raise SystemExit(
            f"{path} line {number}: no user turn with any text in it, so there "
            f"is nothing to ask."
        )

    # Tolerate the quoted form a plain prompt file may use, and a bare one.
    if len(line) > 1 and line[0] == line[-1] and line[0] in "\"'":
        line = line[1:-1]
    return line


def _prompts(path, count=None):
    """One prompt per non-blank line, capped at `count`."""
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError as error:
        raise SystemExit(f"cannot read the eval prompts from {path}: {error.strerror}")

    prompts = []
    for number, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        if not prompts and line[0] == "[":
            raise SystemExit(
                f"{path} looks like one big JSON array. The eval file is read a "
                f"line at a time -- write it as one JSON record per line (JSON "
                f"Lines), or as plain one-prompt-per-line text."
            )
        prompts.append(_prompt_of(line, path, number))
    if not prompts:
        raise SystemExit(f"{path} has no prompts in it")
    return prompts if count is None else prompts[:count]


EVAL_PROMPTS = _prompts(TRAINING_SET, TRAINING_COUNT)

print(f"GPU available: {torch.cuda.is_available()}")
print(f"Baseline: {BASE_MODEL} with no adapter attached")

# ---------------------------------------------------------------------------
# Load the base model. Nothing is attached to it -- that is the whole point.
# ---------------------------------------------------------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ,
    dtype=None,
    load_in_4bit=True,
)

# The line process_run.Progress reads as "the long silent part is over". Said
# in the same words the individuals' scripts say it in, so one progress reader
# serves both; there is no adapter to name, and no rank to report.
print("Active adapter: none (base model, rank 0)")

# Same chat template used during training, so inputs are formatted identically.
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

# Switch to Unsloth's fast inference path (~2x faster generation).
FastLanguageModel.for_inference(model)

# Qwen ships max_length=32768 in its generation_config.json, and transformers
# warns whenever that and max_new_tokens are both set. Clear it so the cap in
# ask() is the only one in play -- max_new_tokens was winning anyway.
model.generation_config.max_length = None


def ask(question, max_new_tokens=250):
    """Send one user turn through the bare base model.

    Identical to template_code.py's ask(), cap included: a control generated
    under different settings would make every improvement score a comparison of
    the settings as much as of the blend.
    """
    msgs = [{"role": "user", "content": question}]
    inputs = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    # Slice off the prompt tokens so we only decode the newly generated reply.
    return tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:],
                            skip_special_tokens=True).strip()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Everything after the script name is treated as one question.
        question = " ".join(sys.argv[1:])
        print(f"\nYOU: {question}")
        print(f"COACH: {ask(question)}")
    else:
        for q in EVAL_PROMPTS:
            print(f"\nYOU: {q}")
            print(f"COACH: {ask(q)}")
