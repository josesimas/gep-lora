#~ TEMPLATE, not a script to run. generate_runs.py reads this file and fills in
#~ the @@MARKERS@@ to produce one run_NNN.py per individual.
#~
#~ Two kinds of marker:
#~   @@NAME@@            inline, replaced inside the line it sits on
#~   a line that is just @@NAME@@ (or "# @@NAME@@") is replaced by a whole block
#~
#~ Lines starting with "#~" are template-only notes and never reach the output.
#~ Everything else is copied through verbatim, which is why this file is kept as
#~ valid Python: your editor, linter and `python -m compileall` all still work on
#~ it, and the generated scripts are exactly what you see here.
"""
@@SCRIPT_NAME@@ - Combine LoRAs the way one GEP tree says to, then chat.

@@PROVENANCE@@
Written in the style of combination.py, which stacks two adapters with a
fixed combination_type; here the tree decides both the shape and the weights.

Expression
    @@EXPRESSION@@

Tree
@@TREE@@

How to read it
    L<i>.w<j>   attach LoRA slot i, blended at weight w<j>
    CAT(a, b)   add_weighted_adapter(..., combination_type="cat")
    SVD(a, b)   add_weighted_adapter(..., combination_type="svd")
    LIN(a, b)   add_weighted_adapter(..., combination_type="linear")

A combined node is itself an adapter, so it feeds its parent like a leaf.
Its children's weights are already folded in, so it enters its parent at
weight 1.0.

Build order (deepest first)
@@BUILD_ORDER@@

@@NOTE@@
Usage
    python @@SCRIPT_NAME@@                          # demo prompts
    python @@SCRIPT_NAME@@ "Help me plan my week."   # your own question
"""

import os
import random
import sys

# Match the training/inference environment: disable Xet download acceleration.
os.environ["HF_HUB_DISABLE_XET"] = "1"

# Unsloth patches transformers and peft as it loads, so it has to be
# imported before them or the optimizations are silently skipped.
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

import torch
from peft import PeftModel

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))   # run/ -> project/ -> loras/

# The base model every adapter was trained on (from adapter_config.json).
BASE_MODEL = "unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit"

# Where each of the 5 LoRAs the trees refer to lives. One independent entry per
# slot: repoint any single line at a different adapter and nothing else in this
# file has to change.
#
# Only two real adapter folders exist on disk today, so L1/L3/L5 currently point
# at test001 and L2/L4 at test002. Loading one folder several times under
# different adapter names is fine -- PEFT keeps them separate -- but two slots
# backed by the same adapter differ only by the weight the tree hands them, so
# the search sees less diversity than the 5 slots suggest.
#
# Any of these may become an absolute path, or a Hub repo id, once there are 5
# genuinely different adapters.
LORA_SLOTS = {
    "L1": os.path.join(_ROOT, "test001", "my_planning_coach-lora_adapter"),
    "L2": os.path.join(_ROOT, "test002", "my_planning_coach-lora_adapter"),
    "L3": os.path.join(_ROOT, "test001", "my_planning_coach-lora_adapter"),
    "L4": os.path.join(_ROOT, "test002", "my_planning_coach-lora_adapter"),
    "L5": os.path.join(_ROOT, "test001", "my_planning_coach-lora_adapter"),
}

# What w1..w5 are worth: a fresh random draw every run, strictly between 0 and
# 1. Set WEIGHT_SEED to an int to repeat one particular draw -- without it the
# same tree scores differently each time it runs.
WEIGHT_SEED = None

_rng = random.Random(WEIGHT_SEED)


def _weight():
    """A blend weight in (0, 1), both ends excluded."""
    # random() yields [0.0, 1.0), so rejecting an exact 0.0 leaves (0.0, 1.0).
    value = 0.0
    while value == 0.0:
        value = _rng.random()
    return value


WEIGHTS = {name: _weight() for name in ("w1", "w2", "w3", "w4", "w5")}

MAX_SEQ = 2048

# The prompts this individual is judged on.
EVAL_PROMPTS = [
    "Help me organize my desktop.",
    "I have three assignments due this week. Just tell me what to do.",
    "Make me a full study schedule for my exams.",
]

EXPRESSION = "@@EXPRESSION@@"

print(f"GPU available: {torch.cuda.is_available()}")
print(f"@@LABEL@@: {EXPRESSION}")
# Print the draw, so a run can be traced back to the weights that produced it.
print("weights: " + ", ".join(f"{k}={v:.4f}" for k, v in WEIGHTS.items()))

# ---------------------------------------------------------------------------
# Load the base model once, with no adapter attached yet.
# ---------------------------------------------------------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ,
    dtype=None,
    load_in_4bit=True,
)

# ---------------------------------------------------------------------------
# Attach the @@LEAF_COUNT@@ leaf adapter(s) the tree names, each under its own name so
# the same slot can appear more than once at different weights.
# ---------------------------------------------------------------------------
#~ One PeftModel.from_pretrained for the first leaf, load_adapter for the rest.
# @@ATTACH_LEAVES@@

# ---------------------------------------------------------------------------
# Fold the tree together, deepest node first. Each call leaves behind a new
# adapter that later calls can use as an input.
# ---------------------------------------------------------------------------
#~ One add_weighted_adapter call per binary node, in post-order.
# @@COMBINE_NODES@@

FINAL_ADAPTER = "@@FINAL_ADAPTER@@"
model.set_adapter(FINAL_ADAPTER)
print(f"Active adapter: {model.active_adapters} (rank @@FINAL_RANK@@)")

# Same chat template used during training, so inputs are formatted identically.
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

# Switch to Unsloth's fast inference path (~2x faster generation).
FastLanguageModel.for_inference(model)

# Qwen ships max_length=32768 in its generation_config.json, and transformers
# warns whenever that and max_new_tokens are both set. Clear it so the cap in
# ask() is the only one in play -- max_new_tokens was winning anyway.
model.generation_config.max_length = None


def ask(question, max_new_tokens=250):
    """Send one user turn through this tree's combined adapter."""
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
        # Score this individual by eyeballing its answers to the eval prompts.
        for q in EVAL_PROMPTS:
            print(f"\nYOU: {q}")
            print(f"COACH: {ask(q)}")
