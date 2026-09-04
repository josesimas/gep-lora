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

import json
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
_PROJECT = os.path.dirname(_HERE)                  # run/ -> project/

# The base model every adapter was trained on (from adapter_config.json).
#~ Whole-line marker, like WEIGHT_SEED and the three below it: generate_runs.py
#~ replaces it with the BASE_MODEL assignment itself, filled from BASE_MODEL in
#~ settings.py. It lives there rather than here so there is one copy of the name
#~ across both templates and the baseline script the llm_judge_baseline
#~ evaluator measures against -- a control loading a different model would make
#~ every improvement score meaningless -- and so the sweep records which model
#~ its fitness numbers were earned on. Another of the names a linter calls
#~ undefined here and finds defined in every file generated from this.
# @@BASE_MODEL@@

# Where each of the 5 LoRAs the trees refer to lives. One independent entry per
# slot, already resolved: repoint a slot in settings.py and every script built
# afterwards follows.
#
# These five were trained at different ranks (r=16, 16, 8, 4, 32), which the code
# handles -- _rank() reads each one's adapter_config.json rather than assuming
# they match. That matters because PEFT's cat sums input ranks, svd takes the
# max, and linear refuses inputs whose ranks differ.
#~ Whole-line marker, like WEIGHT_SEED and TRAINING_SET: generate_runs.py
#~ replaces it with the LORA_SLOTS assignment itself, filled from LORA_SLOTS in
#~ settings.py. The slots are the search space, so they belong with the rest of
#~ the knobs -- and with them there, the sweep records which five adapters its
#~ fitness numbers were earned on. A third name a linter calls undefined here
#~ and finds defined in every file generated from this.
# @@LORA_SLOTS@@

# What w1..w5 are worth: a fresh random draw every run, strictly between 0 and
# 1. Set WEIGHT_SEED to an int to repeat one particular draw -- without it the
# same tree scores differently each time it runs.
#~ The next line is a whole-line marker: generate_runs.py replaces it with the
#~ WEIGHT_SEED assignment itself, so a generated script carries a plain literal.
#~ A sweep fills in the integer it recorded for this individual, so a stored
#~ sweep can be replayed weight for weight; None is what a caller that does not
#~ pin the draw gets, which is the behaviour described above. Since the marker
#~ stands in for the assignment, WEIGHT_SEED is the one name in this template a
#~ linter will call undefined -- it is defined in every file generated from it.
# @@WEIGHT_SEED@@

_rng = random.Random(WEIGHT_SEED)


def _weight():
    """A blend weight in (0, 1), both ends excluded."""
    # random() yields [0.0, 1.0), so rejecting an exact 0.0 leaves (0.0, 1.0).
    value = 0.0
    while value == 0.0:
        value = _rng.random()
    return value


WEIGHTS = {name: _weight() for name in ("w1", "w2", "w3", "w4", "w5")}

def _rank(adapter_dir):
    """The rank PEFT will allocate for this adapter, from its adapter_config.json.

    Mirrors PEFT's own bookkeeping (peft/tuners/lora/model.py): rank_pattern can
    raise the rank above r for individual modules, and PEFT sizes the merged
    adapter for the largest rank it might need.
    """
    with open(os.path.join(adapter_dir, "adapter_config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    return max([config["r"]] + list((config.get("rank_pattern") or {}).values()))


MAX_SEQ = 2048

# The prompts this individual is judged on, read from a file at startup rather
# than baked in, so the eval set can change without regenerating any of these
# scripts.
#~ Whole-line marker, like WEIGHT_SEED above: generate_runs.py replaces it with
#~ the TRAINING_SET assignment itself, so a generated script gets a plain
#~ literal path. The value comes from TRAINING_SET in settings.py -- it lives
#~ there rather than here so the eval set can be repointed without editing
#~ either template, and so the sweep that used it records which file that was.
#~ That is why TRAINING_SET, like WEIGHT_SEED, is a name a linter calls
#~ undefined here and finds defined in every file generated from this.
# @@TRAINING_SET@@

# How many of those records this individual is judged on: the first
# TRAINING_COUNT, or all of them when the file holds fewer or the cap is None.
#~ Whole-line marker again, filled from TRAINING_COUNT in settings.py. The cap
#~ is applied here rather than by handing the script a pre-trimmed list, so a
#~ generated script still reads the eval file itself and still shows how many of
#~ it it used. Fourth and last of the names a linter calls undefined in this
#~ template and finds defined in every file generated from it.
# @@TRAINING_COUNT@@


def _prompt_of(line, path, number):
    """One eval prompt, from one non-blank line of the eval file.

    Two shapes, told apart by the line itself rather than by the file's
    extension:

      * a JSON object carrying a "messages" list -- the shape datasets/*.json
        use and the shape create_lora.py trains on. The prompt is the first
        user turn. The assistant turn sitting beside it is somebody else's
        answer to the same question, and must not reach the model: handing it
        over would be showing the model the answer and then scoring it on the
        reply.
      * anything else -- the line as written, surrounding quotes optional,
        which is the plain one-prompt-per-line file.
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
    """One prompt per non-blank line, capped at `count`.

    `count` keeps the first `count` of them and drops the rest; None keeps all.
    The top of the file rather than a sample of it, because every individual has
    to answer the same questions for their scores to mean anything next to each
    other.
    """
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
            # A whole-file JSON array cannot be read a line at a time, and
            # treating its first line as a prompt would quietly ask the model
            # "[". Say so instead: one record per line is the supported shape.
            raise SystemExit(
                f"{path} looks like one big JSON array. The eval file is read a "
                f"line at a time -- write it as one JSON record per line (JSON "
                f"Lines), or as plain one-prompt-per-line text."
            )
        prompts.append(_prompt_of(line, path, number))
    if not prompts:
        raise SystemExit(f"{path} has no prompts in it")
    # Slicing past the end is not an error, which is exactly the "or all of
    # them" case -- a cap larger than the file needs no special handling.
    return prompts if count is None else prompts[:count]


EVAL_PROMPTS = _prompts(TRAINING_SET, TRAINING_COUNT)

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
# Each adapter's rank is read from its own config, so slots pointing at LoRAs
# trained with different r values are handled correctly.
RANKS = {}


def attach(name, slot):
    """Load LORA_SLOTS[slot] under `name`, and record the rank it carries."""
    global model
    if isinstance(model, PeftModel):
        model.load_adapter(LORA_SLOTS[slot], adapter_name=name)
    else:
        # The first adapter is what turns the base model into a PeftModel.
        model = PeftModel.from_pretrained(model, LORA_SLOTS[slot], adapter_name=name)
    RANKS[name] = _rank(LORA_SLOTS[slot])
    return name


#~ One attach() per leaf, in post-order.
# @@ATTACH_LEAVES@@

# ---------------------------------------------------------------------------
# Fold the tree together, deepest node first. Each call leaves behind a new
# adapter that later calls can use as an input.
# ---------------------------------------------------------------------------
def _compact(name):
    """Copy adapter `name`'s weights off whatever buffer they were sliced from.

    PEFT builds an svd node by running torch.linalg.svd on each module's delta
    weight and handing back `Vh[:new_rank, :]` (peft/tuners/lora/model.py,
    _svd_generalized_task_arithmetic_weighted_adapter), which it assigns
    straight to that module's lora_A. The slice is a *view*, so it pins the
    whole V matrix for as long as the adapter lives -- and V is sized by the
    delta weight, not by the rank. On this base model that is ~306 MB behind
    every down_proj: right around 10 GB across the stack, holding some 18 MB
    of actual weights, which is how an svd node came to cost seven times the
    model it attaches to.

    The slice comes back contiguous, so .contiguous() is a no-op here -- only
    a copy drops the reference to the buffer behind it. Anything already
    sitting on its own storage is left alone.
    """
    for module in model.modules():
        for store in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"):
            entry = getattr(module, store, None)
            if entry is None or name not in entry:
                continue
            held = entry[name]
            # lora_A/lora_B hold Linears; the embedding pair holds bare
            # Parameters. Both are weights to us.
            weight = getattr(held, "weight", held)
            if weight.untyped_storage().nbytes() > weight.numel() * weight.element_size():
                weight.data = weight.data.clone()

    # The buffers just released are the caching allocator's now, not the
    # driver's, and PROCESS_RUN_BATCH_SIZE runs its batch-mates as separate
    # processes -- which cannot see this one's free list. Hand the scratch
    # back so a batch is sized by what its scripts hold rather than by the
    # high-water mark they each passed through.
    torch.cuda.empty_cache()


def combine(name, combination_type, left, right):
    """Fold two adapters into one under `name`, tracking the resulting rank.

    PEFT's rules (peft/tuners/lora/model.py, _check_add_weighted_adapter):
    cat sums the input ranks, svd takes the max, linear demands they match.
    The linear case is checked here so the failure names the node, rather than
    surfacing as a bare ValueError from inside PEFT.
    """
    (left_name, left_weight), (right_name, right_weight) = left, right
    left_rank, right_rank = RANKS[left_name], RANKS[right_name]

    if combination_type == "linear" and left_rank != right_rank:
        raise SystemExit(
            f"{name}: combination_type='linear' needs both inputs at the same rank, "
            f"but {left_name} is rank {left_rank} and {right_name} is rank {right_rank}. "
            f"cat sums its inputs' ranks, which is usually what pushes them apart."
        )

    model.add_weighted_adapter(
        adapters=[left_name, right_name],
        weights=[left_weight, right_weight],
        adapter_name=name,
        combination_type=combination_type,
        # The thin SVD. The first new_rank singular vectors are identical
        # either way, so the full n x n V that PEFT asks for by default is
        # only a bigger buffer to compute and then throw away -- see
        # _compact(), which has to throw it away either way.
        svd_full_matrices=False,
    )

    if combination_type == "svd":
        _compact(name)

    if combination_type == "cat":
        RANKS[name] = left_rank + right_rank
    elif combination_type == "svd":
        RANKS[name] = max(left_rank, right_rank)
    else:
        RANKS[name] = left_rank
    return name


#~ One combine() per binary node, in post-order.
# @@COMBINE_NODES@@

FINAL_ADAPTER = "@@FINAL_ADAPTER@@"
model.set_adapter(FINAL_ADAPTER)
print(f"Active adapter: {model.active_adapters} (rank {RANKS[FINAL_ADAPTER]})")

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
