"""
combination.py - Stack BOTH trained LoRAs on top of the base model, then chat.

You now have two LoRA adapters trained from the same base model:
    Lora001/my_planning_coach-lora_adapter
    Lora002/my_planning_coach-lora_adapter

This script loads the base model once and applies BOTH adapters, one after the
other, then generates exactly like inference.py does.

Why this works / what "one after the other" means here:
    Each LoRA was trained against the SAME frozen base, so each one only adds its
    own low-rank delta:  effective_weight = W_base + dW1 + dW2.
    The deltas are additive, so stacking them is base + dW1 + dW2. We get that with
    PEFT's add_weighted_adapter(..., combination_type="cat"), which concatenates the
    two adapters into one "combined" adapter that fires both at once -- no need to
    dequantize or merge into the 4-bit base weights.

Tune the blend with W1 / W2 below (1.0 each = apply both at full strength).

Usage:
    python combination.py                      # runs a couple of demo prompts
    python combination.py "Help me plan my week."  # ask your own question
"""

import os
import sys

# Match the training/inference environment: disable Xet download acceleration.
os.environ["HF_HUB_DISABLE_XET"] = "1"

import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

_HERE = os.path.dirname(os.path.abspath(__file__))

# The two LoRA folders to stack. Both must share the same base model (they do).
ADAPTER1_DIR = os.path.join(_HERE, "Lora001", "my_planning_coach-lora_adapter")
ADAPTER2_DIR = os.path.join(_HERE, "Lora002", "my_planning_coach-lora_adapter")

# How strongly to apply each adapter. 1.0 = full strength. Lower one to blend.
# Both LoRAs were trained on the same tiny set, so their deltas nearly coincide;
# 0.5 + 0.5 sums to ~1x total (a coherent average) instead of ~2x (which garbles).
W1 = 0.5
W2 = 0.5

MAX_SEQ = 2048

print(f"GPU available: {torch.cuda.is_available()}")
print(f"LoRA #1: {ADAPTER1_DIR}")
print(f"LoRA #2: {ADAPTER2_DIR}")

# Pointing from_pretrained at the FIRST adapter folder loads the base model and
# attaches adapter #1 under the name "default" (this is the known-good path from
# inference.py).
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER1_DIR,
    max_seq_length=MAX_SEQ,
    dtype=None,
    load_in_4bit=True,
)

# Attach the SECOND adapter alongside the first, under its own name.
model.load_adapter(ADAPTER2_DIR, adapter_name="lora2")

# Concatenate the two adapters into a single "combined" adapter and make it
# active. combination_type="cat" preserves each adapter faithfully (base + dW1 +
# dW2) rather than averaging them.
model.add_weighted_adapter(
    adapters=["default", "lora2"],
    weights=[W1, W2],
    adapter_name="combined",
    combination_type="cat",
)
model.set_adapter("combined")
print(f'Active adapter: {model.active_adapters} (weights: lora1={W1}, lora2={W2})')

# Same chat template used during training, so inputs are formatted identically.
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

# Switch to Unsloth's fast inference path (~2x faster generation).
FastLanguageModel.for_inference(model)


def ask(question, max_new_tokens=250):
    """Send one user turn to the combined-LoRA coach and return its reply."""
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
        # No argument: run a few demo prompts so you can eyeball the behavior.
        for q in [
            "Help me organize my desktop.",
            "I have three assignments due this week. Just tell me what to do.",
            "Make me a full study schedule for my exams.",
        ]:
            print(f"\nYOU: {q}")
            print(f"COACH: {ask(q)}")
