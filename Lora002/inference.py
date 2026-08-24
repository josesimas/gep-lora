"""
inference.py - Load the base model + your trained LoRA and chat with it.

The training run (main.py) saved a LoRA adapter to ./my_planning_coach-lora_adapter.
That folder holds ONLY the small low-rank deltas, not the full model. This script
loads the base model (unsloth/Qwen2.5-1.5B-Instruct) and applies the adapter on top,
which is what "combining the LoRA with the model" means in practice.

Usage:
    python inference.py                       # runs a couple of demo prompts
    python inference.py "Help me plan my week."   # ask your own question
"""

import os
import sys

# Match the training environment: disable Xet download acceleration.
os.environ["HF_HUB_DISABLE_XET"] = "1"

import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# This folder is the LoRA that main.py produced. adapter_config.json inside it
# records the base model, so Unsloth pulls the base and attaches the adapter
# in one call -- no need to name the base model again here.
ADAPTER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "my_planning_coach-lora_adapter")
MAX_SEQ = 2048

print(f"GPU available: {torch.cuda.is_available()}")
print(f"Loading base model + LoRA from: {ADAPTER_DIR}")

# Pointing from_pretrained at the adapter folder loads base + LoRA together.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER_DIR,
    max_seq_length=MAX_SEQ,
    dtype=None,
    load_in_4bit=True,
)

# Same chat template used during training, so inputs are formatted identically.
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

# Switch to Unsloth's fast inference path (~2x faster generation).
FastLanguageModel.for_inference(model)


def ask(question, max_new_tokens=250):
    """Send one user turn to the tuned coach and return its reply."""
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
