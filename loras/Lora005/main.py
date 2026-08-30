from datasets import load_dataset, Dataset, concatenate_datasets
from unsloth.chat_templates import get_chat_template
import torch
import os
from trl import SFTTrainer, SFTConfig

# Quick check: this should print True if the T4 is active
print(f"GPU available: {torch.cuda.is_available()}")

from unsloth import FastLanguageModel

# Disable Xet download acceleration to bypass potential 403 errors
os.environ["HF_HUB_DISABLE_XET"] = "1"

# Apache-2.0, ~1.5B, has a maintained Unsloth free-T4 path.
MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct"
MAX_SEQ = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ,
    dtype=None,
    load_in_4bit=True
)

# The seed lives in a GitHub gist, pinned to this exact revision
# (the long hash in the URL) so a later edit to the gist can't silently
# change what interns download.
# To update the seed, edit the gist and
# paste the  NEW revision's raw URL here.
DATA_URL = "https://gist.githubusercontent.com/opencoca/e0ee456fa23f3576c64aa1fab86b432b/raw/c8dfc552fcfcdbf32fa17f98621d75dfb8c28d11/planning-coach-starter.jsonl"

# The try block lets you test a block of code for errors.
try:
    seed_dataset = load_dataset("json", data_files=DATA_URL, split="train")

# The except block lets you handle the error.
except Exception as e:
    print("Couldn't fetch the seed, using a tiny inline sample instead:", e)
    seed_dataset = Dataset.from_list([
        {"messages": [
            {"role": "user", "content": "Help me plan my week."},
            {"role": "assistant", "content": "Happy to. Before we lay anything out, what's the one thing that has to be true by the end of this week for you to feel good about it?"},
        ]},
        {"messages": [
            {"role": "user", "content": "I have a history essay, a math test, and a club project all due soon. Just tell me what to do."},
            {"role": "assistant", "content": "I can help you sort it, but you know your situation best, so let's decide together. Which of the three has the nearest deadline, and which one worries you most?"},
        ]},
        {"messages": [
            {"role": "user", "content": "Make me a full study schedule for my exams."},
            {"role": "assistant", "content": "Let's build it so it's actually yours. First: which subjects are on the exams, and roughly how many days until the first one? Once I know that, we'll pick where to start."},
        ]},
    ])

# Say out loud how many loaded, so a failed download doesn't slip by quietly.
print(f"Loaded {len(seed_dataset)} starter examples.")
if len(seed_dataset) < 20:
    print("That's fewer than expected. The download probably failed and you're on "
          "the small inline sample. Re-run this cell from the top before training.")


# You type two strings; this builds the rest, so you can't misplace a key or a
# bracket. `you` is what the person says, `coach` is the reply.
def ex(you, coach):
    return {"messages": [
        {"role": "user", "content": you},
        {"role": "assistant", "content": coach},
    ]}


# Write YOUR OWN examples here. Aim for 24 to 48, or more if you're on a roll.
# Copy a line, change the two strings, keep going. Same spirit as the starter
# set: the coach asks a question or names a planning trap instead of handing
# over a finished plan.
MY_EXAMPLES = [
    ex("you: I need help studying for my final exams.",
       "coach: Exam time can be stressful. How many exams do you have and when are they?"),
    ex("you: My friend is coming to visit and my house is a mess and I don't know where to begin!",
       "coach: Would you rather start with a specific room or do a quick tidy where you collect all the things that don't belong in each room?"),
    ex("you: I have a group project and need help managing all the moving parts so that we hit our deadline",
       "coach: How many people are in the group? Have the roles been assigned? What is the due date?")
    # Add more ex(...) lines below. Aim for 24 to 48 total. Don't forget to seperate each ex() with a comma and a newline.
]


# An example is ready if it has a user turn and a coach turn, both are real
# text, and neither is still a placeholder. Anything else is skipped so it can't
# quietly train the model on junk. This never stops you; it just says what it
# skipped. The isinstance() checks keep it from crashing on an odd hand-typed entry.
def looks_ready(e):
    msgs = e.get("messages", []) if isinstance(e, dict) else []
    if len(msgs) != 2 or not all(isinstance(m, dict) for m in msgs):
        return False
    if msgs[0].get("role") != "user" or msgs[1].get("role") != "assistant":
        return False
    texts = [m.get("content") for m in msgs]
    if not all(isinstance(t, str) and t.strip() for t in texts):
        return False
    return not any("REPLACE ME" in t for t in texts)


ready = [e for e in MY_EXAMPLES if looks_ready(e)]
skipped = len(MY_EXAMPLES) - len(ready)
if skipped:
    print(f"Skipped {skipped} example(s) that were placeholder, empty, or the wrong shape.")
if len(ready) < 24:
    print(f"You have {len(ready)} ready. Aim for 24 to 48 for a clear result, "
          f"then run this cell again.")

# Rebuild from the seed every run, so running this cell twice never doubles your
# examples. `seed_dataset` comes from Step 3.
if ready:
    dataset = concatenate_datasets([seed_dataset, Dataset.from_list(ready)])
else:
    dataset = seed_dataset

print(f"Training set: {len(dataset)} examples "
      f"({len(seed_dataset)} starter + {len(ready)} yours)")
      
      
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

def to_text(batch):
    return {"text": [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                      for m in batch["messages"]]}

dataset = dataset.map(to_text, batched=True)

model = FastLanguageModel.get_peft_model(
    model, r=32, lora_alpha=16, lora_dropout=0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=3407)
    
# The number of epochs is set to 20. One epoch represents one complete pass through
# all training examples (or how many times your model "reads" through its entire
# training dataset0. It's a seemingly simple parameter, yet its correct setting
# is crucial for achieving optimal performance without falling into common pitfalls.
# As an experiment, feel free to change your epoque to 2 to see how well your model
# is trained compared to 20. If you have time, try 40 epoches and see if it improves things.
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text", max_seq_length=MAX_SEQ,
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        warmup_steps=5, num_train_epochs=15, learning_rate=2e-4,
        logging_steps=1, optim="adamw_8bit", weight_decay=0.01,
        lr_scheduler_type="linear", seed=3407, output_dir="outputs"))
trainer.train()

FastLanguageModel.for_inference(model)

question = "Help me organize my desktop."
msgs = [{"role": "user", "content": question}]
inputs = tokenizer.apply_chat_template(
    msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")

def generate():
    out = model.generate(**inputs, max_length=250, do_sample=False)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

with model.disable_adapter():        # adapter off = the 'before' brain
    before = generate()
after = generate()                   # adapter on = your tuned model. Not the adaper is always
                                     # on after training unless model.delete_adapter() is run.

print("BEFORE (plain model):\n", before)
print("\nAFTER (your coach):\n", after)

# After training your model with Unsloth save as a LoRA and compress and download for later use or save to your drive
model.save_pretrained("my_planning_coach-lora_adapter")
tokenizer.save_pretrained("my_planning_coach-lora_adapter")

