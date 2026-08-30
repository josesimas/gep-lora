"""
test_lora.py - Ask one question twice: once of the bare base model, once with a
LoRA attached, and print both answers next to each other.

The quickest way to see what an adapter actually did. Everything else here
either scores an adapter (the judge, in evaluate_run.py) or blends several of
them (the whole pipeline); this just shows you the difference one makes, in the
words the model uses.

    python test_lora.py "Help me plan my week."
    python test_lora.py                          # keep asking, model stays loaded
    python test_lora.py --lora Lora003 "Describe autumn."   # or loras/Lora003
    python test_lora.py --lora path/to/any_adapter "Hi there!"

Both answers come out of a single base-model load. The adapter is attached once
and switched off for the "before" answer -- `with model.disable_adapter()`, the
same move loras/Lora00*/main.py makes at the end of training. Loading the base twice
would cost twice as long and prove nothing extra, since it is the same weights
either way.

With no question on the command line it stays open and keeps asking, which is
the point: the load is the expensive part, and a comparison is usually worth
making several times over before it tells you anything.

Interpreter: needs the venv one level up, like everything else that loads a
model --

    D:\\sage-is\\loras\\.venv\\Scripts\\python.exe test_lora.py "Hi there!"
"""

import argparse
import json
import os
import sys

# Match the training/inference environment: disable Xet download acceleration.
os.environ["HF_HUB_DISABLE_XET"] = "1"

import create_lora

_HERE = os.path.dirname(os.path.abspath(__file__))

# Where the Lora00N folders live, and the subfolder each one keeps its adapter
# in, so `--lora Lora003` can mean the adapter inside it rather than the folder
# itself -- and can still be written as the bare folder name now that the set
# has moved under loras/.
LORA_DIR = "loras"
ADAPTER_NAME = "my_planning_coach-lora_adapter"
DEFAULT_LORA = os.path.join(LORA_DIR, "Lora001", ADAPTER_NAME)

MAX_SEQ = 2048

# What a bare `python test_lora.py` asks, when nothing is typed at the prompt
# either. Three questions the two answers tend to differ on.
DEMO_PROMPTS = [
    "Help me plan my week.",
    "What's the capital of France?",
    "Tell me about the ocean.",
]


def resolve_adapter(path):
    """An adapter directory, from a path or from a bare Lora00N folder name.

    `--lora Lora003` should find the adapter inside it without the convention
    being spelled out -- neither the loras/ folder the set lives in nor the
    adapter subfolder has to be typed -- but a path that is already an adapter
    always wins. The check is for adapter_config.json, since that is what makes
    a directory an adapter rather than a folder that contains one.
    """
    candidates = [path,
                  os.path.join(path, ADAPTER_NAME),
                  os.path.join(_HERE, path),
                  os.path.join(_HERE, path, ADAPTER_NAME),
                  os.path.join(_HERE, LORA_DIR, path),
                  os.path.join(_HERE, LORA_DIR, path, ADAPTER_NAME)]
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, "adapter_config.json")):
            return os.path.abspath(candidate)
    raise SystemExit(
        "no adapter at %r: none of these holds an adapter_config.json\n  %s"
        % (path, "\n  ".join(os.path.abspath(c) for c in candidates))
    )


def base_model_of(adapter_dir):
    """The base model this adapter was trained against, from its own config.

    Read rather than assumed, so pointing --lora at an adapter from somewhere
    else still loads the right base. It is the same field inference.py leans on
    when it hands Unsloth the adapter folder and lets it find the base itself.
    """
    with open(os.path.join(adapter_dir, "adapter_config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    base = config.get("base_model_name_or_path")
    if not base:
        raise SystemExit(
            "%s/adapter_config.json does not name a base model; pass --base-model."
            % adapter_dir
        )
    return base


def load(adapter_dir, base_model, max_seq):
    """The base model with `adapter_dir` attached, ready to answer either way."""
    # Unsloth patches transformers and peft as it loads, so it comes first.
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    import torch
    from peft import PeftModel

    print("GPU available: %s" % torch.cuda.is_available())
    print("base:    %s" % base_model)
    print("adapter: %s (rank %d)" % (adapter_dir, create_lora.rank_of(adapter_dir)))

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq,
        dtype=None,
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)

    # The template both the training and the pipeline's eval use, so neither
    # answer is being judged on a formatting difference.
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    FastLanguageModel.for_inference(model)
    # Qwen ships max_length in generation_config.json and transformers warns
    # when both caps are set; clearing it leaves max_new_tokens in charge, the
    # same fix the generated scripts carry.
    model.generation_config.max_length = None
    return model, tokenizer


def answer(model, tokenizer, question, max_new_tokens):
    """One reply, from whatever adapter state the model is currently in."""
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    # Slice off the prompt tokens so we only decode the newly generated reply.
    return tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:],
                            skip_special_tokens=True).strip()


def both(model, tokenizer, question, max_new_tokens):
    """The bare answer and the adapted one, in that order.

    disable_adapter() is a context manager that switches the LoRA out for the
    length of the block, so "before" and "after" are the same weights either
    side of one difference.
    """
    with model.disable_adapter():
        base = answer(model, tokenizer, question, max_new_tokens)
    tuned = answer(model, tokenizer, question, max_new_tokens)
    return base, tuned


def show(question, base, tuned):
    """Print the pair, labelled, with the question above them."""
    rule = "=" * 72
    print("\n" + rule)
    print("YOU: %s" % question)
    print(rule)
    print("\n--- BASE (adapter off) " + "-" * 49)
    print(base)
    print("\n--- LORA (adapter on) " + "-" * 50)
    print(tuned)
    if base == tuned:
        # Worth saying: identical output usually means the adapter did not take
        # on this prompt, not that the comparison failed to run.
        print("\n(identical -- the adapter changed nothing on this prompt)")
    print()


def ask_loop(model, tokenizer, max_new_tokens):
    """Keep taking questions until the user is done.

    The base-model load is most of the cost of this script, so staying open is
    what makes a second question nearly free.
    """
    print("\nType a prompt and press Enter. Blank line, 'quit' or Ctrl-C to stop.")
    while True:
        try:
            question = input("\nprompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question or question.lower() in ("quit", "exit"):
            return
        base, tuned = both(model, tokenizer, question, max_new_tokens)
        show(question, base, tuned)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Ask one prompt of the bare model and of the model plus a "
                    "LoRA, and print both answers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
               '  python test_lora.py "Help me plan my week."\n'
               "  python test_lora.py                       # ask repeatedly\n"
               '  python test_lora.py --lora Lora003 "Describe autumn."\n'
               '  python test_lora.py --lora loras/Lora003 "Describe autumn."',
    )
    parser.add_argument(
        "prompt", nargs="*",
        help="the question. Everything after the options is taken as one prompt; "
             "with none given the script stays open and keeps asking.")
    parser.add_argument(
        "--lora", "-l", default=DEFAULT_LORA,
        help="the adapter to compare against the bare model (default %s). Either "
             "an adapter folder or a Lora00N folder holding one, named with or "
             "without the loras/ prefix; any location "
             "works, including one create_lora.py just wrote."
             % DEFAULT_LORA.replace("\\", "/"))
    parser.add_argument(
        "--base-model", default=None,
        help="override the base model. The default is whichever one the "
             "adapter's own adapter_config.json names, which is normally right.")
    parser.add_argument(
        "--max-new-tokens", type=int, default=250,
        help="length cap on each answer (default 250, as in the generated "
             "scripts).")
    parser.add_argument(
        "--max-seq", type=int, default=MAX_SEQ,
        help="max sequence length (default %d)." % MAX_SEQ)
    parser.add_argument(
        "--demo", action="store_true",
        help="run the built-in prompts once instead of asking, then exit.")
    return parser.parse_args(argv)


def main(argv=None):
    options = parse_args(argv)
    create_lora.check_interpreter()

    adapter = resolve_adapter(options.lora)
    base_model = options.base_model or base_model_of(adapter)
    model, tokenizer = load(adapter, base_model, options.max_seq)

    if options.prompt:
        # Everything after the options is one question, so quoting is optional.
        question = " ".join(options.prompt)
        show(question, *both(model, tokenizer, question, options.max_new_tokens))
    elif options.demo:
        for question in DEMO_PROMPTS:
            show(question, *both(model, tokenizer, question, options.max_new_tokens))
    else:
        ask_loop(model, tokenizer, options.max_new_tokens)
    return 0


if __name__ == "__main__":
    sys.exit(main())
