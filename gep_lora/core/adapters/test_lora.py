"""
test_lora.py - Ask one question twice: once of the bare base model, once with a
LoRA attached, and print both answers next to each other.

The quickest way to see what an adapter actually did. Everything else here
either scores an adapter (the judge, in the evaluators package) or blends
several of
them (the whole pipeline); this just shows you the difference one makes, in the
words the model uses.

    python -m gep_lora.core.adapters.test_lora "Help me plan my week."
    python -m gep_lora.core.adapters.test_lora                          # keep asking, model stays loaded
    python -m gep_lora.core.adapters.test_lora --lora Lora003 "Describe autumn."   # or loras/Lora003
    python -m gep_lora.core.adapters.test_lora --lora path/to/any_adapter "Hi there!"
    python -m gep_lora.core.adapters.test_lora --no-unsloth "Hi there!"   # plain transformers, faster import

Both answers come out of a single base-model load. The adapter is attached once
and switched off for the "before" answer -- `with model.disable_adapter()`, the
same move loras/Lora00*/main.py makes at the end of training. Loading the base twice
would cost twice as long and prove nothing extra, since it is the same weights
either way.

With no question on the command line it stays open and keeps asking, which is
the point: the load is the expensive part, and a comparison is usually worth
making several times over before it tells you anything.

Every phase is timed and the table is printed when the script ends -- the
import, the base-model load, attaching the adapter, the inference setup, and
each answer, base and adapted apart, with the tokens it generated. The phase
names are the ones the generated scripts print as TIMING: lines, so the two can
be read side by side. Time spent waiting at the prompt is a phase of its own,
so an interactive session's wall clock is not mistaken for work.

Interpreter: needs the venv one level up, like everything else that loads a
model, and runs as a module from the repo root -- `python adapters\\test_lora.py`
cannot find the `adapters` package it imports from --

    D:\\sage-is\\loras\\.venv\\Scripts\\python.exe -m gep_lora.core.adapters.test_lora "Hi there!"
"""

import argparse
import contextlib
import json
import os
import sys
import time

# Match the training/inference environment: disable Xet download acceleration.
os.environ["HF_HUB_DISABLE_XET"] = "1"

from gep_lora.core.adapters import create_lora
from gep_lora import paths

# The repo folder, one above this one. Every path a setting names is
# resolved against it, so nothing here depends on the cwd a driver was
# started from, or on which sub-folder this module ended up in.
_ROOT = paths.ROOT

# Where the Lora00N folders live, and the subfolder each one keeps its adapter
# in, so `--lora Lora003` can mean the adapter inside it rather than the folder
# itself -- and can still be written as the bare folder name now that the set
# has moved under loras/.
LORA_DIR = "loras"
#ADAPTER_NAME = "my_planning_coach-lora_adapter"
ADAPTER_NAME = "Qwen2.5-1.5B-Instruct-medical-lora_adapter"
DEFAULT_LORA = os.path.join(LORA_DIR, "Lora001", ADAPTER_NAME)

MAX_SEQ = 2048

# What a bare `python -m gep_lora.core.adapters.test_lora` asks, when nothing is typed at the prompt
# either. Three questions the two answers tend to differ on.
DEMO_PROMPTS = [
    "Help me plan my week.",
    "What's the capital of France?",
    "Tell me about the ocean.",
]


class Timings:
    """Seconds per phase, folded as they are measured and printed once at the end.

    A phase keeps its calls, its total, its worst single call and the tokens it
    generated, so a fixed cost paid once reads differently from a per-answer
    cost paid every question. The clock starts when this is built.
    """

    def __init__(self):
        self.started = time.perf_counter()
        self.phases = {}  # name -> [calls, seconds, worst, tokens], in first-seen order

    def add(self, name, seconds, tokens=0):
        row = self.phases.setdefault(name, [0, 0.0, 0.0, 0])
        row[0] += 1
        row[1] += seconds
        row[2] = max(row[2], seconds)
        row[3] += tokens

    @contextlib.contextmanager
    def phase(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - start)

    def report(self):
        """The table: one row per phase, then what they add up to and the wall clock."""
        wall = time.perf_counter() - self.started
        rule = "=" * 72
        print(rule)
        print("TIMINGS")
        print(rule)
        print("%-18s %5s %9s %9s %9s %6s %6s %6s" % (
            "phase", "calls", "total", "mean", "max", "share", "tokens", "tok/s"))
        accounted = 0.0
        for name, (calls, seconds, worst, tokens) in self.phases.items():
            accounted += seconds
            # Blank rather than 0 for a phase that generates nothing: only the
            # generate rows have tokens to count.
            count = "%6d" % tokens if tokens else ""
            rate = "%6.1f" % (tokens / seconds) if tokens and seconds else ""
            print(("%-18s %5d %8.2fs %8.2fs %8.2fs %5.1f%% %6s %6s" % (
                name, calls, seconds, seconds / calls, worst,
                100.0 * seconds / wall if wall else 0.0, count, rate)).rstrip())
        print("-" * 72)
        # What no phase covers: argument parsing, resolving the adapter,
        # tokenising and decoding, printing.
        other = max(wall - accounted, 0.0)
        print("%-18s %5s %8.2fs %9s %9s %5.1f%%" % (
            "other", "", other, "", "", 100.0 * other / wall if wall else 0.0))
        print("%-18s %5s %8.2fs" % ("wall", "", wall))
        print()


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
                  os.path.join(_ROOT, path),
                  os.path.join(_ROOT, path, ADAPTER_NAME),
                  os.path.join(_ROOT, LORA_DIR, path),
                  os.path.join(_ROOT, LORA_DIR, path, ADAPTER_NAME)]
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


def load(adapter_dir, base_model, max_seq, timings, use_unsloth=True,
         template=None):
    """The base model with `adapter_dir` attached, ready to answer either way.

    Three steps, each its own function so base_models_and_loras_comparison can
    take them apart -- it loads a base once and attaches several adapters to
    it in turn -- while still loading exactly the way this script does.
    """
    model, tokenizer = load_base(base_model, max_seq, timings, use_unsloth)
    print("adapter: %s (rank %d)" % (adapter_dir, create_lora.rank_of(adapter_dir)))
    model = attach(model, adapter_dir, timings)
    with timings.phase("inference_setup"):
        tokenizer = chat_template(tokenizer, template, use_unsloth)
        for_inference(model, tokenizer, use_unsloth)
    return model, tokenizer


def load_base(base_model, max_seq, timings, use_unsloth=True):
    """The import and the model load: the base and its tokenizer, nothing attached."""
    if use_unsloth:
        return load_base_unsloth(base_model, max_seq, timings)
    return load_base_plain(base_model, timings)


def describe(torch, base_model, loader):
    print("GPU available: %s" % torch.cuda.is_available())
    print("loader:  %s" % loader)
    print("base:    %s" % base_model)


def load_base_unsloth(base_model, max_seq, timings):
    """The pipeline's way: the same loader the generated scripts use."""
    with timings.phase("import"):
        # Unsloth patches transformers and peft as it loads, so it comes first.
        from unsloth import FastLanguageModel
        import torch
        import peft  # noqa: F401 -- paid here, so attach() times attaching

    describe(torch, base_model, "unsloth")

    with timings.phase("model_load"):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_seq,
            dtype=None,
            load_in_4bit=True,
        )
    return model, tokenizer


def load_base_plain(base_model, timings):
    """The same model through plain transformers + bitsandbytes + peft.

    What --no-unsloth runs. Importing unsloth costs about twice what these three
    do, most of it training libraries this script never touches, plus the
    patching unsloth does to transformers as it loads. The price is that this
    is not the loader the pipeline uses, so an answer here is only evidence
    about a sweep's answers if it comes out the same as without the flag.

    Four things are kept equal on purpose. A pre-quantized repo (the unsloth
    -bnb-4bit ones) is loaded with its own quantization_config, skip list
    included, so the weights are the ones unsloth would load; anything else is
    quantized here as 4-bit NF4 -- where unsloth would swap in its own
    pre-quantized copy or pick its own skip list, so that case is close, not
    identical. The class is the one the repo's config names rather than
    AutoModelForCausalLM, and a repo with a vision tower gets its processor --
    both what unsloth hands back for one (Qwen3.5 loads as
    Qwen3_5ForConditionalGeneration), and the module paths an adapter trained
    on it is keyed to. The chat template is the model's own, as it is for
    unsloth unless --chat-template names one (see chat_template()): an unsloth
    template name cannot be used here. And
    the dtype is the one unsloth's dtype=None would pick. max_seq is unsloth's
    cap alone and has no counterpart here.
    """
    with timings.phase("import"):
        import torch
        import transformers
        from transformers import (AutoConfig, AutoModelForCausalLM, AutoProcessor,
                                  AutoTokenizer, BitsAndBytesConfig)
        import peft  # noqa: F401 -- paid here, so attach() times attaching

    describe(torch, base_model, "transformers + peft (no unsloth)")

    cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float16
    with timings.phase("model_load"):
        config = AutoConfig.from_pretrained(base_model)
        extra = {}
        if not getattr(config, "quantization_config", None):
            # Only when the repo has none of its own. The keyword is left out
            # rather than passed as None otherwise: transformers 5.5 reads an
            # explicit None as a config and fails on it.
            extra["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
        architecture = (getattr(config, "architectures", None) or [""])[0]
        model_class = getattr(transformers, architecture, AutoModelForCausalLM)
        model = model_class.from_pretrained(
            base_model,
            dtype=dtype,
            device_map={"": 0} if cuda else None,
            **extra
        )
        vision = getattr(config, "vision_config", None) is not None
        tokenizer = (AutoProcessor if vision else AutoTokenizer).from_pretrained(base_model)
    return model, tokenizer


def attach(model, adapter_dir, timings):
    """`model` with the adapter at `adapter_dir` attached and active."""
    from peft import PeftModel

    with timings.phase("attach"):
        return PeftModel.from_pretrained(model, adapter_dir)


def chat_template(tokenizer, name=None, use_unsloth=True):
    """The tokenizer, under the chat template the adapter was trained with.

    None keeps the model's own template -- what CHAT_TEMPLATE = None means in
    settings.py, and what create_lora.py trains under by default. A name swaps
    in unsloth's template of that name, which takes unsloth: without it there
    is nothing to read the template from, so --no-unsloth refuses one rather
    than quietly prompting in the model's own.
    """
    if not name:
        return tokenizer
    if not use_unsloth:
        raise SystemExit(
            "--chat-template %s is one of unsloth's templates, and --no-unsloth "
            "loads without unsloth. Drop one of the two." % name)
    from unsloth.chat_templates import get_chat_template

    return get_chat_template(tokenizer, chat_template=name)


def for_inference(model, tokenizer, use_unsloth=True):
    """Switch `model` to answering. Called again after every attach()."""
    if use_unsloth:
        from unsloth import FastLanguageModel

        FastLanguageModel.for_inference(model)
    else:
        model.eval()
    # Qwen ships max_length in generation_config.json and transformers warns
    # when both caps are set; clearing it leaves max_new_tokens in charge, the
    # same fix the generated scripts carry.
    model.generation_config.max_length = None
    # Stop at the end of the turn, as the generated scripts do: a repo with no
    # generation_config.json (unsloth/Qwen3.5-0.8B) stops only on <|endoftext|>,
    # so an answer ending in <|im_end|> ran on into invented turns.
    stops = model.generation_config.eos_token_id
    stops = stops if isinstance(stops, list) else [] if stops is None else [stops]
    end_of_turn = getattr(tokenizer, "tokenizer", tokenizer).eos_token_id
    if end_of_turn is not None and end_of_turn not in stops:
        model.generation_config.eos_token_id = stops + [end_of_turn]


def answer(model, tokenizer, question, max_new_tokens, timings, phase):
    """One reply, from whatever adapter state the model is currently in.

    Only generate() is timed, under `phase`, along with how many tokens it
    produced -- the rate is what says whether the adapter slows decoding.
    """
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    if isinstance(rendered, str):
        # A VLM repo (Qwen3.5 and friends) loads as a processor, and its
        # apply_chat_template ignores return_tensors/return_dict and gives
        # back the rendered string. Tokenise it the ordinary way.
        rendered = tokenizer(text=rendered, return_tensors="pt")
    inputs = rendered.to(model.device)
    prompt_length = inputs["input_ids"].shape[-1]
    start = time.perf_counter()
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    timings.add(phase, time.perf_counter() - start, out.shape[-1] - prompt_length)
    # Slice off the prompt tokens so we only decode the newly generated reply.
    return tokenizer.decode(out[0][prompt_length:], skip_special_tokens=True).strip()


def both(model, tokenizer, question, max_new_tokens, timings):
    """The bare answer and the adapted one, in that order.

    disable_adapter() is a context manager that switches the LoRA out for the
    length of the block, so "before" and "after" are the same weights either
    side of one difference.
    """
    with model.disable_adapter():
        base = answer(model, tokenizer, question, max_new_tokens,
                      timings, "generate.base")
    tuned = answer(model, tokenizer, question, max_new_tokens,
                   timings, "generate.lora")
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


def ask_loop(model, tokenizer, max_new_tokens, timings):
    """Keep taking questions until the user is done.

    The base-model load is most of the cost of this script, so staying open is
    what makes a second question nearly free. The wait at the prompt is timed
    too, so it shows up as itself rather than inflating "other".
    """
    print("\nType a prompt and press Enter. Blank line, 'quit' or Ctrl-C to stop.")
    while True:
        try:
            with timings.phase("waiting for input"):
                question = input("\nprompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question or question.lower() in ("quit", "exit"):
            return
        base, tuned = both(model, tokenizer, question, max_new_tokens, timings)
        show(question, base, tuned)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Ask one prompt of the bare model and of the model plus a "
                    "LoRA, and print both answers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
               '  python -m gep_lora.core.adapters.test_lora "Help me plan my week."\n'
               "  python -m gep_lora.core.adapters.test_lora                       # ask repeatedly\n"
               '  python -m gep_lora.core.adapters.test_lora --lora Lora003 "Describe autumn."\n'
               '  python -m gep_lora.core.adapters.test_lora --lora loras/Lora003 "Describe autumn."\n'
               '  python -m gep_lora.core.adapters.test_lora --no-unsloth "Describe autumn."',
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
    parser.add_argument(
        "--no-unsloth", dest="unsloth", action="store_false",
        help="load through plain transformers + bitsandbytes + peft instead of "
             "unsloth: roughly half the import time, but not the loader the "
             "pipeline uses, so compare its answers with a run without the flag "
             "before trusting them. --max-seq is ignored.")
    parser.add_argument(
        "--chat-template", default=None,
        help="an unsloth chat template name to prompt in, such as qwen-2.5 "
             "(default: the model's own, as CHAT_TEMPLATE = None in settings.py). "
             "Use the one the adapter was trained under: an adapter prompted in "
             "another format answers like a different model.")
    return parser.parse_args(argv)


def main(argv=None):
    options = parse_args(argv)
    create_lora.check_interpreter()

    adapter = resolve_adapter(options.lora)
    base_model = options.base_model or base_model_of(adapter)

    timings = Timings()
    try:
        model, tokenizer = load(adapter, base_model, options.max_seq, timings,
                                use_unsloth=options.unsloth,
                                template=options.chat_template)

        if options.prompt:
            # Everything after the options is one question, so quoting is optional.
            question = " ".join(options.prompt)
            show(question, *both(model, tokenizer, question,
                                 options.max_new_tokens, timings))
        elif options.demo:
            for question in DEMO_PROMPTS:
                show(question, *both(model, tokenizer, question,
                                     options.max_new_tokens, timings))
        else:
            ask_loop(model, tokenizer, options.max_new_tokens, timings)
    finally:
        # In a finally so a load that fails, or a Ctrl-C mid-answer, still
        # says where the time went up to that point.
        timings.report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
