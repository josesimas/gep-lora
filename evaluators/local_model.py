"""
evaluators/local_model.py - The judge, loaded here with unsloth.

The other half of the judge transport. `common.ask_judge()` speaks to a model
either way; this is what it calls when `JUDGE_BACKEND` is `"unsloth"` instead of
`"endpoint"`, and it is the same arrangement the generated scripts already use
to produce the answers being graded -- `FastLanguageModel.from_pretrained`, the
tokeniser's own chat template, one `generate()` per call -- so a sweep can be
run end to end with nothing but this repo and a GPU, and no server up at all.

It is a transport and not an evaluator: it returns the judge's raw reply and
`common.parse_reply()` reads the score out of it, exactly as it does for the
text an endpoint returns. Every rubric, every retry rule and every knob other
than the four below is therefore shared with the endpoint path, which is what
makes the two backends comparable at all.

Three things about it are deliberate.

**The model is loaded on first use, not in `prepare()`.** Preparing is where
`llm_judge_baseline` runs the base model to fill its cache of control answers --
a separate process, holding its own copy of a model -- and loading the judge
before that would have the two of them on the card at once for no reason.

**It is released at the end of the step** (`release()`, called by the evaluate
step and the testing pass). `main.py` runs a whole search in one interpreter, so
the generation after this one spawns scripts that each load the base model; a
judge still resident here would be VRAM taken from every one of them.

**Loaded models are cached and never evicted.** A judge model is loaded once and
grades every answer of the step, which is the entire economy of doing this
locally. `panel` with this backend therefore holds all of `PANEL_MODELS` at the
same time -- N judges resident, the way `PROCESS_RUN_BATCH_SIZE` means N base
models resident -- and says so in the note it prints.
"""

import importlib.util
import sys


# {(model, load_in_4bit, max_seq_length, chat_template): (model, tokenizer)}
# Filled by load(), emptied by release(). Keyed by everything that changes what
# was loaded, so two panel members differing only in precision are two entries
# rather than one wrong one.
_LOADED = {}


def available():
    """Whether unsloth can be imported at all, without importing it.

    Cheap on purpose: this is the check prepare() makes so a missing dependency
    is a sentence at the top of the step rather than a traceback per answer.
    Importing unsloth costs seconds and patches transformers and peft as it
    goes, so nothing here imports it until a model is actually wanted.
    """
    return importlib.util.find_spec("unsloth") is not None


def describe(settings):
    """One line naming what will be loaded, for prepare()'s notes."""
    return "%s (%s, max_seq %d)" % (
        settings["model"],
        "4-bit" if settings["load_in_4bit"] else "16-bit",
        settings["max_seq_length"],
    )


def load(settings):
    """The judge model and its tokeniser, loaded once per process.

    A model that cannot be loaded is a configuration failure and not one
    answer's failure, so this raises SystemExit: the evaluate step stops with a
    sentence naming what it could not load, rather than failing every answer in
    turn with the same message.
    """
    key = (settings["model"], bool(settings["load_in_4bit"]),
           int(settings["max_seq_length"]), settings.get("chat_template"))
    if key in _LOADED:
        return _LOADED[key]

    if not available():
        raise SystemExit(
            "JUDGE_BACKEND = 'unsloth' grades with a model loaded in this "
            "process, and unsloth is not installed in %s. Run the pipeline "
            "under the venv one level up (see CLAUDE.md), or set JUDGE_BACKEND "
            "= 'endpoint' to grade over an API instead." % sys.executable
        )

    # Imported here rather than at the top of the file: unsloth patches
    # transformers and peft as it loads, costs seconds to import and needs a
    # GPU-shaped environment, and none of that should be spent by a sweep that
    # grades over an endpoint or with a local scorer.
    from unsloth import FastLanguageModel

    print("loading judge %s -- this takes a moment, once" % describe(settings))
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=settings["model"],
            max_seq_length=int(settings["max_seq_length"]),
            dtype=None,
            load_in_4bit=bool(settings["load_in_4bit"]),
        )
    except Exception as error:                       # noqa: BLE001 - reported whole
        raise SystemExit(
            "cannot load the judge model %r with unsloth (%s). JUDGE_MODEL has "
            "to name a model this machine can load -- a Hub repo id or a local "
            "folder -- or set JUDGE_BACKEND = 'endpoint' to grade over an API "
            "instead." % (settings["model"], error)
        )

    template = settings.get("chat_template")
    if template:
        # Only when a sweep asks for one. A judge is a stock instruct model and
        # its own template is the one it was trained to answer under; naming one
        # here is for a model whose tokeniser ships none, or ships a wrong one.
        from unsloth.chat_templates import get_chat_template
        tokenizer = get_chat_template(tokenizer, chat_template=template)
    elif not getattr(tokenizer, "chat_template", None):
        raise SystemExit(
            "the judge model %r ships no chat template, so its tokeniser cannot "
            "be handed a system and a user turn. Set JUDGE_LOCAL_CHAT_TEMPLATE "
            "to one unsloth knows (\"qwen-2.5\", \"llama-3.1\", ...), or judge "
            "with a model that carries its own." % settings["model"]
        )

    FastLanguageModel.for_inference(model)
    # Qwen and friends ship a max_length in generation_config.json, and
    # transformers warns whenever that and max_new_tokens are both set. Clear
    # it, exactly as the generated scripts do, so JUDGE_MAX_TOKENS is the only
    # cap in play.
    model.generation_config.max_length = None

    _LOADED[key] = (model, tokenizer)
    return _LOADED[key]


def generate(system_prompt, user_content, settings):
    """One grading call. Returns the judge's raw reply, for parse_reply().

    The mirror of common._request(): everything above it -- the rubric, the
    retries, reading a quality out of what came back -- is the shared path, so
    the only difference between grading here and grading over an endpoint is
    where the tokens are produced.
    """
    model, tokenizer = load(settings)
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}]
    try:
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
            return_dict=True).to(model.device)
        temperature = float(settings["temperature"])
        # do_sample=False is greedy decoding, which is what a temperature of 0
        # means and what grading wants: the same answer graded twice should get
        # the same score.
        out = model.generate(**inputs, max_new_tokens=int(settings["max_tokens"]),
                             do_sample=temperature > 0,
                             temperature=temperature or None)
    except Exception as error:                       # noqa: BLE001 - reported whole
        # Generation failures (an out-of-memory, a prompt over max_seq_length)
        # are this one answer's, the way an endpoint's 500 is: the step counts
        # it, keeps the NULL quality and carries on to the next answer.
        raise RuntimeError("local judge failed to generate: %s" % error)
    # Slice the prompt off, so only the reply is decoded.
    return tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:],
                            skip_special_tokens=True).strip()


def release():
    """Drop every loaded judge and hand its VRAM back. Safe to call always.

    Called at the end of the evaluate step and of the testing pass's scoring
    half, whichever backend ran, because a step that has finished grading has no
    further use for a model and `main.py` goes straight on to a generation whose
    scripts each want the card.
    """
    if not _LOADED:
        return
    _LOADED.clear()
    try:
        import gc

        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
