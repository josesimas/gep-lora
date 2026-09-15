"""
base_models_and_loras_comparison.py - How fast each base model loads and
answers, and what attaching a LoRA to it costs, written up as one markdown file.

Every Lora00N folder under loras/ holds one adapter per base model, each slot
at its own rank. This finds them all by reading each adapter_config.json,
groups them by the base model they name, and runs one worker process per base
model (and loader) that:

  1. loads the base cold -- import, model_load, inference_setup and a first
     generate() -- which is what any script pays before its first answer;
  2. measures the bare base: prefill time and decode speed at a fixed answer
     length, at batch 1 and at the pipeline's batch of 8, plus the prompt set
     answered the way a generated script answers it;
  3. for each of that base's adapters in turn: attaches it (timed), measures
     the same things again, and takes it off before the next one.

The markdown goes into loras/, stamped with the time it was made, with the raw
numbers at the bottom so a later report can be compared against it.

    python -m adapters.base_models_and_loras_comparison
    python -m adapters.base_models_and_loras_comparison --loader both
    python -m adapters.base_models_and_loras_comparison --only 0.8b --slots Lora001 Lora004
    python -m adapters.base_models_and_loras_comparison --tokens 32 --repeats 1   # quick look
    python -m adapters.base_models_and_loras_comparison --list

Why a process per base model: a load is only a load the first time. A second
model in the same interpreter would find torch, transformers and unsloth
already imported, the CUDA context up and the allocator warm, and report a
cold start it never paid. So each base model starts from nothing, the way a
generated script does, and the numbers in the loading table are the ones a
script would see.

Why fixed-length answers: an adapter changes what a model says as well as how
fast it says it. The medical adapters answer in a dozen tokens where the bare
base runs to the 250-token cap, so comparing their natural answers measures
answer length, not speed. The fixed-length test forces every answer to exactly
--tokens new tokens (min_new_tokens = max_new_tokens), so base and adapter do
identical work and the difference is the adapter's own cost. The natural test
is kept beside it because answer length is a real effect on what a sweep
costs -- it is just a different question.

Loading goes through adapters.test_lora's own load_base(), attach(),
chat_template() and for_inference(), so what is measured here is exactly how
test_lora loads a model, with or without unsloth (--loader).

Interpreter: the venv one level up, and run as a module from the repo root --

    D:\\sage-is\\loras\\.venv\\Scripts\\python.exe -m adapters.base_models_and_loras_comparison
"""

import argparse
import datetime
import gc
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback

# Match the training/inference environment: disable Xet download acceleration.
os.environ["HF_HUB_DISABLE_XET"] = "1"

from adapters import create_lora, test_lora
from blends.process_run import CHILD_ENCODING

# The repo folder, one above this one. Every path is resolved against it, so
# nothing here depends on the cwd a driver was started from.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# What the driver launches for each worker; the worker has to be run as a
# module for `from adapters import ...` to resolve.
MODULE = "adapters.base_models_and_loras_comparison"

# Where the report goes unless --out says otherwise: beside what it measures.
REPORT_DIR = os.path.join(_ROOT, test_lora.LORA_DIR)

# The prompts every measurement asks. Fixed here rather than read from
# datasets/, so a report made on another machine asked the same questions.
# Half are medical, which is what the adapters were trained on -- that is what
# makes the natural-answer test show an adapter changing answer length --
# and half are not.
PROMPTS = [
    "What are the early signs of dehydration?",
    "How is high blood pressure usually treated?",
    "What should I do if I think I have the flu?",
    "Why is it important to finish a course of antibiotics?",
    "Help me plan my week.",
    "What's the capital of France?",
    "Tell me about the ocean.",
    "Explain how a bicycle stays upright.",
]

# How many prompts go into one generate() call in the natural test, and its
# length cap: templates/template_code.py's ANSWER_BATCH and max_new_tokens, so
# that test costs what a generated script's answering costs.
NATURAL_BATCH = 8
NATURAL_TOKENS = 250

# The first generate() after a load or an attach, timed on its own: a warm-up
# that pays whatever the first call pays, kept short so it measures that and
# not decoding.
WARMUP_TOKENS = 8


# --------------------------------------------------------------------------
# What there is to measure


def discover(slots=None, only=None):
    """{base model: [adapter]}, from every adapter folder under loras/.

    Each adapter is {slot, name, path, rank}. Grouped by the base model the
    adapter's own config names, since that -- not the folder name -- is what
    it can be attached to. `slots` keeps only those Lora00N folders; `only`
    keeps an adapter when any of its substrings appears in its folder name or
    its base model's name.
    """
    root = os.path.join(_ROOT, test_lora.LORA_DIR)
    groups = {}
    for slot in sorted(os.listdir(root)):
        slot_dir = os.path.join(root, slot)
        if not os.path.isdir(slot_dir) or (slots and slot not in slots):
            continue
        for name in sorted(os.listdir(slot_dir)):
            folder = os.path.join(slot_dir, name)
            if not os.path.isfile(os.path.join(folder, "adapter_config.json")):
                continue
            base = test_lora.base_model_of(folder)
            if only and not any(o.lower() in (name + " " + base).lower() for o in only):
                continue
            groups.setdefault(base, []).append({
                "slot": slot, "name": name, "path": folder,
                "rank": create_lora.rank_of(folder),
            })
    return groups


# --------------------------------------------------------------------------
# The worker: one base model, one loader, one fresh interpreter


def say(message):
    """A progress line for the driver to echo. Everything else on stdout -- the
    unsloth banner, transformers' notices -- is the libraries', and ignored."""
    print("BENCH: " + message, flush=True)


class Bench:
    """Timed generate() calls against whatever the model currently is."""

    def __init__(self, model, tokenizer, torch):
        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch
        # A processor (the Qwen3.5 VLM) wraps the tokenizer that pads.
        text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        # Left padding, as the generated scripts batch: a decoder continues
        # from the last column, so a right-padded row would continue its padding.
        text_tokenizer.padding_side = "left"
        if text_tokenizer.pad_token_id is None:
            text_tokenizer.pad_token = text_tokenizer.eos_token
        self.pad_id = text_tokenizer.pad_token_id

    def inputs(self, prompts):
        texts = [self.tokenizer.apply_chat_template(
                     [{"role": "user", "content": prompt}],
                     add_generation_prompt=True, tokenize=False)
                 for prompt in prompts]
        # `text=` by name, which a tokenizer and a processor both take.
        return self.tokenizer(text=texts, padding=True,
                              return_tensors="pt").to(self.model.device)

    def sync(self):
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()

    def generate(self, inputs, new_tokens, fixed):
        """-> (seconds, tokens generated across the batch, longest row).

        `fixed` forces every row to exactly `new_tokens` by suppressing the
        end of sequence until then; otherwise rows stop where the model stops
        them, and the tokens are the ones before the padding a finished row is
        filled with.
        """
        options = dict(max_new_tokens=new_tokens, do_sample=False,
                       pad_token_id=self.pad_id)
        if fixed:
            options["min_new_tokens"] = new_tokens
        self.sync()
        start = time.perf_counter()
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, **options)
        self.sync()
        seconds = time.perf_counter() - start
        new = out[:, inputs["input_ids"].shape[-1]:]
        per_row = (new != self.pad_id).sum(dim=-1).tolist()
        return seconds, int(sum(per_row)), int(max(per_row))


def batch_of(size):
    """`size` prompts, cycling through PROMPTS when it asks for more."""
    return [PROMPTS[i % len(PROMPTS)] for i in range(size)]


def measure(bench, spec, label):
    """Everything measured about one configuration -- the bare base or one adapter."""
    torch = bench.torch
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    entry = {"resting_vram": torch.cuda.memory_allocated() if cuda else None}

    entry["first_generate"] = bench.generate(
        bench.inputs(batch_of(1)), WARMUP_TOKENS, fixed=True)[0]

    tokens, repeats = spec["tokens"], spec["repeats"]
    fixed = {}
    for size in spec["batches"]:
        inputs = bench.inputs(batch_of(size))
        # A batch shape can pay a one-off cost the first time it is seen --
        # Qwen3.5 under unsloth spent ~9 s compiling for b=8 -- and without
        # this it would land in whichever measurement went first. Kept, since
        # a script that answers in batches of that size pays it too.
        warmup = bench.generate(inputs, WARMUP_TOKENS, fixed=True)[0]
        prefill = statistics.median(
            bench.generate(inputs, 1, fixed=True)[0] for _ in range(repeats))
        runs = [bench.generate(inputs, tokens, fixed=True) for _ in range(repeats)]
        total = statistics.median(seconds for seconds, _, _ in runs)
        generated = runs[0][1]
        # Decoding is what is left once the prompt has been read and the first
        # token produced; the first token belongs to prefill. Noise can put a
        # short answer's total under its prefill, and then there is no rate
        # to report rather than a huge one.
        decode = total - prefill
        fixed[str(size)] = {
            "warmup": warmup, "prefill": prefill, "total": total,
            "tokens": generated, "forced": generated == size * tokens,
            "decode_tps": size * (tokens - 1) / decode if decode > 0 else None,
        }
    entry["fixed"] = fixed

    natural = {"seconds": 0.0, "tokens": 0, "longest": 0, "calls": 0}
    for start in range(0, len(PROMPTS), NATURAL_BATCH):
        inputs = bench.inputs(PROMPTS[start:start + NATURAL_BATCH])
        bench.generate(inputs, WARMUP_TOKENS, fixed=True)  # its shape, as above
        seconds, generated, longest = bench.generate(
            inputs, spec["natural_tokens"], fixed=False)
        natural["seconds"] += seconds
        natural["tokens"] += generated
        natural["longest"] = max(natural["longest"], longest)
        natural["calls"] += 1
    natural["prompts"] = len(PROMPTS)
    entry["natural"] = natural
    entry["peak_vram"] = torch.cuda.max_memory_allocated() if cuda else None

    one = fixed.get("1") or next(iter(fixed.values()))
    say("%s: prefill %.0f ms, decode %s tok/s, natural %.1fs for %d tokens"
        % (label, one["prefill"] * 1000, rate(one["decode_tps"]),
           natural["seconds"], natural["tokens"]))
    return entry


def environment(torch, loader):
    import peft
    import transformers

    env = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "unsloth": getattr(sys.modules.get("unsloth"), "__version__", None)
                   if loader == "unsloth" else None,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        env["gpu"] = props.name
        env["gpu_memory"] = props.total_memory
    return env


def work(spec):
    """One base model under one loader: the cold load, the base, each adapter.

    Prints a RESULT line holding everything it managed. An adapter that fails
    is recorded against that adapter and the rest carry on -- unless taking it
    off failed too, since every later adapter would then be measured on top of
    it.
    """
    result = {"base_model": spec["base_model"], "loader": spec["loader"],
              "ok": False, "configs": []}
    use_unsloth = spec["loader"] == "unsloth"
    try:
        timings = test_lora.Timings()
        model, tokenizer = test_lora.load_base(
            spec["base_model"], spec["max_seq"], timings, use_unsloth)
        import torch

        result["environment"] = environment(torch, spec["loader"])
        with timings.phase("inference_setup"):
            tokenizer = test_lora.chat_template(tokenizer, use_unsloth)
            test_lora.for_inference(model, use_unsloth)
        result["cold"] = {name: row[1] for name, row in timings.phases.items()}
        result["cold"]["vram_after_load"] = (
            torch.cuda.memory_allocated() if torch.cuda.is_available() else None)
        say("loaded: import %.1fs, model_load %.1fs"
            % (result["cold"].get("import", 0), result["cold"].get("model_load", 0)))

        base = measure(Bench(model, tokenizer, torch), spec, "base")
        base.update(name="base", slot=None, rank=None)
        result["configs"].append(base)
        result["cold"]["first_generate"] = base["first_generate"]

        for adapter in spec["adapters"]:
            label = "%s r%d" % (adapter["slot"], adapter["rank"])
            entry = {"name": adapter["name"], "slot": adapter["slot"],
                     "rank": adapter["rank"]}
            result["configs"].append(entry)
            wrapped = None
            try:
                attached = test_lora.Timings()
                wrapped = test_lora.attach(model, adapter["path"], attached)
                test_lora.for_inference(wrapped, use_unsloth)
                entry["attach"] = attached.phases["attach"][1]
                entry.update(measure(Bench(wrapped, tokenizer, torch), spec, label))
            except Exception as error:
                entry["error"] = "%s: %s" % (type(error).__name__, error)
                say("%s failed: %s" % (label, entry["error"]))
                traceback.print_exc()
            # Off again whether or not it measured, so the next adapter is
            # attached to the bare base and measured with only itself on it.
            if wrapped is not None:
                start = time.perf_counter()
                model = wrapped.unload()
                entry["detach"] = time.perf_counter() - start
                wrapped = None
            if any("lora_A" in name for name, _ in model.named_modules()):
                raise RuntimeError("%s is still attached after unload()" % label)
            gc.collect()
            torch.cuda.empty_cache()

        # The bare base once more, after every adapter has been on and off.
        # How far it lands from the first measurement is the noise floor --
        # GPU clocks, the allocator, whatever unload() left behind -- and a
        # LoRA effect smaller than that gap is not one.
        test_lora.for_inference(model, use_unsloth)
        again = measure(Bench(model, tokenizer, torch), spec, "base, again")
        again.update(name="base_again", slot=None, rank=None)
        result["configs"].append(again)
        result["ok"] = True
    except Exception as error:
        result["error"] = "%s: %s" % (type(error).__name__, error)
        traceback.print_exc()
    print("RESULT: " + json.dumps(result), flush=True)
    return 0 if result["ok"] else 1


# --------------------------------------------------------------------------
# The driver

# GPU utilization, in percent, above which a card is already doing something
# else. A speed test on a shared GPU measures the sharing: LM Studio's
# llama-server generating in the background halved every unsloth decode rate
# while this was being written.
BUSY_UTILIZATION = 10.0


def gpu_load(samples=3, interval=1.0):
    """What the GPU is doing right now, from nvidia-smi -- or None without one.

    -> {"utilization": mean % over the samples, "memory_used": bytes}. Sampled
    a few times because utilization is a moment's reading, and a generator
    between tokens reads as idle.
    """
    readings = []
    try:
        for sample in range(samples):
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15).stdout
            utilization, used = (float(v) for v in out.strip().splitlines()[0].split(","))
            readings.append((utilization, used))
            if sample < samples - 1:
                time.sleep(interval)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None
    return {"utilization": statistics.mean(u for u, _ in readings),
            "memory_used": max(m for _, m in readings) * 1024 * 1024}


def busy(load):
    return bool(load) and load["utilization"] > BUSY_UTILIZATION


def run_worker(spec, timeout):
    """Run one worker in a fresh interpreter. -> its result, failures included."""
    env = dict(os.environ)
    env[CHILD_ENCODING] = "utf-8"
    started = time.perf_counter()
    result = None
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as err:
        child = subprocess.Popen(
            [sys.executable, "-u", "-m", MODULE, "--worker", json.dumps(spec)],
            cwd=_ROOT, stdout=subprocess.PIPE, stderr=err, env=env,
            text=True, encoding="utf-8", errors="replace",
        )
        killer = threading.Timer(timeout, child.kill) if timeout else None
        if killer:
            killer.start()
        try:
            for line in child.stdout:
                if line.startswith("RESULT: "):
                    result = json.loads(line[len("RESULT: "):])
                elif line.startswith("BENCH: "):
                    print("    " + line[len("BENCH: "):].rstrip(), flush=True)
            code = child.wait()
        finally:
            if killer:
                killer.cancel()
        err.seek(0)
        stderr = err.read()
    if result is None:
        result = {"base_model": spec["base_model"], "loader": spec["loader"],
                  "ok": False, "configs": [],
                  "error": "worker exited %s without a result" % code}
    result["process_seconds"] = time.perf_counter() - started
    if not result["ok"]:
        # The traceback is in the worker's stderr, not its result line.
        result["stderr_tail"] = "\n".join(stderr.strip().splitlines()[-25:])
    return result


# --------------------------------------------------------------------------
# The report


def seconds(value):
    return "–" if value is None else "%.2f s" % value


def millis(value):
    return "–" if value is None else "%.0f ms" % (value * 1000)


def rate(value):
    return "–" if value is None else "%.1f" % value


def gigabytes(value):
    return "–" if value is None else "%.2f GB" % (value / 1e9)


def change(value, reference):
    """How `value` compares with the base's, as a signed percentage."""
    if value is None or not reference:
        return "–"
    return "%+.0f%%" % (100.0 * (value - reference) / reference)


def fixed_of(config, size):
    return (config.get("fixed") or {}).get(str(size)) or {}


def cold_total(cold):
    parts = [cold.get(k) for k in ("import", "model_load", "inference_setup", "first_generate")]
    return None if any(p is None for p in parts) else sum(parts)


def short(base_model):
    return base_model.split("/")[-1]


def title_of(result):
    return "%s (%s)" % (short(result["base_model"]), result["loader"])


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def adapters_of(result):
    return [c for c in result["configs"] if c.get("slot")]


def base_of(result, name="base"):
    return next((c for c in result["configs"] if c["name"] == name), None)


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def summary_section(results, sizes):
    size = sizes[0]
    rows = []
    for r in results:
        base = base_of(r)
        if not r.get("cold") or not base:
            rows.append([title_of(r), "failed"] + [""] * 6)
            continue
        base_tps = fixed_of(base, size).get("decode_tps")
        lora_tps = mean(fixed_of(a, size).get("decode_tps") for a in adapters_of(r))
        again = base_of(r, "base_again")
        rows.append([
            title_of(r),
            seconds(cold_total(r["cold"])),
            seconds(mean(a.get("attach") for a in adapters_of(r))),
            rate(base_tps), rate(lora_tps), change(lora_tps, base_tps),
            change(fixed_of(again, size).get("decode_tps"), base_tps) if again else "–",
            gigabytes(r["cold"].get("vram_after_load")),
        ])
    return "\n".join([
        "## Summary", "",
        table(["base model (loader)", "time to first answer", "LoRA attach (mean)",
               "decode tok/s, base (b=%d)" % size, "decode tok/s, with LoRA (mean)",
               "LoRA effect", "noise", "VRAM after load"], rows),
        "",
        "*Time to first answer* is import + model load + inference setup + the "
        "first generate(); add *LoRA attach* for the same with an adapter on. "
        "*LoRA effect* is the mean over this base's adapters of the change in "
        "decode speed at a fixed answer length -- negative is slower. *noise* is "
        "the same change for the bare base measured a second time, at the end: a "
        "LoRA effect no bigger than it is not one.",
    ])


def loading_section(results):
    rows = []
    for r in results:
        cold = r.get("cold") or {}
        load = r.get("gpu_before")
        rows.append([
            title_of(r), seconds(cold.get("import")), seconds(cold.get("model_load")),
            seconds(cold.get("inference_setup")), seconds(cold.get("first_generate")),
            "**%s**" % seconds(cold_total(cold)) if cold_total(cold) is not None else "–",
            gigabytes(cold.get("vram_after_load")),
            seconds(r.get("process_seconds")),
            "–" if not load else "%s%.0f%%, %s" % (
                "**busy** " if busy(load) else "", load["utilization"],
                gigabytes(load["memory_used"])),
        ])
    return "\n".join([
        "## 1. Loading a base model, cold", "",
        table(["base model (loader)", "import", "model_load", "inference_setup",
               "first generate (%d tok)" % WARMUP_TOKENS, "time to first answer",
               "VRAM after load", "whole worker", "GPU before"], rows),
        "",
        "Each row is a fresh interpreter, so nothing was imported or loaded before "
        "it. *whole worker* is that process from launch to exit, every measurement "
        "below included; the gap between it and the phases before it is also the "
        "interpreter starting up, which no phase covers. *GPU before* is the card's "
        "utilization and memory in use just before the worker started, by anything: "
        "on an idle card it is near 0%%, and anything above %.0f%% means the row was "
        "measured while sharing the GPU." % BUSY_UTILIZATION,
    ])


def attach_section(results):
    slots = sorted({a["slot"] for r in results for a in adapters_of(r)})
    if not slots:
        return ""
    header = ["slot"] + [title_of(r) for r in results]
    rows = []
    for slot in slots:
        row = [slot]
        for r in results:
            match = next((a for a in adapters_of(r) if a["slot"] == slot), None)
            if match is None:
                row.append("")
            elif match.get("attach") is None:
                row.append("r%d · failed" % match["rank"])
            else:
                row.append("r%d · %s" % (match["rank"], seconds(match["attach"])))
        rows.append(row)
    return "\n".join([
        "## 2. Attaching a LoRA", "",
        table(header, rows), "",
        "Each cell is the adapter's rank and how long `PeftModel.from_pretrained()` "
        "took to read it and wrap the base with it. Every adapter was attached to "
        "the bare base on its own, and taken off again before the next.",
    ])


def inference_section(results, sizes):
    parts = ["## 3. Inference, with and without a LoRA", ""]
    for r in results:
        base = base_of(r)
        parts.append("### %s" % title_of(r))
        parts.append("")
        if base is None or "fixed" not in base:
            parts.append("Not measured: %s" % (r.get("error") or "the base did not load"))
            parts.append("")
            continue
        header = ["", "rank"]
        for size in sizes:
            header += ["prefill b=%d" % size, "decode tok/s b=%d" % size, "vs base"]
        header += ["natural: seconds", "tokens", "vs base", "peak VRAM"]
        rows = []
        again = base_of(r, "base_again")
        for config in [base] + adapters_of(r) + ([again] if again else []):
            name = ("**base**" if config is base else
                    "base, again" if config is again else config["slot"])
            if "error" in config or "fixed" not in config:
                rows.append([name, "r%s" % config.get("rank"), "failed: %s"
                             % config.get("error", "not measured")]
                            + [""] * (len(header) - 3))
                continue
            row = [name, "r%d" % config["rank"] if config.get("rank") else ""]
            for size in sizes:
                mine, theirs = fixed_of(config, size), fixed_of(base, size)
                row += [millis(mine.get("prefill")), rate(mine.get("decode_tps")),
                        "" if config is base else change(mine.get("decode_tps"),
                                                         theirs.get("decode_tps"))]
            natural, base_natural = config["natural"], base["natural"]
            row += [seconds(natural["seconds"]), natural["tokens"],
                    "" if config is base else change(natural["seconds"],
                                                     base_natural["seconds"]),
                    gigabytes(config.get("peak_vram"))]
            rows.append(row)
        parts.append(table(header, rows))
        unforced = [c.get("slot") or "base" for c in r["configs"]
                    for f in (c.get("fixed") or {}).values() if not f.get("forced", True)]
        if unforced:
            parts.append("")
            parts.append("Warning: generate() stopped short of the forced length for "
                         "%s, so those decode rates are over fewer tokens than the "
                         "others." % ", ".join(sorted(set(unforced))))
        parts.append("")
    parts.append(
        "*base, again* is the bare base measured a second time, after every "
        "adapter had been on and off: how far it is from the first *base* row "
        "is the noise in these numbers, and an adapter's *vs base* within that "
        "distance is not a measurable effect.")
    parts.append("")
    parts.append(
        "*prefill* is a generate() of one new token: reading the prompt. *decode "
        "tok/s* is the tokens after that one, divided by the time after it, summed "
        "over the batch -- so b=8 is throughput across eight answers at once. Both "
        "are at a fixed answer length, the same work with and without the adapter. "
        "*natural* is the %d prompts answered in generate() calls of %d, stopping "
        "where the model stops (at most %d tokens), the way a generated script "
        "answers: an adapter that changes how long the answers are shows up there "
        "and not in the decode columns." % (len(PROMPTS), NATURAL_BATCH, NATURAL_TOKENS))
    return "\n".join(parts)


def rank_section(results, sizes):
    ranks = sorted({a["rank"] for r in results for a in adapters_of(r)})
    if not ranks:
        return ""
    header = ["rank"]
    for r in results:
        header += ["%s b=%d" % (title_of(r), size) for size in sizes]
    rows = []
    for rank in ranks:
        row = ["r%d" % rank]
        for r in results:
            base = base_of(r)
            for size in sizes:
                reference = fixed_of(base, size).get("decode_tps") if base else None
                speed = mean(fixed_of(a, size).get("decode_tps")
                             for a in adapters_of(r) if a["rank"] == rank)
                row.append(change(speed, reference))
        rows.append(row)
    return "\n".join([
        "## 4. LoRA effect on decode speed, by rank", "",
        table(header, rows), "",
        "The change in decode speed against the bare base, at a fixed answer "
        "length. Where two slots share a rank, the two are averaged.",
    ])


def failures_section(results):
    failed = [r for r in results if not r["ok"]]
    broken = [(r, c) for r in results for c in r["configs"] if "error" in c]
    if not failed and not broken:
        return ""
    parts = ["## Failures", ""]
    for r in failed:
        parts.append("**%s**: %s" % (title_of(r), r.get("error")))
        if r.get("stderr_tail"):
            parts += ["", "```", r["stderr_tail"], "```"]
        parts.append("")
    for r, c in broken:
        parts.append("- %s, %s (r%d): %s" % (title_of(r), c["slot"], c["rank"], c["error"]))
    return "\n".join(parts)


def report(results, spec, command, elapsed):
    env = next((r["environment"] for r in results if r.get("environment")), {})
    versions = " / ".join("%s %s" % (k, env[k]) for k in
                          ("torch", "transformers", "peft", "unsloth") if env.get(k))
    sizes = spec["batches"]
    header = [
        "# Base models and LoRAs: speed comparison", "",
        "Made %s by `%s`, in %s." % (
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), command,
            "%dm %02ds" % divmod(int(elapsed), 60)),
        "",
        table(["", ""], [
            ["GPU", "%s (%s)" % (env.get("gpu", "none"), gigabytes(env.get("gpu_memory")))],
            ["Python / libraries", "%s / %s" % (env.get("python", "?"), versions)],
            ["fixed-length test", "%d new tokens, median of %d, batch %s"
             % (spec["tokens"], spec["repeats"], " and ".join(map(str, sizes)))],
            ["natural test", "%d prompts, generate() calls of %d, up to %d new tokens"
             % (len(PROMPTS), NATURAL_BATCH, spec["natural_tokens"])],
            ["base models", ", ".join("`%s`" % m for m in
                                      dict.fromkeys(r["base_model"] for r in results))],
        ]),
    ]
    shared = [r for r in results if busy(r.get("gpu_before"))]
    if shared:
        header += ["", "> **Warning: the GPU was not idle.** Before %d of %d workers it "
                   "was already busy (up to %.0f%%, with up to %s in use) -- something "
                   "else was running on it, and every number below includes sharing "
                   "the card with it. Stop whatever that is (an LM Studio model, "
                   "another sweep) and run this again before trusting a speed here."
                   % (len(shared), len(results),
                      max(r["gpu_before"]["utilization"] for r in shared),
                      gigabytes(max(r["gpu_before"]["memory_used"] for r in shared)))]
    sections = [
        "\n".join(header),
        summary_section(results, sizes),
        loading_section(results),
        attach_section(results),
        inference_section(results, sizes),
        rank_section(results, sizes),
        failures_section(results),
        "\n".join([
            "## Prompts", "",
            "\n".join("%d. %s" % (i + 1, p) for i, p in enumerate(PROMPTS)),
        ]),
        "\n".join([
            "## Raw results", "",
            "<details><summary>JSON, one entry per worker</summary>", "",
            "```json", json.dumps(results, indent=1), "```", "", "</details>",
        ]),
    ]
    return "\n\n".join(s for s in sections if s) + "\n"


def default_out():
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(REPORT_DIR, "base_models_and_loras_comparison_%s.md" % stamp)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Time loading and inference for every base model under loras/, "
                    "with and without each of its LoRAs, and write a markdown report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
               "  python -m adapters.base_models_and_loras_comparison\n"
               "  python -m adapters.base_models_and_loras_comparison --loader both\n"
               "  python -m adapters.base_models_and_loras_comparison --only 1.5b --slots Lora001\n"
               "  python -m adapters.base_models_and_loras_comparison --tokens 32 --repeats 1",
    )
    parser.add_argument(
        "--loader", choices=("unsloth", "plain", "both"), default="unsloth",
        help="how to load each base: unsloth (the pipeline's way, default), plain "
             "transformers + peft (test_lora's --no-unsloth), or both, one after "
             "the other.")
    parser.add_argument(
        "--only", nargs="+", metavar="TEXT",
        help="keep only adapters whose folder name or base model contains one of "
             "these, e.g. --only 0.8b 1.5b.")
    parser.add_argument(
        "--slots", nargs="+", metavar="LORA00N",
        help="keep only these Lora00N folders (default: all of them).")
    parser.add_argument(
        "--tokens", type=int, default=64,
        help="new tokens per answer in the fixed-length test (default 64).")
    parser.add_argument(
        "--batches", type=int, nargs="+", default=[1, NATURAL_BATCH],
        help="batch sizes for the fixed-length test (default 1 %d)." % NATURAL_BATCH)
    parser.add_argument(
        "--repeats", type=int, default=3,
        help="runs of each fixed-length measurement; the median is kept (default 3).")
    parser.add_argument(
        "--natural-tokens", type=int, default=NATURAL_TOKENS,
        help="length cap in the natural test (default %d, as the generated "
             "scripts)." % NATURAL_TOKENS)
    parser.add_argument(
        "--max-seq", type=int, default=test_lora.MAX_SEQ,
        help="unsloth's max sequence length (default %d)." % test_lora.MAX_SEQ)
    parser.add_argument(
        "--timeout", type=float, default=1800,
        help="seconds one worker may take before it is killed (default 1800, 0 = "
             "no limit).")
    parser.add_argument(
        "--out", default=None,
        help="where to write the report (default loras/"
             "base_models_and_loras_comparison_<time>.md).")
    parser.add_argument(
        "--list", action="store_true",
        help="print the base models and adapters that would be measured, and stop.")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    options = parser.parse_args(argv)
    if options.tokens < 2:
        parser.error("--tokens must be at least 2: the first token is prefill")
    if options.repeats < 1:
        parser.error("--repeats must be at least 1")
    return options


def main(argv=None):
    options = parse_args(argv)
    if options.worker:
        return work(json.loads(options.worker))

    groups = discover(options.slots, options.only)
    if not groups:
        raise SystemExit("no adapters under %s match"
                         % os.path.join(_ROOT, test_lora.LORA_DIR))
    loaders = ["unsloth", "plain"] if options.loader == "both" else [options.loader]
    for base, adapters in groups.items():
        print("%s\n    %s" % (base, "\n    ".join(
            "%s  r%-3d %s" % (a["slot"], a["rank"], a["name"]) for a in adapters)))
    if options.list:
        return 0
    create_lora.check_interpreter()

    common = {"tokens": options.tokens, "batches": options.batches,
              "repeats": options.repeats, "natural_tokens": options.natural_tokens,
              "max_seq": options.max_seq}
    work_list = [dict(common, base_model=base, loader=loader, adapters=adapters)
                 for base, adapters in groups.items() for loader in loaders]
    started = time.perf_counter()
    results = []
    for number, spec in enumerate(work_list, 1):
        print("\n[%d/%d] %s, %s loader, %d adapters"
              % (number, len(work_list), spec["base_model"], spec["loader"],
                 len(spec["adapters"])), flush=True)
        load = gpu_load()
        if busy(load):
            print("    warning: the GPU is already %.0f%% busy with %s in use before "
                  "this worker starts -- something else is running on it, and these "
                  "numbers will include sharing it"
                  % (load["utilization"], gigabytes(load["memory_used"])), flush=True)
        result = run_worker(spec, options.timeout)
        result["gpu_before"] = load
        if not result["ok"]:
            print("    failed: %s" % result.get("error"), flush=True)
        results.append(result)

    command = "python -m %s %s" % (MODULE, " ".join(argv if argv is not None else sys.argv[1:]))
    out = os.path.abspath(options.out or default_out())
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        handle.write(report(results, common, command.strip(),
                            time.perf_counter() - started))
    print("\nreport: %s" % out)
    # Only a run where nothing measured is a failure; one failed base model is
    # a result, and is in the report as one.
    return 0 if any(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
