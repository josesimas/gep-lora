"""
lora_server.py - One base model, held open, blending LoRAs on request.

The expensive half of a sweep is not the arithmetic; it is the fixed price each
generated script pays before it can do any. On this machine that is ~13s of
`import unsloth` plus ~19s of model load, per individual, per generation --
about 54% of an average script, and a price a cheaper search cannot escape by
picking simpler trees.

This is the other way to pay it: one process that loads the base model once and
then answers requests. A client says "build this blend" and then "answer these
prompts", and the model, the tokeniser and the chat template are already there.

    POST /build      {base_model, slots, plan, final} -> {rank, timings, builds}
    POST /generate   {prompts, max_new_tokens}        -> {replies, seconds}
    GET  /health                                      -> {ready, builds, ...}
    POST /shutdown                                    -> stop

What it is not
--------------
It is not a change to what an individual *is*. templates/template_remote_code.py
still renders one script per individual, still gets stored in script_source,
still prints the same YOU:/COACH:/TIMING: lines on stdout and still has an exit
code -- it just becomes a thin client for the part that was expensive. Every
guarantee the pipeline hangs off that contract (one execution per script,
transcripts parsed from stdout, partial output kept on a timeout, an individual
that crashes being a result rather than a stopped sweep) is untouched, which is
why this lives beside the scripts rather than inside the driver.

Requests are served one at a time on purpose: a server owns one copy of the
model on one card, so two builds at once would be two blends fighting over the
same adapters. Parallelism comes from running several of these -- see
server_pool.py, which is what the process step starts and hands out.

    python -m blends.lora_server --base-model <id> --port 8770

Where the blend-building lives
------------------------------
Blend below is a second implementation of what template_code.py does inline:
attach, combine, the rank rule, and the compact() that gives back the buffers
PEFT's svd pins. It has to be a second one -- a generated script is standalone
by design and can import nothing from this repo -- so the rule is the one the
two templates already live under: **a change to the blend arithmetic in
template_code.py belongs here too, and the other way round.** What differs
between them is only what a long-lived process has to do and a one-shot script
does not: reset() between builds, and a model load that happens once at startup
rather than once per blend.

The drift that buys
-------------------
A cold process per individual is a strong guarantee: every result came from a
model that had never seen another blend. Here it did. So a build says which
server produced it and how many blends that server had built before -- the
client prints both, they land in the execution's stdout, and server_pool.py's
recycle_after bounds the number. That is a smaller guarantee honestly recorded,
rather than the same one quietly weakened.
"""

import argparse
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

# Match the training/inference environment: disable Xet download acceleration.
# Set before unsloth or transformers is imported, exactly as the templates do.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# The same two facts about this machine that template_code.py holds as
# constants: how long a sequence the model is loaded with, and whether it is
# quantised. They belong to the machine rather than to a sweep, so they are
# constants here too -- with flags for the one caller who wants to try
# something else.
MAX_SEQ = 2048
LOAD_IN_4BIT = True


def _log(message):
    """Say something on the server's own stdout -- its log, not a transcript."""
    print("[lora_server] %s" % message, flush=True)


class Blend:
    """The base model, held open, with whatever blend was last asked for on it.

    Everything below the load is template_code.py's attach/combine/_compact,
    with the two differences a long-lived process forces: the model is loaded
    once in load() rather than per blend, and reset() takes the previous blend
    off before the next one goes on.
    """

    def __init__(self, base_model, max_seq=MAX_SEQ, load_in_4bit=LOAD_IN_4BIT):
        self.base_model = base_model
        self.max_seq = max_seq
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None
        self.ranks = {}          # adapter name -> the rank PEFT gave it
        self.built = []          # this blend's adapters, in build order
        self.final = None
        self.builds = 0          # how many blends this server has been through
        self.ready = False
        self.cuda = False
        self._torch = None
        self._fast = None

    # --- startup -----------------------------------------------------------

    def load(self):
        """Load the base model, once. Everything after this is cheap by
        comparison, which is the entire reason this process exists."""
        started = time.perf_counter()
        # Unsloth patches transformers and peft as it loads, so it has to be
        # imported before them -- the same ordering rule the templates keep.
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
        import torch

        self._torch = torch
        self._fast = FastLanguageModel
        self.cuda = torch.cuda.is_available()
        _log("GPU available: %s" % self.cuda)

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model,
            max_seq_length=self.max_seq,
            dtype=None,
            load_in_4bit=self.load_in_4bit,
        )

        # Same chat template used during training, so inputs are formatted
        # identically -- and, as in the templates, left padding, because a
        # decoder continues from the last column of what it is given and a
        # right-padded row would be continuing from its own padding.
        self.tokenizer = get_chat_template(self.tokenizer, chat_template="qwen-2.5")
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Qwen ships max_length=32768 in its generation_config.json, and
        # transformers warns whenever that and max_new_tokens are both set.
        self.model.generation_config.max_length = None

        self.ready = True
        seconds = time.perf_counter() - started
        _log("loaded %s in %.1fs" % (self.base_model, seconds))
        return seconds

    # --- one blend ---------------------------------------------------------

    def reset(self):
        """Take the last blend off, and give its VRAM back. -> seconds.

        Deleted newest first, so a combined adapter goes before the inputs it
        was folded from. empty_cache() afterwards for the reason compact()
        calls it: the buffers are the caching allocator's now, and the other
        servers in a pool are separate processes that cannot see this one's
        free list.
        """
        if self.model is None or not self.built:
            self.built, self.ranks, self.final = [], {}, None
            return 0.0
        started = time.perf_counter()
        for name in reversed(self.built):
            try:
                self.model.delete_adapter(name)
            except Exception as error:          # already gone, or never made
                _log("could not delete adapter %s: %s" % (name, error))
        self.built, self.ranks, self.final = [], {}, None
        self._torch.cuda.empty_cache()
        return time.perf_counter() - started

    def _rank(self, adapter_dir):
        """The rank PEFT will allocate for this adapter, from its own config.

        template_code.py's _rank(), and PEFT's own bookkeeping: rank_pattern can
        raise the rank above r for individual modules.
        """
        with open(os.path.join(adapter_dir, "adapter_config.json"),
                  encoding="utf-8") as handle:
            config = json.load(handle)
        return max([config["r"]] + list((config.get("rank_pattern") or {}).values()))

    def attach(self, name, path):
        """Load the adapter at `path` under `name`. -> seconds."""
        started = time.perf_counter()
        from peft import PeftModel
        if isinstance(self.model, PeftModel):
            self.model.load_adapter(path, adapter_name=name)
        else:
            # The first adapter of this server's life is what turns the base
            # model into a PeftModel; after that the wrapper stays and only the
            # adapters on it come and go.
            self.model = PeftModel.from_pretrained(self.model, path,
                                                   adapter_name=name)
        self.ranks[name] = self._rank(path)
        self.built.append(name)
        return time.perf_counter() - started

    def compact(self, name):
        """Copy adapter `name`'s weights off whatever buffer they were sliced
        from -- template_code.py's _compact(), which see for the whole story.

        PEFT builds an svd node by handing back Vh[:new_rank, :], a *view* into
        a V sized by the delta weight rather than by the rank, which the adapter
        then pins for its whole life: ~10 GB across this stack for ~18 MB of
        weights. The slice is already contiguous, so only a copy releases the
        buffer behind it.
        """
        started = time.perf_counter()
        for module in self.model.modules():
            for store in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"):
                entry = getattr(module, store, None)
                if entry is None or name not in entry:
                    continue
                held = entry[name]
                weight = getattr(held, "weight", held)
                if weight.untyped_storage().nbytes() > weight.numel() * weight.element_size():
                    weight.data = weight.data.clone()
        self._torch.cuda.empty_cache()
        return time.perf_counter() - started

    def combine(self, name, combination_type, left, right):
        """Fold two adapters into one under `name`. -> (seconds, compact seconds).

        PEFT's rules (peft/tuners/lora/model.py, _check_add_weighted_adapter):
        cat sums the input ranks, svd takes the max, linear demands they match.
        The linear case is checked here so the failure names the node, exactly
        as it is checked in a generated script -- the client checks it too, from
        the ranks on disk, so a bad tree normally never reaches a server.
        """
        (left_name, left_weight), (right_name, right_weight) = left, right
        left_rank, right_rank = self.ranks[left_name], self.ranks[right_name]

        if combination_type == "linear" and left_rank != right_rank:
            raise ValueError(
                "%s: combination_type='linear' needs both inputs at the same rank, "
                "but %s is rank %d and %s is rank %d. cat sums its inputs' ranks, "
                "which is usually what pushes them apart."
                % (name, left_name, left_rank, right_name, right_rank))

        started = time.perf_counter()
        self.model.add_weighted_adapter(
            adapters=[left_name, right_name],
            weights=[left_weight, right_weight],
            adapter_name=name,
            combination_type=combination_type,
            # The thin SVD: the first new_rank singular vectors are identical
            # either way, so the full n x n V is only a bigger buffer to compute
            # and then throw away.
            svd_full_matrices=False,
        )
        self.built.append(name)

        compacted = self.compact(name) if combination_type == "svd" else None

        if combination_type == "cat":
            self.ranks[name] = left_rank + right_rank
        elif combination_type == "svd":
            self.ranks[name] = max(left_rank, right_rank)
        else:
            self.ranks[name] = left_rank
        return time.perf_counter() - started, compacted

    def build(self, slots, plan, final):
        """Put one blend on the model. -> {rank, final, timings, builds, cuda}.

        `plan` is the generated script's own build order, in the same post-order
        the template writes its attach()/combine() calls in, so what runs here
        is the plan that script describes and nothing else.

        The timings come back rather than being printed: the client re-prints
        them as its own TIMING: lines, which is what keeps phase_timings working
        unchanged. model_load simply stops appearing among them, which is the
        whole point.
        """
        timings = []
        seconds = self.reset()
        if seconds:
            timings.append({"phase": "reset", "seconds": seconds})

        for step in plan:
            operation = step.get("op")
            name = step["name"]
            if operation == "attach":
                slot = step["slot"]
                path = slots.get(slot)
                if not path:
                    raise ValueError("no path given for slot %s" % slot)
                timings.append({"phase": "attach", "seconds": self.attach(name, path),
                                "label": "%s=%s" % (name, slot)})
            elif operation == "combine":
                kind = step["type"]
                took, compacted = self.combine(
                    name, kind, tuple(step["left"]), tuple(step["right"]))
                if compacted is not None:
                    timings.append({"phase": "compact", "seconds": compacted,
                                    "label": name})
                timings.append({"phase": "combine.%s" % kind, "seconds": took,
                                "label": name})
            else:
                raise ValueError("unknown build step %r" % operation)

        if final not in self.ranks:
            raise ValueError("the plan never builds its final adapter %r" % final)

        started = time.perf_counter()
        self.model.set_adapter(final)
        # Unsloth's fast inference path, re-applied per blend: the adapters
        # under it have just changed.
        self._fast.for_inference(self.model)
        self.model.generation_config.max_length = None
        timings.append({"phase": "inference_setup",
                        "seconds": time.perf_counter() - started})

        self.final = final
        self.builds += 1
        return {"rank": self.ranks[final], "final": final, "timings": timings,
                "builds": self.builds, "cuda": self.cuda}

    def memory(self):
        """What this server is holding on the card, in bytes, or None.

        Asked of torch rather than of nvidia-smi because that is the only thing
        that can answer it per process: Windows' WDDM driver reports no
        per-process figure at all. It is on /health because the question a warm
        server has to keep answering is whether its resting cost is flat --
        build the same chromosome as the 1st and the 300th blend and watch this
        rather than guess.
        """
        if self._torch is None or not self.cuda:
            return None
        return {"allocated": self._torch.cuda.memory_allocated(),
                "reserved": self._torch.cuda.memory_reserved(),
                "peak": self._torch.cuda.max_memory_allocated()}

    # --- answering ---------------------------------------------------------

    def generate(self, prompts, max_new_tokens=250):
        """Answer several questions in one generate() call -- the templates'
        answer(), unchanged. -> (replies in order, seconds)."""
        if self.final is None:
            raise ValueError("no blend is built on this server yet; POST /build first")
        started = time.perf_counter()
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
            return_dict=True, padding=True
        ).to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                  do_sample=False,
                                  pad_token_id=self.tokenizer.pad_token_id)
        # Padded on the left, so every row's reply starts at the same column and
        # one width slices the prompt off all of them.
        width = inputs["input_ids"].shape[-1]
        replies = [self.tokenizer.decode(row[width:], skip_special_tokens=True).strip()
                   for row in out]
        return replies, time.perf_counter() - started


# --- the HTTP surface ------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    """The four endpoints. One request at a time -- see the module docstring."""

    protocol_version = "HTTP/1.1"
    blend = None                                # bound by serve()
    stopping = None

    def log_message(self, fmt, *args):          # one line per request, ours
        _log("%s %s" % (self.command, self.path))

    # -- plumbing --

    def _send(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    # -- endpoints --

    def do_GET(self):
        if self.path.split("?")[0] != "/health":
            return self._send(404, {"error": "no such endpoint: %s" % self.path})
        blend = self.blend
        self._send(200, {"ready": blend.ready, "base_model": blend.base_model,
                         "builds": blend.builds, "final": blend.final,
                         "cuda": blend.cuda, "pid": os.getpid(),
                         "memory": blend.memory()})

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/shutdown":
            self._send(200, {"stopping": True})
            self.stopping.append(True)
            return
        blend = self.blend
        if not blend.ready:
            return self._send(503, {"error": "the base model is still loading"})
        try:
            payload = self._body()
        except ValueError as error:
            return self._send(400, {"error": "bad JSON: %s" % error})

        try:
            if path == "/build":
                wanted = payload.get("base_model")
                # A pool pointed at one model and scripts built for another
                # would blend adapters onto the wrong base and then score it.
                # Say so instead: this is the one mismatch a client cannot see.
                if wanted and wanted != blend.base_model:
                    raise ValueError(
                        "this server holds %s but the script asks for %s"
                        % (blend.base_model, wanted))
                self._send(200, blend.build(payload.get("slots") or {},
                                            payload.get("plan") or [],
                                            payload.get("final")))
            elif path == "/generate":
                replies, seconds = blend.generate(
                    payload.get("prompts") or [],
                    int(payload.get("max_new_tokens") or 250))
                self._send(200, {"replies": replies, "seconds": seconds})
            else:
                self._send(404, {"error": "no such endpoint: %s" % path})
        except Exception as error:
            # Every failure is one client's failure, never the server's: the
            # next individual gets a clean build. The same rule the pipeline
            # keeps about an individual that crashes.
            #
            # The client is sent one line, because one line is all an
            # execution's stderr should carry about somebody else's process.
            # The traceback goes in this server's log, which is the only place
            # it can go: an OOM inside add_weighted_adapter is the sort of
            # thing you need the frames for, and the sweep that lost the
            # individual has no way to ask for them later.
            _log("%s failed: %s: %s\n%s"
                 % (path, type(error).__name__, error, traceback.format_exc()))
            self._send(400, {"error": "%s: %s" % (type(error).__name__, error)})


def serve(base_model, host="127.0.0.1", port=8770, max_seq=MAX_SEQ,
          load_in_4bit=LOAD_IN_4BIT):
    """Load the model, then answer requests until told to stop."""
    blend = Blend(base_model, max_seq, load_in_4bit)
    stopping = []

    handler = type("_Bound", (_Handler,), {"blend": blend, "stopping": stopping})
    server = HTTPServer((host, port), handler)
    # Up *before* the load, so the pool can tell "still loading" from "never
    # started" -- /health answers all the way through the load, and a load that
    # takes twenty minutes is not a startup that failed.
    _log("listening on http://%s:%d" % (host, port))
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.2},
                     daemon=True).start()

    try:
        blend.load()
    except Exception as error:
        _log("could not load %s: %s" % (base_model, error))
        server.shutdown()
        raise
    _log("ready")

    try:
        while not stopping:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    _log("stopping after %d build(s)" % blend.builds)
    server.shutdown()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="One base model, held open, blending LoRAs on request.")
    parser.add_argument("--base-model", required=True,
                        help="the model every adapter was trained on")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--max-seq", type=int, default=MAX_SEQ,
                        help="context length to load with (default %d)" % MAX_SEQ)
    parser.add_argument("--no-4bit", action="store_true",
                        help="load at full precision instead of 4-bit")
    args = parser.parse_args(argv)
    return serve(args.base_model, args.host, args.port, args.max_seq,
                 not args.no_4bit)


if __name__ == "__main__":
    sys.exit(main())
