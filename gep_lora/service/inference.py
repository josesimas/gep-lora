"""
inference.py - Live blends: loaded on first use, cached, streamed from.

A deployment's first request loads its base model and builds its blend; the
model then stays loaded for MODEL_TTL seconds after its *last* request and is
unloaded by a janitor thread when that passes. The next request after that
pays the load again.

The cache is keyed by what a load costs -- (engine, base model, chat template)
-- and not by deployment: two live individuals of the same base model share one
loaded model, and switching between them costs a blend rebuild (a few seconds
of attach/combine) rather than a model load. A loaded model answers one request
at a time, for the reason a lora server does: two blends built on one model at
once would be fighting over the same adapters.

The blend arithmetic is not repeated here. `UnslothEngine` holds a
`gep_lora.core.blends.lora_server.Blend` -- the class the lora servers build every search
blend with -- and adds only what a server does not do: stream the tokens out
as they are generated, and unload.

`MockEngine` is what a deployment from a mocked sweep gets: no model, a
made-up answer streamed a word at a time, so the whole API runs on a machine
with no GPU the way a mocked sweep does.
"""

import gc
import threading
import time

from gep_lora.service import settings


class MockEngine:
    """Answers without a model, for deployments of mocked sweeps."""

    delay = 0.01

    def __init__(self, spec):
        self.spec = spec
        self.loaded = False
        self.built = None

    def load(self):
        self.loaded = True

    def build(self, spec):
        self.spec = spec
        self.built = spec["chromosome"]

    def stream(self, prompt, max_new_tokens):
        words = ("[mock %s] You asked: %s. A real deployment would answer here."
                 % (self.built, prompt.strip())).split()
        for index, word in enumerate(words[:max_new_tokens]):
            time.sleep(self.delay)
            yield word if index == 0 else " " + word

    def unload(self):
        self.loaded = False


class UnslothEngine:
    """A base model held by a lora_server Blend, with streaming on top."""

    def __init__(self, spec):
        from gep_lora.core.blends.lora_server import Blend      # imports nothing heavy until load()
        self.blend = Blend(spec["base_model"], chat_template=spec["chat_template"])
        self.built = None

    def load(self):
        self.blend.load()

    def build(self, spec):
        self.blend.build(spec["slots"], spec["plan"], spec["final"])
        self.built = spec["chromosome"]

    def stream(self, prompt, max_new_tokens):
        # After load(): unsloth has to be imported before transformers.
        from transformers import TextIteratorStreamer
        blend = self.blend
        text = blend.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            tokenize=False)
        inputs = blend.text_tokenizer([text], return_tensors="pt",
                                      add_special_tokens=False).to(blend.model.device)
        streamer = TextIteratorStreamer(blend.text_tokenizer, skip_prompt=True,
                                        skip_special_tokens=True)
        failure = []

        def generate():
            try:
                blend.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                     do_sample=False, streamer=streamer,
                                     pad_token_id=blend.text_tokenizer.pad_token_id)
            except Exception as error:             # noqa: BLE001 - reported below
                failure.append(error)
                streamer.end()

        worker = threading.Thread(target=generate, daemon=True)
        worker.start()
        try:
            for piece in streamer:
                if piece:
                    yield piece
        finally:
            # Hold the model until generate() is really done, even when the
            # client has gone: the next request must not start on top of it.
            worker.join()
        if failure:
            raise failure[0]

    def unload(self):
        blend = self.blend
        torch = blend._torch
        blend.model = blend.tokenizer = blend.text_tokenizer = None
        blend.ready, blend.final, blend.built, blend.ranks = False, None, [], {}
        gc.collect()
        if torch is not None and blend.cuda:
            torch.cuda.empty_cache()


ENGINES = {"mock": MockEngine, "unsloth": UnslothEngine}


class _Entry:
    def __init__(self, engine):
        self.engine = engine
        self.lock = threading.Lock()
        self.loaded = False
        self.built_for = None          # the deployment whose blend is on it
        self.last_used = time.monotonic()
        self.busy = 0


class ModelCache:
    """Loaded models, by (engine, base model, chat template), with a TTL."""

    def __init__(self, ttl=None, engines=None, sweep_every=None):
        self.ttl = settings.MODEL_TTL if ttl is None else ttl
        self.engines = engines or ENGINES
        self._entries = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        every = sweep_every if sweep_every is not None else max(1.0, min(30.0, self.ttl / 5))
        self._janitor = threading.Thread(target=self._sweep_loop, args=(every,), daemon=True)
        self._janitor.start()

    @staticmethod
    def key(spec):
        return (spec["engine"], spec["base_model"], spec["chat_template"])

    def _entry(self, spec):
        key = self.key(spec)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                entry = self._entries[key] = _Entry(self.engines[spec["engine"]](spec))
            entry.busy += 1
            return entry

    def stream(self, deployment_id, spec, prompt, max_new_tokens):
        """Yield the answer's text pieces. Loads and builds as needed.

        The load and the build happen before the first piece is yielded -- in
        the generator's first next() -- so a caller that wants to fail with a
        status code rather than mid-stream calls next() once before sending
        headers.
        """
        entry = self._entry(spec)
        try:
            with entry.lock:
                if not entry.loaded:
                    try:
                        entry.engine.load()
                    except BaseException:
                        self._drop(entry)
                        raise
                    entry.loaded = True
                    entry.built_for = None
                if entry.built_for != deployment_id:
                    entry.built_for = None
                    entry.engine.build(spec)
                    entry.built_for = deployment_id
                yield ""                               # loaded and built: headers may go
                for piece in entry.engine.stream(prompt, max_new_tokens):
                    yield piece
                entry.last_used = time.monotonic()
        finally:
            with self._lock:
                entry.busy -= 1
                entry.last_used = time.monotonic()

    def _drop(self, entry):
        with self._lock:
            for key, held in list(self._entries.items()):
                if held is entry:
                    del self._entries[key]

    def forget(self, deployment_id):
        """A deployment went away: its blend must not be served again."""
        with self._lock:
            entries = list(self._entries.values())
        for entry in entries:
            if entry.built_for == deployment_id:
                entry.built_for = None

    def sweep(self, now=None):
        """Unload every idle model past its TTL. -> how many."""
        now = time.monotonic() if now is None else now
        with self._lock:
            stale = [(key, entry) for key, entry in self._entries.items()
                     if entry.busy == 0 and now - entry.last_used >= self.ttl]
            for key, _ in stale:
                del self._entries[key]
        for _, entry in stale:
            with entry.lock:
                if entry.loaded:
                    entry.engine.unload()
                    entry.loaded = False
        return len(stale)

    def status(self):
        now = time.monotonic()
        with self._lock:
            return [{"engine": key[0], "base_model": key[1], "chat_template": key[2],
                     "loaded": entry.loaded, "built_for": entry.built_for,
                     "busy": entry.busy,
                     "unloads_in": max(0.0, self.ttl - (now - entry.last_used))}
                    for key, entry in self._entries.items()]

    def _sweep_loop(self, every):
        while not self._stop.wait(every):
            try:
                self.sweep()
            except Exception:                          # noqa: BLE001 - keep sweeping
                pass

    def close(self):
        self._stop.set()
        self.sweep(now=float("inf"))
