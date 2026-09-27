"""
golive.py - Putting a finished job's individual up for inference.

Going live is two things, kept apart:

  * **the blend spec** -- what to build, derived once from the job's database
    at the moment it goes live and stored with the deployment: the base model,
    the chat template, the adapter folders, the build plan with every
    weight already a number, and the final adapter. It is the same plan a
    generated `template_remote_code.py` script sends a lora server's /build,
    so a live blend is built by the code that built it during the search. A
    deployment is then self-contained: it does not re-read the job database
    for every request, and a later edit of settings.py cannot change what a
    token serves.

  * **the target** -- where it goes live. A target publishes a spec, streams
    answers from it and retires it. `local` is the one that exists: the model
    is loaded into the API server's own process on first use and cached for
    MODEL_TTL (see inference.py). Another target -- a remote inference server,
    an upload somewhere -- is a class with the same three methods added to
    TARGETS; the token, the registry row and the endpoints do not change.

The weights are drawn from the individual's own weight seed exactly as its
script drew them (`random.Random(seed)`, ten draws, zero excluded), rather than
read from the execution's `weights` column, which holds the four-decimal
figures the script printed.
"""

import random

from blends import generate_runs
from blends import process_run
from blends import server_pool
from search.generate_population import VARIABLES, decode
import start_run
from storage import store

from async_api import inference
from async_api import results

WEIGHT_NAMES = VARIABLES      # the templates draw all of them, in this order


class GoLiveError(ValueError):
    """Why this individual cannot go live."""


def draw_weights(seed):
    """The template's own draw: {w1..w10}, each in (0, 1)."""
    rng = random.Random(seed)

    def one():
        value = 0.0
        while value == 0.0:
            value = rng.random()
        return value

    return {name: one() for name in WEIGHT_NAMES}


def _weight_of(expression, weights):
    """A plan Step's weight expression ('WEIGHTS["w3"]' or '1.0') as a number."""
    for name in WEIGHT_NAMES:
        if expression == 'WEIGHTS["%s"]' % name:
            return weights[name]
    return float(expression)


def build_plan(chromosome, ranks, weights):
    """The lora_server /build plan for one chromosome. -> (plan, final, rank)."""
    steps, final = generate_runs.plan(decode(chromosome)[0], ranks)
    broken = [step.name for step in steps if step.broken]
    if broken:
        raise GoLiveError("%s cannot be built: linear over mismatched ranks at %s"
                          % (chromosome, ", ".join(broken)))
    # Every attach, then every combine, each in post-order: the order a
    # generated script's ATTACH_LEAVES and COMBINE_NODES blocks run in. A leaf's
    # weight is carried by the combine above it, not by its attach.
    plan = [{"op": "attach", "name": step.name, "slot": step.symbol}
            for step in steps if step.kind == "leaf"]
    plan += [{"op": "combine", "name": step.name,
              "type": generate_runs.COMBINATION_TYPE[step.symbol],
              "left": [step.left[0], _weight_of(step.left[1], weights)],
              "right": [step.right[0], _weight_of(step.right[1], weights)]}
             for step in steps if step.kind == "combine"]
    return plan, final, steps[-1].rank


def blend_spec(db_path, run_id, number=None):
    """The spec for one individual of a finished sweep (the best by default)."""
    try:
        detail = results.detail(db_path, run_id)
    except results.NoResults:
        raise GoLiveError("the job's database is gone")
    population = detail["population"]
    if number is None:
        chosen = detail["best"]
        if chosen is None:
            raise GoLiveError("the sweep has no scored individual to put live")
    else:
        chosen = next((one for one in population if one["number"] == number), None)
        if chosen is None:
            raise GoLiveError("the sweep holds no individual %d" % number)
    if chosen["state"] == "BAD":
        raise GoLiveError("individual %d is BAD: PEFT cannot build it" % chosen["number"])

    conf = detail["settings"]
    conn = store.connect(db_path)
    try:
        row = next(one for one in store.individuals(conn, run_id)
                   if one["number"] == chosen["number"])
        source = row["script_source"] or ""
        seed = row["weight_seed"]
        if seed is None:
            seed = row["weight_pin"]            # a blend saved by hand keeps its own
    finally:
        conn.close()
    if seed is None:
        # Never rendered: derive it the way the runs step would have.
        seed = random.Random("%s:%d" % (conf["WEIGHT_MASTER_SEED"], chosen["number"])
                             ).randrange(start_run._SEED_LIMIT)

    slots = generate_runs.lora_slots(conf.get("LORA_SLOTS"))
    try:
        ranks = generate_runs.slot_ranks(slots)
    except SystemExit as error:
        raise GoLiveError(str(error))
    weights = draw_weights(seed)
    plan, final, rank = build_plan(chosen["chromosome"], ranks, weights)
    # What the search ran decides how it is served: a mocked sweep never loaded
    # a model, so its deployment does not either. A test on the script, as the
    # process step makes it, not on what settings.py says now.
    mocked = bool(source) and not (process_run.imports_unsloth(source)
                                   or server_pool.wanted(source))
    return {"number": chosen["number"], "chromosome": chosen["chromosome"],
            "fitness": chosen["fitness"], "quality": chosen["quality"],
            "base_model": generate_runs.base_model_name(conf.get("BASE_MODEL")),
            "chat_template": generate_runs.chat_template_name(conf),
            "slots": slots, "plan": plan, "final": final, "rank": rank,
            "weight_seed": seed, "weights": weights,
            "engine": "mock" if mocked else "unsloth"}


class LocalTarget:
    """Served from this process, through the shared model cache."""

    name = "local"

    def __init__(self, cache):
        self.cache = cache

    def publish(self, deployment_id, spec):
        """Nothing to upload; the model loads on the first request."""
        return {"loaded_on": "first request", "ttl_seconds": self.cache.ttl}

    def stream(self, deployment_id, spec, prompt, max_new_tokens):
        return self.cache.stream(deployment_id, spec, prompt, max_new_tokens)

    def retire(self, deployment_id, spec):
        self.cache.forget(deployment_id)


TARGETS = {"local": LocalTarget}


def targets(cache=None):
    """One instance of every target, sharing one model cache."""
    cache = cache or inference.ModelCache()
    return {name: kind(cache) for name, kind in TARGETS.items()}
