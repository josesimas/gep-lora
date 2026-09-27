"""
drawn.py - A blend drawn by hand, run as a sweep of one and verified.

The visual guide (visual_guide.html) lets a person draw a blend instead of
searching for one: a tree of their own LoRAs folded together by CAT, SVD and
LIN, each LoRA at a weight. This module is the API's half of that -- what the
drawing *is* in the pipeline's terms, whether it can be built, and the job
that tests it -- and invents none of the pipeline:

  * **A drawing is a chromosome.** `encode()` turns the page's tree into the
    K-expression and LORA_SLOTS a search would have held it as, through
    generate_population's own Node and encode(), and `check()` puts it through
    generate_population.check() and generate_runs.plan() -- the grammar and the
    rank rule every individual meets. A LIN over mismatched ranks is BAD here
    for the reason it is BAD in a sweep.
  * **Its weights are a seed's.** A leaf names a weight symbol (w1..w10), and
    what the symbols are worth is the draw a sweep makes for individual
    `number` (NUMBER, 1, unless said) under its WEIGHT_MASTER_SEED --
    start_run's derivation, golive.draw_weights()'s values. So the page shows
    the numbers the scripts will use, a new seed is a new draw, and the sweep
    stored for it reproduces the blend as any sweep does. A searched blend
    opened as a drawing (`opened()`) keeps its search's seed *and its own
    number*, which is what makes its weights the ones it was scored with.
  * **Testing it is a verification.** `create()` writes a prepared sweep with
    submit.prepare() -- the chosen dataset as its training split, COUNT 1 --
    puts the one individual in through start_run's own `trees` and `runs`
    steps, and settles the job `done` (registry.BLEND: never queued, never
    requeued). Then it queues a verification of that individual on those
    questions, which the worker runs with
    testing/evaluate_chromosome_against_loras.py: the blend, and each LoRA it
    uses alone, answering the same questions under the sweep's evaluator. The
    page reads it back with GET /verifications/{id}, as any verification.

The tree, as the page and the chat hold it:

    {"op": "CAT", "children": [<node or null>, <node or null>]}
    {"lora": <catalogue id>, "weight": "w3"}

A null child is a place not filled yet. The top may be anything a place may
hold, as a chromosome's root may: a fold, one LoRA on its own (a blend of one,
at full strength -- no fold above it applies its weight), or null, a drawing
with nothing in it yet. A new drawing starts from an empty CAT, since most
blends fold something, but that is where it starts rather than a rule. The same LoRA may appear at several leaves, at the same weight or not;
it takes one slot, L1..Ln numbered in the order the drawing is read (level
order, left to right), which is the order a Karva expression is written in.
"""

import json
import random
import shutil

from adapters import catalog as lora_catalog
from async_api import golive
from async_api import registry as reg
from async_api import results
from async_api import submit
from async_api import verify
from blends import generate_runs
from config import settings as config
from search.generate_population import (
    BINARY_OPS, MAX_SLOTS, UNARY_OPS, VARIABLES, Node, check as check_expression,
    encode as encode_tree, decode, random_tree, slot_key)
from storage import db_datasets
from storage import store

# The fold a new drawing starts from (see the module note).
ROOT = "CAT"

# The one individual a drawn blend's sweep holds, unless it was opened from a
# search and keeps its number there. Its number is what the weight seed is
# derived from, as for any individual of any sweep.
NUMBER = 1
MAX_NUMBER = 10 ** 9

# A drawing is for a person to read and a GPU to build: past this many leaves
# it is neither, and the page says so rather than the worker failing later.
MAX_LEAVES = 16

# The most questions a test asks, and how many by default.
MAX_QUESTIONS = 200
DEFAULT_QUESTIONS = 20

MOCKED_TEMPLATE = "template_code_mocked.py"

# The settings a test may ask for beside what the drawing decides: the rubric
# and the judge. Everything else is the drawing's, or config/settings.py's.
TEST_SETTINGS = ("EVALUATOR", "JUDGE_BACKEND", "JUDGE_MODEL", "JUDGE_BASE_URL",
                 "COMPOSITE_EVALUATORS", "COMPOSITE_AGGREGATE", "PANEL_MODELS")

# PEFT's names for the three folds, and what each does to the ranks below it.
FOLDS = {"CAT": "cat: sums the ranks", "SVD": "svd: the larger rank",
         "LIN": "linear: needs equal ranks"}


class DrawnError(ValueError):
    """What is wrong with a drawing or a test of one, in words for the person."""


# --- the drawing -------------------------------------------------------------


def _path(parent, index):
    return str(index) if parent == "" else "%s.%d" % (parent, index)


def place(path):
    """A path as a person reads it: "the top", "left", "right › left"."""
    if not path:
        return "the top"
    return " › ".join("left" if part == "0" else "right" for part in path.split("."))


def walk(tree):
    """Every place in a drawing, checked. -> [(path, node or None)] in level
    order; the root's path is "" and a child's is its parent's plus its index,
    so "1.0" is the root's second child's first. The top is checked like any
    other place: a fold, a LoRA, or None for a drawing still empty. Raises
    DrawnError."""
    places, frontier, leaves = [], [("", tree)], 0
    while frontier:
        path, node = frontier.pop(0)
        places.append((path, node))
        if node is None:
            continue
        if not isinstance(node, dict):
            raise DrawnError("the node at %s is not a node" % (path or "the top"))
        if "op" in node:
            if node["op"] not in BINARY_OPS:
                raise DrawnError("%r is not a fold; there are %s"
                                 % (node["op"], ", ".join(BINARY_OPS)))
            children = node.get("children")
            if not isinstance(children, list) or len(children) != 2:
                raise DrawnError("a %s folds exactly two things" % node["op"])
            frontier.extend((_path(path, index), child) for index, child in enumerate(children))
        elif "lora" in node:
            leaves += 1
            lora = node["lora"]
            if isinstance(lora, bool) or not isinstance(lora, int):
                raise DrawnError("a leaf names one of your LoRAs by its id")
            if node.get("weight") not in VARIABLES:
                raise DrawnError("a leaf's weight is one of %s..%s"
                                 % (VARIABLES[0], VARIABLES[-1]))
        else:
            raise DrawnError("the node at %s is neither a fold nor a LoRA" % (path or "the top"))
    if leaves > MAX_LEAVES:
        raise DrawnError("a drawing holds at most %d LoRAs; this one has %d"
                         % (MAX_LEAVES, leaves))
    return places


def loras_in(tree):
    """The LoRA ids a drawing uses, each once, in the order it is read."""
    seen = []
    for _, node in walk(tree):
        if node is not None and "lora" in node and node["lora"] not in seen:
            seen.append(node["lora"])
    return seen


def encode(tree, slot_of=None):
    """A complete drawing as a sweep holds it. -> (chromosome, {slot: lora id}).

    Slots are handed out in reading order; a LoRA used twice keeps its slot.
    `slot_of` ({lora id: slot}) gives them instead -- a run's own LORA_SLOTS,
    for a drawing saved into that run. Raises DrawnError for a place left
    empty or a drawing past MAX_SLOTS LoRAs.
    """
    places = walk(tree)
    empty = [path for path, node in places if node is None]
    if empty:
        raise DrawnError("the drawing has %d empty place(s) left to fill" % len(empty))
    ids = loras_in(tree)
    if len(ids) > MAX_SLOTS:
        raise DrawnError("a blend uses at most %d different LoRAs" % MAX_SLOTS)
    if slot_of is None:
        slot_of = {lora: UNARY_OPS[index] for index, lora in enumerate(ids)}
    else:
        slot_of = {lora: slot_of[lora] for lora in ids}

    def build(node):
        if "lora" in node:
            leaf = Node(slot_of[node["lora"]])
            leaf.children.append(Node(node["weight"]))
            return leaf
        made = Node(node["op"])
        made.children.extend(build(child) for child in node["children"])
        return made

    chromosome = encode_tree(build(tree))
    check_expression(chromosome)            # the grammar's own word on it
    return chromosome, {slot: lora for lora, slot in slot_of.items()}


def decode_tree(chromosome, slots):
    """A sweep's chromosome as a drawing, {slot: lora id} naming the LoRAs:
    the way back, so a drawn blend (or a searched one) can be opened again."""
    root, _ = decode(chromosome)

    def said(node):
        if node.symbol in UNARY_OPS:
            return {"lora": slots[node.symbol], "weight": node.children[0].symbol}
        return {"op": node.symbol, "children": [said(child) for child in node.children]}
    return said(root)


# --- the weights ---------------------------------------------------------------


def pin_of(value):
    """A pinned weight seed, or None: the draw a blend saved by hand keeps."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DrawnError("pin is a weight seed, a whole number, 0 or more")
    return value


def number_of(value):
    """The individual a drawing's weights are drawn for: NUMBER when not said."""
    if value is None:
        return NUMBER
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_NUMBER:
        raise DrawnError("number is the individual the weights are drawn for, 1 or more")
    return value


def weight_seed(master, number=NUMBER):
    """The seed individual `number` draws its weights from under `master` -- the
    derivation start_run.step_runs makes for every individual."""
    import start_run                # the pipeline driver: imported where it is needed
    return random.Random("%s:%d" % (master, number)).randrange(start_run._SEED_LIMIT)


def draw(master=None, number=None, pin=None):
    """-> {seed, number, pin, weight_seed, weights}: what w1..w10 are worth for
    individual `number` under `master`, a new master seed when None -- drawn as
    a sweep draws one it was not given. A `pin` is a weight seed kept instead
    of that derivation (a blend saved by hand, as start_run.step_runs keeps it)."""
    import start_run
    number, pin = number_of(number), pin_of(pin)
    if master is None:
        master = random.randrange(start_run._SEED_LIMIT)
    if isinstance(master, bool) or not isinstance(master, int) or master < 0:
        raise DrawnError("the seed is a whole number, 0 or more")
    seed = pin if pin is not None else weight_seed(master, number)
    return {"seed": master, "number": number, "pin": pin, "weight_seed": seed,
            "weights": {name: round(value, 4)
                        for name, value in golive.draw_weights(seed).items()}}


# A random drawing is redrawn until it can be built, at most this many times.
RANDOM_ATTEMPTS = 60


# --- is it buildable -------------------------------------------------------------


def _rows(catalog, user, ids):
    """The user's own ready catalogue rows for `ids`. Raises DrawnError."""
    rows = {}
    for lora in ids:
        row = submit.own_lora(catalog, user, lora)
        if row is None:
            raise DrawnError("no LoRA of yours with id %s" % lora)
        if row["status"] != lora_catalog.READY:
            raise DrawnError("%s is %s, not ready" % (row["name"], row["status"]))
        rows[lora] = row
    return rows


def _together(rows):
    """LoRAs that can be built into one model: one base model, one chat template."""
    models = sorted({row["base_model"] or "?" for row in rows})
    if len(models) > 1:
        return ("those LoRAs were trained on different base models (%s); a blend "
                "needs them all on one" % ", ".join(models))
    templates = sorted({str(row["chat_template"]) for row in rows})
    if len(templates) > 1:
        return ("those LoRAs were trained under different chat templates (%s); a "
                "blend needs one" % ", ".join(templates))
    return None


def _post_order(tree):
    """The paths of a complete drawing's L*/fold nodes in the order
    generate_runs.plan() builds them, so its steps can be put back on the
    drawing one for one."""
    out = []

    def visit(node, path):
        if "op" in node:
            for index, child in enumerate(node["children"]):
                visit(child, _path(path, index))
        out.append(path)
    visit(tree, "")
    return out


def check(catalog, user, tree, seed=None, number=None, pin=None):
    """Everything the page draws beside a drawing, and whether it can be built.

    -> {complete, empty, state, problems, chromosome, slots, nodes, rank,
        formula, loras, base_model, chat_template, mock, seed, number,
        weight_seed, weights}. `seed` and `number` say which draw the weights
        are (see draw()). `state` is "incomplete" while a place is empty, "BAD" when
    PEFT cannot build it (the rank rule, or LoRAs that do not go together) and
    "ok" when it can. `nodes` is {path: {rank, broken, fold}} -- a leaf's rank
    is its adapter's, a fold's what PEFT gives it. Raises DrawnError only for
    a drawing that is not a drawing; anything else is a problem it reports.
    """
    places = walk(tree)
    drawn = draw(seed, number, pin)
    ids = loras_in(tree)
    rows = _rows(catalog, user, ids)
    empty = [path for path, node in places if node is None]
    problems = []
    clash = _together(list(rows.values())) if rows else None
    if clash:
        problems.append(clash)
    if len(ids) > MAX_SLOTS:
        problems.append("a blend uses at most %d different LoRAs" % MAX_SLOTS)
    first = next(iter(rows.values()), None)
    out = {"complete": not empty, "empty": empty, "state": "incomplete",
           "problems": problems, "chromosome": None, "slots": {}, "nodes": {},
           "rank": None, "formula": None,
           "loras": {str(lora): {"id": lora, "name": row["name"], "rank": row["rank"],
                                 "practice_run": bool(row["mock"])}
                     for lora, row in rows.items()},
           "base_model": first["base_model"] if first else None,
           "chat_template": first["chat_template"] if first else None,
           "mock": any(row["mock"] for row in rows.values())}
    out.update(drawn)

    # A leaf's rank is known before the drawing is whole; a fold's is not.
    ranks_by_id = {}
    for lora, row in rows.items():
        try:
            ranks_by_id[lora] = generate_runs.slot_ranks({"L1": row["folder"]})["L1"]
        except SystemExit as error:
            problems.append("%s: %s" % (row["name"], error))
    for path, node in places:
        if node is not None and "lora" in node and node["lora"] in ranks_by_id:
            out["nodes"][path] = {"rank": ranks_by_id[node["lora"]], "broken": False}
    if empty or len(ids) > MAX_SLOTS or len(ranks_by_id) != len(rows):
        out["state"] = "BAD" if (problems and not empty) else "incomplete"
        return out

    chromosome, slots = encode(tree)
    steps, _ = generate_runs.plan(decode(chromosome)[0],
                                  {slot: ranks_by_id[lora] for slot, lora in slots.items()})
    for path, step in zip(_post_order(tree), steps):
        out["nodes"][path] = {"rank": step.rank, "broken": step.broken,
                              "fold": FOLDS.get(step.symbol)}
    for path, step in zip(_post_order(tree), steps):
        if step.broken:
            problems.append("the LIN at %s mixes rank %d and rank %d, which PEFT cannot do "
                            "-- make both sides the same rank, or use CAT or SVD there"
                            % (place(path), step.left[2], step.right[2]))
    from async_api_agent import blending          # the chromosome in words
    names = {slot: rows[lora]["name"] for slot, lora in slots.items()}
    out.update(chromosome=chromosome, rank=steps[-1].rank,
               slots={slot: {"id": lora, "name": rows[lora]["name"]}
                      for slot, lora in slots.items()},
               formula=blending.formula(chromosome, names, drawn["weights"]),
               state="BAD" if problems else "ok")
    return out


# --- a random one ------------------------------------------------------------


def random_drawing(catalog, user, base_model=None, rng=None):
    """A drawing the way a search draws an individual. -> {tree, seed, check}.

    Up to MAX_SLOTS of the user's ready LoRAs on `base_model` (else on the base
    model most of them are on) under one chat template, shuffled into slots,
    and a tree grown by generate_population.random_tree() under config's
    MAX_DEPTH, BRANCH_PROB and ROOT_LEAF_PROB -- the search's own draw, so
    what comes out is something a first population could hold. Drawn again,
    up to RANDOM_ATTEMPTS times, until check() says it can be built (a LIN
    over two ranks is BAD here as in a sweep); the last one otherwise. A new
    weight seed each time. Raises DrawnError when there is nothing to draw."""
    rng = rng or random.Random()
    rows = catalog.all(owner=user["name"], status=lora_catalog.READY)
    groups = {}
    for row in rows:
        groups.setdefault((row["base_model"], str(row["chat_template"])), []).append(row)
    if base_model:
        groups = {key: value for key, value in groups.items() if key[0] == base_model}
    if not groups:
        raise DrawnError("you have no ready LoRAs%s to draw a blend from"
                         % (" on %s" % base_model if base_model else ""))
    pool = max(groups.values(), key=len)
    last = None
    for _ in range(RANDOM_ATTEMPTS):
        chosen = rng.sample(pool, min(len(pool), MAX_SLOTS))
        slots = {UNARY_OPS[index]: row["id"] for index, row in enumerate(chosen)}
        expression = encode_tree(random_tree(rng, rng.randint(1, config.MAX_DEPTH),
                                             config.BRANCH_PROB, len(chosen),
                                             config.ROOT_LEAF_PROB))
        tree = decode_tree(expression, slots)
        try:
            found = check(catalog, user, tree)
        except DrawnError:              # past MAX_LEAVES: draw again
            continue
        last = {"tree": tree, "seed": found["seed"], "check": found}
        if found["state"] == "ok":
            break
    if last is None:
        raise DrawnError("could not draw a blend small enough to show")
    return last


# --- testing it --------------------------------------------------------------


def _count(value):
    if value in (None, ""):
        return DEFAULT_QUESTIONS
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_QUESTIONS:
        raise DrawnError("count is how many questions, from 1 to %d" % MAX_QUESTIONS)
    return value


def dataset_lines(value, lora_path):
    """The questions a test asks, as the lines of a dataset file. -> (lines, label).

    {"file": name} a shared dataset; {"lora": id} the data one of the user's own
    LoRAs was trained on -- `lora_path(value)` is the server's, which knows whose
    is whose, and returns (path, label); {"text", "name"?} pasted or uploaded.
    """
    if not isinstance(value, dict):
        raise DrawnError('the dataset is {"file": name}, {"lora": id} or {"text": ...}')
    if set(value) == {"file"}:
        try:
            return submit.dataset_lines("training", value), value["file"]
        except submit.SubmissionError as error:
            raise DrawnError(str(error))
    if set(value) == {"lora"}:
        try:
            path, label = lora_path(value)
        except verify.VerifyError as error:
            raise DrawnError(str(error))
        with open(path, encoding="utf-8-sig", errors="replace") as handle:
            return [line.strip() for line in handle if line.strip()], label
    text = value.get("text")
    if isinstance(text, str) and set(value) <= {"text", "name"}:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        name = value.get("name") if isinstance(value.get("name"), str) else None
        return lines, name or "the dataset you gave"
    raise DrawnError('the dataset is {"file": name}, {"lora": id} or {"text": ...}')


def _label(payload, found):
    label = payload.get("label")
    if label is not None and not isinstance(label, str):
        raise DrawnError("label must be a string")
    if label and label.strip():
        return label.strip()[:80]
    return "drawn: %s" % (found["formula"] or found["chromosome"])[:72]


def create(registry, catalog, user, payload, lora_path):
    """A test of a drawing: the sweep of one, and its verification queued.

    -> (job row, verification row, check()). The body:

        {"tree": <drawing>, "seed": n?, "number": n?, "dataset": {...},
         "count": n?, "label"?, "mock"?: bool, "settings"?: {EVALUATOR, JUDGE_*...}}

    `number` is the individual the sweep holds it as (NUMBER unless said), so a
    blend opened from a search is tested with the weights it was scored with.

    Refused before anything is written unless the drawing is complete and
    buildable. The job is the owner's like any other, and deleting its run
    takes the verification with it.
    """
    if not isinstance(payload, dict):
        raise DrawnError("the body is a JSON object")
    unknown = sorted(set(payload) - {"tree", "seed", "number", "pin", "dataset", "count",
                                     "label", "mock", "settings"})
    if unknown:
        raise DrawnError("unknown field(s): %s" % ", ".join(unknown))
    seed = payload.get("seed")
    found = check(catalog, user, payload.get("tree"), seed, payload.get("number"),
                  payload.get("pin"))
    if found["state"] != "ok":
        raise DrawnError("; ".join(found["problems"]) or "the drawing is not finished yet")
    count = _count(payload.get("count"))
    lines, source = dataset_lines(payload.get("dataset"), lora_path)
    if not lines:
        raise DrawnError("that dataset is empty")
    extra = payload.get("settings") or {}
    if not isinstance(extra, dict) or set(extra) - set(TEST_SETTINGS):
        raise DrawnError("settings may name only: %s" % ", ".join(TEST_SETTINGS))

    rows = _rows(catalog, user, loras_in(payload["tree"]))
    mocked = bool(payload.get("mock")) or found["mock"]
    first = next(iter(rows.values()))
    wanted = dict(extra)
    wanted.update({
        "LORA_SLOTS": {slot: one["id"] for slot, one in found["slots"].items()},
        "BASE_MODEL": first["base_model"] or config.BASE_MODEL,
        "CHAT_TEMPLATE": first["chat_template"],
        "COUNT": 1, "GENERATIONS": 1,
        "WEIGHT_MASTER_SEED": found["seed"],
        "TRAINING_COUNT": min(count, len(lines))})
    if mocked:
        wanted["TEMPLATE"] = MOCKED_TEMPLATE
    body = {"label": _label(payload, found), "settings": wanted,
            "datasets": {"training": "\n".join(lines)}, "options": {"no_test": True}}

    def settle(job, run_id, conf):
        _one_individual(registry.database(job), run_id, found["chromosome"],
                        found["number"], found["pin"])
        return registry.settle_drawn(job["id"], run_id)

    try:
        job = submit.prepare(registry, user, body, settle)
    except submit.SubmissionError as error:
        raise DrawnError(str(error))
    try:
        offered = verify.choices(registry.database(job), job["run_id"])
        number, options = verify.options_for(
            {"individual": found["number"], "split": "training",
             "count": min(count, len(lines))},
            offered)
        options["dataset_label"] = source
        verification = registry.add_verification(job, number, found["chromosome"], options)
    except BaseException as error:
        # A test that cannot be queued leaves no run behind: the job exists only
        # to be verified.
        registry.delete_job(job["id"])
        shutil.rmtree(registry.folder(job), ignore_errors=True)
        if isinstance(error, verify.VerifyError):
            raise DrawnError(str(error))
        raise
    return job, verification, found


def _one_individual(db_path, run_id, chromosome, number=NUMBER, pin=None):
    """Put the drawing in as the sweep's only individual, numbered `number`, through start_run's
    own `trees` and `runs` steps -- its tree, state, rank, weight seed and
    script are the ones a searched individual gets -- and leave no script
    file behind: the database is the sweep."""
    import start_run
    conn = store.connect(db_path)
    try:
        store.add_individuals(conn, run_id, [chromosome], first=number, weight_pin=pin)
        conf = store.get_settings(conn, run_id)
        context = start_run.Context(conn, run_id, conf, db_datasets.run_folder(conn, run_id),
                                    generate_runs.template_path(conf.get("TEMPLATE")), None)
        start_run.step_trees(context)
        start_run.step_runs(context)
        store.remove_scripts(conn, run_id, context.run_dir)
        row = store.individuals(conn, run_id)[0]
        if row["state"] != "ok":
            raise DrawnError("the pipeline could not build %s" % chromosome)
    except SystemExit as error:
        raise DrawnError(str(error))
    finally:
        conn.close()


# --- opening one again -------------------------------------------------------------


def _slot_ids(catalog, user, conf):
    """{slot: the user's catalogue id} for a sweep's LORA_SLOTS. Raises DrawnError."""
    slots = {}
    for slot, folder in generate_runs.lora_slots(conf.get("LORA_SLOTS")).items():
        row = catalog.by_folder(folder)
        if row is None or row["owner"] != user["name"]:
            raise DrawnError("slot %s's LoRA is no longer in your catalogue" % slot)
        slots[slot] = row["id"]
    return slots


def opened(registry, catalog, user, job, number=None):
    """One blend of a job as a page draws it again. -> {job, label, task,
    number, pin, tree, seed, count, chromosome, drawn, state, fitness,
    quality, verification, questions}. `pin` is the weight seed a blend saved
    by hand keeps (None for any other), which the weights come from instead. `chromosome` is the one the job holds and
    `drawn` the tree's own encoding -- the same blend, its slots numbered in
    reading order -- which is what a page compares a later drawing with.

    A drawn blend's job holds one individual, `number` or not. A search's
    blend is individual `number` of it (its best when None): the drawing is
    read back out of the sweep (decode_tree over the stored chromosome, each
    slot's folder back to the user's catalogue row) and keeps the sweep's
    WEIGHT_MASTER_SEED *and the individual's number*, so its weights are the
    ones it was scored with and a test of it tests that blend. `verification`
    is the latest of that individual."""
    try:
        conn = results.connect(registry.database(job))
    except results.NoResults:
        raise DrawnError("job %d's run is gone" % job["id"])
    try:
        conf = store.get_settings(conn, job["run_id"])
        population = results.population(conn, job["run_id"])
        pins = {row["number"]: row["weight_pin"] for row in store.individuals(conn, job["run_id"])}
    finally:
        conn.close()
    if not population:
        raise DrawnError("job %d holds no blend yet" % job["id"])
    if job["task"] == reg.BLEND and number is None:
        # The blend it was drawn as; any saved beside it are opened by number.
        chosen = population[0]
    elif number is None:
        chosen = results.best(population)
        if chosen is None:
            raise DrawnError("job %d has no scored blend yet; name one by its number"
                             % job["id"])
    else:
        chosen = next((one for one in population if one["number"] == number), None)
        if chosen is None:
            raise DrawnError("job %d holds no blend #%d" % (job["id"], number))
    slots = _slot_ids(catalog, user, conf)
    tree = decode_tree(chosen["chromosome"], slots)
    found = [row for row in registry.verifications(job_id=job["id"])
             if row["number"] == chosen["number"]]
    latest = found[0] if found else None
    options = json.loads(latest["options"] or "{}") if latest else {}
    return {"job": job["id"], "label": job["label"], "task": job["task"],
            "number": chosen["number"], "pin": pins.get(chosen["number"]),
            "tree": tree,
            "seed": conf.get("WEIGHT_MASTER_SEED"), "count": conf.get("TRAINING_COUNT"),
            "chromosome": chosen["chromosome"], "drawn": encode(tree)[0],
            "state": chosen["state"],
            "fitness": chosen["fitness"], "quality": chosen["quality"],
            "verification": latest["id"] if latest else None,
            "questions": options.get("dataset_label")}


# A job's run takes a saved blend only when nothing else is writing to it.
SAVABLE = (reg.DONE, reg.STOPPED, reg.FAILED, reg.CANCELLED)


def save(registry, catalog, user, job, payload):
    """A drawing saved into the run of `job` as a brand new individual.

        {"tree": <drawing>, "seed": n, "number": n?, "pin": n?}

    -> {job, number, chromosome, drawn, check}. The drawing is the one a page
    opened from that job and edited; it is saved with the run's own slots for
    its LoRAs (so every LoRA in it must be one of the run's), the next number
    (store.append_individual) and a *pin*: the weight seed it was drawn and
    shown under, which start_run.step_runs keeps rather than deriving one from
    the new number -- so what was saved is exactly what runs. Its tree and
    script are made by start_run's own `trees` and `runs` steps for it alone,
    and no file is left behind. It joins the search like any individual: a
    resumed search runs, scores, selects, mutates or culls it. Refused unless
    the drawing can be built and the job's run is at rest."""
    if not isinstance(payload, dict):
        raise DrawnError("the body is a JSON object")
    unknown = sorted(set(payload) - {"tree", "seed", "number", "pin"})
    if unknown:
        raise DrawnError("unknown field(s): %s" % ", ".join(unknown))
    if job["status"] not in SAVABLE:
        raise DrawnError("job %d is %s; a blend can be saved into it once it is not running"
                         % (job["id"], job["status"]))
    tree = payload.get("tree")
    found = check(catalog, user, tree, payload.get("seed"), payload.get("number"),
                  payload.get("pin"))
    if found["state"] != "ok":
        raise DrawnError("; ".join(found["problems"]) or "the drawing is not finished yet")
    try:
        conn = results.connect(registry.database(job))
    except results.NoResults:
        raise DrawnError("job %d's run is gone" % job["id"])
    import start_run
    try:
        conf = store.get_settings(conn, job["run_id"])
        slot_of = {}
        for slot, lora in sorted(_slot_ids(catalog, user, conf).items(),
                                 key=lambda pair: slot_key(pair[0])):
            slot_of.setdefault(lora, slot)
        missing = [found["loras"][str(lora)]["name"] for lora in loras_in(tree)
                   if lora not in slot_of]
        if missing:
            raise DrawnError("%s %s not among the LoRAs job %d blends; a blend saved into a "
                             "run uses that run's LoRAs"
                             % (", ".join(missing), "is" if len(missing) == 1 else "are",
                                job["id"]))
        chromosome, _ = encode(tree, slot_of)
        number = store.append_individual(conn, job["run_id"], chromosome,
                                         weight_pin=found["weight_seed"])
        context = start_run.Context(conn, job["run_id"], conf,
                                    db_datasets.run_folder(conn, job["run_id"]),
                                    generate_runs.template_path(conf.get("TEMPLATE")), None)
        try:
            start_run.step_trees(context, numbers={number})
            start_run.step_runs(context, numbers={number})
        except SystemExit as error:
            raise DrawnError(str(error))
    finally:
        conn.close()
    return {"job": job["id"], "number": number, "chromosome": chromosome,
            "drawn": found["chromosome"], "check": found}


def sources(registry, catalog, user):
    """Every blend of the user's that a page can open, job by job, newest job
    first. -> [{job, label, task, status, created_at, base_model, blends:
    [{number, chromosome, formula, state, fitness, quality, is_best}]}], a
    search's blends best first and those that cannot be built last.

    A job whose run is gone, or which holds nobody yet, is left out; so is one
    whose LoRAs are no longer all the user's, since it could not be drawn."""
    from async_api_agent import blending          # the chromosome in words
    out = []
    for job in registry.jobs(user["id"]):
        if job["status"] == reg.DELETED or job["run_id"] is None:
            continue
        try:
            conn = results.connect(registry.database(job))
        except results.NoResults:
            continue
        try:
            conf = store.get_settings(conn, job["run_id"])
            population = results.population(conn, job["run_id"])
        finally:
            conn.close()
        if not population:
            continue
        try:
            _slot_ids(catalog, user, conf)
        except DrawnError:
            continue
        names = blending.slot_names(conf, catalog, user)
        top = results.best(population)
        population.sort(key=lambda one: (one["state"] == "BAD", -(one["fitness"] or 0.0),
                                         one["number"]))
        out.append({"job": job["id"], "label": job["label"], "task": job["task"],
                    "status": job["status"], "created_at": job["created_at"],
                    "base_model": conf.get("BASE_MODEL"),
                    "blends": [{"number": one["number"], "chromosome": one["chromosome"],
                                "formula": blending.formula(one["chromosome"], names),
                                "state": one["state"], "fitness": one["fitness"],
                                "quality": one["quality"],
                                "is_best": top is not None and one["number"] == top["number"]}
                               for one in population]})
    return out
