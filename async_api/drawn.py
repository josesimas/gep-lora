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
    what the symbols are worth is the draw a sweep makes for individual 1 under
    its WEIGHT_MASTER_SEED -- start_run's derivation, golive.draw_weights()'s
    values. So the page shows the numbers the scripts will use, a new seed is a
    new draw, and the sweep stored for it reproduces the blend as any sweep does.
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
from async_api import submit
from async_api import verify
from blends import generate_runs
from config import settings as config
from search.generate_population import (
    BINARY_OPS, MAX_SLOTS, UNARY_OPS, VARIABLES, Node, check as check_expression,
    encode as encode_tree, decode)
from storage import db_datasets
from storage import store

# The fold a new drawing starts from (see the module note).
ROOT = "CAT"

# The one individual a drawn blend's sweep holds. Its number is what the weight
# seed is derived from, as for any individual of any sweep.
NUMBER = 1

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


def encode(tree):
    """A complete drawing as a sweep holds it. -> (chromosome, {slot: lora id}).

    Slots are handed out in reading order; a LoRA used twice keeps its slot.
    Raises DrawnError for a place left empty or a drawing past MAX_SLOTS LoRAs.
    """
    places = walk(tree)
    empty = [path for path, node in places if node is None]
    if empty:
        raise DrawnError("the drawing has %d empty place(s) left to fill" % len(empty))
    ids = loras_in(tree)
    if len(ids) > MAX_SLOTS:
        raise DrawnError("a blend uses at most %d different LoRAs" % MAX_SLOTS)
    slot_of = {lora: UNARY_OPS[index] for index, lora in enumerate(ids)}

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


def weight_seed(master):
    """The seed individual NUMBER draws its weights from under `master` -- the
    derivation start_run.step_runs makes for every individual."""
    import start_run                # the pipeline driver: imported where it is needed
    return random.Random("%s:%d" % (master, NUMBER)).randrange(start_run._SEED_LIMIT)


def draw(master=None):
    """-> {seed, weight_seed, weights}: what w1..w10 are worth under `master`,
    a new master seed when None -- drawn as a sweep draws one it was not given."""
    import start_run
    if master is None:
        master = random.randrange(start_run._SEED_LIMIT)
    if isinstance(master, bool) or not isinstance(master, int) or master < 0:
        raise DrawnError("the seed is a whole number, 0 or more")
    seed = weight_seed(master)
    return {"seed": master, "weight_seed": seed,
            "weights": {name: round(value, 4)
                        for name, value in golive.draw_weights(seed).items()}}


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


def check(catalog, user, tree, seed=None):
    """Everything the page draws beside a drawing, and whether it can be built.

    -> {complete, empty, state, problems, chromosome, slots, nodes, rank,
        formula, loras, base_model, chat_template, mock, seed, weight_seed,
        weights}. `state` is "incomplete" while a place is empty, "BAD" when
    PEFT cannot build it (the rank rule, or LoRAs that do not go together) and
    "ok" when it can. `nodes` is {path: {rank, broken, fold}} -- a leaf's rank
    is its adapter's, a fold's what PEFT gives it. Raises DrawnError only for
    a drawing that is not a drawing; anything else is a problem it reports.
    """
    places = walk(tree)
    drawn = draw(seed)
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

        {"tree": <drawing>, "seed": n?, "dataset": {...}, "count": n?,
         "label"?, "mock"?: bool, "settings"?: {EVALUATOR, JUDGE_*...}}

    Refused before anything is written unless the drawing is complete and
    buildable. The job is the owner's like any other, and deleting its run
    takes the verification with it.
    """
    if not isinstance(payload, dict):
        raise DrawnError("the body is a JSON object")
    unknown = sorted(set(payload) - {"tree", "seed", "dataset", "count", "label",
                                     "mock", "settings"})
    if unknown:
        raise DrawnError("unknown field(s): %s" % ", ".join(unknown))
    seed = payload.get("seed")
    found = check(catalog, user, payload.get("tree"), seed)
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
        _one_individual(registry.database(job), run_id, found["chromosome"])
        return registry.settle_drawn(job["id"], run_id)

    try:
        job = submit.prepare(registry, user, body, settle)
    except submit.SubmissionError as error:
        raise DrawnError(str(error))
    try:
        offered = verify.choices(registry.database(job), job["run_id"])
        number, options = verify.options_for(
            {"individual": NUMBER, "split": "training", "count": min(count, len(lines))},
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


def _one_individual(db_path, run_id, chromosome):
    """Put the drawing in as the sweep's only individual, through start_run's
    own `trees` and `runs` steps -- its tree, state, rank, weight seed and
    script are the ones a searched individual gets -- and leave no script
    file behind: the database is the sweep."""
    import start_run
    conn = store.connect(db_path)
    try:
        store.add_individuals(conn, run_id, [chromosome])
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


def opened(registry, catalog, user, job):
    """A drawn blend's job as the page draws it again. -> {job, tree, seed,
    count, verification, questions}: the drawing read back out of its sweep
    (decode_tree over the stored chromosome, each slot's folder back to the
    user's catalogue row), the seed its weights came from, and its latest
    verification. A search is refused: its blends were not drawn, and their
    weights come from their own individual's number, not NUMBER's."""
    if job["task"] != reg.BLEND:
        raise DrawnError("job %d is a search, not a blend drawn here" % job["id"])
    conn = store.connect(registry.database(job))
    try:
        conf = store.get_settings(conn, job["run_id"])
        rows = store.individuals(conn, job["run_id"])
    finally:
        conn.close()
    if not rows:
        raise DrawnError("job %d holds no blend" % job["id"])
    slots = {}
    for slot, folder in generate_runs.lora_slots(conf.get("LORA_SLOTS")).items():
        row = catalog.by_folder(folder)
        if row is None or row["owner"] != user["name"]:
            raise DrawnError("slot %s's LoRA is no longer in your catalogue" % slot)
        slots[slot] = row["id"]
    found = registry.verifications(job_id=job["id"])
    latest = found[0] if found else None
    options = json.loads(latest["options"] or "{}") if latest else {}
    return {"job": job["id"], "tree": decode_tree(rows[0]["chromosome"], slots),
            "seed": conf.get("WEIGHT_MASTER_SEED"), "count": conf.get("TRAINING_COUNT"),
            "verification": latest["id"] if latest else None,
            "questions": options.get("dataset_label")}
