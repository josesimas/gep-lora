"""
blending.py - The second half of the guide: the user's LoRAs, combined.

Training gives a person LoRAs; a search blends them. This module is to that
half what planner.py is to the first: it turns what the chat has set up into
the body of one request the API already takes -- a POST /jobs -- and says how
long that should take. Nothing here submits anything: the page sends the body
and watches the job through GET /jobs/{id}/status, exactly as the demo page's
Jobs tab does.

What the person decides lives in the session's `blend` part
(planner.session_of checks it on the way in):

    {"loras":       [id, ...]   1-5 of their own ready LoRAs, one base model
     "generations": n           after the first, so the search is 1 + n
     "population":  n           individuals in each generation
     "questions":   n           how many questions every blend is judged on
     "source":      {"lora": id} | {"file": name} | None
                                where the questions come from; None is the
                                conversation's dataset, else the chosen
                                LoRAs' own training data
     "label":       text}       the job's label

**A user blends their own LoRAs and nobody else's.** Every id is looked up
among the user's catalogue rows (another user's is "no LoRA of yours", as a
missing one is), and the job names them by id, which submit.own_slots() checks
again: the server's own LORA_SLOTS -- the command line's set under
loras/Lora00N -- are never used.

The grammar blends exactly five slots, L1-L5. Fewer LoRAs than that go round
the slots again (two LoRAs are L1=A, L2=B, L3=A, ...), which the search already
allows: one slot may appear in a chromosome many times, and so may one
adapter.
"""

import json
import math
import os

from adapters import catalog as lora_catalog
from async_api import results
from async_api import train
from async_api_agent import analysis
from async_api_agent import planner
from async_api_agent import settings
from config import settings as config
from search import generate_population
from search.generate_population import UNARY_OPS as SLOTS

MOCKED_TEMPLATE = "template_code_mocked.py"


class BlendError(ValueError):
    """Why a blend cannot be planned, in words for the person."""


# --- the session's blend part ------------------------------------------------


def _whole(name, value, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value) \
            or not low <= value <= high:
        raise BlendError("%s must be a whole number from %d to %d" % (name, low, high))
    return int(value)


def blend_of(raw):
    """The session's blend part, checked. -> the full blend, defaults filled."""
    raw = raw if isinstance(raw, dict) else {}
    loras = raw.get("loras") or []
    if isinstance(loras, int) and not isinstance(loras, bool):
        loras = [loras]
    if not isinstance(loras, list) or not all(
            isinstance(one, int) and not isinstance(one, bool) for one in loras):
        raise BlendError("loras must be a list of LoRA ids")
    if len(loras) > len(SLOTS):
        raise BlendError("a search blends at most %d LoRAs" % len(SLOTS))
    source = raw.get("source")
    if source is not None:
        if not isinstance(source, dict) or len(source) != 1 or not (
                (isinstance(source.get("lora"), int) and not isinstance(source.get("lora"), bool))
                or isinstance(source.get("file"), str)):
            raise BlendError("the questions' source must be {\"lora\": id} or {\"file\": name}")
    label = raw.get("label")
    if label is not None and (not isinstance(label, str) or not label.strip()):
        raise BlendError("a label must be some text")
    return {
        "loras": list(dict.fromkeys(loras)),
        "generations": _whole("generations", raw.get("generations", settings.BLEND_GENERATIONS),
                              1, settings.MAX_BLEND_GENERATIONS),
        "population": _whole("population", raw.get("population", settings.BLEND_POPULATION),
                             settings.MIN_BLEND_POPULATION, settings.MAX_BLEND_POPULATION),
        "questions": _whole("questions", raw.get("questions", settings.BLEND_QUESTIONS),
                            1, settings.MAX_BLEND_QUESTIONS),
        "source": source,
        "label": " ".join(label.split())[:80] if label else None,
    }


# --- the user's LoRAs ----------------------------------------------------------


def describe(row):
    """One LoRA as the page and the model are shown it."""
    return {"id": row["id"], "name": row["name"], "rank": row["rank"],
            "base_model": row["base_model"], "chat_template": row["chat_template"],
            "status": row["status"], "final_loss": row["final_loss"],
            "records": row["records"], "dataset": row["dataset"],
            "practice_run": bool(row["mock"]), "trained_at": row["trained_at"]}


def mine(catalog, user):
    """The user's LoRAs that can be blended: their own, and ready. Newest first."""
    rows = catalog.all(owner=user["name"], status=lora_catalog.READY)
    return sorted(rows, key=lambda row: (row["trained_at"] or row["created_at"] or "", row["id"]),
                  reverse=True)


def own(catalog, user, key):
    """One of the user's ready LoRAs, by id or name. -> the row. Raises BlendError."""
    row = None
    if isinstance(key, int) and not isinstance(key, bool):
        row = catalog.get(key)
    elif isinstance(key, str) and key.strip():
        wanted = key.strip()
        row = catalog.by_name(wanted, user["name"])
        if row is None and wanted.isdigit():
            row = catalog.get(int(wanted))
        if row is None:
            close = [one for one in mine(catalog, user) if wanted.lower() in one["name"].lower()]
            if len(close) == 1:
                row = close[0]
    if row is None or row["owner"] != user["name"]:
        raise BlendError("no LoRA of yours called %r" % (key,))
    if row["status"] != lora_catalog.READY:
        raise BlendError("%s is %s, not ready" % (row["name"], row["status"]))
    return row


def check_together(rows):
    """LoRAs that can be blended in one search: 1 to 5, one base model and one
    chat template, since a search runs one model under one template."""
    if not rows:
        raise BlendError("choose at least one of your LoRAs to blend")
    if len(rows) > len(SLOTS):
        raise BlendError("a search blends at most %d LoRAs" % len(SLOTS))
    models = sorted({row["base_model"] or "?" for row in rows})
    if len(models) > 1:
        raise BlendError("those LoRAs were trained on different base models (%s); a blend "
                         "needs them all on one" % ", ".join(models))
    templates = sorted({str(row["chat_template"]) for row in rows})
    if len(templates) > 1:
        raise BlendError("those LoRAs were trained under different chat templates (%s); a "
                         "blend needs one" % ", ".join(templates))
    return rows


def default_pick(catalog, user, prefer=()):
    """The LoRAs to offer when none are chosen: `prefer` (the ones this
    conversation just trained) when they can go together, else the newest of
    the user's on the base model they have the most ready LoRAs for."""
    ready = mine(catalog, user)
    by_id = {row["id"]: row for row in ready}
    wanted = [by_id[one] for one in prefer if one in by_id][:len(SLOTS)]
    if wanted:
        try:
            return check_together(wanted)
        except BlendError:
            pass
    if not ready:
        return []
    groups = {}
    for row in ready:
        groups.setdefault((row["base_model"], str(row["chat_template"])), []).append(row)
    best = max(groups.values(), key=lambda rows: (len(rows), rows[0]["trained_at"] or ""))
    return best[:len(SLOTS)]


def chosen(catalog, user, blend, prefer=()):
    """The rows the blend names, checked -- or the default pick. -> [row]."""
    if blend["loras"]:
        return check_together([own(catalog, user, one) for one in blend["loras"]])
    return default_pick(catalog, user, prefer)


def slots(rows):
    """{L1..L5: id}, going round the LoRAs again when there are fewer than five."""
    return {slot: rows[index % len(rows)]["id"] for index, slot in enumerate(SLOTS)}


# --- the questions ---------------------------------------------------------------


def _file_text(name):
    from async_api_agent import routes           # the shared-dataset rule lives there
    return routes._read(routes._shared_path(name)), name


def _lora_text(catalog, registry, user, key):
    row = own(catalog, user, key)
    training = registry.training_for_lora(row["id"]) if registry is not None else None
    if training is not None and (training["folder"] != row["folder"]
                                 or training["user_id"] != user["id"]):
        training = None
    path = train.dataset_path(row, training, registry)
    if path is None:
        raise BlendError("no copy of %s's training data is kept, so its questions cannot be "
                         "used; pick a demo dataset for them instead" % row["name"])
    with open(path, encoding="utf-8-sig", errors="replace") as handle:
        return handle.read(), "%s's training data" % row["name"]


def source_text(catalog, registry, user, blend, rows, dataset=None):
    """The dataset the blends are judged on. -> (text, where it came from).

    The source the chat chose, else the conversation's own dataset, else the
    first chosen LoRA that kept a copy of what it was trained on."""
    source = blend["source"]
    if source and source.get("file"):
        return _file_text(source["file"])
    if source and source.get("lora") is not None:
        return _lora_text(catalog, registry, user, source["lora"])
    if dataset is not None:
        text, name, shared = dataset
        return text, name or "the dataset in this conversation"
    for row in rows:
        try:
            return _lora_text(catalog, registry, user, row["id"])
        except BlendError:
            continue
    raise BlendError("none of those LoRAs kept a copy of its training data; pick a demo "
                     "dataset for the questions")


def split(lines, questions):
    """-> (training lines, testing lines): the first `questions` records are
    what every blend is judged on, and up to settings.BLEND_TEST_QUESTIONS of
    the rest are the testing pass -- questions the search never saw."""
    training = lines[:questions]
    testing = lines[questions:questions + settings.BLEND_TEST_QUESTIONS]
    return training, testing


# --- the plan -------------------------------------------------------------------


def estimate(population, generations, mock=False, template=None):
    """How long a search takes: at most `population` individuals run in each
    of 1 + generations, a few at once. A rough guess, and said to be one."""
    template = template or config.TEMPLATE
    at_once = (config.LORA_SERVER_COUNT if template == "template_remote_code.py"
               else config.PROCESS_RUN_BATCH_SIZE if template == "template_code.py" else 1)
    per = settings.MOCK_SECONDS_PER_INDIVIDUAL if mock else settings.SECONDS_PER_INDIVIDUAL
    rounds = (1 + generations) * math.ceil(population / float(max(1, at_once)))
    seconds = settings.BLEND_OVERHEAD_SECONDS * (0 if mock else 1) + rounds * per
    return {"generations": 1 + generations, "population": population,
            "individuals": (1 + generations) * population, "at_once": at_once,
            "seconds": round(seconds, 1), "time": planner.human(seconds),
            "source": ("a practice run's pace" if mock else
                       "a rough guess: about %ds a blend, %d at a time" % (per, at_once))}


def plan(catalog, registry, user, session, mock=False, dataset=None, prefer=()):
    """The POST /jobs body that blends the session's LoRAs, and what it is.

    -> {job, loras, slots, questions, testing, source, estimate, mock, template}.
    Mocked when asked to be, and whenever a chosen LoRA is a practice run: a
    practice LoRA has no weights, and only the mocked template runs one.
    """
    blend = session["blend"]
    rows = chosen(catalog, user, blend, prefer)
    if not rows:
        raise BlendError("you have no ready LoRAs yet -- train some first")
    rows = check_together(rows)
    text, where = source_text(catalog, registry, user, blend, rows, dataset)
    try:
        found = analysis.analyse(text, where)
    except analysis.DatasetError as error:
        raise BlendError("the questions could not be read: %s" % error)
    if not found["usable"]:
        raise BlendError("%s has no questions with answers to judge a blend by" % where)
    training, testing = split(found["lines"], blend["questions"])
    mocked = bool(mock) or any(row["mock"] for row in rows)
    template = MOCKED_TEMPLATE if mocked else config.TEMPLATE
    first = rows[0]
    wanted = {"LORA_SLOTS": slots(rows), "BASE_MODEL": first["base_model"] or config.BASE_MODEL,
              "CHAT_TEMPLATE": first["chat_template"],
              "GENERATIONS": blend["generations"], "COUNT": blend["population"],
              "TRAINING_COUNT": len(training)}
    if mocked:
        wanted["TEMPLATE"] = MOCKED_TEMPLATE
    datasets = {"training": "\n".join(training)}
    if testing:
        datasets["testing"] = "\n".join(testing)
    label = blend["label"] or settings.BLEND_LABEL.format(
        stem=planner.stem_of(first["name"]).rsplit("-r", 1)[0], loras=len(rows))
    job = {"label": label, "settings": wanted, "datasets": datasets,
           "options": {"no_test": not testing}}
    names = {row["id"]: row["name"] for row in rows}
    return {"job": job, "loras": [describe(row) for row in rows],
            "slots": {slot: names[lora_id] for slot, lora_id in wanted["LORA_SLOTS"].items()},
            "questions": len(training), "testing": len(testing),
            "source": where, "total_records": found["records"],
            "estimate": estimate(blend["population"], blend["generations"], mocked, template),
            "mock": mocked, "template": template, "label": label}


# --- reading a search back ---------------------------------------------------------


def formula(chromosome, names, weights=None):
    """A chromosome in words: CAT(SVD(poem-r8 ×0.42, poem-r16 ×0.9), ...).

    `names` is {slot: LoRA name}; `weights` the individual's drawn {w: value},
    else the weight symbols are left as they are."""
    try:
        root, _ = generate_population.decode(chromosome)
    except ValueError:
        return chromosome
    verbs = {"CAT": "stack", "SVD": "merge", "LIN": "mix"}

    def said(node):
        if node.symbol in SLOTS:
            weight = node.children[0].symbol
            value = (weights or {}).get(weight)
            return "%s ×%s" % (names.get(node.symbol, node.symbol),
                               "%.2f" % value if isinstance(value, (int, float)) else weight)
        return "%s(%s)" % (verbs.get(node.symbol, node.symbol),
                           ", ".join(said(child) for child in node.children))
    return said(root)


def outcome(registry, job, user, catalog):
    """What a blend search came to, from the user's own job. -> facts."""
    database = registry.database(job)
    try:
        found = results.detail(database, job["run_id"])
    except results.NoResults:
        return {"status": job["status"], "error": job["error"], "best": None}
    conf = found["settings"]
    names = {}
    for slot, folder in (conf.get("LORA_SLOTS") or {}).items():
        row = catalog.by_folder(lora_catalog.absolute(folder))
        names[slot] = row["name"] if row is not None and row["owner"] == user["name"] \
            else os.path.basename(str(folder).rstrip("/\\"))
    best = found["best"]
    weights = None
    if best is not None:
        one = results.individual(database, job["run_id"], best["number"])
        weights = ((one or {}).get("execution") or {}).get("weights")
    history = [{"generation": entry.get("generation"), "best": entry.get("best"),
                "mean": entry.get("mean")} for entry in found["fitness_history"]]
    testing = found["testing"]["summary"]
    tested = None
    if best is not None:
        tested = next((row for row in found["testing"]["individuals"]
                       if row.get("number") == best["number"]), None)
    return {
        "status": job["status"], "error": job["error"],
        "mock": conf.get("TEMPLATE") == MOCKED_TEMPLATE,
        "evaluator": conf.get("EVALUATOR"),
        "slots": names,
        "generations": len(history), "population": len(found["population"]),
        "best": None if best is None else {
            "number": best["number"], "chromosome": best["chromosome"],
            "fitness": best["fitness"], "quality": best["quality"],
            "formula": formula(best["chromosome"], names, weights),
            "tested_quality": (tested or {}).get("quality"),
            "weights": weights},
        "history": history,
        "blocked": sum(1 for one in found["population"] if one["state"] == "BAD"),
        "testing": json.loads(json.dumps(testing, default=str)) if testing else None,
    }
