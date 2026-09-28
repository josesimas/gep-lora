"""
catalog.py - Every LoRA adapter on this machine: the `loras` table.

    python -m gep_lora.core.adapters.catalog scan [--owner ze]   # register what is under loras/
    python -m gep_lora.core.adapters.catalog list [--owner ze]
    python -m gep_lora.core.adapters.catalog show 3              # by id or by name
    python -m gep_lora.core.adapters.catalog own ze 3 4 5        # give rows to an API user
    python -m gep_lora.core.adapters.catalog own ze --unowned    # ...every row nobody owns
    python -m gep_lora.core.adapters.catalog forget 3            # the row, never the folder
    python -m gep_lora.core.adapters.catalog models              # base models a LoRA can be trained on

The adapters were only ever known by their folders -- LORA_SLOTS names five of
them, and loras/ holds however many sets have been trained. This is the index
over those folders: a `loras` table, one row per adapter folder, saying what it
was trained on, at what rank, from which data, and how far its training got.

**It lives in the API's database**, `api_jobs/api.sqlite` (the async API's
JOBS_DIR; $GEP_LORA_CATALOG or --catalog points elsewhere), beside the users who
own the rows and the trainings that fill them -- one file for everything the API
knows. This module owns the table and nothing else in the file; registry.py owns
the rest and creates this table through SCHEMA, so either can open the file first.

**The folder is the truth and the row is an index.** Everything a row says
about a finished adapter is read off its folder -- the rank and base model from
adapter_config.json, the rest from the `training.json` create_lora.py writes
beside the weights (or, for an adapter trained before that existed, from the
trainer_state.json of its newest checkpoint) -- so `scan` can rebuild the whole
table from disk, and deleting the file loses nothing but the rows of trainings
still under way.

Rows are written from three places, each for what only it knows:

    scan()               folders found on disk (origin 'scanned')
    create_lora.py       an adapter it is training: 'training', then
                         'ready' or 'failed' (origin 'trained')
    gep_lora.service     a training it has queued ('queued') and, when the
                         process running it was killed, how it ended

**Every row can have an owner** -- the name of the async API user it belongs
to, and the only user the API shows it to. A LoRA trained through the API is
its trainer's from the moment it is queued; one found by `scan` or trained by
hand belongs to nobody until `own` (or `scan --owner`, `create_lora.py
--owner`) says whose it is, and nobody sees it through the API until then.
Names are unique per owner, not across the catalogue, so choosing a name can
never tell one user what another has. An owner is a user's name, and `own`
refuses a name the file's `users` table does not hold.

This is the third module that imports sqlite3 -- store.py owns a sweep and
registry.py owns the API's queue; neither is the place for a list of adapters
that outlives every sweep and every job, and create_lora.py writes it with no
API running at all.
"""

import argparse
import glob
import json
import os
import sqlite3
import sys
import time
from gep_lora import paths

_ROOT = paths.ROOT
LORA_DIR = os.path.join(_ROOT, "loras")

# The file create_lora.py writes into every adapter folder it trains: the
# recipe, the dataset's fingerprint, the loss history and how it ended.
RECIPE = "training.json"

QUEUED, TRAINING, READY, FAILED, CANCELLED, MISSING = (
    "queued", "training", "ready", "failed", "cancelled", "missing")
STATUSES = (QUEUED, TRAINING, READY, FAILED, CANCELLED, MISSING)
# A training under way: its row is the only record of it, so scan() leaves it be.
IN_FLIGHT = (QUEUED, TRAINING)

# all(owner=ANYONE) is every row; all(owner=None) is only the rows nobody owns.
ANYONE = object()

SCHEMA = """
CREATE TABLE IF NOT EXISTS loras (
    -- Never reused: the API's trainings point at these ids, and a reused one
    -- would hand an old training -- and who may delete its folder -- to a new row.
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,         -- unique per owner, see below
    folder          TEXT NOT NULL UNIQUE,  -- relative to the repo when inside it
    base_model      TEXT,                  -- as adapter_config.json records it
    rank            INTEGER,               -- max(r, *rank_pattern), as PEFT allocates
    alpha           REAL,
    target_modules  TEXT,                  -- JSON: a list, or PEFT's regex string
    chat_template   TEXT,                  -- the unsloth template trained under
    status          TEXT NOT NULL,
    origin          TEXT NOT NULL,         -- scanned | trained
    mock            INTEGER NOT NULL DEFAULT 0,
    weights         INTEGER NOT NULL DEFAULT 0,  -- adapter_model.* is on disk
    recipe          TEXT NOT NULL DEFAULT '{}',  -- JSON: every training option
    dataset         TEXT,                  -- where its data came from
    dataset_sha256  TEXT,
    records         INTEGER,
    steps           INTEGER,
    max_steps       INTEGER,
    epochs          REAL,
    final_loss      REAL,
    seconds         REAL,
    sample          TEXT,                  -- the smoke test's answer
    owner           TEXT,                  -- the API user it belongs to
    error           TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    trained_at      TEXT
);
CREATE INDEX IF NOT EXISTS loras_by_base ON loras(base_model, name);
-- One namespace per owner; every row nobody owns shares the one namespace.
CREATE UNIQUE INDEX IF NOT EXISTS loras_name_per_owner ON loras(COALESCE(owner, ''), name);
CREATE INDEX IF NOT EXISTS loras_by_owner ON loras(owner, base_model, name);
"""

COLUMNS = ("name", "folder", "base_model", "rank", "alpha", "target_modules",
           "chat_template", "status", "origin", "mock", "weights", "recipe",
           "dataset", "dataset_sha256", "records", "steps", "max_steps", "epochs",
           "final_loss", "seconds", "sample", "owner", "error", "trained_at")
JSON_COLUMNS = ("target_modules", "recipe")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def catalog_path(path=None):
    """The file the table is in: `path`, $GEP_LORA_CATALOG, or the API's database."""
    path = path or os.environ.get("GEP_LORA_CATALOG")
    if not path:
        # Where the API's database is -- gep_lora.paths, which the service's
        # registry reads too, so core need not import the service to know.
        return paths.api_database()
    return os.path.abspath(path if os.path.isabs(path) else os.path.join(_ROOT, path))


def stored_folder(folder):
    """How a folder is written in the table: relative, with forward slashes,
    when it is inside the repo -- the way LORA_SLOTS writes it -- else absolute."""
    folder = os.path.abspath(folder)
    try:
        relative = os.path.relpath(folder, _ROOT)
    except ValueError:                        # another drive, on Windows
        return folder
    return folder if relative.startswith("..") else relative.replace(os.sep, "/")


def absolute(folder):
    return os.path.normpath(folder if os.path.isabs(folder) else os.path.join(_ROOT, folder))


def name_for(folder):
    """A default name: the folder under loras/ (`Lora001/QWen2.5-0.5b-lora_adapter`)."""
    folder = os.path.abspath(folder)
    try:
        relative = os.path.relpath(folder, LORA_DIR)
    except ValueError:
        relative = ".."
    if relative.startswith(".."):
        return os.path.basename(folder)
    return relative.replace(os.sep, "/")


# --- reading a folder -------------------------------------------------------------

def _json(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def rank_of(config):
    """max(r, *rank_pattern.values()), the rank PEFT allocates -- the same
    reading create_lora.rank_of() and the pipeline's slot_ranks() make."""
    return max([config.get("r") or 0] + list((config.get("rank_pattern") or {}).values()))


def newest_checkpoint_state(folder):
    """The trainer_state.json of the newest checkpoint beside an adapter, or None.

    Every adapter trained by a Trainer leaves `<folder>_outputs/checkpoint-N/`,
    and its trainer_state.json is the only record of how an adapter trained
    before training.json existed got there -- its steps, epochs and loss.
    """
    found = glob.glob(os.path.join(folder + "_outputs", "checkpoint-*", "trainer_state.json"))

    def step(path):
        try:
            return int(os.path.basename(os.path.dirname(path)).split("-")[-1])
        except ValueError:
            return -1
    return _json(max(found, key=step)) if found else None


def history(folder):
    """[{step, loss, epoch, lr}], one per logged step, for a loss curve."""
    folder = absolute(folder)
    recipe = _json(os.path.join(folder, RECIPE))
    if recipe and recipe.get("history"):
        return recipe["history"]
    state = newest_checkpoint_state(folder) or {}
    return [{"step": entry.get("step"), "loss": entry.get("loss"),
             "epoch": entry.get("epoch"), "lr": entry.get("learning_rate")}
            for entry in state.get("log_history") or [] if "loss" in entry]


def weights_on_disk(folder):
    return any(os.path.exists(os.path.join(folder, name))
               for name in ("adapter_model.safetensors", "adapter_model.bin"))


def describe(folder):
    """What an adapter folder says about itself. -> fields for a row, or None
    when the folder holds no adapter_config.json."""
    folder = absolute(folder)
    config = _json(os.path.join(folder, "adapter_config.json"))
    if config is None:
        return None
    weights = [os.path.join(folder, name)
               for name in ("adapter_model.safetensors", "adapter_model.bin")
               if os.path.exists(os.path.join(folder, name))]
    stamp = os.path.getmtime(weights[0] if weights else os.path.join(folder, "adapter_config.json"))
    out = {"base_model": config.get("base_model_name_or_path"),
           "rank": rank_of(config), "alpha": config.get("lora_alpha"),
           "target_modules": config.get("target_modules"),
           "weights": 1 if weights else 0,
           "mock": 1 if config.get("mock") else 0,
           "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stamp))}

    recipe = _json(os.path.join(folder, RECIPE))
    if recipe:
        options = recipe.get("options") or {}
        result = recipe.get("result") or {}
        out.update({
            "recipe": options, "mock": 1 if recipe.get("mock") else out["mock"],
            "chat_template": options.get("chat_template"),
            "dataset": (recipe.get("dataset") or {}).get("source"),
            "dataset_sha256": (recipe.get("dataset") or {}).get("sha256"),
            "records": (recipe.get("dataset") or {}).get("records"),
            "steps": result.get("steps"), "max_steps": result.get("max_steps"),
            "epochs": result.get("epochs"), "final_loss": result.get("final_loss"),
            "seconds": result.get("seconds"), "sample": result.get("sample"),
        })
        return out

    state = newest_checkpoint_state(folder)
    if state:
        losses = [entry["loss"] for entry in state.get("log_history") or [] if "loss" in entry]
        out.update({"steps": state.get("global_step"), "max_steps": state.get("max_steps"),
                    "epochs": state.get("epoch"),
                    "final_loss": losses[-1] if losses else None,
                    "recipe": {"epochs": state.get("num_train_epochs"),
                               "batch_size": state.get("train_batch_size"),
                               "logging_steps": state.get("logging_steps")}})
    return out


def adapter_folders(root=None):
    """Every folder under `root` (default loras/) holding an adapter_config.json.

    A Trainer's `<name>_outputs/` and the checkpoints in it hold one too, and
    are skipped: they are the scratch beside an adapter, not adapters.
    """
    root = os.path.abspath(root or LORA_DIR)
    found = []
    for folder, children, files in os.walk(root):
        children[:] = sorted(child for child in children
                             if not child.endswith("_outputs")
                             and not child.startswith("checkpoint-"))
        if "adapter_config.json" in files:
            found.append(folder)
            children[:] = []                 # an adapter holds no other adapter
    return found


# --- base models --------------------------------------------------------------------

def hub_cache():
    """The Hugging Face hub cache folder this machine downloads models into."""
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    return os.path.join(home, "hub")


# Only these heads generate text, so only these are base models a LoRA of this
# pipeline can be trained on; a GGUF repo has no config.json at all.
TRAINABLE = ("ForCausalLM", "ForConditionalGeneration")


def local_models(cache=None):
    """The generative models already downloaded here. -> [{id, architectures, bytes}]

    Read from the hub cache's own layout (`models--org--name/snapshots/*/`)
    rather than through huggingface_hub, so any Python can answer it -- the
    API server runs under whatever it was started with.
    """
    cache = cache or hub_cache()
    try:
        names = sorted(os.listdir(cache))
    except OSError:
        return []
    out = []
    for entry in names:
        if not entry.startswith("models--"):
            continue
        configs = glob.glob(os.path.join(cache, entry, "snapshots", "*", "config.json"))
        config = _json(configs[0]) if configs else None
        architectures = (config or {}).get("architectures") or []
        if not any(arch.endswith(TRAINABLE) for arch in architectures):
            continue
        size = 0
        for blob in glob.glob(os.path.join(cache, entry, "blobs", "*")):
            try:
                size += os.path.getsize(blob)
            except OSError:
                pass
        out.append({"id": entry[len("models--"):].replace("--", "/"),
                    "architectures": architectures, "bytes": size})
    return out


# --- the table ------------------------------------------------------------------------

class Catalog:
    """The catalogue file. Opens a connection per call, so any thread -- and the
    API server, the worker and create_lora.py at once -- may use it."""

    def __init__(self, path=None):
        self.path = catalog_path(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(SCHEMA)

    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return _Closing(conn)

    @staticmethod
    def _encode(fields):
        unknown = sorted(set(fields) - set(COLUMNS))
        if unknown:
            raise ValueError("no catalogue column(s) %s" % ", ".join(unknown))
        return {name: (json.dumps(value) if name in JSON_COLUMNS and value is not None
                       else value) for name, value in fields.items()}

    def add(self, name, folder, status, origin, **fields):
        """A new row. -> it. The folder is unique, and the name within its owner."""
        fields = self._encode(dict(fields, name=name, folder=stored_folder(folder),
                                   status=status, origin=origin))
        names = sorted(fields)
        with self._connect() as conn:
            try:
                cursor = conn.execute(
                    "INSERT INTO loras (%s, created_at, updated_at) VALUES (%s, ?, ?)"
                    % (", ".join(names), ", ".join("?" * len(names))),
                    [fields[name] for name in names] + [now(), now()])
            except sqlite3.IntegrityError:
                raise ValueError("there is already a LoRA called %r%s, or one in %s"
                                 % (name, " of %s's" % fields["owner"] if fields.get("owner")
                                    else "", fields["folder"]))
            return self.get(cursor.lastrowid)

    def update(self, lora_id, **fields):
        if not fields:
            return self.get(lora_id)
        fields = self._encode(fields)
        names = sorted(fields)
        with self._connect() as conn:
            conn.execute("UPDATE loras SET %s, updated_at = ? WHERE id = ?"
                         % ", ".join("%s = ?" % name for name in names),
                         [fields[name] for name in names] + [now(), lora_id])
        return self.get(lora_id)

    def get(self, lora_id):
        with self._connect() as conn:
            return conn.execute("SELECT * FROM loras WHERE id = ?", (lora_id,)).fetchone()

    def by_folder(self, folder):
        with self._connect() as conn:
            return conn.execute("SELECT * FROM loras WHERE folder = ?",
                                (stored_folder(folder),)).fetchone()

    def slot_names(self, slots, owner):
        """{slot: LoRA name} for a sweep's LORA_SLOTS ({slot: folder}): the
        catalogue name where the folder is one of `owner`'s rows, else the
        folder's own name -- so nobody learns another user's names this way."""
        names = {}
        for slot, folder in sorted((slots or {}).items()):
            row = self.by_folder(absolute(folder))
            names[slot] = row["name"] if row is not None and row["owner"] == owner \
                else os.path.basename(str(folder).rstrip("/\\"))
        return names

    def by_name(self, name, owner=None):
        """The row called `name` among `owner`'s (None: among the unowned)."""
        with self._connect() as conn:
            return conn.execute("SELECT * FROM loras WHERE name = ? AND owner IS ?",
                                (name, owner)).fetchone()

    def find(self, key):
        """A row by id (a number) or by name, whoever owns it -- for the
        command line, which answers to whoever runs it."""
        if isinstance(key, int) or (isinstance(key, str) and key.isdigit()):
            return self.get(int(key))
        with self._connect() as conn:
            return conn.execute("SELECT * FROM loras WHERE name = ? ORDER BY id",
                                (key,)).fetchone()

    def all(self, base_model=None, status=None, owner=ANYONE):
        sql, args = "SELECT * FROM loras WHERE 1 = 1", []
        if owner is not ANYONE:
            sql += " AND owner IS ?"
            args.append(owner)
        if base_model:
            sql += " AND base_model = ?"
            args.append(base_model)
        if status:
            sql += " AND status = ?"
            args.append(status)
        with self._connect() as conn:
            return conn.execute(sql + " ORDER BY base_model, name", args).fetchall()

    def remove(self, lora_id):
        with self._connect() as conn:
            return conn.execute("DELETE FROM loras WHERE id = ?", (lora_id,)).rowcount

    def own(self, lora_ids, owner):
        """Give rows to `owner` (None: to nobody). -> how many changed.

        Refuses, changing nothing, when a name would clash with one the new
        owner already holds -- rename one of them first.
        """
        rows = [self.get(lora_id) for lora_id in lora_ids]
        missing = [str(lora_id) for lora_id, row in zip(lora_ids, rows) if row is None]
        if missing:
            raise ValueError("no LoRA %s" % ", ".join(missing))
        if owner is not None:
            # In the API's database the users are in the same file: a typo in a
            # name would otherwise hand the rows to somebody who never logs in.
            with self._connect() as conn:
                has_users = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table'"
                                         " AND name = 'users'").fetchone()
                known = has_users and conn.execute("SELECT 1 FROM users WHERE name = ?",
                                                   (owner,)).fetchone()
            if has_users and not known:
                raise ValueError("there is no user called %r in %s" % (owner, self.path))
        moving = {row["id"] for row in rows}
        clash, seen = set(), set()
        for row in rows:
            held = self.by_name(row["name"], owner)
            if row["name"] in seen or (held is not None and held["id"] not in moving):
                clash.add(row["name"])
            seen.add(row["name"])
        if clash:
            raise ValueError("%s would hold two LoRAs called %s; rename one first"
                             % (owner or "nobody", ", ".join(sorted(clash))))
        with self._connect() as conn:
            return conn.execute("UPDATE loras SET owner = ?, updated_at = ? WHERE id IN (%s)"
                                % ",".join("?" * len(rows)),
                                [owner, now()] + [row["id"] for row in rows]).rowcount

    def free_name(self, name, owner=None):
        """`name`, or `name-2`, `name-3`... -- the first `owner` holds no row of."""
        candidate, number = name, 1
        while self.by_name(candidate, owner) is not None:
            number += 1
            candidate = "%s-%d" % (name, number)
        return candidate

    def record(self, folder, status, origin="trained", name=None, owner=None, **fields):
        """Upsert the row for `folder`. -> it. What create_lora.py calls.

        `owner` is only for a new row: an existing one keeps whoever it has --
        the API's reservation already says whose a queued training is.
        """
        row = self.by_folder(folder)
        if row is None:
            return self.add(self.free_name(name or name_for(folder), owner), folder,
                            status, origin, owner=owner, **fields)
        return self.update(row["id"], status=status, **fields)

    def refresh(self, row):
        """Re-read one row's folder. -> the row as it is now."""
        described = describe(absolute(row["folder"]))
        if described is None:
            if row["status"] in IN_FLIGHT or row["status"] in (FAILED, CANCELLED):
                return row
            return self.update(row["id"], status=MISSING, weights=0)
        if row["status"] in IN_FLIGHT or row["status"] in (FAILED, CANCELLED):
            return self.update(row["id"], **described)
        # An adapter is usable once its weights are there -- or, for a mocked
        # one, once its adapter_config.json is, which is all a mock has.
        usable = described["weights"] or described["mock"]
        return self.update(row["id"], status=READY if usable else MISSING, **described)

    def scan(self, root=None, owner=None):
        """Register every adapter folder under `root` (default loras/), and
        re-read the rows already there. -> {"added", "updated", "missing"}: names.

        A folder found for the first time belongs to `owner` (default nobody);
        rows already there keep whoever they have.

        A row in flight is left alone -- its folder is being written and its row
        is the only record of the training -- and so is a failed or cancelled
        one, whose folder may hold a half-written adapter.
        """
        report = {"added": [], "updated": [], "missing": []}
        seen = set()
        for folder in adapter_folders(root):
            row = self.by_folder(folder)
            seen.add(stored_folder(folder))
            if row is None:
                described = describe(folder)
                status = READY if described["weights"] or described["mock"] else MISSING
                row = self.add(self.free_name(name_for(folder), owner), folder, status,
                               "scanned", owner=owner, **described)
                report["added"].append(row["name"])
            elif row["status"] not in IN_FLIGHT + (FAILED, CANCELLED):
                self.refresh(row)
                report["updated"].append(row["name"])
        for row in self.all():
            if row["folder"] in seen or row["status"] in IN_FLIGHT + (FAILED, CANCELLED):
                continue
            if self.refresh(row)["status"] == MISSING:
                report["missing"].append(row["name"])
        return report


def as_dict(row):
    """A row as JSON-ready values, with the family it belongs to.

    The family is the adapter folder's own name: loras/Lora001..Lora005 each
    hold one member of the same set under the same name, which is how the five
    slots of one search were trained together.
    """
    out = {key: row[key] for key in row.keys()}
    for name in JSON_COLUMNS:
        try:
            out[name] = json.loads(out[name]) if out[name] else ({} if name == "recipe" else None)
        except ValueError:
            pass
    out["mock"] = bool(out["mock"])
    out["weights"] = bool(out["weights"])
    out["family"] = os.path.basename(row["folder"].rstrip("/\\"))
    return out


class _Closing:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self.conn

    def __exit__(self, *exc):
        self.conn.close()
        return False


# --- the command line ---------------------------------------------------------------

def _line(row):
    loss = "" if row["final_loss"] is None else "loss %.3f" % row["final_loss"]
    return "%4d  %-10s %-8s r%-3s %-44s %-46s %s" % (
        row["id"], row["status"], (row["owner"] or "-")[:8],
        row["rank"] if row["rank"] is not None else "?",
        row["name"][:44], (row["base_model"] or "?")[:46], loss)


def main(argv=None):
    parser = argparse.ArgumentParser(description="The catalogue of LoRA adapters on this machine.")
    parser.add_argument("--catalog", default=None,
                        help="the database the table is in (default the API's, api_jobs/api.sqlite; "
                             "or $GEP_LORA_CATALOG)")
    commands = parser.add_subparsers(dest="command", required=True)
    scan = commands.add_parser("scan", help="register every adapter folder under loras/")
    scan.add_argument("root", nargs="?", default=None, help="where to look (default loras/)")
    scan.add_argument("--owner", default=None,
                      help="the API user the folders found for the first time belong to "
                           "(default: nobody, which the API shows to no one)")
    listing = commands.add_parser("list", help="every row")
    listing.add_argument("--owner", default=None, help="only this user's rows")
    own = commands.add_parser("own", help="give rows to an API user -- the only one "
                                          "the API will show them to")
    own.add_argument("owner", help="the user's name, as `python -m gep_lora.service.users list` "
                                   "prints it; 'nobody' takes rows back")
    own.add_argument("loras", nargs="*", help="ids or names")
    own.add_argument("--unowned", action="store_true", help="every row nobody owns")
    show = commands.add_parser("show", help="one row, whole")
    show.add_argument("lora", help="its id or its name")
    forget = commands.add_parser("forget", help="drop a row; the folder is left alone")
    forget.add_argument("lora", help="its id or its name")
    commands.add_parser("models", help="base models downloaded here that a LoRA can be trained on")
    args = parser.parse_args(argv)

    if args.command == "models":
        for model in local_models():
            print("%-60s %6.2f GB  %s" % (model["id"], model["bytes"] / 1e9,
                                          ", ".join(model["architectures"])))
        return 0

    catalog = Catalog(args.catalog)
    if args.command == "own":
        owner = None if args.owner == "nobody" else args.owner
        rows = [catalog.find(key) for key in args.loras]
        if any(row is None for row in rows):
            raise SystemExit("no LoRA %s in %s" % (", ".join(
                key for key, row in zip(args.loras, rows) if row is None), catalog.path))
        if args.unowned:
            rows += list(catalog.all(owner=None))
        if not rows:
            raise SystemExit("name the LoRAs to give %s, or pass --unowned" % args.owner)
        try:
            changed = catalog.own(sorted({row["id"] for row in rows}), owner)
        except ValueError as error:
            raise SystemExit(str(error))
        print("%d LoRA(s) now belong to %s" % (changed, owner or "nobody"))
        return 0
    if args.command == "scan":
        report = catalog.scan(args.root, owner=args.owner)
        for kind in ("added", "updated", "missing"):
            print("%s: %d%s" % (kind, len(report[kind]),
                                ("  " + ", ".join(report[kind])) if report[kind] and kind != "updated" else ""))
        print("catalogue: %s" % catalog.path)
        return 0
    if args.command == "list":
        rows = catalog.all() if args.owner is None else catalog.all(owner=args.owner)
        for row in rows:
            print(_line(row))
        if not rows:
            print("the catalogue is empty; `python -m gep_lora.core.adapters.catalog scan` fills it from loras/")
        return 0
    row = catalog.find(args.lora)
    if row is None:
        raise SystemExit("no LoRA %r in %s" % (args.lora, catalog.path))
    if args.command == "show":
        print(json.dumps(as_dict(row), indent=2))
        return 0
    catalog.remove(row["id"])
    print("forgot %s (the folder %s is untouched)" % (row["name"], row["folder"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
