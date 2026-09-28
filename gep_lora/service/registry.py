"""
registry.py - Who submitted what, in which order, and what is live.

One sqlite file, `<JOBS_DIR>/api.sqlite`, beside the job folders -- everything
the API knows, in one place:

    users        a name and the hash of its API key
    jobs         one submitted sweep: its owner, its status, and the folder
                 holding its database. `id` is arrival order, which is the
                 order the worker takes them in. `task` is what the worker is
                 to do with it next -- run the search, resume it, or evaluate
                 its answers -- since a job goes back into the queue for the
                 last two (see requeue()).
    deployments  a job's individual put live: the hash of the token that
                 reaches it, where it went, and the blend spec it serves
    verifications  a blend of a finished job queued to be run beside the LoRAs
                 it is made of (see verify.py). Work to be done, like a job,
                 and taken by the same worker -- after the jobs, since a
                 verification is a question about a search that has finished
                 and a job is one still waiting to start.
    trainings    a LoRA queued to be trained (see train.py): the same kind of
                 work, taken by the same worker. The adapter itself is a row
                 of `loras`; this row is only the work of making it.
    loras        every LoRA adapter on the machine, and whose each is. Owned
                 by gep_lora/core/adapters/catalog.py, which create_lora.py writes through
                 with no API running, and created here from its SCHEMA so the
                 file is whole whichever opens it first.
    guide_defaults  one user's defaults for the LoRA guide (see
                 gep_lora/assistant/guide_defaults.py): a JSON document per
                 user, holding only the values they set.

It was two files until the LoRAs arrived -- `jobs.sqlite3` here and a
catalogue under loras/. A registry that finds a `jobs.sqlite3` and no
`api.sqlite` copies it across once (sqlite's backup, so a live WAL comes too)
and renames the old file `jobs.sqlite3.merged`; the LoRA rows are rebuilt from
disk by `python -m gep_lora.core.adapters.catalog scan`, since the folders are their truth.

What a job *is* lives in its own database (see submit.py) -- the same file
store.py writes for any sweep. This file only knows about jobs as work to be
done and results to be found, so a job folder copied somewhere else is still a
whole sweep without it.

The API server and the worker are separate processes on the same file, so
every write is a short transaction and claiming a job is atomic: two workers
can never take the same one.
"""

import hashlib
import json
import os
import secrets
import sqlite3
import time

from gep_lora.core.adapters import catalog as lora_catalog
from gep_lora.service import settings
from gep_lora import paths

_ROOT = paths.ROOT

PREPARING = "preparing"
QUEUED, RUNNING, DONE, FAILED, CANCELLED, STOPPED, DELETED = (
    "queued", "running", "done", "failed", "cancelled", "stopped", "deleted")
# A job in one of these will never change again on its own.
FINISHED = (DONE, FAILED, CANCELLED, STOPPED, DELETED)
# A search that did not get to its end, for whatever reason -- cancelled before
# it started, stopped while it ran, or failed -- and can be carried on from
# where it got to: the sweep database says how far that was.
RESUMABLE = (STOPPED, CANCELLED, FAILED)
# What the worker does with a job it claims. `search` runs the prepared sweep
# from the top; `resume` carries a started one on; `evaluate` grades the
# answers it already holds; `test` puts its blends in front of its testing
# split (see testpass.py). See worker.command().
SEARCH, RESUME, EVALUATE, TEST = "search", "resume", "evaluate", "test"
TASKS = (SEARCH, RESUME, EVALUATE, TEST)
# A blend drawn by hand on the visual guide (drawn.py): a sweep of one
# individual that no search produced. It is never queued -- it is `done` the
# moment its database is whole -- and never requeued, since there is no search
# to resume, evaluate or test; what is asked of it is a verification.
BLEND = "blend"
# Which statuses each requeued task may be asked of: resuming a finished
# search has nothing left to do, evaluating asks only for answers, which a
# finished search has and a stopped one may, and testing asks for the blends
# a search *found* -- a half-finished search's are not that.
REQUEUE_FROM = {RESUME: RESUMABLE, EVALUATE: (DONE,) + RESUMABLE, TEST: (DONE,)}

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    key_hash    TEXT NOT NULL UNIQUE,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id               INTEGER PRIMARY KEY,     -- arrival order
    user_id          INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    label            TEXT,
    status           TEXT NOT NULL DEFAULT 'queued',
    created_at       TEXT NOT NULL,
    started_at       TEXT,
    finished_at      TEXT,
    folder           TEXT NOT NULL,           -- relative to JOBS_DIR
    run_id           INTEGER,                 -- the sweep inside job.sqlite3
    options          TEXT NOT NULL DEFAULT '{}',  -- JSON, gep_lora/core/pipeline/main.py options
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    pid              INTEGER,                 -- gep_lora/core/pipeline/main.py's, while running
    exit_code        INTEGER,
    error            TEXT,
    task             TEXT NOT NULL DEFAULT 'search',  -- search | resume | evaluate | test
    task_options     TEXT NOT NULL DEFAULT '{}',      -- JSON, the task's own
    requeued_from    TEXT                     -- the status a requeue took it from
);

CREATE TABLE IF NOT EXISTS deployments (
    id          INTEGER PRIMARY KEY,
    job_id      INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash  TEXT NOT NULL UNIQUE,
    target      TEXT NOT NULL,
    number      INTEGER NOT NULL,             -- the individual that went live
    chromosome  TEXT NOT NULL,
    spec        TEXT NOT NULL,                -- JSON, see golive.blend_spec()
    created_at  TEXT NOT NULL,
    revoked_at  TEXT
);

CREATE TABLE IF NOT EXISTS verifications (
    id               INTEGER PRIMARY KEY,     -- arrival order, as for jobs
    job_id           INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    user_id          INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    status           TEXT NOT NULL DEFAULT 'queued',
    created_at       TEXT NOT NULL,
    started_at       TEXT,
    finished_at      TEXT,
    number           INTEGER NOT NULL,        -- the individual being verified
    chromosome       TEXT NOT NULL,           -- as it was when queued
    options          TEXT NOT NULL DEFAULT '{}',  -- JSON, see verify.py
    pid              INTEGER,
    exit_code        INTEGER,
    error            TEXT
);

CREATE TABLE IF NOT EXISTS trainings (
    id               INTEGER PRIMARY KEY,     -- arrival order, as for jobs
    user_id          INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    lora_id          INTEGER REFERENCES loras(id) ON DELETE SET NULL,
    name             TEXT NOT NULL,
    folder           TEXT NOT NULL,           -- the adapter folder it writes
    status           TEXT NOT NULL DEFAULT 'preparing',
    created_at       TEXT NOT NULL,
    started_at       TEXT,
    finished_at      TEXT,
    options          TEXT NOT NULL DEFAULT '{}',  -- JSON, see train.py
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    pid              INTEGER,
    exit_code        INTEGER,
    error            TEXT
);

CREATE TABLE IF NOT EXISTS guide_defaults (
    user_id     INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    doc         TEXT NOT NULL DEFAULT '{}',   -- JSON, {key: value}, only what was set
    updated_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS jobs_by_status ON jobs(status, id);
CREATE INDEX IF NOT EXISTS trainings_by_status ON trainings(status, id);
CREATE INDEX IF NOT EXISTS trainings_by_lora ON trainings(lora_id);
CREATE INDEX IF NOT EXISTS verifications_by_status ON verifications(status, id);
CREATE INDEX IF NOT EXISTS verifications_by_job ON verifications(job_id, id);
CREATE INDEX IF NOT EXISTS jobs_by_user ON jobs(user_id, id);
"""


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def digest(secret):
    """What is stored for a key or a token: never the secret itself."""
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


DATABASE = paths.API_DATABASE
# What the database was called before the LoRAs moved into it.
OLD_DATABASE = "jobs.sqlite3"


def jobs_dir(value=None):
    value = value or settings.JOBS_DIR
    return os.path.abspath(value if os.path.isabs(value) else os.path.join(_ROOT, value))


def database_path(root=None):
    """The API's database: users, jobs, trainings, deployments and LoRAs."""
    return os.path.join(jobs_dir(root), DATABASE)


def merge_old(root):
    """Carry a `jobs.sqlite3` over into a new `api.sqlite`, once. -> True if it did.

    Through sqlite's backup rather than a file copy, so the pages still in a
    WAL beside it come across. The old file is then renamed, not deleted: if
    another process still holds it (Windows will say so), it is left as it is
    and only the copy is new.
    """
    new, old = os.path.join(root, DATABASE), os.path.join(root, OLD_DATABASE)
    if os.path.exists(new) or not os.path.exists(old):
        return False
    source = sqlite3.connect(old)
    target = sqlite3.connect(new)
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()
    for suffix in ("", "-wal", "-shm"):
        try:
            os.replace(old + suffix, old + ".merged" + suffix)
        except OSError:
            pass
    return True


class Registry:
    """The registry file. Opens a connection per call, so any thread may use it."""

    def __init__(self, root=None):
        self.root = jobs_dir(root)
        os.makedirs(self.root, exist_ok=True)
        merge_old(self.root)
        self.path = database_path(self.root)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(lora_catalog.SCHEMA)
            conn.executescript(SCHEMA)
            self._migrate(conn)
        self.catalog = lora_catalog.Catalog(self.path)

    # Columns added after a registry may already have been written. CREATE
    # TABLE IF NOT EXISTS leaves an existing table as it is, so each is added
    # here, once, with the default a job from before it existed means.
    ADDED = (("jobs", "task", "TEXT NOT NULL DEFAULT 'search'"),
             ("jobs", "task_options", "TEXT NOT NULL DEFAULT '{}'"),
             ("jobs", "requeued_from", "TEXT"))

    def _migrate(self, conn):
        for table, column, declaration in self.ADDED:
            held = {row["name"] for row in conn.execute("PRAGMA table_info(%s)" % table)}
            if column not in held:
                conn.execute("ALTER TABLE %s ADD COLUMN %s %s" % (table, column, declaration))

    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return _Closing(conn)

    def folder(self, job):
        """The absolute folder of a job row."""
        return os.path.normpath(os.path.join(self.root, job["folder"]))

    def database(self, job):
        """The absolute path of a job's sweep database."""
        return os.path.join(self.folder(job), "job.sqlite3")

    def verification_folder(self, job, verification_id):
        """Where one verification's answers, scores and report go.

        Inside the job's own folder, so deleting the run takes its
        verifications with it -- they are readings of that sweep and mean
        nothing without it -- and named for the verification, so two of them
        (another judge, another blend) cannot tread on each other's answers.
        """
        return os.path.join(self.folder(job), "verify%d" % verification_id)

    # --- users -------------------------------------------------------------

    def add_user(self, name):
        """-> the new user's API key. Shown once; only its hash is kept."""
        key = "gep_" + secrets.token_urlsafe(32)
        with self._connect() as conn:
            conn.execute("INSERT INTO users (name, key_hash, created_at) VALUES (?, ?, ?)",
                         (name, digest(key), now()))
        return key

    def rotate_key(self, name):
        key = "gep_" + secrets.token_urlsafe(32)
        with self._connect() as conn:
            done = conn.execute("UPDATE users SET key_hash = ? WHERE name = ?",
                                (digest(key), name))
        return key if done.rowcount else None

    def remove_user(self, name):
        with self._connect() as conn:
            return conn.execute("DELETE FROM users WHERE name = ?", (name,)).rowcount

    def users(self):
        with self._connect() as conn:
            return conn.execute("SELECT id, name, created_at FROM users ORDER BY id").fetchall()

    def user_for_key(self, key):
        if not key:
            return None
        with self._connect() as conn:
            return conn.execute("SELECT * FROM users WHERE key_hash = ?",
                                (digest(key),)).fetchone()

    # --- the guide's defaults ----------------------------------------------

    def guide_defaults(self, user_id):
        """-> (the user's saved defaults, when they were saved), or ({}, None)."""
        with self._connect() as conn:
            row = conn.execute("SELECT doc, updated_at FROM guide_defaults WHERE user_id = ?",
                               (user_id,)).fetchone()
        if row is None:
            return {}, None
        try:
            doc = json.loads(row["doc"])
        except ValueError:
            doc = {}
        return (doc if isinstance(doc, dict) else {}), row["updated_at"]

    def set_guide_defaults(self, user_id, doc):
        """Replace the user's saved defaults whole; an empty one removes the row."""
        with self._connect() as conn:
            if not doc:
                conn.execute("DELETE FROM guide_defaults WHERE user_id = ?", (user_id,))
                return None
            stamp = now()
            conn.execute("INSERT INTO guide_defaults (user_id, doc, updated_at) VALUES (?, ?, ?) "
                         "ON CONFLICT(user_id) DO UPDATE SET doc = excluded.doc, "
                         "updated_at = excluded.updated_at",
                         (user_id, json.dumps(doc, sort_keys=True), stamp))
            return stamp

    # --- jobs --------------------------------------------------------------

    def reserve_job(self, user_id, label, options):
        """A job row in no queue yet. -> the row.

        Status `preparing` is not in the queue: the database is written after
        the row exists (the folder is named for the id), and the worker must
        not see a job whose database is half there. enqueue() finishes it.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO jobs (user_id, label, status, created_at, folder, options)"
                " VALUES (?, ?, 'preparing', ?, '', ?)",
                (user_id, label, now(), json.dumps(options or {})))
            job_id = cursor.lastrowid
            conn.execute("UPDATE jobs SET folder = ? WHERE id = ?",
                         ("user%d/job%d" % (user_id, job_id), job_id))
        return self.job(job_id)

    def enqueue(self, job_id, run_id):
        with self._connect() as conn:
            conn.execute("UPDATE jobs SET status = 'queued', run_id = ? WHERE id = ?",
                         (run_id, job_id))
        return self.job(job_id)

    def settle_drawn(self, job_id, run_id):
        """Finish a reserved job as a drawn blend: done, with nothing to run.
        The row was `preparing` until its database was whole, as a submitted
        job's is, so the worker never saw it."""
        stamp = now()
        with self._connect() as conn:
            conn.execute("UPDATE jobs SET status = 'done', task = ?, run_id = ?,"
                         " started_at = ?, finished_at = ?, exit_code = 0 WHERE id = ?",
                         (BLEND, run_id, stamp, stamp, job_id))
        return self.job(job_id)

    def discard(self, job_id):
        """Drop a job row that never made it into the queue."""
        with self._connect() as conn:
            conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

    def job(self, job_id, user_id=None):
        sql, args = "SELECT * FROM jobs WHERE id = ?", [job_id]
        if user_id is not None:
            sql += " AND user_id = ?"
            args.append(user_id)
        with self._connect() as conn:
            return conn.execute(sql, args).fetchone()

    def jobs(self, user_id, status=None):
        sql, args = "SELECT * FROM jobs WHERE user_id = ?", [user_id]
        if status:
            sql += " AND status = ?"
            args.append(status)
        with self._connect() as conn:
            return conn.execute(sql + " ORDER BY id DESC", args).fetchall()

    def queue_position(self, job_id):
        """1 for the next job the worker takes, None when not queued."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS ahead FROM jobs WHERE status = 'queued' AND id <= ?"
                " AND EXISTS (SELECT 1 FROM jobs WHERE id = ? AND status = 'queued')",
                (job_id, job_id)).fetchone()
        return row["ahead"] or None

    def claim_next(self):
        """Take the oldest queued job and mark it running. -> the row, or None."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute("SELECT id FROM jobs WHERE status = 'queued'"
                                   " ORDER BY id LIMIT 1").fetchone()
                if row is None:
                    conn.execute("COMMIT")
                    return None
                conn.execute("UPDATE jobs SET status = 'running', started_at = ?"
                             " WHERE id = ?", (now(), row["id"]))
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return self.job(row["id"])

    def rename_job(self, job_id, label):
        with self._connect() as conn:
            conn.execute("UPDATE jobs SET label = ? WHERE id = ?", (label, job_id))

    def set_pid(self, job_id, pid):
        with self._connect() as conn:
            conn.execute("UPDATE jobs SET pid = ? WHERE id = ?", (pid, job_id))

    def finish(self, job_id, status, exit_code=None, error=None):
        with self._connect() as conn:
            conn.execute("UPDATE jobs SET status = ?, finished_at = ?, exit_code = ?,"
                         " error = ?, pid = NULL, requeued_from = NULL WHERE id = ?",
                         (status, now(), exit_code, error, job_id))

    def request_cancel(self, job_id):
        """-> the job's status after asking.

        A queued job is cancelled on the spot -- or, when it was queued again
        to be resumed or evaluated, put back in the status it was queued from.
        A running one is flagged, and the worker that holds it kills gep_lora/core/pipeline/main.py
        and marks it (stopped, for a search: see worker.run_job). Anything else
        has nothing to cancel and is returned as it stands.
        """
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT status, task, requeued_from FROM jobs WHERE id = ?",
                               (job_id,)).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return None
            if row["status"] == QUEUED and row["requeued_from"]:
                # A resume or an evaluation not started yet: taking it back out
                # of the queue leaves the job as it was before it was asked.
                status = row["requeued_from"]
                conn.execute("UPDATE jobs SET status = ?, finished_at = ?,"
                             " requeued_from = NULL, error = ? WHERE id = ?",
                             (status, now(),
                              "the %s was cancelled before it started"
                              % {EVALUATE: "evaluation", TEST: "testing pass"}.get(
                                  row["task"], row["task"]),
                              job_id))
            elif row["status"] in (QUEUED, PREPARING):
                conn.execute("UPDATE jobs SET status = 'cancelled', finished_at = ?,"
                             " cancel_requested = 1 WHERE id = ?", (now(), job_id))
                status = CANCELLED
            elif row["status"] == RUNNING:
                conn.execute("UPDATE jobs SET cancel_requested = 1 WHERE id = ?", (job_id,))
                status = RUNNING
            else:
                status = row["status"]
            conn.execute("COMMIT")
        return status

    def requeue(self, job_id, task, task_options=None):
        """Put a job that has stopped back in the queue for `task`. -> the row,
        or None when its status does not allow that task.

        Atomic, like a claim, so a resume asked twice queues it once. The job
        keeps its id, and with it its place: the queue is in arrival order, and
        it arrived before anything submitted since. What it was taken from is
        kept in `requeued_from`, so a cancel before the worker gets to it puts
        it back.
        """
        allowed = REQUEUE_FROM[task]
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute("SELECT status, task FROM jobs WHERE id = ?",
                                   (job_id,)).fetchone()
                if row is None or row["status"] not in allowed or row["task"] == BLEND:
                    conn.execute("COMMIT")
                    return None
                conn.execute(
                    "UPDATE jobs SET status = 'queued', task = ?, task_options = ?,"
                    " requeued_from = ?, cancel_requested = 0, pid = NULL,"
                    " exit_code = NULL, error = NULL, finished_at = NULL WHERE id = ?",
                    (task, json.dumps(task_options or {}), row["status"], job_id))
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return self.job(job_id)

    def cancel_requested(self, job_id):
        row = self.job(job_id)
        return bool(row and row["cancel_requested"])

    def orphaned(self):
        """Jobs marked running with no worker behind them (a worker that died)."""
        with self._connect() as conn:
            return conn.execute("SELECT * FROM jobs WHERE status = 'running'"
                                " ORDER BY id").fetchall()

    def mark_deleted(self, job_id):
        with self._connect() as conn:
            conn.execute("UPDATE jobs SET status = 'deleted' WHERE id = ?", (job_id,))

    def delete_job(self, job_id):
        with self._connect() as conn:
            conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

    # --- verifications -----------------------------------------------------

    def add_verification(self, job, number, chromosome, options):
        """Queue one verification of a job's blend. -> the row.

        Queued outright, unlike a job: there is nothing to write first. The
        sweep it reads is the job's own database, already on disk.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO verifications (job_id, user_id, status, created_at,"
                " number, chromosome, options) VALUES (?, ?, 'queued', ?, ?, ?, ?)",
                (job["id"], job["user_id"], now(), number, chromosome,
                 json.dumps(options or {})))
            row_id = cursor.lastrowid
        return self.verification(row_id)

    def verification(self, verification_id, user_id=None):
        sql, args = "SELECT * FROM verifications WHERE id = ?", [verification_id]
        if user_id is not None:
            sql += " AND user_id = ?"
            args.append(user_id)
        with self._connect() as conn:
            return conn.execute(sql, args).fetchone()

    def verifications(self, job_id=None, user_id=None):
        sql, args = "SELECT * FROM verifications WHERE 1 = 1", []
        if job_id is not None:
            sql += " AND job_id = ?"
            args.append(job_id)
        if user_id is not None:
            sql += " AND user_id = ?"
            args.append(user_id)
        with self._connect() as conn:
            return conn.execute(sql + " ORDER BY id DESC", args).fetchall()

    def claim_next_verification(self):
        """Take the oldest queued verification. -> the row, or None.

        The same atomic claim jobs get, and for the same reason: the server and
        the worker are separate processes on one file.
        """
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute("SELECT id FROM verifications WHERE status = 'queued'"
                                   " ORDER BY id LIMIT 1").fetchone()
                if row is None:
                    conn.execute("COMMIT")
                    return None
                conn.execute("UPDATE verifications SET status = 'running', started_at = ?"
                             " WHERE id = ?", (now(), row["id"]))
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return self.verification(row["id"])

    def set_verification_pid(self, verification_id, pid):
        with self._connect() as conn:
            conn.execute("UPDATE verifications SET pid = ? WHERE id = ?",
                         (pid, verification_id))

    def finish_verification(self, verification_id, status, exit_code=None, error=None):
        with self._connect() as conn:
            conn.execute("UPDATE verifications SET status = ?, finished_at = ?,"
                         " exit_code = ?, error = ?, pid = NULL WHERE id = ?",
                         (status, now(), exit_code, error, verification_id))

    def orphaned_verifications(self):
        """Verifications left running by a worker that died."""
        with self._connect() as conn:
            return conn.execute("SELECT * FROM verifications WHERE status = 'running'"
                                " ORDER BY id").fetchall()

    # --- trainings ---------------------------------------------------------

    def training_folder(self, row):
        """Where one training's dataset and log live: beside the user's jobs.

        Not the adapter folder, which is the catalogue's and outlives the API;
        this is the work of making it, and goes when the training row does.
        """
        return os.path.join(self.root, "user%d" % row["user_id"], "lora%d" % row["id"])

    def reserve_training(self, user_id, name, folder, options):
        """A training row in no queue yet (its dataset is written next). -> it."""
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO trainings (user_id, name, folder, status, created_at, options)"
                " VALUES (?, ?, ?, 'preparing', ?, ?)",
                (user_id, name, folder, now(), json.dumps(options or {})))
            row_id = cursor.lastrowid
        return self.training(row_id)

    def enqueue_training(self, training_id, lora_id):
        with self._connect() as conn:
            conn.execute("UPDATE trainings SET status = 'queued', lora_id = ? WHERE id = ?",
                         (lora_id, training_id))
        return self.training(training_id)

    def discard_training(self, training_id):
        with self._connect() as conn:
            conn.execute("DELETE FROM trainings WHERE id = ?", (training_id,))

    def training(self, training_id, user_id=None):
        sql, args = "SELECT * FROM trainings WHERE id = ?", [training_id]
        if user_id is not None:
            sql += " AND user_id = ?"
            args.append(user_id)
        with self._connect() as conn:
            return conn.execute(sql, args).fetchone()

    def training_for_lora(self, lora_id):
        """The newest training that filled a catalogue row, or None."""
        with self._connect() as conn:
            return conn.execute("SELECT * FROM trainings WHERE lora_id = ?"
                                " ORDER BY id DESC LIMIT 1", (lora_id,)).fetchone()

    def trainings(self, user_id=None, status=None):
        sql, args = "SELECT * FROM trainings WHERE 1 = 1", []
        if user_id is not None:
            sql += " AND user_id = ?"
            args.append(user_id)
        if status:
            sql += " AND status = ?"
            args.append(status)
        with self._connect() as conn:
            return conn.execute(sql + " ORDER BY id DESC", args).fetchall()

    def training_queue_position(self, training_id):
        """1 for the next training the worker takes, None when not queued."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS ahead FROM trainings WHERE status = 'queued' AND id <= ?"
                " AND EXISTS (SELECT 1 FROM trainings WHERE id = ? AND status = 'queued')",
                (training_id, training_id)).fetchone()
        return row["ahead"] or None

    def claim_next_training(self):
        """Take the oldest queued training. -> the row, or None."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute("SELECT id FROM trainings WHERE status = 'queued'"
                                   " ORDER BY id LIMIT 1").fetchone()
                if row is None:
                    conn.execute("COMMIT")
                    return None
                conn.execute("UPDATE trainings SET status = 'running', started_at = ?"
                             " WHERE id = ?", (now(), row["id"]))
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return self.training(row["id"])

    def set_training_pid(self, training_id, pid):
        with self._connect() as conn:
            conn.execute("UPDATE trainings SET pid = ? WHERE id = ?", (pid, training_id))

    def finish_training(self, training_id, status, exit_code=None, error=None):
        with self._connect() as conn:
            conn.execute("UPDATE trainings SET status = ?, finished_at = ?, exit_code = ?,"
                         " error = ?, pid = NULL WHERE id = ?",
                         (status, now(), exit_code, error, training_id))

    def request_training_cancel(self, training_id):
        """-> the training's status after asking: cancelled on the spot when
        queued, flagged for the worker when running, as it was otherwise."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT status FROM trainings WHERE id = ?",
                               (training_id,)).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return None
            if row["status"] in (QUEUED, PREPARING):
                conn.execute("UPDATE trainings SET status = 'cancelled', finished_at = ?,"
                             " cancel_requested = 1, error = ? WHERE id = ?",
                             (now(), "cancelled before it started", training_id))
                status = CANCELLED
            elif row["status"] == RUNNING:
                conn.execute("UPDATE trainings SET cancel_requested = 1 WHERE id = ?",
                             (training_id,))
                status = RUNNING
            else:
                status = row["status"]
            conn.execute("COMMIT")
        return status

    def training_cancel_requested(self, training_id):
        row = self.training(training_id)
        return bool(row and row["cancel_requested"])

    def orphaned_trainings(self):
        """Trainings left running by a worker that died."""
        with self._connect() as conn:
            return conn.execute("SELECT * FROM trainings WHERE status = 'running'"
                                " ORDER BY id").fetchall()

    def delete_training(self, training_id):
        with self._connect() as conn:
            conn.execute("DELETE FROM trainings WHERE id = ?", (training_id,))

    # --- deployments -------------------------------------------------------

    def add_deployment(self, job, target, number, chromosome, spec):
        """-> (token, deployment row). The token is shown once."""
        token = "live_" + secrets.token_urlsafe(32)
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO deployments (job_id, user_id, token_hash, target, number,"
                " chromosome, spec, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (job["id"], job["user_id"], digest(token), target, number,
                 chromosome, json.dumps(spec), now()))
            row = conn.execute("SELECT * FROM deployments WHERE id = ?",
                               (cursor.lastrowid,)).fetchone()
        return token, row

    def deployment_for_token(self, token):
        """The live deployment a token reaches, or None (unknown or revoked)."""
        if not token:
            return None
        with self._connect() as conn:
            return conn.execute(
                "SELECT d.* FROM deployments d JOIN jobs j ON j.id = d.job_id"
                " WHERE d.token_hash = ? AND d.revoked_at IS NULL", (digest(token),)).fetchone()

    def deployments(self, user_id=None, job_id=None, live_only=True):
        sql, args = "SELECT * FROM deployments WHERE 1 = 1", []
        if user_id is not None:
            sql += " AND user_id = ?"
            args.append(user_id)
        if job_id is not None:
            sql += " AND job_id = ?"
            args.append(job_id)
        if live_only:
            sql += " AND revoked_at IS NULL"
        with self._connect() as conn:
            return conn.execute(sql + " ORDER BY id DESC", args).fetchall()

    def deployment(self, deployment_id, user_id=None):
        sql, args = "SELECT * FROM deployments WHERE id = ?", [deployment_id]
        if user_id is not None:
            sql += " AND user_id = ?"
            args.append(user_id)
        with self._connect() as conn:
            return conn.execute(sql, args).fetchone()

    def revoke(self, deployment_ids):
        if not deployment_ids:
            return 0
        with self._connect() as conn:
            return conn.execute(
                "UPDATE deployments SET revoked_at = ? WHERE revoked_at IS NULL AND id IN (%s)"
                % ",".join("?" * len(deployment_ids)),
                [now()] + list(deployment_ids)).rowcount


class _Closing:
    """`with registry._connect() as conn:` that closes, which sqlite3's own
    context manager does not (it only commits)."""

    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self.conn

    def __exit__(self, *exc):
        self.conn.close()
        return False
