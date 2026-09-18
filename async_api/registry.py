"""
registry.py - Who submitted what, in which order, and what is live.

One small sqlite file, `<JOBS_DIR>/jobs.sqlite3`, beside the job folders:

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

from async_api import settings

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
# answers it already holds. See worker.command().
SEARCH, RESUME, EVALUATE = "search", "resume", "evaluate"
TASKS = (SEARCH, RESUME, EVALUATE)
# Which statuses each requeued task may be asked of: resuming a finished
# search has nothing left to do, and evaluating asks only for answers, which a
# finished search has and a stopped one may.
REQUEUE_FROM = {RESUME: RESUMABLE, EVALUATE: (DONE,) + RESUMABLE}

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
    options          TEXT NOT NULL DEFAULT '{}',  -- JSON, main.py options
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    pid              INTEGER,                 -- main.py's, while running
    exit_code        INTEGER,
    error            TEXT,
    task             TEXT NOT NULL DEFAULT 'search',  -- search | resume | evaluate
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

CREATE INDEX IF NOT EXISTS jobs_by_status ON jobs(status, id);
CREATE INDEX IF NOT EXISTS verifications_by_status ON verifications(status, id);
CREATE INDEX IF NOT EXISTS verifications_by_job ON verifications(job_id, id);
CREATE INDEX IF NOT EXISTS jobs_by_user ON jobs(user_id, id);
"""


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def digest(secret):
    """What is stored for a key or a token: never the secret itself."""
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def jobs_dir(value=None):
    value = value or settings.JOBS_DIR
    return os.path.abspath(value if os.path.isabs(value) else os.path.join(_ROOT, value))


class Registry:
    """The registry file. Opens a connection per call, so any thread may use it."""

    def __init__(self, root=None):
        self.root = jobs_dir(root)
        os.makedirs(self.root, exist_ok=True)
        self.path = os.path.join(self.root, "jobs.sqlite3")
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(SCHEMA)
            self._migrate(conn)

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
        A running one is flagged, and the worker that holds it kills main.py
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
                              % ("evaluation" if row["task"] == EVALUATE else row["task"]),
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
                row = conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
                if row is None or row["status"] not in allowed:
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
