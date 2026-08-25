"""
store.py - The sqlite database behind main_sqlite.py.

The text mode scatters a sweep across run/: population.txt, trees.txt,
index.txt, run_NNN.py, output_NNN.txt, output_result_NNN.json, results.txt.
That is easy to read and impossible to query -- "which chromosomes scored above
0.7, and under which weights" means opening a hundred files by hand, and a
second sweep overwrites the first.

This module holds the same information in one file, with the sweep itself as a
row so sweeps accumulate instead of replacing each other:

    runs          one sweep: when, which template, which interpreter, which commit
      settings    every knob it ran under, including the seeds
      individuals the population: chromosome, tree, rank, verdict, and the
                  generated script in full
        executions  one per time that individual was actually run: exit code,
                    seconds, the weight seed and the weights it drew, stdout,
                    stderr
          exchanges the questions and answers, and the judge's score for each

Everything a scorer needs is reachable from one query, and nothing is derived
from a filename. The only thing that still has to exist on disk is the generated
run_NNN.py scripts, because process runs them as subprocesses -- and those are a
cache of `individuals.script_source`, written out by materialise() whenever they
are missing.

Repeatability comes from the settings table: the population seed, and the weight
seed stamped into each individual's script. Re-running a sweep with the stored
values reproduces both the chromosomes and the blends they were judged under.

Usage as a module:

    conn = store.connect("run/gep.sqlite3")
    run_id = store.create_run(conn, template="template_code.py")
    store.save_settings(conn, run_id, settings.snapshot())

Usage from the command line:

    python store.py --list                  # the sweeps in the database
    python store.py --show 3                # one sweep, summarised
    python store.py --export 3 --into dump  # write it back out as text files
"""

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))


# --- schema ----------------------------------------------------------------

# `state`, `verdict` and the NNN numbering deliberately match the text mode's
# vocabulary, so a query reads the way index.txt and results.txt do.
SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY,
    created_at  TEXT    NOT NULL,
    label       TEXT,
    template    TEXT    NOT NULL,
    status      TEXT    NOT NULL DEFAULT 'open',   -- open | done | failed
    git_commit  TEXT,
    interpreter TEXT
);

CREATE TABLE IF NOT EXISTS settings (
    run_id  INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    key     TEXT    NOT NULL,
    value   TEXT,
    PRIMARY KEY (run_id, key)
);

CREATE TABLE IF NOT EXISTS individuals (
    id            INTEGER PRIMARY KEY,
    run_id        INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    number        INTEGER NOT NULL,        -- 1-based; the NNN in run_NNN.py
    chromosome    TEXT    NOT NULL,
    tree          TEXT,                    -- the drawing draw_trees.draw() makes
    state         TEXT,                    -- ok | BAD (PEFT's equal-rank rule)
    rank          INTEGER,                 -- rank of the final adapter
    script_name   TEXT,
    script_source TEXT,
    weight_seed   INTEGER,                 -- stamped into the script above
    UNIQUE (run_id, number)
);

CREATE TABLE IF NOT EXISTS executions (
    id            INTEGER PRIMARY KEY,
    individual_id INTEGER NOT NULL REFERENCES individuals(id) ON DELETE CASCADE,
    started_at    TEXT    NOT NULL,
    seconds       REAL,
    exit_code     INTEGER,                 -- NULL means it hit the timeout
    verdict       TEXT,                    -- ok | timeout | exit N
    weight_seed   INTEGER,
    weights       TEXT,                    -- JSON {"w1": .., .. } as drawn
    stdout        TEXT,
    stderr        TEXT
);

CREATE TABLE IF NOT EXISTS exchanges (
    id           INTEGER PRIMARY KEY,
    execution_id INTEGER NOT NULL REFERENCES executions(id) ON DELETE CASCADE,
    position     INTEGER NOT NULL,         -- 1-based, in the order asked
    question     TEXT    NOT NULL,
    answer       TEXT    NOT NULL DEFAULT '',
    quality      REAL,                     -- 0..1, NULL until judged
    reason       TEXT,
    judged_at    TEXT,
    judge_model  TEXT,
    UNIQUE (execution_id, position)
);

CREATE INDEX IF NOT EXISTS individuals_by_run ON individuals(run_id, number);
CREATE INDEX IF NOT EXISTS executions_by_individual ON executions(individual_id);
CREATE INDEX IF NOT EXISTS exchanges_by_execution ON exchanges(execution_id);

-- The fitness view: one row per individual, over its most recent execution.
-- Mean quality is what a selection step would sort on.
CREATE VIEW IF NOT EXISTS individual_quality AS
SELECT i.run_id      AS run_id,
       i.number      AS number,
       i.chromosome  AS chromosome,
       i.state       AS state,
       i.rank        AS rank,
       e.id          AS execution_id,
       e.verdict     AS verdict,
       e.weight_seed AS weight_seed,
       e.weights     AS weights,
       COUNT(x.id)                                   AS answers,
       SUM(CASE WHEN x.quality IS NULL THEN 1 ELSE 0 END) AS unscored,
       AVG(x.quality)                                AS quality
FROM individuals i
LEFT JOIN executions e
       ON e.id = (SELECT id FROM executions
                   WHERE individual_id = i.id
                   ORDER BY id DESC LIMIT 1)
LEFT JOIN exchanges x ON x.execution_id = e.id
GROUP BY i.id;
"""


class Database(sqlite3.Connection):
    """A connection that remembers which file it opened.

    sqlite3.Connection is a C type with no __dict__, so `conn.path = ...` is not
    allowed on one; a subclass can hold it, and every error message in the
    pipeline can then name the database it is talking about.
    """

    path = None


def connect(db_path):
    """Open (creating if need be) the database, with the schema in place."""
    path = db_path if os.path.isabs(db_path) else os.path.join(_HERE, db_path)
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    conn = sqlite3.connect(path, factory=Database)
    conn.path = path
    conn.row_factory = sqlite3.Row
    # Cascading deletes are off by default, and the schema leans on them.
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def _now():
    """A sortable timestamp. Stored as text: sqlite has no date type, and the
    adapters that used to paper over that are deprecated in 3.12+."""
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _git_commit():
    """The commit the code was at, or None outside a checkout.

    Worth recording: a stored sweep is only reproducible alongside the code that
    produced it, and this is the cheapest way to say which code that was.
    """
    try:
        done = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=_HERE, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() or None if done.returncode == 0 else None


# --- runs and their settings ----------------------------------------------


def create_run(conn, template, label=None):
    """Start a new sweep. Returns its id."""
    cursor = conn.execute(
        "INSERT INTO runs (created_at, label, template, status, git_commit, interpreter)"
        " VALUES (?, ?, ?, 'open', ?, ?)",
        (_now(), label, template, _git_commit(), sys.executable))
    conn.commit()
    return cursor.lastrowid


def finish_run(conn, run_id, status="done"):
    conn.execute("UPDATE runs SET status = ? WHERE id = ?", (status, run_id))
    conn.commit()


def latest_run(conn):
    """The most recent sweep's id, or None if the database is empty."""
    row = conn.execute("SELECT id FROM runs ORDER BY id DESC LIMIT 1").fetchone()
    return row["id"] if row else None


def get_run(conn, run_id):
    return conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()


def save_settings(conn, run_id, mapping):
    """Record what a sweep ran under. Values are stored as JSON so ints, floats,
    bools, None and strings all come back as themselves."""
    conn.executemany(
        "INSERT INTO settings (run_id, key, value) VALUES (?, ?, ?)"
        " ON CONFLICT(run_id, key) DO UPDATE SET value = excluded.value",
        [(run_id, key, json.dumps(value)) for key, value in sorted(mapping.items())])
    conn.commit()


def get_settings(conn, run_id):
    """The recorded settings, decoded back into Python values."""
    rows = conn.execute("SELECT key, value FROM settings WHERE run_id = ?",
                        (run_id,)).fetchall()
    out = {}
    for row in rows:
        try:
            out[row["key"]] = json.loads(row["value"])
        except (ValueError, TypeError):
            out[row["key"]] = row["value"]      # written by hand, keep the text
    return out


# --- the population --------------------------------------------------------


def add_individuals(conn, run_id, chromosomes):
    """Store a freshly drawn population. Replaces any already held for this run."""
    conn.execute("DELETE FROM individuals WHERE run_id = ?", (run_id,))
    conn.executemany(
        "INSERT INTO individuals (run_id, number, chromosome) VALUES (?, ?, ?)",
        [(run_id, number, chromosome)
         for number, chromosome in enumerate(chromosomes, 1)])
    conn.commit()


def individuals(conn, run_id, state=None):
    """The population in number order, optionally only those in one state."""
    sql = "SELECT * FROM individuals WHERE run_id = ?"
    args = [run_id]
    if state is not None:
        sql += " AND state = ?"
        args.append(state)
    return conn.execute(sql + " ORDER BY number", args).fetchall()


def set_tree(conn, individual_id, tree):
    conn.execute("UPDATE individuals SET tree = ? WHERE id = ?", (tree, individual_id))


def set_script(conn, individual_id, state, rank, script_name, source, weight_seed):
    conn.execute(
        "UPDATE individuals SET state = ?, rank = ?, script_name = ?,"
        " script_source = ?, weight_seed = ? WHERE id = ?",
        (state, rank, script_name, source, weight_seed, individual_id))


# --- executions and what they said ----------------------------------------


def add_execution(conn, individual_id, seconds, exit_code, verdict,
                  weight_seed, weights, stdout, stderr):
    """Record one execution of one individual. Returns its id.

    A row per execution rather than per individual: the same chromosome run
    again under a different weight seed is a second result, not a correction of
    the first.
    """
    cursor = conn.execute(
        "INSERT INTO executions (individual_id, started_at, seconds, exit_code,"
        " verdict, weight_seed, weights, stdout, stderr)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (individual_id, _now(), seconds, exit_code, verdict, weight_seed,
         json.dumps(weights or {}), stdout, stderr))
    return cursor.lastrowid


def add_exchanges(conn, execution_id, transcript):
    """Store the question/answer pairs of one execution.

    A mocked run arrives already carrying "quality" and "reason", so those are
    taken here when present and the judge never has to be asked -- the same way
    the text mode lands them in the JSON transcript.
    """
    conn.executemany(
        "INSERT INTO exchanges (execution_id, position, question, answer,"
        " quality, reason, judge_model) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(execution_id, position, item.get("question", ""), item.get("answer", ""),
          item.get("quality"), item.get("reason"),
          "generated" if "quality" in item else None)
         for position, item in enumerate(transcript, 1)])


def latest_execution(conn, individual_id):
    return conn.execute(
        "SELECT * FROM executions WHERE individual_id = ?"
        " ORDER BY id DESC LIMIT 1", (individual_id,)).fetchone()


def exchanges_to_score(conn, run_id, force=False):
    """Every exchange of the latest execution of each individual that still
    needs a score -- or all of them, with force."""
    sql = """
        SELECT x.*, i.number AS number, i.chromosome AS chromosome
        FROM exchanges x
        JOIN executions e ON e.id = x.execution_id
        JOIN individuals i ON i.id = e.individual_id
        WHERE i.run_id = ?
          AND e.id = (SELECT id FROM executions
                       WHERE individual_id = i.id ORDER BY id DESC LIMIT 1)
    """
    if not force:
        sql += " AND x.quality IS NULL"
    return conn.execute(sql + " ORDER BY i.number, x.position", (run_id,)).fetchall()


def score_exchange(conn, exchange_id, quality, reason, model):
    conn.execute(
        "UPDATE exchanges SET quality = ?, reason = ?, judged_at = ?, judge_model = ?"
        " WHERE id = ?", (quality, reason, _now(), model, exchange_id))


def quality_rows(conn, run_id):
    """The fitness view for one sweep, best first."""
    return conn.execute(
        "SELECT * FROM individual_quality WHERE run_id = ?"
        " ORDER BY quality IS NULL, quality DESC, number", (run_id,)).fetchall()


# --- the one thing that must be a file ------------------------------------


def materialise(conn, run_id, run_dir, force=False):
    """Write the generated scripts to disk, because process has to run them.

    They are a cache of individuals.script_source, not a second copy of the
    truth: an existing file is left alone unless it differs, and a deleted one
    simply comes back. Returns how many were written.

    run_dir must stay exactly one level below the project folder -- a generated
    script finds the LoRA folders and training_set.txt by going up one from
    itself, so a deeper folder would break every path in it.
    """
    os.makedirs(run_dir, exist_ok=True)
    written = 0
    for row in individuals(conn, run_id):
        if not row["script_source"]:
            continue                            # not through the runs step yet
        path = os.path.join(run_dir, row["script_name"])
        if not force and os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                if handle.read() == row["script_source"]:
                    continue
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(row["script_source"])
        written += 1
    return written


def remove_scripts(conn, run_id, run_dir, names=None):
    """Delete generated script files from disk. Returns how many went.

    The inverse of materialise(), and safe for the same reason it is: the text
    of every script is in individuals.script_source, so a deleted file comes
    back the moment anything needs it. Once a sweep has been processed the files
    have served their purpose -- they exist only because a script has to be a
    file to be launched -- and leaving a population's worth of them lying about
    only invites someone to run a stale one by hand.

    `names` limits it to those scripts; None means every one this run owns.
    Either way only names this run recorded are touched, never whatever else
    happens to be sitting in run_dir.
    """
    wanted = None if names is None else set(names)
    gone = 0
    for row in individuals(conn, run_id):
        name = row["script_name"]
        if not name or (wanted is not None and name not in wanted):
            continue
        try:
            os.remove(os.path.join(run_dir, name))
            gone += 1
        except OSError:
            # Already deleted, or held open by something: not worth failing a
            # finished sweep over a file that was only ever a cache.
            pass
    return gone


# --- reading a sweep back out ---------------------------------------------


def export_run(conn, run_id, out_dir):
    """Write one sweep out in the text mode's layout.

    The database is the store; this is for the times you want to read a sweep
    the old way -- diff two populations, grep a transcript, hand someone a
    folder. Returns the list of paths written.
    """
    os.makedirs(out_dir, exist_ok=True)
    rows = individuals(conn, run_id)
    if not rows:
        raise SystemExit("run %s holds no individuals" % run_id)
    written = []

    def write(name, text):
        path = os.path.join(out_dir, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        written.append(path)

    write("population.txt", "".join(row["chromosome"] + "\n" for row in rows))

    drawn = [row for row in rows if row["tree"]]
    if drawn:
        write("trees.txt", "\n\n\n".join(
            "#%d\n%s" % (row["number"], row["tree"]) for row in drawn) + "\n")

    scripted = [row for row in rows if row["script_source"]]
    if scripted:
        write("index.txt", "script      state rank  expression\n" + "\n".join(
            "%s  %-4s rank %-4d %s" % (row["script_name"], row["state"],
                                       row["rank"], row["chromosome"])
            for row in scripted) + "\n")
        for row in scripted:
            write(row["script_name"], row["script_source"])

    results = []
    for row in rows:
        execution = latest_execution(conn, row["id"])
        if execution is None:
            continue
        number = "%03d" % row["number"]
        write("output_%s.txt" % number,
              "# %s\n# %s\n# predicted rank %s, index says %s\n\n%s%s"
              % (row["script_name"], row["chromosome"], row["rank"], row["state"],
                 execution["stdout"] or "", execution["stderr"] or ""))

        transcript = [
            {key: item[key] for key in ("question", "answer", "quality", "reason")
             if item[key] is not None}
            for item in conn.execute(
                "SELECT * FROM exchanges WHERE execution_id = ? ORDER BY position",
                (execution["id"],)).fetchall()
        ]
        write("output_result_%s.json" % number,
              json.dumps({"chromosome": row["chromosome"],
                          "weights": json.loads(execution["weights"] or "{}"),
                          "exchanges": transcript},
                         indent=2, ensure_ascii=False) + "\n")
        results.append("%-14s %-5s %-8s %7.1f %5d  %-26s %s"
                       % (row["script_name"], row["state"], execution["verdict"],
                          execution["seconds"] or 0.0, len(transcript),
                          "output_result_%s.json" % number, row["chromosome"]))

    if results:
        write("results.txt", "%-14s %-5s %-8s %7s %5s  %-26s %s\n"
              % ("script", "state", "result", "secs", "qa", "transcript", "expression")
              + "\n".join(results) + "\n")
    return written


# --- driver ---------------------------------------------------------------


def summarise(conn, run_id):
    """Print one sweep: what it ran under, and how it came out."""
    run = get_run(conn, run_id)
    if run is None:
        raise SystemExit("no run %s in this database" % run_id)
    print("run %d  %s  %s  template=%s  commit=%s"
          % (run["id"], run["created_at"], run["status"], run["template"],
             run["git_commit"] or "?"))

    stored = get_settings(conn, run_id)
    if stored:
        print("\nsettings")
        for key, value in sorted(stored.items()):
            printable = str(value)
            if len(printable) > 60:
                printable = printable[:57] + "..."
            print("    %-20s %s" % (key, printable))

    rows = quality_rows(conn, run_id)
    print("\n%d individual(s)" % len(rows))
    print("    %-4s %-5s %-9s %-6s %-7s %s"
          % ("#", "state", "verdict", "answers", "quality", "chromosome"))
    for row in rows:
        print("    %-4d %-5s %-9s %-6d %-7s %s"
              % (row["number"], row["state"] or "-", row["verdict"] or "-",
                 row["answers"] or 0,
                 "-" if row["quality"] is None else "%.3f" % row["quality"],
                 row["chromosome"]))

    scored = [row["quality"] for row in rows if row["quality"] is not None]
    if scored:
        print("\nquality across %d individual(s): min %.3f, max %.3f, mean %.3f"
              % (len(scored), min(scored), max(scored), sum(scored) / len(scored)))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect the sweep database.")
    parser.add_argument("--db", default=None,
                        help="database file (default: settings.DB_PATH)")
    parser.add_argument("--list", action="store_true", help="list the sweeps")
    parser.add_argument("--show", type=int, metavar="RUN",
                        help="summarise one sweep (0 = the most recent)")
    parser.add_argument("--export", type=int, metavar="RUN",
                        help="write one sweep out as text files (0 = most recent)")
    parser.add_argument("--into", default="export",
                        help="folder for --export (default export)")
    args = parser.parse_args(argv)

    if args.db is None:
        import settings as _settings
        args.db = _settings.DB_PATH
    conn = connect(args.db)

    def resolve(value):
        chosen = latest_run(conn) if value == 0 else value
        if chosen is None:
            raise SystemExit("%s holds no runs yet" % conn.path)
        return chosen

    if args.export is not None:
        run_id = resolve(args.export)
        written = export_run(conn, run_id, args.into)
        print("wrote %d file(s) from run %d to %s" % (len(written), run_id, args.into))
        return 0

    if args.show is not None:
        summarise(conn, resolve(args.show))
        return 0

    rows = conn.execute("SELECT * FROM runs ORDER BY id").fetchall()
    if not rows:
        print("%s holds no runs yet" % conn.path)
        return 0
    print("%-4s %-20s %-7s %-26s %s" % ("id", "created", "status", "template", "label"))
    for row in rows:
        print("%-4d %-20s %-7s %-26s %s"
              % (row["id"], row["created_at"], row["status"], row["template"],
                 row["label"] or ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
