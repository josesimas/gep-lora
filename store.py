"""
store.py - The sqlite database behind main.py.

A sweep scattered across a folder -- a population file, a tree file, an index,
a script, an output and a transcript per individual -- is easy to read and
impossible to query: "which chromosomes scored above 0.7, and under which
weights" means opening a hundred files by hand, and the next sweep overwrites
them all.

This module holds the whole of a sweep in one file, with the sweep itself as a
row so sweeps accumulate instead of replacing each other:

    runs          one sweep: when, which template, which interpreter, which commit
      settings    every knob it ran under, including the seeds
      datasets    the training/validation/testing records it was given, saved
                  the moment the sweep was created
      individuals the population: chromosome, tree, rank, verdict, and the
                  generated script in full
        executions  one per time that individual was actually run: exit code,
                    seconds, the weight seed and the weights it drew, stdout,
                    stderr
          exchanges the questions and answers, and the judge's score for each
      fitness_history  what each individual's fitness was at the end of each
                    generation, and when it was worked out

    baselines     what the base model itself answered, per model and question.
                  The one table outside that tree: it belongs to the model
                  rather than to a sweep, so every sweep that grades against
                  the base model reads the same cache instead of producing it
                  again.

Everything a scorer needs is reachable from one query, and nothing is derived
from a filename. The only thing that still has to exist on disk is the generated
run_NNN.py scripts, because process runs them as subprocesses -- and those are a
cache of `individuals.script_source`, written out by materialise() whenever they
are missing.

Repeatability comes from the settings table: the population seed, and the weight
seed stamped into each individual's script. Re-running a sweep with the stored
values reproduces both the chromosomes and the blends they were judged under.

Usage as a module:

    conn = store.connect("run_db/gep.sqlite3")
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

# `state` is ok | BAD as the generator decides it, `verdict` is what the exit
# code came to, and the NNN numbering is the individual's own number -- the same
# vocabulary the generated scripts and their output use.
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

-- The dataset the sweep was given, one row per record, saved at the moment the
-- sweep is created and never touched again -- for the reason the settings are:
-- a fitness number means "this blend, under these knobs, on these questions",
-- and the file the questions came from goes on being edited, repointed and
-- regenerated long after the sweep that was judged on it.
--
-- `split` is which of the three the row belongs to. Only the training split is
-- read by the search today (it is TRAINING_SET, capped by TRAINING_COUNT, and
-- what every generated script asks); validation and testing are stored when
-- settings name them so that a later pass has the questions this sweep was
-- built beside rather than whatever those files hold by then.
--
-- Every record of the file is stored, uncapped: TRAINING_COUNT says how many
-- an individual is judged on, which is a fact about the sweep and already in
-- the settings table, not a fact about the dataset. `position` is 1-based over
-- non-blank lines from the top, so it lines up with exchanges.position the way
-- generate_runs.eval_records() does. `content` is the line as it stood --
-- question and reference are this repo's reading of it, and keeping the line
-- means a later reader can disagree.
CREATE TABLE IF NOT EXISTS datasets (
    id        INTEGER PRIMARY KEY,
    run_id    INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    split     TEXT    NOT NULL
              CHECK (split IN ('training', 'validation', 'testing')),
    source    TEXT    NOT NULL,      -- the file it was read from, resolved
    position  INTEGER NOT NULL,      -- 1-based, non-blank lines from the top
    question  TEXT    NOT NULL,      -- the user turn: what a script would ask
    reference TEXT,                  -- the assistant turn, when the record has one
    content   TEXT    NOT NULL,      -- the whole line, as it was read
    UNIQUE (run_id, split, position)
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
    fitness       REAL DEFAULT 0.0,
    is_best       INTEGER DEFAULT 0,
    has_changed   INTEGER DEFAULT 0,
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

-- Fitness as it stood at the end of each generation, one row per individual
-- per generation. individuals.fitness only ever holds the latest value and
-- mutation clears it outright, so without this a sweep can say how fit its
-- population is now and nothing whatever about how it got there.
--
-- A row is self-contained for the reason an execution is: it keeps the
-- chromosome and the verdict it was scored under, because the individual it
-- names goes on being rewritten -- mutation replaces the chromosome, the next
-- generation replaces the fitness -- and a history that has to join back to the
-- current population to be read would report the past in terms of the present.
CREATE TABLE IF NOT EXISTS fitness_history (
    id          INTEGER PRIMARY KEY,
    run_id      INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    generation  INTEGER NOT NULL,        -- 1-based, in the order fitness ran
    population  INTEGER NOT NULL,        -- individuals the generation held
    recorded_at TEXT    NOT NULL,        -- when the fitness step wrote the row
    number      INTEGER NOT NULL,        -- the individual's own number
    chromosome  TEXT    NOT NULL,        -- as it was then, not as it is now
    state       TEXT,                    -- ok | BAD, as it was then
    fitness     REAL,                    -- what individuals.fitness was given
    answers     INTEGER,                 -- exchanges the mean was taken over
    unscored    INTEGER,                 -- how many of those carried no quality
    UNIQUE (run_id, generation, number)
);

-- What the base model, with no adapter attached, says to each eval prompt.
--
-- The one table here that hangs off no run, on purpose: a base-model answer
-- belongs to the model and the question and to nothing else, so the sweep that
-- happened to pay for it is not part of its identity. Every later sweep on the
-- same base model reads these rather than loading the model again, which is the
-- whole reason they are stored -- producing them costs a model load and one
-- generate() per prompt.
--
-- `model` is BASE_MODEL as the sweep recorded it, with a "mock:" prefix for
-- answers a mocked baseline invented, so a dry run can never leave made-up
-- answers where a real sweep would find them. `question_key` is the question
-- reduced to what makes two of them the same question (see question_key), and
-- is what the lookup matches on; `question` keeps the wording as it was asked.
CREATE TABLE IF NOT EXISTS baselines (
    id           INTEGER PRIMARY KEY,
    model        TEXT NOT NULL,          -- BASE_MODEL, or mock:<BASE_MODEL>
    question_key TEXT NOT NULL,          -- the question, normalised for matching
    question     TEXT NOT NULL,          -- as it was asked
    answer       TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL,
    source       TEXT,                   -- which template produced it
    UNIQUE (model, question_key)
);

-- What a finished individual says on a dataset it was never scored on.
--
-- The search judges every individual on the training split, over and over, and
-- an individual that scores well there has been selected *for* that split. The
-- question a testing pass asks is the other one: does the blend hold up on
-- questions it was never picked for? test_run_with_dataset.py takes the ones
-- worth asking it of, re-points their own scripts at another dataset and runs
-- them again, and this is where the answers land.
--
-- Deliberately not a second `executions` table hanging off `individuals`:
-- fitness, elitism and selection all read the *latest execution* of an
-- individual, so a test run stored there would be picked up as that
-- individual's current result and would decide the next generation on the
-- strength of questions the search is not judged on. Its own table cannot be
-- read by mistake.
--
-- A row is self-contained the way an execution row is, and for the reason
-- fitness_history rows are: the individual it names goes on being rewritten by
-- mutation and selection, so the number, the chromosome, the weights and the
-- quality that got it picked are all kept as they were at the moment it was
-- tested. `exchanges` is the whole transcript as JSON -- question, answer, and
-- whatever quality the run itself carried -- because a testing pass is read
-- back whole and a script that crashed still leaves a row saying so, with an
-- empty transcript and its exit code. The test_answers view below unnests it
-- for the times a query wants one row per question.
CREATE TABLE IF NOT EXISTS test_results (
    id            INTEGER PRIMARY KEY,
    run_id        INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    individual_id INTEGER NOT NULL REFERENCES individuals(id) ON DELETE CASCADE,
    number        INTEGER NOT NULL,      -- the individual's own number, as it was
    chromosome    TEXT    NOT NULL,      -- what the script that ran builds
    dataset       TEXT    NOT NULL,      -- the file it was tested on, resolved
    records       INTEGER,               -- how many of its questions it was asked
    selected_on   REAL,                  -- the training quality that got it picked
    started_at    TEXT    NOT NULL,
    seconds       REAL,
    exit_code     INTEGER,               -- NULL means it hit the timeout
    verdict       TEXT,                  -- ok | timeout | exit N
    weight_seed   INTEGER,
    weights       TEXT,                  -- JSON {"w1": .., .. } as drawn
    stdout        TEXT,
    stderr        TEXT,
    exchanges     TEXT    NOT NULL DEFAULT '[]',  -- JSON, the whole transcript
    -- Filled by the scoring pass, which is a second visit: a pass stores the
    -- answers first and grades them after, so these are NULL until it has.
    quality       REAL,                   -- mean over this row's scored answers
    evaluator     TEXT,                   -- which one gave those scores
    judge_model   TEXT,                   -- its label: a model id, or a method
    scored_at     TEXT
);

CREATE INDEX IF NOT EXISTS individuals_by_run ON individuals(run_id, number);
CREATE INDEX IF NOT EXISTS executions_by_individual ON executions(individual_id);
CREATE INDEX IF NOT EXISTS exchanges_by_execution ON exchanges(execution_id);
CREATE INDEX IF NOT EXISTS fitness_history_by_run ON fitness_history(run_id, generation);
CREATE INDEX IF NOT EXISTS test_results_by_run ON test_results(run_id, number);

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

-- One row per answer of a testing pass, out of the JSON each test_results row
-- carries. The table stores a transcript whole because that is how a pass is
-- written and read back; this is for the questions that want it the other way
-- round -- which chromosome said what to which question.
CREATE VIEW IF NOT EXISTS test_answers AS
SELECT t.id         AS test_result_id,
       t.run_id     AS run_id,
       t.number     AS number,
       t.chromosome AS chromosome,
       t.dataset    AS dataset,
       t.verdict    AS verdict,
       j.key + 1    AS position,
       json_extract(j.value, '$.question') AS question,
       json_extract(j.value, '$.answer')   AS answer,
       json_extract(j.value, '$.quality')  AS quality,
       json_extract(j.value, '$.reason')   AS reason
FROM test_results t, json_each(t.exchanges) j;

CREATE VIEW IF NOT EXISTS individual_stats AS
select i.number, i.chromosome, SUM(ex.quality) AS SumQuality, Max(ex.quality) AS MaxQuality, MIN(ex.quality) AS MinQuality, AVG(ex.quality) AS AvgQuality
from individuals i inner join executions e on i.id = e.individual_id inner join exchanges ex on e.id = ex.execution_id
where i.state = 'ok'
group by i.number, chromosome
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


# --- the dataset the sweep was given --------------------------------------

# The three splits, in the order they are worked through and reported. The
# schema names them too (datasets.split is CHECK'd against exactly these), so
# a fourth split is a schema change rather than a new string somewhere.
SPLITS = ("training", "validation", "testing")


def save_dataset(conn, run_id, split, source, records):
    """Store one split of a sweep's dataset. -> how many records were written.

    `records` is generate_runs.dataset_records() output -- position, question,
    reference and the line it came from -- and `source` the file they were read
    from, resolved, so the row says where it came from as well as what it said.

    Written whole, replacing whatever that split held: a dataset is saved once,
    at sweep creation, and re-saving one is either a re-run of that moment or a
    correction, never half of each. The rows themselves are never edited
    afterwards -- the point of them is that they say what this sweep was built
    on, and a sweep's dataset changing under it would be the thing they exist to
    prevent.
    """
    if split not in SPLITS:
        raise ValueError("unknown dataset split %r; the table holds %s"
                         % (split, ", ".join(SPLITS)))
    conn.execute("DELETE FROM datasets WHERE run_id = ? AND split = ?",
                 (run_id, split))
    conn.executemany(
        "INSERT INTO datasets (run_id, split, source, position, question,"
        " reference, content) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(run_id, split, source, record["position"], record["question"],
          record["reference"], record["content"]) for record in records])
    conn.commit()
    return len(records)


def dataset(conn, run_id, split):
    """One split's records, in file order."""
    return conn.execute(
        "SELECT * FROM datasets WHERE run_id = ? AND split = ? ORDER BY position",
        (run_id, split)).fetchall()


def dataset_summary(conn, run_id):
    """[{split, source, records, references}, ...] for the splits a sweep holds.

    In SPLITS order rather than in whatever order they were saved, and silent
    about the ones a sweep was never given -- a sweep with no validation set has
    no validation rows, which is not the same as an empty one.
    """
    rows = conn.execute(
        "SELECT split, source, COUNT(*) AS records,"
        "       SUM(CASE WHEN reference IS NULL THEN 0 ELSE 1 END) AS references_"
        "  FROM datasets WHERE run_id = ? GROUP BY split, source",
        (run_id,)).fetchall()
    found = {row["split"]: row for row in rows}
    return [{"split": name,
             "source": found[name]["source"],
             "records": found[name]["records"],
             "references": found[name]["references_"] or 0}
            for name in SPLITS if name in found]



# --- the population --------------------------------------------------------


def add_individuals(conn, run_id, chromosomes):
    """Store a freshly drawn population. Replaces any already held for this run."""
    conn.execute("DELETE FROM individuals WHERE run_id = ?", (run_id,))
    conn.executemany(
        "INSERT INTO individuals (run_id, number, chromosome) VALUES (?, ?, ?)",
        [(run_id, number, chromosome)
         for number, chromosome in enumerate(chromosomes, 1)])
    conn.commit()


def append_copies(conn, run_id, rows):
    """Add a copy of each of `rows` to a population. -> the numbers they got.

    The counterpart of add_individuals(), which replaces: numbering continues
    from the highest already stored, so the copies are new individuals rather
    than a re-drawn generation, and every number an existing script, execution
    or transcript refers to still means what it meant.

    A copy is the parent field for field -- tree, state, rank, script, weight
    seed and fitness, not just the chromosome. Only `id` and `number` are its
    own, since those are what make it a row of its own; everything else it
    inherits and keeps until something changes it.

    Except is_best, which every copy arrives with at 0. That flag does not
    describe an individual, it picks one out of the population -- the single
    one this generation carries forward -- so it is not the parent's to hand on.
    Copying it would give a sweep two elites, or six, and mark_best() exists
    precisely to keep that from ever being true. A copy of the elite is not
    itself the elite; the next election decides that, on the fitness it earns.

    The column list comes from the table rather than being written out here, so
    a field added to individuals is copied too instead of being quietly dropped
    from every copy the search makes.
    """
    if not rows:
        return []
    columns = [column["name"] for column in conn.execute("PRAGMA table_info(individuals)")
               if column["name"] not in ("id", "number")]
    top = conn.execute("SELECT MAX(number) AS top FROM individuals WHERE run_id = ?",
                       (run_id,)).fetchone()["top"] or 0
    numbers = list(range(top + 1, top + 1 + len(rows)))
    conn.executemany(
        "INSERT INTO individuals (number, %s) VALUES (%s)"
        % (", ".join(columns), ", ".join("?" * (len(columns) + 1))),
        [[number] + [0 if column == "is_best" else row[column]
                     for column in columns]
         for number, row in zip(numbers, rows)])
    conn.commit()
    return numbers


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
    taken here when present and the judge never has to be asked.
    """
    conn.executemany(
        "INSERT INTO exchanges (execution_id, position, question, answer,"
        " quality, reason, judge_model) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(execution_id, position, item.get("question", ""), item.get("answer", ""),
          item.get("quality"), item.get("reason"),
          "generated" if "quality" in item else None)
         for position, item in enumerate(transcript, 1)])


def executed(conn, run_id):
    """The ids of the individuals in one sweep that have ever been run.

    An individual not in here has no execution at all, so there is nothing
    stored about it whatever its flags say -- which is what separates "already
    done" from "not done yet" when deciding what needs running.
    """
    return {row["individual_id"] for row in conn.execute(
        "SELECT DISTINCT e.individual_id FROM executions e"
        " JOIN individuals i ON i.id = e.individual_id WHERE i.run_id = ?",
        (run_id,))}


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


# --- a testing pass -------------------------------------------------------


def add_test_result(conn, run_id, individual, chromosome, dataset, records,
                    selected_on, seconds, exit_code, verdict, weights, stdout,
                    stderr, transcript):
    """Record one individual's run against a testing dataset. Returns its id.

    `individual` is the row it was built from -- its number and weight seed are
    copied in rather than joined to later, because the row has to go on meaning
    what it meant after mutation has rewritten that individual.

    `chromosome` is passed separately, and is the one the script that ran
    actually builds, which is not always individuals.chromosome: mutation
    rewrites that column and leaves the script describing what the individual
    used to be. What was tested is what ran, so that is what is stored.

    Stored whatever happened: a script that crashed leaves a row with its exit
    code and an empty transcript, the way a failed execution does, because "it
    could not answer these questions" is a result of a testing pass and not an
    absence of one.
    """
    cursor = conn.execute(
        "INSERT INTO test_results (run_id, individual_id, number, chromosome,"
        " dataset, records, selected_on, started_at, seconds, exit_code, verdict,"
        " weight_seed, weights, stdout, stderr, exchanges)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, individual["id"], individual["number"], chromosome,
         dataset, records, selected_on, _now(), seconds, exit_code, verdict,
         individual["weight_seed"], json.dumps(weights or {}), stdout, stderr,
         json.dumps(transcript or [])))
    return cursor.lastrowid


def test_results_to_score(conn, run_id, dataset=None, force=False):
    """The testing rows holding answers that still need a quality.

    Whole rows rather than answers, because a transcript is stored whole: the
    caller unpacks the JSON, scores what is missing and writes the row back.
    With `force`, every row that has any answer at all, so a pass can be
    re-graded by another evaluator.

    A row whose script crashed carries an empty transcript and is not pending
    -- there is nothing in it to grade, and it would come back forever.
    """
    rows = test_results(conn, run_id, dataset)
    pending = []
    for row in rows:
        transcript = json.loads(row["exchanges"] or "[]")
        if not transcript:
            continue
        if force or any(item.get("quality") is None for item in transcript):
            pending.append(row)
    return pending


def score_test_result(conn, result_id, transcript, evaluator, judge_model):
    """Write a graded transcript back, with the mean it comes to. -> that mean.

    The scores live inside the JSON, beside the answers they belong to, so the
    test_answers view reads them without another join and a row goes on being
    the whole story of one individual against one dataset. `quality` is the
    mean over the answers that have one -- the same average over the same
    answers that individual_quality takes for a training run, so a testing
    number and a training number mean the same thing.

    None when nothing in the row could be scored: an average of no scores is
    not zero, and storing 0.0 would say the blend answered badly rather than
    that nobody managed to grade it.
    """
    scored = [item["quality"] for item in transcript
              if item.get("quality") is not None]
    mean = round(sum(scored) / len(scored), 4) if scored else None
    conn.execute(
        "UPDATE test_results SET exchanges = ?, quality = ?, evaluator = ?,"
        " judge_model = ?, scored_at = ? WHERE id = ?",
        (json.dumps(transcript), mean, evaluator, judge_model, _now(), result_id))
    return mean


def test_quality(conn, run_id, dataset=None):
    """How the tested individuals did, beside how they did in the search.

    One row each: the training quality that got it picked (selected_on, as it
    was then) and the quality it earned on the dataset it was never scored on.
    The two columns side by side are the whole point of a testing pass, and the
    difference between them is the only thing that says whether a blend was
    selected for the questions or for the job.
    """
    sql = ("SELECT id, number, chromosome, dataset, verdict, records,"
           "       selected_on, quality, evaluator,"
           "       json_array_length(exchanges) AS answers"
           "  FROM test_results WHERE run_id = ?")
    args = [run_id]
    if dataset is not None:
        sql += " AND dataset = ?"
        args.append(dataset)
    return conn.execute(sql + " ORDER BY id", args).fetchall()


def test_results(conn, run_id, dataset=None):
    """A sweep's testing rows, newest pass last, optionally for one dataset."""
    sql = "SELECT * FROM test_results WHERE run_id = ?"
    args = [run_id]
    if dataset is not None:
        sql += " AND dataset = ?"
        args.append(dataset)
    return conn.execute(sql + " ORDER BY id", args).fetchall()


def test_summary(conn, run_id):
    """[{dataset, tested, ok, answers, last}, ...], one per dataset tested on.

    The read side of a testing pass at a glance: which datasets this sweep's
    individuals have been put in front of, how many of them ran, and how many
    answers came back. Silent about a sweep that has never been tested.
    """
    rows = conn.execute(
        "SELECT dataset,"
        "       COUNT(*)                                        AS tested,"
        "       SUM(CASE WHEN verdict = 'ok' THEN 1 ELSE 0 END) AS ok_,"
        "       SUM(json_array_length(exchanges))               AS answers,"
        "       AVG(quality)                                    AS quality,"
        "       MAX(started_at)                                 AS last"
        "  FROM test_results WHERE run_id = ?"
        " GROUP BY dataset ORDER BY last", (run_id,)).fetchall()
    return [{"dataset": row["dataset"], "tested": row["tested"],
             "ok": row["ok_"] or 0, "answers": row["answers"] or 0,
             "quality": row["quality"], "last": row["last"]} for row in rows]


# --- the base model's own answers, cached across sweeps --------------------


def question_key(text):
    """A question reduced to what makes two of them the same question.

    The baselines table matches on this rather than on the wording, because the
    question reaching it comes back out of a transcript the generated script
    printed, and a line that lost its trailing spaces on the way is still the
    same question. Lives here rather than with the evaluator for the same reason
    the table does: it is how a row is addressed.
    """
    return " ".join((text or "").split()).strip().lower()


def baselines(conn, model):
    """{question_key: answer} for one base model -- the cache, read whole.

    One query rather than one per question: a transcript is a few dozen
    exchanges of the same eval set, so the whole of a model's cache is smaller
    than the transcript being scored against it.
    """
    rows = conn.execute(
        "SELECT question_key, answer FROM baselines WHERE model = ?", (model,)).fetchall()
    return {row["question_key"]: row["answer"] for row in rows}


def add_baselines(conn, model, items, source=None):
    """Store base-model answers. -> how many rows are new.

    `items` is [(question, answer)]. An answer already cached for that model and
    question is left as it was: it is the same model answering the same question
    with do_sample off, so a second copy would say the same thing, and quietly
    replacing it would move the ground every earlier score in the database was
    measured against. An empty answer is not stored at all -- that is a run that
    failed to answer, not the base model's reply.
    """
    added = 0
    for question, answer in items:
        if not (answer or "").strip():
            continue
        cursor = conn.execute(
            "INSERT OR IGNORE INTO baselines"
            " (model, question_key, question, answer, created_at, source)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (model, question_key(question), question, answer, _now(), source))
        added += cursor.rowcount or 0
    conn.commit()
    return added


def baseline_rows(conn, model):
    """Every cached answer for one base model, in the order they were stored."""
    return conn.execute(
        "SELECT * FROM baselines WHERE model = ? ORDER BY id", (model,)).fetchall()


def set_fitness(conn, run_id, number, fitness):
    """Record one individual's fitness -- the number selection sorts on.

    Keyed by (run_id, number) rather than by row id, because the caller works
    from the individual_quality view, whose rows are individuals seen through
    their latest execution.
    """
    conn.execute(
        "UPDATE individuals SET fitness = ? WHERE run_id = ? AND number = ?",
        (fitness, run_id, number))


# --- fitness, generation by generation ------------------------------------


def fitness_generation(conn, run_id, population):
    """Which generation a fitness snapshot over `population` individuals is.

    Nothing in the schema counts generations, because until now nothing needed
    to: a sweep is a population that keeps being rewritten in place. The count
    is derived here instead, from the one thing that dates a round -- the size
    of the population, which selection grows by appending and never shrinks.
    The same derivation selection and mutation seed their generators from.

    So: the first snapshot is generation 1, and a later one is a new generation
    if the population has grown since the last was taken, and the *same*
    generation restated if it has not. That is what makes re-running

        python main.py fitness

    -- pure arithmetic over stored transcripts, and a reasonable thing to redo
    after a re-scored `evaluate` -- rewrite the current generation rather than
    invent another one.

    The blind spot is a generation that appends nobody: SELECTION_COUNT at 0,
    or a wheel with nothing to spin. Such a round is indistinguishable from the
    one before it here and is recorded as a restatement of it. Both cases are
    already dead ends for the search -- an all-zero population stops at the
    elitism step -- so the history losing a row there costs nothing a running
    sweep would have wanted.
    """
    row = conn.execute(
        "SELECT generation, population FROM fitness_history WHERE run_id = ?"
        " ORDER BY generation DESC LIMIT 1", (run_id,)).fetchone()
    if row is None:
        return 1
    return row["generation"] + (0 if row["population"] == population else 1)


def record_fitness(conn, run_id, generation, entries):
    """Store one generation's fitness for a whole population. -> the timestamp.

    Every row of the generation carries the same `recorded_at`, because they are
    one snapshot taken at one moment and not a stream of separate events; the
    stamp dates the generation, and is the only place a sweep says when its
    generations happened.

    Anything already held for this generation is replaced, so a re-run of the
    fitness step restates the round with what it now knows instead of failing on
    the unique key or leaving half the old snapshot behind.

    `entries` are dicts with the columns below, which is what calculate_fitness
    builds out of the individual_quality view.
    """
    stamp = _now()
    conn.execute("DELETE FROM fitness_history WHERE run_id = ? AND generation = ?",
                 (run_id, generation))
    conn.executemany(
        "INSERT INTO fitness_history (run_id, generation, population, recorded_at,"
        " number, chromosome, state, fitness, answers, unscored)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(run_id, generation, len(entries), stamp, entry["number"],
          entry["chromosome"], entry["state"], entry["fitness"],
          entry["answers"], entry["unscored"]) for entry in entries])
    conn.commit()
    return stamp


def fitness_history(conn, run_id, generation=None):
    """The recorded history of one sweep, oldest generation first."""
    sql = "SELECT * FROM fitness_history WHERE run_id = ?"
    args = [run_id]
    if generation is not None:
        sql += " AND generation = ?"
        args.append(generation)
    return conn.execute(sql + " ORDER BY generation, number", args).fetchall()


def fitness_by_generation(conn, run_id):
    """One row per generation: when it was, and how its fitness came out.

    The shape you want to watch a search in -- whether the mean is climbing, and
    whether the best chromosome is getting better or the population is merely
    getting larger.
    """
    return conn.execute(
        "SELECT generation, population, MIN(recorded_at) AS recorded_at,"
        " COUNT(*) AS individuals, MIN(fitness) AS worst, MAX(fitness) AS best,"
        " AVG(fitness) AS mean,"
        " SUM(CASE WHEN fitness > 0 THEN 1 ELSE 0 END) AS scored"
        " FROM fitness_history WHERE run_id = ?"
        " GROUP BY generation ORDER BY generation", (run_id,)).fetchall()


def best_of_generation(conn, run_id, generation):
    """The fittest individual of one recorded generation, lowest number on a tie.

    Elitism's rule, applied to the history rather than to the population, so the
    two agree about which chromosome a generation was won by.
    """
    return conn.execute(
        "SELECT * FROM fitness_history WHERE run_id = ? AND generation = ?"
        " ORDER BY fitness DESC, number LIMIT 1", (run_id, generation)).fetchone()


def set_changed(conn, individual_id, has_changed):
    """Record whether the last mutation round altered this individual.

    has_changed = 1 clears the fitness, for the reason set_chromosome() gives.
    """
    if has_changed:
        conn.execute(
            "UPDATE individuals SET has_changed = 1, fitness = NULL WHERE id = ?",
            (individual_id,))
    else:
        conn.execute("UPDATE individuals SET has_changed = 0 WHERE id = ?",
                     (individual_id,))


def set_chromosome(conn, individual_id, chromosome):
    """Replace an individual's chromosome. It has changed, and it has no fitness.

    All three go together. A new chromosome *is* the change, so has_changed
    follows from it rather than being decided separately; and the fitness beside
    it was earned by the chromosome that just went away, which makes it not
    merely stale but wrong -- it would let an individual be elected, or win a
    slice of the roulette wheel, on a score belonging to a blend it no longer
    describes. NULL is the honest value: no fitness yet, the way an individual
    that has never run has none.

    The tree, script and rank are stale too, but they are only descriptions and
    they are re-derived wholesale by trees and runs. Fitness has to be re-earned
    through process and evaluate, so it is cleared here rather than left to
    mislead until then.
    """
    conn.execute(
        "UPDATE individuals SET chromosome = ?, has_changed = 1, fitness = NULL"
        " WHERE id = ?", (chromosome, individual_id))


def mark_best(conn, run_id, number):
    """Make one individual the sweep's elite, and the only one.

    Both halves in one statement, because they are one fact: is_best says which
    individual this generation carries forward, and a sweep with two of them --
    or with last generation's still set -- says nothing at all. Clearing first
    and setting after would leave that window open on the way through.
    """
    conn.execute(
        "UPDATE individuals SET is_best = (number = ?) WHERE run_id = ?",
        (number, run_id))


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
    script finds the LoRA folders by going up one from itself, so a deeper
    folder would break every path in it. The eval prompts are exempt: their path
    comes from TRAINING_SET in settings.py and is stamped in absolute.
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
    """Write one sweep out as a folder of text files.

    The database is the store; this is for the times a folder is what you want
    -- diff two populations, grep a transcript, hand someone the scripts.
    Everything written here is derived from the database, so the folder is a
    view of a sweep and never the sweep itself. Returns the paths written.
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

    # Each split written back as the lines it was read from, so the file in the
    # folder can be diffed against the one on disk today and say whether the
    # dataset has moved under the sweep.
    for entry in dataset_summary(conn, run_id):
        write("dataset_%s.txt" % entry["split"],
              "".join(item["content"] + "\n"
                      for item in dataset(conn, run_id, entry["split"])))

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

    # The one file here that is not a view of the population as it stands now:
    # every generation the sweep went through, in the order it went through
    # them, with the chromosomes and the scores as they were at the time.
    history = fitness_history(conn, run_id)
    if history:
        write("fitness_history.txt",
              "%-5s %-20s %-6s %-7s %-7s %-7s\n"
              % ("gen", "recorded", "pop", "best", "mean", "worst")
              + "".join("%-5d %-20s %-6d %-7.3f %-7.3f %-7.3f\n"
                        % (entry["generation"], entry["recorded_at"],
                           entry["population"], entry["best"] or 0.0,
                           entry["mean"] or 0.0, entry["worst"] or 0.0)
                        for entry in fitness_by_generation(conn, run_id))
              + "\n%-5s %-4s %-5s %-7s %-7s %s\n"
              % ("gen", "#", "state", "fitness", "answers", "chromosome")
              + "".join("%-5d %-4d %-5s %-7.3f %-7d %s\n"
                        % (row["generation"], row["number"], row["state"] or "-",
                           row["fitness"] or 0.0, row["answers"] or 0,
                           row["chromosome"])
                        for row in history))
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

    splits = dataset_summary(conn, run_id)
    if splits:
        print("\ndataset")
        for entry in splits:
            print("    %-11s %4d record(s), %4d with a reference  %s"
                  % (entry["split"], entry["records"], entry["references"],
                     entry["source"]))

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

    # The population above is the sweep as it stands; this is the sweep as it
    # went, which is the half that says whether the search was working.
    history = fitness_by_generation(conn, run_id)
    if history:
        print("\nfitness by generation")
        print("    %-5s %-20s %-6s %-7s %-7s %-7s %s"
              % ("gen", "recorded", "pop", "best", "mean", "worst",
                 "fittest chromosome"))
        for entry in history:
            best = best_of_generation(conn, run_id, entry["generation"])
            print("    %-5d %-20s %-6d %-7.3f %-7.3f %-7.3f %s"
                  % (entry["generation"], entry["recorded_at"], entry["population"],
                     entry["best"] or 0.0, entry["mean"] or 0.0,
                     entry["worst"] or 0.0, best["chromosome"] if best else "-"))

    # And this is the sweep against questions it was never scored on --
    # test_run_with_dataset.py, if it has been run. Silent otherwise: most
    # sweeps have never been tested, and an empty heading says nothing.
    tested = test_summary(conn, run_id)
    if tested:
        print("\ntesting passes")
        print("    %-20s %-7s %-5s %-8s %-7s %s"
              % ("last", "tested", "ok", "answers", "quality", "dataset"))
        for entry in tested:
            print("    %-20s %-7d %-5d %-8d %-7s %s"
                  % (entry["last"], entry["tested"], entry["ok"],
                     entry["answers"],
                     "-" if entry["quality"] is None else "%.3f" % entry["quality"],
                     entry["dataset"]))


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
