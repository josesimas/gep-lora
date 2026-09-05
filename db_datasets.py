"""db_datasets.py - a sweep's dataset, read back out of the database as a file,
and the one folder on disk a database-driven run is allowed to touch.

`add_dataset.py` is the way in: every sweep stores the splits its settings name
the moment it is created, one row per record, keeping the line exactly as it
was read. This is the way back out.

It exists because of a gap between two things that are both true. A sweep reads
the settings it was created with, so TRAINING_SET goes on naming the file the
questions came from -- and that file is a file: it can be edited, moved, or
simply never have existed on this machine. The rows in `datasets` are the only
copy of the eval set that belongs to the sweep rather than to the disk, and a
database handed to somebody else carries them and nothing else.

So `repoint()` writes each stored split back out and hands back the sweep's
settings with the three `*_SET` values pointing at what it wrote. Every existing
reader then works unchanged -- generate_runs.eval_records(), the evaluators that
grade against a reference, and above all the generated scripts, which are handed
a path as a literal and open it at startup. Nothing had to learn about sqlite to
be fed from it.

Where it writes is the other half of the job. `run_folder()` is one folder per
sweep, **beside the database and named for it** --

    dbtemplates/
        test_new_run.sqlite3
        test_new_run_run1/
            training.jsonl          the questions, out of the rows
            run_001.py ...          the generated scripts, until they have run
            testing_scripts/        the testing pass's re-pointed copies

-- and it is where everything a run of this kind puts on disk goes: the
materialised splits, the generated scripts, the caches the children drop in
their cwd. A run driven from a database writes nothing into `run_db/`,
`run_testing/` or anywhere else in the repo, so two prepared databases cannot
tread on each other and neither can tread on an ordinary sweep. Delete the
folder and the sweep is still whole, because all of it is derived.

Two more things are deliberate.

It repoints the settings **in memory only**. The `settings` table goes on saying
where the questions originally came from, which is the provenance those rows
exist to keep; a sweep that rewrote its own TRAINING_SET to name a cache would
lose the one record of what it was built on.

And a split the sweep does not hold has its setting **cleared** rather than left
alone. The whole point of this mode is that the local files are not consulted,
so a TESTING_SET naming a file no testing rows were ever stored from names
nothing, and saying so is better than quietly reading it.
"""

import os

import add_dataset
import store


def db_folder(conn):
    """The folder the database lives in."""
    return os.path.dirname(os.path.abspath(conn.path))


def _stem(conn):
    """The database's name without its extension: gep.sqlite3 -> gep."""
    return os.path.splitext(os.path.basename(os.path.abspath(conn.path)))[0]


def run_folder(conn, run_id):
    """The one folder on disk this sweep's database-driven run uses.

    Beside the database and named for it and the sweep -- `gep_run3` next to
    `gep.sqlite3` -- so everything a run puts on disk is in one place that says
    which sweep of which database it belongs to. Not created here: whoever
    writes into it makes it, the way store.materialise() already does.
    """
    return os.path.join(db_folder(conn), "%s_run%d" % (_stem(conn), run_id))


def testing_folder(conn, run_id):
    """Where the testing pass's re-pointed scripts go: under the run folder.

    Its own subfolder rather than the run folder itself, because the pass names
    a script after the individual exactly as the search does, and a testing
    run_007.py sitting where the search's run_007.py was is one confusion worth
    a directory. Named `testing_scripts` and not `testing`, which is already
    half of `testing.jsonl` sitting beside it -- the questions and the scripts
    that ask them should not be one glance apart.
    """
    return os.path.join(run_folder(conn, run_id), "testing_scripts")


def _extension(rows):
    """`.jsonl` or `.txt`, from the records themselves.

    The readers tell the two shapes apart by the line rather than by the name
    -- a JSON record per line, or plain one-prompt-per-line text -- so this is
    only ever for the person who finds the file. Which is reason enough to get
    it right.
    """
    first = (rows[0]["content"] or "").lstrip()
    return ".jsonl" if first.startswith("{") else ".txt"


def export(conn, run_id, split, folder=None):
    """Write one stored split out as a file. -> (path, records), or None.

    None means the sweep holds no rows for that split, which is not an error:
    a sweep with no validation set simply has none.

    The `content` column is the line as it was read, so what comes out is what
    went in -- the same records, in the same order, parsed by the same readers
    into the same questions and the same references. Rewritten whole every
    time, because it is a cache of the rows and never the other way round.
    """
    rows = store.dataset(conn, run_id, split)
    if not rows:
        return None
    where = os.path.abspath(folder) if folder else run_folder(conn, run_id)
    path = os.path.join(where, "%s%s" % (split, _extension(rows)))
    os.makedirs(where, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write("%s\n" % row["content"])
    return path, len(rows)


def repoint(conn, run_id, conf, folder=None, say=print):
    """This sweep's settings, reading its dataset from the database. -> a conf.

    A copy: the caller's `conf` is left alone, and nothing is written to the
    settings table. Each split the sweep holds is written into the run folder
    and its setting points at the file; each split it does not hold has its
    setting cleared, so no local file is consulted for it either.

    A sweep with no training rows at all cannot be run this way, and that is
    said here -- before a population is drawn, rather than an hour later when
    every generated script fails to find its prompts. Sweeps created before the
    `datasets` table existed are the ones this happens to.
    """
    out = dict(conf)
    written = []
    for split in store.SPLITS:
        setting = add_dataset.SPLIT_SETTINGS[split]
        made = export(conn, run_id, split, folder)
        if made is None:
            if out.get(setting):
                written.append((split, None, out[setting]))
            out[setting] = None
        else:
            path, count = made
            written.append((split, path, count))
            out[setting] = path

    if not out.get(add_dataset.SPLIT_SETTINGS["training"]):
        raise SystemExit(
            "run %d in %s holds no training dataset, so there are no questions "
            "to read from the database. Store them first: python add_dataset.py "
            "<file> --db %s --run %d --split training"
            % (run_id, conn.path, conn.path, run_id))

    if say:
        say("datasets from the database (run %d in %s):"
            % (run_id, os.path.basename(conn.path)))
        for split, path, detail in written:
            if path is None:
                say("  %-10s the sweep holds none; %s (%s) is ignored"
                    % (split, add_dataset.SPLIT_SETTINGS[split], detail))
            else:
                say("  %-10s %3d record(s) -> %s" % (split, detail, path))
    return out
