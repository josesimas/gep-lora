"""
add_dataset.py - Put a dataset file into a sweep's `datasets` table.

The one way records get into that table, from either end of a sweep's life:

  * `save_all()` stores the splits a sweep's settings name, and is what
    `main.new_sweep()` calls as the sweep is created -- alongside the settings
    and for the same reason. A fitness number means *this blend, under these
    knobs, on these questions*, and a settings table that records TRAINING_SET
    records only **where** the questions were; the files go on being edited,
    repointed and regenerated.
  * `add()`, and the command line over it, is the same act performed
    afterwards, by hand -- for a split the sweep was never given, a validation
    or testing set decided on after it started, so a later pass has the
    questions beside the run they belong to:

        python add_dataset.py datasets/validation.json --split validation
        python add_dataset.py extra.json --split testing --db run_db1/gep.sqlite3 --run 3

Both go through `add()`, so a split added by hand is stored exactly as one the
driver stored -- there is no second reader of a dataset file and no second
INSERT. Replacing a split that already holds rows is possible but never the
default: those rows are what the sweep says it was built on, and a dataset
changing under a finished sweep is the thing the table exists to prevent.

Nothing here parses or stores anything itself either: `generate_runs`
`.dataset_records()` reads the file (same two shapes, same 1-based positions,
uncapped -- the cap is a fact about the sweep, not about the file) and
`store.save_dataset()` writes it.
"""

import argparse
import os
import sys

import generate_runs
import store

_HERE = os.path.dirname(os.path.abspath(__file__))


def resolve_dataset(name):
    """Absolute path of the dataset file named on the command line.

    A path typed at a shell means what the shell means by it, so the cwd comes
    first; the repo folder is the fallback, because that is what a *setting*
    naming the same file would mean (generate_runs.training_set_path()) and
    `add_dataset.py datasets/x.json` should work from either place. Missing in
    both is said here rather than reaching the reader as an OSError.
    """
    if os.path.isabs(name):
        return os.path.abspath(name)
    here = os.path.abspath(name)
    if os.path.exists(here):
        return here
    beside = os.path.abspath(os.path.join(_HERE, name))
    if os.path.exists(beside):
        return beside
    raise SystemExit("no dataset file at %s%s"
                     % (here, "" if beside == here else " (nor at %s)" % beside))


def add(conn, run_id, split, path, replace=False, say=print):
    """Read one dataset file and store it as `split` of run `run_id`.

    -> (records written, how many carry a reference answer).

    `path` is absolute or relative to this file, the way a settings value is;
    save_all() passes TRAINING_SET straight in and the command line passes what
    it resolved against the cwd first.

    Refuses a split that already holds rows unless `replace`, and refuses a
    file with no records at all: save_dataset() writes a split whole, so an
    empty read would quietly leave the sweep with no dataset rather than with
    the one that was meant.
    """
    if split not in store.SPLITS:
        raise SystemExit("unknown split %r; the table holds %s"
                         % (split, ", ".join(store.SPLITS)))
    if store.get_run(conn, run_id) is None:
        raise SystemExit("%s holds no run %d" % (conn.path, run_id))

    held = [entry for entry in store.dataset_summary(conn, run_id)
            if entry["split"] == split]
    if held and not replace:
        entry = held[0]
        raise SystemExit(
            "run %d already holds a %s split -- %d record(s) from %s. Pass "
            "--replace to overwrite it; those rows say what that sweep was "
            "built on." % (run_id, split, entry["records"], entry["source"]))

    # Absolute already (the command line resolves before calling), or a
    # settings value like "datasets/x.json", which means what it means to every
    # other reader of one: beside this file, never beside the cwd.
    path = generate_runs.training_set_path(path)
    records = generate_runs.dataset_records(path)
    if not records:
        raise SystemExit("%s holds no records; nothing to store." % path)

    count = store.save_dataset(conn, run_id, split, path, records)
    with_reference = sum(1 for one in records if one["reference"])
    say("%-10s %3d record(s), %d with a reference answer, from %s"
        % (split, count, with_reference, path))
    if held:
        say("           replaced %d record(s) that split already held"
            % held[0]["records"])
    return count, with_reference


# The setting each split of the dataset is named by. Only the training one is
# read by the search -- it is the eval set every generated script asks and every
# fitness number is earned on; the other two are stored when they are named and
# nothing reads them yet.
SPLIT_SETTINGS = {"training": "TRAINING_SET",
                  "validation": "VALIDATION_SET",
                  "testing": "TESTING_SET"}


def save_all(conn, run_id, conf, say=print):
    """Store every split a sweep's settings name -> datasets. -> how many.

    Called by main.new_sweep() the moment a sweep is created, before its first
    step: the files a sweep was judged on are half of what its fitness numbers
    mean, and they are not going to sit still.

    Every record of each file, uncapped -- TRAINING_COUNT decides how many of
    the training records an individual is judged on, and that is already stored
    as a setting. A split whose setting is None is simply not part of this
    sweep and leaves no rows; a split naming a file that cannot be read, or one
    with nothing in it, stops the sweep here, at its first second, rather than
    an hour into it.
    """
    stored = 0
    for split in store.SPLITS:
        where = conf.get(SPLIT_SETTINGS[split])
        if not where:
            continue
        count, _ = add(conn, run_id, split, where, say=say)
        stored += count
    return stored


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Add a dataset file to a sweep's datasets table.")
    parser.add_argument("dataset",
                        help="the dataset file, JSON Lines or one prompt per line")
    parser.add_argument("--split", required=True, choices=list(store.SPLITS),
                        help="which split the records belong to")
    parser.add_argument("--db", default=None,
                        help="database file (default: settings.DB_PATH)")
    parser.add_argument("--run", type=int, default=0, metavar="RUN",
                        help="the sweep to add it to (0 = the most recent, default)")
    parser.add_argument("--replace", action="store_true",
                        help="overwrite the split if the sweep already holds one")
    args = parser.parse_args(argv)

    if args.db is None:
        import settings
        args.db = settings.DB_PATH
    # connect() would create an empty database with the schema in it, which for
    # a typo'd --db means being told a fresh file holds no runs instead of that
    # the file does not exist.
    path = args.db if os.path.isabs(args.db) else os.path.join(_HERE, args.db)
    if not os.path.exists(path):
        raise SystemExit("no database at %s" % os.path.abspath(path))

    conn = store.connect(args.db)
    run_id = store.latest_run(conn) if args.run == 0 else args.run
    if run_id is None:
        raise SystemExit("%s holds no runs yet" % conn.path)

    dataset = resolve_dataset(args.dataset)
    add(conn, run_id, args.split, dataset, replace=args.replace)
    print("run %d in %s" % (run_id, conn.path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
