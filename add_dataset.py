"""
add_dataset.py - Put a dataset file into an existing sweep's `datasets` table.

`main.save_datasets()` stores the three splits a sweep was created with, once,
at the moment it is created. This is the same act performed afterwards, by
hand: point it at a database, a dataset file and one of the split names and it
reads the file the way the pipeline reads it and files the records under a run.

    python add_dataset.py datasets/validation.json --split validation
    python add_dataset.py extra.json --split testing --db run_db1/gep.sqlite3 --run 3

It is for the splits a sweep was never given -- a validation or testing set
decided on after the sweep started, so a later pass has the questions beside
the run they belong to. Replacing a split that already holds rows is possible
but not the default: those rows are what the sweep says it was built on, and a
dataset changing under a finished sweep is the thing the table exists to
prevent, so it takes --replace to say that is meant.

Nothing here parses or stores anything itself: `generate_runs.dataset_records()`
reads the file (same two shapes, same 1-based positions, uncapped -- the cap is
a fact about the sweep, not about the file) and `store.save_dataset()` writes
it, so a dataset added here is indistinguishable from one the driver stored.
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
