"""
main.py - A sweep from nothing to the end of the search, in one command.

    start_run.py     draws a population and takes it through one generation
    continue_run.py  carries that sweep on for GENERATIONS more
    test_run_with_dataset.py  puts what the search found in front of the
                     testing split, and grades it
    main.py          all of that, in that order, against the same sweep

    python main.py

which is:

    python start_run.py
    python continue_run.py --run <the sweep start_run.py just made>

so the whole search is 1 + GENERATIONS generations -- start_run.py's own turn of the
crank, then the ones continue_run.py adds. --generations controls the second
half; there is no way to have fewer than the one start_run.py runs, because drawing
a population and leaving it unjudged would not be a generation.

It calls the two drivers as libraries, in this interpreter. That matters: the
process step launches every generated script with sys.executable, so a subprocess
would be one more chance to run the search under the wrong Python. Whatever you
start this with is what the whole sweep uses.

The sweep is handed on by id, not by "the latest one" -- start_run.py's new sweep is
looked up once it exists and named explicitly, so a database that gains a sweep
from somewhere else in between cannot be picked up by mistake.

Options are forwarded to whichever driver understands them: --label to start_run.py,
--generations and --set to continue_run.py, and the rest -- --db, --run-dir,
--limit, --include-blocked, --include-unchanged, --keep-scripts, --timeout,
--force -- to both, meaning there what they mean there.

    python main.py --generations 3
    python main.py --db run_real/gep.sqlite3 --label "overnight"
    python main.py --limit 2 --generations 1     # a smoke test of the lot

Running a database that is already a sweep
-----------------------------------------------------------------------------

    python main.py --db dbtemplates/test_new_run.sqlite3

A database can be *prepared* rather than produced: a run row, its settings, its
dataset, and no individuals. Everything a search needs and none of the search.
Handed one of those, this runs it rather than starting a second sweep beside it
-- `--db <a prepared database>` means "run this", and the alternative would be
a stranger's sweep appearing in somebody's experiment file. --run 0 (or --run N)
says the same thing out loud, and takes any sweep by id.

The rule is the latest sweep having no individuals. Anything else is a database
to start a new sweep in, exactly as before: one that does not exist yet, one
holding no sweeps, one whose latest sweep has a population. --label suppresses
it too, since only a sweep being created can be given a label.

From there the database is the only thing the run reads itself out of --

  * the settings are the sweep's stored ones, which is what resuming already
    meant, so the evaluator, the base model, the adapters, the seeds, the batch
    size and GENERATIONS are the ones written down beside the questions;
  * the questions are the sweep's stored dataset rows, not the files those
    settings name. settings.py and datasets/ are never opened.

and it is **isolated on disk as well**. Everything such a run writes goes into
one folder beside the database, named for it and the sweep --

    dbtemplates/
        test_new_run.sqlite3
        test_new_run_run1/
            training.jsonl      the questions, out of the rows
            run_001.py ...      the generated scripts, until they have run
            testing_scripts/    the testing pass's re-pointed copies

-- so run_db/, run_testing/ and the rest of the repo are untouched, two prepared
databases cannot tread on each other, and the results go back into the same
file, since --db is where the sweep lives. A prepared database is then a whole
experiment in one file plus one folder: hand it to another machine and the
search it runs there is the one it describes, not the one that machine's
settings.py happens to say. --run-dir still overrides, if you want it elsewhere.

An adopted sweep must hold no individuals yet -- start_run.py's half of the run draws
a population, and drawing one into a search that already has one would leave two
side by side. continue_run.py --from-db is what carries a started sweep on.

--from-db is the flag underneath all of this, and the three drivers take it on
their own too:

    python start_run.py --db <prepared> --run 1 --from-db
    python continue_run.py --db <prepared> --run 1 --from-db
    python -m testing.test_run_with_dataset --db <prepared> --run 1 --from-db

--no-test and --test-min-quality go to the testing pass, and --db, --limit,
--keep-scripts, --timeout and --force reach it too.

A failing half stops the run: if start_run.py cannot produce a sweep there is nothing
to continue, and its exit code comes straight back out.

The testing pass at the end runs only when the sweep has a TESTING_SET, because
that setting is the only statement anyone has made about which questions the
search was not judged on. It costs a base-model load per individual above
TESTING_MIN_QUALITY, so --no-test skips it and --test-min-quality changes how
many that is. It is deliberately the *last* thing, and it is why the search's last
generation stops after `fitness`: the individuals worth testing are the ones
that were actually scored, and stopping there is what leaves the population
holding exactly those rather than a generation selection and mutation have
already moved on to.

    python main.py --no-test                 # search only, as before
    python main.py --test-min-quality 0.7    # test fewer of them
"""

import argparse
import os
import sys
import time

from config import settings as config
import continue_run
import start_run
from storage import store
from testing import test_run_with_dataset

_HERE = os.path.dirname(os.path.abspath(__file__))


def forwarded(options, adopted=False):
    """The arguments both drivers understand, as an argv fragment."""
    argv = ["--db", options.db, "--timeout", str(options.timeout)]
    if adopted:
        # Both halves read the adopted sweep's questions out of the sweep, and
        # work in its own folder beside the database rather than in run_db/.
        argv.append("--from-db")
    if options.run_dir:
        argv += ["--run-dir", options.run_dir]
    if options.limit:
        argv += ["--limit", str(options.limit)]
    if options.include_blocked:
        argv.append("--include-blocked")
    if options.include_unchanged:
        argv.append("--include-unchanged")
    if options.keep_scripts:
        argv.append("--keep-scripts")
    if options.force:
        argv.append("--force")
    return argv


def call(driver, argv):
    """Run one driver in this process. -> its exit code.

    SystemExit is what a driver raises when it stops itself -- an unknown run,
    a population that is not there. Caught here so the second half is skipped
    rather than the whole thing unwinding through an exception nobody reads.
    """
    try:
        code = driver(argv)
    except SystemExit as error:
        if error.code in (0, None):
            return 0
        print(error)
        return 1
    return code or 0


def sweep_after(db_path, before):
    """The id of the sweep start_run.py just created, or None if it made none."""
    conn = store.connect(db_path)
    try:
        latest = store.latest_run(conn)
    finally:
        conn.close()
    return latest if latest != before else None


def prepared(options):
    """The sweep a bare --db should run rather than add to, or None.

    A database can be *prepared* rather than produced: a run row, its settings,
    its dataset, and no individuals -- everything a search needs and none of
    the search. Handed one of those, `python main.py --db it.sqlite3` means
    "run this", not "start a second sweep in the same file". So the latest run
    is looked at, and adopted when it is waiting to be run.

    Nothing else changes: a database that does not exist yet, holds no sweeps,
    or whose latest sweep has a population is a database to start a new sweep
    in, exactly as before -- `main.py --db run_real/gep.sqlite3` goes on
    meaning what it meant. --label suppresses this too, since a label is
    something only a sweep being created can be given.

    --run says it out loud and takes any sweep by id; this is the same decision
    made for somebody who did not.
    """
    if options.run is not None or options.label:
        return None
    where = (options.db if os.path.isabs(options.db)
             else os.path.join(_HERE, options.db))
    if not os.path.exists(where):
        return None
    conn = store.connect(options.db)
    try:
        run_id = store.latest_run(conn)
        if run_id is None or store.individuals(conn, run_id):
            return None
        # Settings are what make it a sweep rather than an empty row; without
        # them there is nothing to run it under and adopt() would say so.
        return run_id if store.get_settings(conn, run_id) else None
    finally:
        conn.close()


def adopt(options, run=None):
    """The prepared sweep to run. -> (run_id, its settings, its datasets).

    `run` is the id, from --run or from prepared(); 0 means the latest.

    A sweep is *prepared* when the database already holds its settings and its
    dataset and no individuals: everything a search needs, and none of the
    search. start_run.py then draws the population into it and takes it through its
    first generation exactly as it would into a sweep it had just created --
    except that the knobs and the questions are the ones already written down,
    so settings.py and the files under datasets/ are never opened, and the run
    keeps to its own folder beside the database rather than to run_db/.

    Refused, rather than guessed past:

    A path that is not already a database. store.connect() creates what it
    cannot open, which for a mistyped --db would mean being told a brand-new
    empty file holds no run 1.

    A sweep that already holds individuals. The population step appends, so
    joining a search that has one would draw a second population beside the
    first; continue_run.py is what carries one of those on. Only --run can
    reach this, since prepared() picks no such sweep in the first place.
    """
    run = options.run if run is None else run
    where = (options.db if os.path.isabs(options.db)
             else os.path.join(_HERE, options.db))
    if not os.path.exists(where):
        raise SystemExit("no database at %s" % os.path.abspath(where))

    conn = store.connect(options.db)
    try:
        run_id = store.latest_run(conn) if run == 0 else run
        if run_id is None:
            raise SystemExit(
                "%s holds no sweeps, so there is none to run. Drop --run to "
                "start one from settings.py." % conn.path)
        if store.get_run(conn, run_id) is None:
            raise SystemExit("no run %d in %s. Try: python -m storage.store --list"
                             % (run_id, conn.path))
        held = store.individuals(conn, run_id)
        if held:
            raise SystemExit(
                "run %d in %s already holds %d individual(s). main.py starts "
                "a search rather than joining one, and its first half would draw "
                "a second population beside that one.\n"
                "    Carry it on instead: python continue_run.py --db %s --run %d "
                "--from-db"
                % (run_id, conn.path, len(held), options.db, run_id))
        conf = store.get_settings(conn, run_id)
        if not conf:
            raise SystemExit(
                "run %d in %s has no stored settings, so there is nothing to run "
                "it under." % (run_id, conn.path))
        splits = store.dataset_summary(conn, run_id)
    finally:
        conn.close()
    return run_id, conf, splits


def describe(options, run_id, conf, splits, asked):
    """Say what the adopted sweep is, before a single step runs.

    `asked` is whether --run named it. When it did not, the reason it was picked
    is worth a line: this is the one decision main.py makes on its own, and
    the alternative it turned down -- starting a second sweep in somebody's
    prepared database -- is exactly what a person would want to be told about.
    """
    if asked:
        print("adopting run %d in %s -- its settings and its questions, not "
              "settings.py's" % (run_id, options.db))
    else:
        print("run %d in %s is prepared -- settings and a dataset, no "
              "individuals -- so this runs it" % (run_id, options.db))
        print("    rather than starting a second sweep beside it. Its settings "
              "and its questions, not settings.py's.")
    print("    %d stored setting(s); evaluator %s, base model %s"
          % (len(conf), conf.get("EVALUATOR"), conf.get("BASE_MODEL")))
    if splits:
        for entry in splits:
            print("    %-10s %3d record(s), %d with a reference answer"
                  % (entry["split"], entry["records"], entry["references"]))
    else:
        print("    no dataset rows -- the run will stop when it looks for them")
    print()


def cli(argv=None):
    options = parse(argv)

    # --run names a sweep to run; without it, a database that is already a
    # prepared sweep is one too. Anything else starts a new sweep, as before.
    asked = options.run is not None
    target = options.run if asked else prepared(options)
    adopted = adopt(options, target) if target is not None else None
    if adopted:
        run_id, conf, splits = adopted
        # The sweep's own GENERATIONS, the way continue_run.py reads it, so a
        # prepared database says how long its own search is.
        generations = continue_run.generation_count(options, conf)
        before = None
    else:
        run_id = None
        generations = (config.GENERATIONS if options.generations is None
                       else options.generations)
        conn = store.connect(options.db)
        try:
            before = store.latest_run(conn)
        finally:
            conn.close()

    if generations < 1:
        raise SystemExit(
            "%d generation(s) for the second half; start_run.py's generation would be "
            "the whole run. Use start_run.py on its own for that." % generations)

    started = time.time()
    print("=" * 70)
    print("full run: start_run.py, then continue_run.py for %d more generation(s)"
          % generations)
    print("=" * 70)
    print()
    if adopted:
        describe(options, run_id, conf, splits, asked)

    # --next-generation because start_run.py's generation is this run's *first*,
    # not its last: continue_run.py has at least one more to run (--generations
    # is refused below 1), and it needs a population selection and mutation
    # have moved on. Only the run's last generation stops after fitness, and
    # that one belongs to continue_run.py.
    first = ["--next-generation"]
    if adopted:
        first += ["--run", str(run_id)]
    elif options.label:
        first += ["--label", options.label]
    code = call(start_run.main, first + forwarded(options, bool(adopted)))
    if code:
        print()
        print("start_run.py failed; there is no sweep to continue.")
        return code

    if not adopted:
        run_id = sweep_after(options.db, before)
        if run_id is None:
            print()
            print("start_run.py made no new sweep in %s; nothing to continue."
                  % options.db)
            return 1

    print()
    print("#" * 70)
    print("# start_run.py done -- run %d. Continuing it for %d more generation(s)."
          % (run_id, generations))
    print("#" * 70)
    print()

    second = ["--run", str(run_id), "--generations", str(generations)]
    for assignment in options.settings:
        second += ["--set", assignment]
    code = call(continue_run.cli, second + forwarded(options, bool(adopted)))

    print()
    print("=" * 70)
    print("full run of %d generation(s) %s in %.1fs -- run %d in %s"
          % (generations + 1, "finished" if not code else "STOPPED",
             time.time() - started, run_id, options.db))
    print("python -m storage.store --show %d" % run_id)
    print("=" * 70)

    # The search is over; this is the one question it could not answer about
    # itself. Skipped when the search stopped early -- a half-finished sweep's
    # best individual is not what the search found -- and when the sweep names
    # no testing set, which is the only statement of which questions it was
    # never judged on.
    if not code:
        code = test(options, run_id, bool(adopted))
    return code


def testing_set(db_path, run_id):
    """The sweep's own TESTING_SET, or None if it named none.

    The sweep's stored setting rather than settings.py's, for the reason every
    step reads stored settings: the file it recorded at creation is the one its
    `testing` split holds, and settings.py may have been repointed since.
    """
    conn = store.connect(db_path)
    try:
        return store.get_settings(conn, run_id).get("TESTING_SET")
    finally:
        conn.close()


def testing_split(db_path, run_id):
    """How many testing records the sweep itself holds, or None for none.

    The question TESTING_SET answers for a sweep run from files, asked of the
    database instead -- because an adopted sweep reads its questions out of its
    own rows, and a setting naming a file those rows never came from is not a
    statement about this run. A sweep whose settings name a testing file it
    never stored has none as far as this pass is concerned, which is the same
    thing db_datasets.repoint() decides.
    """
    conn = store.connect(db_path)
    try:
        held = [entry for entry in store.dataset_summary(conn, run_id)
                if entry["split"] == "testing"]
    finally:
        conn.close()
    return held[0]["records"] if held else None


def test(options, run_id, adopted=False):
    """Put the finished search in front of its testing split. -> an exit code.

    Called as a library, in this interpreter, for the reason the other two are:
    the pass launches each script with sys.executable, so a subprocess would be
    one more chance to run it under the wrong Python.

    A pass that stops itself -- nothing scored above the bar, no judge to grade
    with -- is reported and returned, not swallowed: the search finished and
    this did not, and saying so is the difference between a testing pass that
    was skipped and one that failed.
    """
    if options.no_test:
        return 0

    # An adopted sweep is asked the same question of its rows rather than of
    # its settings, and answers it with --from-db: the questions it was never
    # judged on are the ones in its own `testing` split.
    if adopted:
        records = testing_split(options.db, run_id)
        if not records:
            print()
            print("no testing pass: run %d holds no testing split, so nothing "
                  "says which questions it was never judged on." % run_id)
            print("    add one with: python -m storage.add_dataset <file> --db %s "
                  "--run %d --split testing" % (options.db, run_id))
            return 0
        dataset = "its own testing split (%d record(s))" % records
        first = ["--from-db"]
    else:
        dataset = testing_set(options.db, run_id)
        if not dataset:
            print()
            print("no testing pass: run %d names no TESTING_SET, so nothing says "
                  "which questions it was never judged on." % run_id)
            print("    set TESTING_SET in settings.py before a sweep, or run "
                  "test_run_with_dataset.py against this one.")
            return 0
        first = [dataset]

    print()
    print("#" * 70)
    print("# the search is done -- testing run %d against %s" % (run_id, dataset))
    print("#" * 70)
    print()

    argv = first + ["--db", options.db, "--run", str(run_id),
                    "--timeout", str(options.timeout)]
    if options.test_min_quality is not None:
        argv += ["--min-quality", str(options.test_min_quality)]
    if options.limit:
        # Means there what it means here: only the first few individuals, so a
        # smoke test of the whole thing stays a smoke test.
        argv += ["--limit", str(options.limit)]
    if options.keep_scripts:
        argv.append("--keep-scripts")
    if options.force:
        argv.append("--force")

    code = call(test_run_with_dataset.main, argv)
    if code:
        print()
        print("the search finished; the testing pass did not.")
    return code


def parse(argv):
    parser = argparse.ArgumentParser(
        description="Run a whole search: start_run.py for a new sweep and its first "
                    "generation, then continue_run.py for the rest.")
    parser.add_argument("--db", default=config.DB_PATH,
                        help="database file (default %s)" % config.DB_PATH)
    parser.add_argument("--run", type=int, default=None, metavar="ID",
                        help="run the sweep the database already holds instead of "
                             "creating one (0 = the latest). Its stored settings "
                             "and its stored dataset are what the search uses; "
                             "settings.py and the files under datasets/ are never "
                             "read. The sweep must hold no individuals yet.")
    parser.add_argument("--label", default=None,
                        help="a note stored with the sweep, to find it again later")
    parser.add_argument("--generations", type=int, default=None, metavar="N",
                        help="generations for continue_run.py, on top of the one "
                             "start_run.py runs (default: with --run, the sweep's own "
                             "stored GENERATIONS; otherwise settings.py's, "
                             "currently %d)" % config.GENERATIONS)
    parser.add_argument("--set", action="append", default=[], dest="settings",
                        metavar="NAME=VALUE",
                        help="change one of the new sweep's stored settings before "
                             "continuing it, e.g. --set SELECTION_COUNT=3")
    parser.add_argument("--run-dir", default=None,
                        help="folder for the generated scripts (default %s)"
                             % config.DB_RUN_DIR)
    parser.add_argument("--limit", type=int, default=0,
                        help="process only the first N individuals of a generation")
    parser.add_argument("--include-blocked", action="store_true",
                        help="also run the ones marked BAD")
    parser.add_argument("--include-unchanged", action="store_true",
                        help="also run individuals whose chromosome has not changed "
                             "since their last execution")
    parser.add_argument("--keep-scripts", action="store_true",
                        help="leave the generated scripts on disk after processing them")
    parser.add_argument("--timeout", type=int, default=900,
                        help="seconds to allow each script (default 900)")
    parser.add_argument("--force", action="store_true",
                        help="re-score answers that already have a quality")
    parser.add_argument("--no-test", action="store_true",
                        help="skip the testing pass at the end (it runs when the "
                             "sweep has a testing split, and costs a base-model "
                             "load per individual it tests)")
    parser.add_argument("--test-min-quality", type=float, default=None, metavar="Q",
                        help="test the individuals scoring above this (default: "
                             "the sweep's own TESTING_MIN_QUALITY, falling back to "
                             "settings.py's, currently %.2f)"
                             % config.TESTING_MIN_QUALITY)
    options = parser.parse_args(argv)
    if options.generations is not None and options.generations < 1:
        parser.error("--generations is %d; start_run.py's generation would be the whole "
                     "run. Use start_run.py on its own for that." % options.generations)
    if options.run is not None and options.label:
        # create_run() is what writes a label, and --run creates nothing.
        parser.error("--label names a sweep as it is created, and --run adopts one "
                     "that already exists. Label it when you prepare the database.")
    return options


if __name__ == "__main__":
    sys.exit(cli())
