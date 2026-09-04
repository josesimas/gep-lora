"""
full_run.py - A sweep from nothing to the end of the search, in one command.

    main.py          draws a population and takes it through one generation
    continue_run.py  carries that sweep on for GENERATIONS more
    test_run_with_dataset.py  puts what the search found in front of the
                     testing split, and grades it
    full_run.py      all of that, in that order, against the same sweep

    python full_run.py

which is:

    python main.py
    python continue_run.py --run <the sweep main.py just made>

so the whole search is 1 + GENERATIONS generations -- main.py's own turn of the
crank, then the ones continue_run.py adds. --generations controls the second
half; there is no way to have fewer than the one main.py runs, because drawing
a population and leaving it unjudged would not be a generation.

It calls the two drivers as libraries, in this interpreter. That matters: the
process step launches every generated script with sys.executable, so a subprocess
would be one more chance to run the search under the wrong Python. Whatever you
start this with is what the whole sweep uses.

The sweep is handed on by id, not by "the latest one" -- main.py's new sweep is
looked up once it exists and named explicitly, so a database that gains a sweep
from somewhere else in between cannot be picked up by mistake.

Options are forwarded to whichever driver understands them: --label to main.py,
--generations and --set to continue_run.py, and the rest -- --db, --run-dir,
--limit, --include-blocked, --include-unchanged, --keep-scripts, --timeout,
--force -- to both, meaning there what they mean there.

    python full_run.py --generations 3
    python full_run.py --db run_real/gep.sqlite3 --label "overnight"
    python full_run.py --limit 2 --generations 1     # a smoke test of the lot

--no-test and --test-min-quality go to the testing pass, and --db, --limit,
--keep-scripts, --timeout and --force reach it too.

A failing half stops the run: if main.py cannot produce a sweep there is nothing
to continue, and its exit code comes straight back out.

The testing pass at the end runs only when the sweep has a TESTING_SET, because
that setting is the only statement anyone has made about which questions the
search was not judged on. It costs a base-model load per individual above
TESTING_MIN_QUALITY, so --no-test skips it and --test-min-quality changes how
many that is. It is deliberately the *last* thing: a search that has finished
ends in mutation, and the individuals worth testing are the ones that were
actually scored -- which is what their stored scripts still describe.

    python full_run.py --no-test                 # search only, as before
    python full_run.py --test-min-quality 0.7    # test fewer of them
"""

import argparse
import sys
import time

import continue_run
import main
import settings as config
import store
import test_run_with_dataset


def forwarded(options):
    """The arguments both drivers understand, as an argv fragment."""
    argv = ["--db", options.db, "--timeout", str(options.timeout)]
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
    """The id of the sweep main.py just created, or None if it made none."""
    conn = store.connect(db_path)
    try:
        latest = store.latest_run(conn)
    finally:
        conn.close()
    return latest if latest != before else None


def cli(argv=None):
    options = parse(argv)
    generations = (config.GENERATIONS if options.generations is None
                   else options.generations)

    conn = store.connect(options.db)
    try:
        before = store.latest_run(conn)
    finally:
        conn.close()

    started = time.time()
    print("=" * 70)
    print("full run: main.py, then continue_run.py for %d more generation(s)"
          % generations)
    print("=" * 70)
    print()

    first = ["--label", options.label] if options.label else []
    code = call(main.main, first + forwarded(options))
    if code:
        print()
        print("main.py failed; there is no sweep to continue.")
        return code

    run_id = sweep_after(options.db, before)
    if run_id is None:
        print()
        print("main.py made no new sweep in %s; nothing to continue." % options.db)
        return 1

    print()
    print("#" * 70)
    print("# main.py done -- run %d. Continuing it for %d more generation(s)."
          % (run_id, generations))
    print("#" * 70)
    print()

    second = ["--run", str(run_id), "--generations", str(generations)]
    for assignment in options.settings:
        second += ["--set", assignment]
    code = call(continue_run.cli, second + forwarded(options))

    print()
    print("=" * 70)
    print("full run of %d generation(s) %s in %.1fs -- run %d in %s"
          % (generations + 1, "finished" if not code else "STOPPED",
             time.time() - started, run_id, options.db))
    print("python store.py --show %d" % run_id)
    print("=" * 70)

    # The search is over; this is the one question it could not answer about
    # itself. Skipped when the search stopped early -- a half-finished sweep's
    # best individual is not what the search found -- and when the sweep names
    # no testing set, which is the only statement of which questions it was
    # never judged on.
    if not code:
        code = test(options, run_id)
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


def test(options, run_id):
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
    dataset = testing_set(options.db, run_id)
    if not dataset:
        print()
        print("no testing pass: run %d names no TESTING_SET, so nothing says "
              "which questions it was never judged on." % run_id)
        print("    set TESTING_SET in settings.py before a sweep, or run "
              "test_run_with_dataset.py against this one.")
        return 0

    print()
    print("#" * 70)
    print("# the search is done -- testing run %d against %s" % (run_id, dataset))
    print("#" * 70)
    print()

    argv = [dataset, "--db", options.db, "--run", str(run_id),
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
        description="Run a whole search: main.py for a new sweep and its first "
                    "generation, then continue_run.py for the rest.")
    parser.add_argument("--db", default=config.DB_PATH,
                        help="database file (default %s)" % config.DB_PATH)
    parser.add_argument("--label", default=None,
                        help="a note stored with the sweep, to find it again later")
    parser.add_argument("--generations", type=int, default=None, metavar="N",
                        help="generations for continue_run.py, on top of the one "
                             "main.py runs (default GENERATIONS in settings.py, "
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
                             "sweep names a TESTING_SET, and costs a base-model "
                             "load per individual it tests)")
    parser.add_argument("--test-min-quality", type=float, default=None, metavar="Q",
                        help="test the individuals scoring above this (default "
                             "TESTING_MIN_QUALITY in settings.py, currently %.2f)"
                             % config.TESTING_MIN_QUALITY)
    options = parser.parse_args(argv)
    if options.generations is not None and options.generations < 1:
        parser.error("--generations is %d; main.py's generation would be the whole "
                     "run. Use main.py on its own for that." % options.generations)
    return options


if __name__ == "__main__":
    sys.exit(cli())
