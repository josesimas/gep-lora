"""
continue_run.py - Run the evolutionary loop on, generation after generation.

    main.py          one sweep, from a fresh population through one generation
    continue_run.py  that sweep, carried on for as many generations as you ask

One generation is every step but population, in pipeline order:

    trees -> runs -> process -> evaluate -> fitness -> elitism -> selection
    -> mutation

which is a complete turn of the crank: describe the chromosomes, build them,
run them, judge them, score them, keep the best, breed from the fit, and vary
the offspring. The population that comes out is the one the next generation
goes in with.

Everything comes out of the database
------------------------------------
There is no population step here and no new sweep: this driver continues one
that already exists, reads the settings *that* sweep was created with, and
writes back into it. Point it at a database and it picks up the most recent
sweep in it; name one with --run. Nothing about the run is taken from
settings.py as it stands today except GENERATIONS, which is a property of this
invocation rather than of the sweep.

    python continue_run.py                            # the latest sweep, GENERATIONS times
    python continue_run.py --generations 3            # three turns instead
    python continue_run.py --db run_real/gep.sqlite3  # a sweep in another file
    python continue_run.py --run 3                    # a particular sweep

Which also means editing settings.py does nothing to a sweep already under way.
That is the point for the seeds and the template -- change those and it is not
the same sweep -- but a knob you only discover you wanted after the first
generation is a different matter, so --set changes one *on* the sweep:

    python continue_run.py --set SELECTION_COUNT=3

The new value goes into the settings table, so the sweep still records what it
ran under rather than being read past.

Watch the size
--------------
Selection appends: it adds picks without removing anyone. With SELECTION_COUNT
left at None it adds as many individuals as the population already holds, so the
population *doubles* every generation -- 10 becomes 20, 40, 80, and after ten
generations 10240 -- and process loads the base model once per individual, every
generation, for the whole population and not only the part of it that changed.
The driver prints that projection before it starts anything, so the cost is on
screen rather than discovered three hours in. It prints and carries on -- there
is no prompt to answer, so this stays runnable from a script or a scheduled job.
Fixing SELECTION_COUNT to a number makes the growth linear instead: on this
sweep with --set, since editing settings.py only reaches the next one.

Why the name
------------
`continue` is a Python keyword, so a continue.py could be run but never
imported. The _run suffix sidesteps that and puts it with process_run.py and
evaluate_run.py besides, which is the company it keeps.

    python continue_run.py --help
"""

import argparse
import json
import sys
import time

import main
import settings as config
import store

# One generation: every step but population, in the order main.py defines them.
# Named rather than sliced, so a step inserted at the front of STEPS does not
# silently join the loop.
GENERATION = ("trees", "runs", "process", "evaluate", "fitness", "elitism",
              "selection", "mutation")


def generation_steps():
    """The steps of one generation, in pipeline order."""
    wanted = set(GENERATION)
    steps = [step for step in main.STEPS if step.name in wanted]
    missing = wanted - {step.name for step in steps}
    if missing:
        raise SystemExit("main.py has no step(s) named: %s" % ", ".join(sorted(missing)))
    return steps


def projection(size, generations, selection_count):
    """What the population grows to, generation by generation.

    Selection appends `selection_count` individuals, or as many as the
    population holds when that is None -- which is a doubling, and worth seeing
    written out before it happens rather than after.
    """
    sizes = [size]
    for _ in range(generations):
        size += size if selection_count is None else selection_count
        sizes.append(size)
    return sizes


def override(conn, run_id, conf, assignments):
    """Change settings on the stored sweep before continuing it. -> what changed.

    A sweep normally reads the settings it was created with, which is what makes
    a continued sweep still be that sweep -- and what makes editing settings.py
    have no effect on one already under way. That is right for the seeds and the
    template, and wrong for a knob you only discover you wanted after the first
    generation, SELECTION_COUNT being the obvious one.

    So the change is written *into the sweep* rather than read past it: the
    settings table still says what the sweep used from here on, and the next
    step to read it sees the new value. The name has to be one the sweep already
    holds, so a typo is caught rather than quietly stored.
    """
    changed = {}
    for assignment in assignments:
        name, separator, raw = assignment.partition("=")
        name = name.strip()
        if not separator:
            raise SystemExit("--set wants NAME=VALUE, not %r" % assignment)
        if name not in conf:
            raise SystemExit("run %d has no setting called %s. It has: %s"
                             % (run_id, name, ", ".join(sorted(conf))))
        try:
            changed[name] = json.loads(raw)
        except ValueError:
            changed[name] = raw                 # a bare word is a string
    if changed:
        conf.update(changed)
        store.save_settings(conn, run_id, changed)
        for name in sorted(changed):
            print("set %s = %r on run %d (stored with the sweep)"
                  % (name, changed[name], run_id))
        print()
    return changed


def announce(conn, run_id, conf, generations):
    """Say what is about to happen, and what it will cost. -> the population."""
    rows = store.individuals(conn, run_id)
    if not rows:
        raise SystemExit("run %d holds no individuals -- there is nothing to "
                         "continue. Start a sweep with: python main.py" % run_id)

    selection_count = conf.get("SELECTION_COUNT")
    sizes = projection(len(rows), generations, selection_count)
    print("continuing run %d in %s" % (run_id, conn.path))
    print("population %d, %d generation(s) to run" % (sizes[0], generations))
    print("one generation: %s" % " -> ".join(GENERATION))
    print()
    print("population by generation: %s" % " -> ".join(str(size) for size in sizes))
    print("individuals through process in total: %d" % sum(sizes[:-1]))
    if selection_count is None:
        print("that is a doubling each time: this sweep stored SELECTION_COUNT as")
        print("None, so selection appends as many individuals as the population")
        print("already holds. Editing settings.py will not change that -- a sweep")
        print("keeps the settings it was created with. To change it for this one:")
        print()
        print("    python continue_run.py --set SELECTION_COUNT=3")
    if conf.get("TEMPLATE") != "template_code_mocked.py":
        print("every one of those loads the base model in its own process.")
    print()
    return rows


def evolve(conn, run_id, conf, generations, options):
    """Run `generations` generations over one sweep. -> an exit code.

    Stops at the first generation that fails, for the reason main.run() stops at
    the first step that fails: each one builds on the one before it, and a
    generation grown out of a broken one is not a result.
    """
    steps = generation_steps()
    context = main.context_for(conn, run_id, conf, options)
    started = time.time()

    for number in range(1, generations + 1):
        size = len(store.individuals(conn, run_id))
        # Which generation this is, for the step banners inside it: the header
        # below scrolls away during process, which is the part that takes the
        # hours. It is display only -- no step reads it, and nothing stores it.
        context.generation = "%d/%d" % (number, generations)
        print("#" * 70)
        print("# generation %d of %d -- population %d" % (number, generations, size))
        print("#" * 70)
        print()

        code = main.run(steps, context)
        if code:
            print()
            print("stopped in generation %d of %d, after %.1fs"
                  % (number, generations, time.time() - started))
            print("the sweep is marked failed; what the earlier generations did is")
            print("still in the database -- python store.py --show %d" % run_id)
            return code

        grown = len(store.individuals(conn, run_id))
        print("# generation %d done: population %d -> %d" % (number, size, grown))
        print()

    summarise(conn, run_id, generations, time.time() - started)
    return 0


def summarise(conn, run_id, generations, seconds):
    """Where the sweep got to."""
    rows = store.individuals(conn, run_id)
    best = [row for row in rows if row["is_best"]]
    print("=" * 70)
    print("%d generation(s) in %.1fs; population %d"
          % (generations, seconds, len(rows)))
    if best:
        row = best[0]
        print("best individual: #%d, fitness %s"
              % (row["number"],
                 "-" if row["fitness"] is None else "%.3f" % row["fitness"]))
        print("    %s" % row["chromosome"])
    print("run %d in %s -- python store.py --show %d" % (run_id, conn.path, run_id))
    print("=" * 70)


def parse(argv):
    parser = argparse.ArgumentParser(
        description="Continue an existing sweep for a number of generations: %s."
                    % " -> ".join(GENERATION))
    parser.add_argument("--db", default=config.DB_PATH,
                        help="database file holding the sweep (default %s)"
                             % config.DB_PATH)
    parser.add_argument("--run", type=int, default=0, metavar="ID",
                        help="which sweep to continue (default 0 = the latest)")
    parser.add_argument("--generations", type=int, default=None, metavar="N",
                        help="how many generations to run (default GENERATIONS in "
                             "settings.py, currently %d)" % config.GENERATIONS)
    parser.add_argument("--run-dir", default=None,
                        help="folder for the generated scripts (default: the "
                             "sweep's own DB_RUN_DIR)")
    # The steps read these off the options, exactly as they do when main.py is
    # the one driving.
    parser.add_argument("--limit", type=int, default=0,
                        help="process only the first N individuals of each generation")
    parser.add_argument("--include-blocked", action="store_true",
                        help="also run the ones marked BAD")
    parser.add_argument("--include-unchanged", action="store_true",
                        help="also re-run individuals whose chromosome has not "
                             "changed since their last execution")
    parser.add_argument("--keep-scripts", action="store_true",
                        help="leave the generated scripts on disk after processing them")
    parser.add_argument("--timeout", type=int, default=900,
                        help="seconds to allow each script (default 900)")
    parser.add_argument("--force", action="store_true",
                        help="re-score answers that already have a quality")
    parser.add_argument("--set", action="append", default=[], dest="settings",
                        metavar="NAME=VALUE",
                        help="change one of the sweep's stored settings before "
                             "continuing, e.g. --set SELECTION_COUNT=3. The value "
                             "is written into the sweep, so it still records what "
                             "it ran under. Repeatable.")
    return parser.parse_args(argv)


def cli(argv=None):
    options = parse(argv)
    generations = (config.GENERATIONS if options.generations is None
                   else options.generations)
    if generations < 1:
        raise SystemExit("--generations is %d; there is nothing to run" % generations)

    conn = store.connect(options.db)
    try:
        run_id = store.latest_run(conn) if options.run == 0 else options.run
        if run_id is None:
            raise SystemExit("%s holds no sweeps to continue. Start one with: "
                             "python main.py" % conn.path)
        if store.get_run(conn, run_id) is None:
            raise SystemExit("no run %d in %s. Try: python store.py --list"
                             % (run_id, conn.path))

        # The settings the sweep was created with, never settings.py as it
        # stands now -- that is what makes a continued sweep still be that sweep.
        conf = store.get_settings(conn, run_id)
        # ...unless you say otherwise, in writing, into the sweep itself.
        override(conn, run_id, conf, options.settings)
        # What it will cost is printed, not asked about: the projection is there
        # to be read, and stopping to wait for an answer would make the driver
        # useless to anything that is not a person at a terminal.
        announce(conn, run_id, conf, generations)
        return evolve(conn, run_id, conf, generations, options)
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(cli())
