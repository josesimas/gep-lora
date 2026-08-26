"""
main.py - Run the whole pipeline end to end against a sqlite database.

    population -> trees -> runs -> process -> evaluate -> fitness

Nothing is left scattered across a folder afterwards. The population, every
setting the sweep ran under, every seed, every generated script, every
transcript, every score and the fitness they come to go into one database,
described in store.py:

    runs -> settings, individuals -> executions -> exchanges

Why a database
--------------
A folder of a hundred files answers "what happened" only by being read one file
at a time, and the next sweep overwrites it. A database keeps sweeps side by
side and lets you ask across them:

    SELECT chromosome, quality FROM individual_quality
     WHERE run_id = 3 ORDER BY quality DESC LIMIT 5;

and, because the seeds are stored too, it can say exactly what a stored sweep
was: which chromosomes (the population seed) and which blend weights each of
them was judged under (a per-individual weight seed, stamped into its script).
A stored sweep can therefore be repeated, which is the point of storing it.

What still touches the disk
---------------------------
The generated run_NNN.py scripts, and only those, and only until they have run:
process launches them as subprocesses, so they have to be real files, and it
deletes each one it has processed once the sweep is through. They are a cache of
individuals.script_source, rewritten from the database whenever they are missing
or stale, so nothing is lost -- what is gained is that run_db/ does not fill up
with spent scripts, and nobody can launch a stale one by hand a week later.
Pass --keep-scripts to leave them.

They have to sit exactly one level below the project folder -- a generated
script finds the LoRA folders and training_set.txt by going up one from itself
-- so they land in run_db/, alongside the database itself.

Steps run in the order listed and stop at the first failure. An individual that
crashes is not a failure: it is a row in executions with its exit code.

Resuming
--------
Every step but population can be re-run against a sweep already in the database,
and the settings it reads are the ones that sweep was created with, not whatever
settings.py says today. That is what makes a resumed sweep still be the same
sweep. Name a run with --run, or leave it out and the most recent one is used.

Usage:
    python main.py                    # a new sweep, every step
    python main.py --list             # show the steps without running them
    python main.py population trees runs   # the fast half, before a model load
    python main.py process evaluate   # resume the latest sweep
    python main.py evaluate --run 3   # score sweep 3
    python main.py --limit 3          # smoke test: only the first few

Run it with the **venv's python**: process launches each generated script with
sys.executable, so the wrong interpreter fails every individual. That does not
apply to a sweep generated from template_code_mocked.py, which loads nothing.

Reading a sweep back:
    python store.py --list
    python store.py --show 0
    python store.py --export 0 --into export
"""

import argparse
import os
import random
import sys
import time
from collections import namedtuple

import calculate_fitness
import draw_trees
import evaluate_run
import generate_population
import generate_runs
import process_run
import settings as config
import store

_HERE = os.path.dirname(os.path.abspath(__file__))

# The upper bound for a drawn seed. Kept well inside 2**32 so the value is a
# plain int in every sqlite column and in every generated script.
_SEED_LIMIT = 2 ** 31 - 1


class Context:
    """What every step needs: the database, the sweep, and the options."""

    __slots__ = ("conn", "run_id", "conf", "run_dir", "template", "options")

    def __init__(self, conn, run_id, conf, run_dir, template, options):
        self.conn = conn
        self.run_id = run_id
        self.conf = conf              # the settings this sweep runs under
        self.run_dir = run_dir
        self.template = template      # absolute path to the template it fills
        self.options = options        # the parsed command line


# --- the settings a sweep runs under --------------------------------------


def _template_path(name):
    """Absolute path of the template to fill, from a name or None."""
    name = name or "template_code.py"
    if not os.path.isabs(name) and not os.path.exists(name):
        return os.path.join(_HERE, name)
    return os.path.abspath(name)


def judge_snapshot():
    """The judge settings, recorded alongside the rest.

    The system prompt is in here on purpose: it is the rubric the whole search
    selects on, so a stored sweep that did not record it could not be explained
    later, let alone compared with another.
    """
    return {
        "JUDGE_BASE_URL": evaluate_run.BASE_URL,
        "JUDGE_MODEL": evaluate_run.MODEL,
        "JUDGE_TEMPERATURE": evaluate_run.TEMPERATURE,
        "JUDGE_MAX_TOKENS": evaluate_run.MAX_TOKENS,
        "JUDGE_TIMEOUT": evaluate_run.TIMEOUT,
        "JUDGE_SYSTEM_PROMPT": evaluate_run.SYSTEM_PROMPT,
    }


def new_sweep(conn, label):
    """Create a run row and freeze the settings it will use. -> (run_id, conf).

    Seeds left as None in settings.py are drawn *here* and stored as the number
    that was drawn, rather than passed along as None. A sweep is then repeatable
    even when it was never asked to be: whatever it used is written down.
    """
    conf = config.snapshot()
    conf.update(judge_snapshot())
    if conf.get("SEED") is None:
        conf["SEED"] = random.randrange(_SEED_LIMIT)
    if conf.get("WEIGHT_MASTER_SEED") is None:
        conf["WEIGHT_MASTER_SEED"] = random.randrange(_SEED_LIMIT)

    run_id = store.create_run(conn, template=conf.get("TEMPLATE") or "template_code.py",
                              label=label)
    store.save_settings(conn, run_id, conf)
    return run_id, conf


# --- the steps -------------------------------------------------------------


def step_population(context):
    """Draw the chromosomes -> individuals."""
    conf = context.conf
    rng = random.Random(conf["SEED"])
    chromosomes = generate_population.build_population(
        conf["COUNT"], rng, conf["MAX_DEPTH"], conf["BRANCH_PROB"], conf["UNIQUE"])
    store.add_individuals(context.conn, context.run_id, chromosomes)

    sizes = [len(one.split(".")) for one in chromosomes]
    print("stored %d individuals in run %d (seed %s)"
          % (len(chromosomes), context.run_id, conf["SEED"]))
    print("symbols per individual: min %d, max %d, mean %.1f"
          % (min(sizes), max(sizes), sum(sizes) / len(sizes)))


def step_trees(context):
    """Draw each chromosome -> individuals.tree."""
    rows = store.individuals(context.conn, context.run_id)
    if not rows:
        raise SystemExit("run %d holds no individuals. Run the population step first."
                         % context.run_id)
    bad = 0
    for row in rows:
        try:
            drawing = "\n".join(draw_trees.draw(row["chromosome"]))
        except ValueError as error:
            # An undrawable individual is recorded with its complaint, not
            # dropped: it is still part of the population.
            drawing = "%s\n\n!! cannot draw: %s" % (row["chromosome"], error)
            bad += 1
        store.set_tree(context.conn, row["id"], drawing)
    context.conn.commit()

    print("drew %d trees into run %d" % (len(rows), context.run_id))
    if bad:
        print("%d individual(s) could not be drawn -- see the !! markers" % bad)


def step_runs(context):
    """Fill the template per individual -> script_source, state, rank, weight seed."""
    conn, run_id = context.conn, context.run_id
    rows = store.individuals(conn, run_id)
    if not rows:
        raise SystemExit("run %d holds no individuals. Run the population step first."
                         % run_id)

    template_lines = generate_runs.load_template(context.template)
    # Each slot's rank comes from its own adapter_config.json -- they may differ.
    ranks = generate_runs.slot_ranks(context.run_dir, context.template)
    # The generated scripts read their prompts at startup; fail here if they cannot.
    prompts_path, prompt_count = generate_runs.eval_prompt_count(
        context.run_dir, context.template)

    master = context.conf["WEIGHT_MASTER_SEED"]
    runnable = 0
    for row in rows:
        steps, final = generate_runs.plan(
            generate_population.decode(row["chromosome"])[0], ranks)
        # Derived from the master seed and the individual's own number, so it is
        # the same seed however the population is walked, and re-generating a
        # sweep from its stored settings reproduces the blends exactly.
        weight_seed = random.Random("%s:%d" % (master, row["number"])).randrange(_SEED_LIMIT)
        name = "run_%03d.py" % row["number"]
        source = generate_runs.render(
            row["chromosome"], steps, final,
            script_name=name,
            provenance="Generated by main.py from run %d, individual %d of %s."
                       % (run_id, row["number"], os.path.basename(conn.path)),
            label="Individual %d" % row["number"],
            template_lines=template_lines,
            weight_seed=weight_seed,
        )
        broken = any(step.broken for step in steps)
        runnable += not broken
        store.set_script(conn, row["id"], "BAD" if broken else "ok",
                         steps[-1].rank, name, source, weight_seed)
    conn.commit()

    written = store.materialise(conn, run_id, context.run_dir)
    print("stored %d scripts in run %d (from %s)"
          % (len(rows), run_id, os.path.basename(context.template)))
    print("slot ranks: %s" % ", ".join("%s=%d" % pair for pair in sorted(ranks.items())))
    print("eval prompts: %d from %s" % (prompt_count, os.path.basename(prompts_path)))
    print("%d runnable, %d blocked by PEFT's equal-rank rule for linear"
          % (runnable, len(rows) - runnable))
    print("wrote %d script file(s) to %s" % (written, context.run_dir))


def step_process(context):
    """Execute each script -> executions and exchanges."""
    conn, run_id, options = context.conn, context.run_id, context.options
    rows = [row for row in store.individuals(conn, run_id) if row["script_source"]]
    if not rows:
        raise SystemExit("run %d holds no generated scripts. Run the runs step first."
                         % run_id)

    # The scripts have to exist as files to be launched; rewrite any that are
    # missing or out of date.
    store.materialise(conn, run_id, context.run_dir)

    # Whether the venv is needed is a property of the template that was filled:
    # mocked scripts load nothing, and demanding the venv for them would put a
    # GPU-less machine out of reach.
    real = process_run.imports_unsloth(rows[0]["script_source"])
    if real:
        process_run.check_interpreter()

    selected = [row for row in rows
                if options.include_blocked or row["state"] != "BAD"]
    skipped = len(rows) - len(selected)
    if options.limit:
        selected = selected[:options.limit]
    if not selected:
        raise SystemExit("nothing to run: all %d individuals are marked BAD. "
                         "Pass --include-blocked to run them anyway." % len(rows))

    print("running %d of %d individuals%s"
          % (len(selected), len(rows),
             " (%d skipped as BAD)" % skipped if skipped else ""))
    print("each one loads the base model, so this takes a while\n" if real
          else "these are mocked scripts: nothing is loaded, so this is quick\n")

    failures = 0
    started = time.time()
    for number, row in enumerate(selected, 1):
        print("[%d/%d] %s  %s" % (number, len(selected), row["script_name"],
                                  row["chromosome"]))
        code, seconds, out, err = process_run.launch(
            context.run_dir, row["script_name"], options.timeout)
        verdict = process_run.verdict_of(code)
        if code != 0:
            failures += 1

        transcript = process_run.exchanges(out)
        execution_id = store.add_execution(
            conn, row["id"], seconds, code, verdict, row["weight_seed"],
            process_run.drawn_weights(out), out, err)
        store.add_exchanges(conn, execution_id, transcript)
        # Commit per individual, so an interrupted sweep keeps what it has done.
        conn.commit()

        print("        %-9s %6.1fs  %d exchange(s) -> execution %d"
              % (verdict, seconds, len(transcript), execution_id))
        if code != 0:
            # Show why, so a systemic problem is obvious without a query.
            tail = [line for line in (out + err).splitlines() if line.strip()][-1:]
            if tail:
                print("        %s" % tail[0][:100])

    print("\nran %d in %.1fs, %d failed" % (len(selected), time.time() - started, failures))

    # The scripts have done their job: everything they printed is in the
    # database, and their source is in individuals.script_source, so the files
    # themselves are spent. Clearing them keeps run_db/ to the database plus
    # whatever is still waiting to run, and makes it impossible to launch a
    # stale one by hand later. This happens whatever the individuals did --
    # a failed run is still a processed one, and its output is stored either way.
    if options.keep_scripts:
        print("kept the scripts in %s (--keep-scripts)" % context.run_dir)
    else:
        gone = store.remove_scripts(conn, run_id, context.run_dir,
                                    [row["script_name"] for row in selected])
        print("removed %d processed script(s) from %s -- they are still in the "
              "database (python main.py runs, or python store.py --export)"
              % (gone, context.run_dir))

    # A chromosome that cannot run is a result, not a pipeline failure. Only a
    # sweep where nothing at all worked points at something systemic.
    if failures == len(selected):
        raise SystemExit("every individual failed -- try: python store.py --show %d"
                         % run_id)


def step_evaluate(context):
    """Score every answer with the judge -> quality and reason on each exchange."""
    conn, run_id, options = context.conn, context.run_id, context.options
    pending = store.exchanges_to_score(conn, run_id, options.force)
    if not pending:
        total = conn.execute(
            "SELECT COUNT(*) AS n FROM exchanges x JOIN executions e ON e.id = x.execution_id"
            " JOIN individuals i ON i.id = e.individual_id WHERE i.run_id = ?",
            (run_id,)).fetchone()["n"]
        if not total:
            raise SystemExit("run %d holds no answers to score. Run the process step first."
                             % run_id)
        # Only the most recent execution of each individual is scored -- that is
        # the current result. An older one keeps whatever score it was given.
        print("judge: not contacted -- every answer in the latest execution of each "
              "individual already has a quality")
        return

    # Only reach for the judge if something actually needs grading, so a mocked
    # sweep -- which arrives already scored -- needs no endpoint up. An answer
    # that is blank is scored here without asking anyone.
    needs_judge = any(row["answer"].strip() for row in pending)
    judge_settings = {
        "base_url": context.conf.get("JUDGE_BASE_URL", evaluate_run.BASE_URL),
        "api_key": evaluate_run.API_KEY,
        "timeout": context.conf.get("JUDGE_TIMEOUT", evaluate_run.TIMEOUT),
        "model": context.conf.get("JUDGE_MODEL"),
    }
    if needs_judge and not judge_settings["model"]:
        judge_settings["model"] = evaluate_run.discover_model(
            judge_settings["base_url"], judge_settings["api_key"],
            judge_settings["timeout"])

    if needs_judge:
        print("judge: %s at %s"
              % (judge_settings["model"], judge_settings["base_url"]))
    else:
        print("judge: not contacted -- no answer needs grading")
    print("scoring %d answer(s)%s\n"
          % (len(pending), " (--force: re-scoring)" if options.force else ""))

    scored = failed = 0
    current = None
    started = time.time()
    for row in pending:
        if row["number"] != current:
            current = row["number"]
            print("individual %d  %s" % (row["number"], row["chromosome"]))

        answer = row["answer"] or ""
        if not answer.strip():
            # Nothing to grade: an unanswered question is worth nothing, and
            # asking the judge about an empty string just wastes a call.
            store.score_exchange(conn, row["id"], 0.0, "no answer given", None)
            scored += 1
            print("    [%d] 0.00  (no answer)" % row["position"])
        else:
            try:
                quality, reason = evaluate_run.judge(row["question"], answer,
                                                     judge_settings)
                store.score_exchange(conn, row["id"], round(quality, 3), reason,
                                     judge_settings["model"])
                scored += 1
                print("    [%d] %.2f  %s" % (row["position"], round(quality, 3), reason))
            except (RuntimeError, ValueError) as error:
                failed += 1
                print("    [%d] FAILED  %s" % (row["position"], error))
        # Saved as we go, so an interrupted scoring run resumes where it stopped.
        conn.commit()

    print("\nscored %d, %d failed, in %.1fs" % (scored, failed, time.time() - started))

    qualities = [row["quality"] for row in store.quality_rows(conn, run_id)
                 if row["quality"] is not None]
    if qualities:
        print("quality across %d individual(s): min %.3f, max %.3f, mean %.3f"
              % (len(qualities), min(qualities), max(qualities),
                 sum(qualities) / len(qualities)))

    if failed and not scored:
        raise SystemExit("nothing could be scored -- check the judge endpoint")


def step_fitness(context):
    """Fold each transcript into one number -> individuals.fitness."""
    conn, run_id = context.conn, context.run_id
    rows = calculate_fitness.assign(conn, run_id)
    if not rows:
        raise SystemExit("run %d holds no individuals. Run the population step first."
                         % run_id)
    if not any(row["quality"] is not None for row in rows):
        raise SystemExit("no answer in run %d carries a quality yet -- run the "
                         "evaluate step first." % run_id)

    print("fitness = mean quality across the exchanges of the latest execution\n")
    print("    %-4s %-5s %-7s %-7s %s"
          % ("#", "state", "answers", "fitness", "chromosome"))
    values = []
    for row in rows:                            # already best first
        value = calculate_fitness.fitness_of(row)
        values.append(value)
        print("    %-4d %-5s %-7d %-7.3f %s"
              % (row["number"], row["state"] or "-", row["answers"] or 0,
                 value, row["chromosome"]))

    print("\nwrote fitness for %d individual(s): min %.3f, max %.3f, mean %.3f"
          % (len(values), min(values), max(values), sum(values) / len(values)))

    # An individual averaged over part of its transcript still gets a fitness,
    # but it is a weaker claim than one averaged over all of it -- say so.
    partial = [row["number"] for row in rows
               if row["quality"] is not None and (row["unscored"] or 0)]
    if partial:
        print("%d individual(s) averaged over a partly scored transcript: %s"
              % (len(partial), ", ".join(str(number) for number in partial)))
    # 0.0 here is a decision, not a gap -- see calculate_fitness.py.
    empty = [row["number"] for row in rows if row["quality"] is None]
    if empty:
        print("%d individual(s) had no judged answer and scored 0.0: %s"
              % (len(empty), ", ".join(str(number) for number in empty)))


Step = namedtuple("Step", "name run description")

STEPS = [
    Step("population", step_population,
         "draw the chromosomes -> individuals"),
    Step("trees", step_trees,
         "draw each chromosome as a tree -> individuals.tree"),
    Step("runs", step_runs,
         "fill the template per individual -> individuals.script_source + run_db/"),
    # The expensive one: a base-model load per individual. Keep COUNT small
    # while iterating, or run the earlier steps on their own.
    Step("process", step_process,
         "execute each script -> executions, exchanges; then delete the scripts"),
    # Also slow, and needs the judge endpoint up: one grading call per answer.
    Step("evaluate", step_evaluate,
         "score every answer with the judge -> exchanges.quality"),
    # Cheap, and pure arithmetic over what evaluate stored: no model, no judge.
    Step("fitness", step_fitness,
         "average each transcript's qualities -> individuals.fitness"),
]


# --- driver ----------------------------------------------------------------


def run(steps, context):
    """Run `steps` in order. Returns the exit code for the process.

    A sweep that stops part way is left marked 'failed', so `python store.py
    --list` says so rather than presenting a half-finished run as a result.
    """
    started = time.time()
    for number, step in enumerate(steps, 1):
        print("=" * 70)
        print("[%d/%d] %s -- %s" % (number, len(steps), step.name, step.description))
        print("=" * 70)
        step_started = time.time()
        try:
            step.run(context)
        except SystemExit as error:
            if error.code not in (0, None):
                print("\nSTOPPED in step '%s': %s" % (step.name, error))
                print("Later steps were skipped, since they build on this one.")
                store.finish_run(context.conn, context.run_id, "failed")
                return 1
        except Exception as error:                      # noqa: BLE001 - report and stop
            print("\nSTOPPED in step '%s': %s: %s"
                  % (step.name, type(error).__name__, error))
            store.finish_run(context.conn, context.run_id, "failed")
            return 1
        print("  (%s took %.1fs)\n" % (step.name, time.time() - step_started))

    store.finish_run(context.conn, context.run_id, "done")
    print("=" * 70)
    print("done: %s in %.1fs" % (", ".join(step.name for step in steps),
                                 time.time() - started))
    print("run %d in %s -- python store.py --show %d"
          % (context.run_id, context.conn.path, context.run_id))
    print("=" * 70)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the pipeline into a sqlite database: %s."
                    % " -> ".join(step.name for step in STEPS)
    )
    parser.add_argument("steps", nargs="*", metavar="STEP",
                        help="steps to run (default: all of them, in order)")
    parser.add_argument("--list", action="store_true",
                        help="list the steps and exit")
    parser.add_argument("--db", default=config.DB_PATH,
                        help="database file (default %s)" % config.DB_PATH)
    parser.add_argument("--run", type=int, default=None, metavar="ID",
                        help="resume this sweep instead of starting one (0 = the latest)")
    parser.add_argument("--label", default=None,
                        help="a note stored with the sweep, to find it again later")
    parser.add_argument("--run-dir", default=None,
                        help="folder for the generated scripts (default %s)"
                             % config.DB_RUN_DIR)
    parser.add_argument("--limit", type=int, default=0,
                        help="process only the first N individuals (0 = all)")
    parser.add_argument("--include-blocked", action="store_true",
                        help="also run the ones marked BAD")
    parser.add_argument("--keep-scripts", action="store_true",
                        help="leave the generated scripts on disk after processing "
                             "them (default: delete them; the source is in the database)")
    parser.add_argument("--timeout", type=int, default=900,
                        help="seconds to allow each script (default 900)")
    parser.add_argument("--force", action="store_true",
                        help="re-score answers that already have a quality")
    args = parser.parse_args(argv)

    known = {step.name: step for step in STEPS}

    if args.list:
        for step in STEPS:
            print("%-12s %s" % (step.name, step.description))
        return 0

    if args.steps:
        unknown = [name for name in args.steps if name not in known]
        if unknown:
            parser.error("unknown step(s): %s. Known steps: %s"
                         % (", ".join(unknown), ", ".join(known)))
        # Keep the order the pipeline defines, not the order they were typed.
        selected = [step for step in STEPS if step.name in set(args.steps)]
    else:
        selected = list(STEPS)

    conn = store.connect(args.db)

    # A sweep that draws a population is a new sweep; one that does not is
    # continuing an existing one, and must use the settings that one was
    # created with rather than whatever settings.py says now.
    if args.run is None and any(step.name == "population" for step in selected):
        run_id, conf = new_sweep(conn, args.label)
        print("new run %d in %s\n" % (run_id, conn.path))
    else:
        run_id = store.latest_run(conn) if args.run in (None, 0) else args.run
        if run_id is None:
            raise SystemExit("%s holds no runs yet -- start one with the population "
                             "step." % conn.path)
        if store.get_run(conn, run_id) is None:
            raise SystemExit("no run %d in %s. Try: python store.py --list"
                             % (run_id, conn.path))
        conf = store.get_settings(conn, run_id)
        print("resuming run %d in %s\n" % (run_id, conn.path))

    run_dir = args.run_dir or conf.get("DB_RUN_DIR") or config.DB_RUN_DIR
    if not os.path.isabs(run_dir):
        run_dir = os.path.join(_HERE, run_dir)
    # Absolute from here on: each script is launched with cwd set to this
    # folder, so a relative path would be resolved against itself a second time.
    run_dir = os.path.abspath(run_dir)

    context = Context(conn, run_id, conf, run_dir,
                      _template_path(conf.get("TEMPLATE")), args)
    try:
        return run(selected, context)
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
