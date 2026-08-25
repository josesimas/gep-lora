"""
main.py - Run the whole pipeline end to end.

    population -> trees -> runs -> process -> evaluate

Each stage is one entry in STEPS below. A step names a callable and the
arguments to hand it; the callable is just the `main(argv)` of one of the
scripts in this folder, so a step behaves exactly as if you had run that script
from the command line, and nothing is duplicated here.

Adding a step later
-------------------
Append a Step to the list. It can be another script's main:

    Step("score", score_runs.main, ["--input", "run/index.txt"],
         "score every runnable individual -> run/scores.txt"),

or any plain function that accepts a list of arguments:

    Step("archive", archive_run, [],
         "copy run/ aside so the next run does not overwrite it"),

Steps run in the order listed and stop at the first failure, so a later step can
rely on the output of an earlier one.

The last two steps are the slow ones. `process` runs the generated scripts,
loading the base model once per individual; `evaluate` then makes one grading
call per answer against the judge model configured in evaluate_run.py, which
must be reachable. `python main.py` is therefore a long operation, and
`python main.py population trees runs` stops short of both.

Both of those costs disappear if you set TEMPLATE below to
"template_code_mocked.py": the runs step then generates scripts that load
nothing and answer at random, so a full `python main.py` takes seconds and needs
neither a GPU nor the judge. That checks the plumbing, not the blends.

Usage:
    python main.py                  # every step, in order
    python main.py runs             # just that step
    python main.py trees runs       # a subset, in the listed order
    python main.py --list           # show the steps without running them
"""

import argparse
import os
import sys
import time
from collections import namedtuple

import draw_trees
import evaluate_run
import generate_population
import generate_runs
import process_run

_HERE = os.path.dirname(os.path.abspath(__file__))

# --- settings for a complete run ------------------------------------------

# How many individuals the population holds.
COUNT = 30

# Seed for the population draw. An int repeats the same population every run;
# None grows a fresh one each time. Note this is separate from the LoRA blend
# weights, which each generated script draws for itself at runtime.
SEED = 42

# Reject duplicate chromosomes when building the population.
UNIQUE = True

# Which template generate_runs.py fills. None means its own default,
# template_code.py -- the real thing. Set it to "template_code_mocked.py" for a
# dry run: same trees, same ranks, same BAD verdicts, but no model load, random
# answers, and scores that arrive with the transcript, so the whole pipeline
# finishes in seconds on a machine with no GPU and no judge running. Mocked
# scores are noise; never read one as a result.
TEMPLATE = "template_code_mocked.py"


# --- the pipeline ----------------------------------------------------------

Step = namedtuple("Step", "name run args description")


def _population_args():
    """CLI arguments for the population step, from the settings above."""
    args = ["--count", str(COUNT)]
    if SEED is not None:
        args += ["--seed", str(SEED)]
    if UNIQUE:
        args.append("--unique")
    return args


def _runs_args():
    """CLI arguments for the runs step: which template to fill, if not the default."""
    return ["--template", TEMPLATE] if TEMPLATE else []


STEPS = [
    Step("population", generate_population.main, _population_args(),
         "grow %d random chromosomes -> run/population.txt" % COUNT),
    Step("trees", draw_trees.main, [],
         "draw each chromosome as a tree -> run/trees.txt"),
    Step("runs", generate_runs.main, _runs_args(),
         "fill %s, one runnable script per chromosome -> run/"
         % (TEMPLATE or "template_code.py")),
    # The expensive one: a base-model load per individual. Keep COUNT small
    # while iterating, or run the earlier steps on their own.
    Step("process", process_run.main, [],
         "execute each generated script -> run/output_NNN.txt, run/results.txt"),
    # Also slow, and needs the judge endpoint up: one grading call per answer.
    Step("evaluate", evaluate_run.main, [],
         "score every answer with the judge -> quality in run/output_result_NNN.json"),
]


# --- driver ----------------------------------------------------------------


def run(steps):
    """Run `steps` in order. Returns the exit code for the process."""
    started = time.time()
    for number, step in enumerate(steps, 1):
        print("=" * 70)
        print("[%d/%d] %s -- %s" % (number, len(steps), step.name, step.description))
        print("=" * 70)
        step_started = time.time()
        try:
            step.run(step.args)
        except SystemExit as error:
            # The scripts raise SystemExit with a message when they cannot go on.
            if error.code not in (0, None):
                print("\nSTOPPED in step '%s': %s" % (step.name, error))
                print("Later steps were skipped, since they build on this one.")
                return 1
        except Exception as error:                      # noqa: BLE001 - report and stop
            print("\nSTOPPED in step '%s': %s: %s"
                  % (step.name, type(error).__name__, error))
            return 1
        print("  (%s took %.1fs)\n" % (step.name, time.time() - step_started))

    print("=" * 70)
    print("done: %s in %.1fs" % (", ".join(step.name for step in steps),
                                 time.time() - started))
    print("=" * 70)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the pipeline: %s." % " -> ".join(step.name for step in STEPS)
    )
    parser.add_argument("steps", nargs="*", metavar="STEP",
                        help="steps to run (default: all of them, in order)")
    parser.add_argument("--list", action="store_true",
                        help="list the steps and exit")
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

    return run(selected)


if __name__ == "__main__":
    sys.exit(main())
