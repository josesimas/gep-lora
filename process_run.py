"""
process_run.py - Execute every generated script and collect what it says.

generate_runs.py writes run/run_001.py ... run_NNN.py but does not run them.
This does: it launches each one the way you would by hand, captures everything
it prints, and writes a summary.

    run/output_001.txt   everything run_001.py printed (replies included)
    run/results.txt      one row per individual: state, exit code, seconds

Each script is a separate process, because each loads the base model at import
and attaches its own adapters -- they cannot share an interpreter. That also
means this step is the expensive one: a model load per individual, so a
population of 100 is a long sweep. Keep COUNT small in main.py while iterating.

Individuals that fail are recorded, not fatal: a chromosome that cannot run is
a result, the same as one that can. Only a sweep where nothing at all ran
returns a failing exit code, since that points at something systemic.

By default the ones index.txt marks BAD are skipped -- they stop at their bad
combine step, but only after paying for a full model load. Pass
--include-blocked to run them anyway and capture the error.

Usage:
    python process_run.py                     # every runnable individual
    python process_run.py --limit 3           # just the first few, to smoke test
    python process_run.py --include-blocked   # BAD ones too
"""

import argparse
import os
import re
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))

INDEX_NAME = "index.txt"
RESULTS_NAME = "results.txt"


class Individual:
    """One row of index.txt: which script, and what generation predicted."""

    __slots__ = ("script", "state", "rank", "expression")

    def __init__(self, script, state, rank, expression):
        self.script = script
        self.state = state            # "ok" or "BAD"
        self.rank = rank
        self.expression = expression


def read_index(run_dir):
    """Parse run/index.txt -> list of Individual, in file order."""
    path = os.path.join(run_dir, INDEX_NAME)
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except OSError as error:
        raise SystemExit("cannot read %s (%s). Run generate_runs.py first."
                         % (path, error.strerror))

    individuals = []
    for line in lines[1:]:                      # first line is the header
        fields = line.split()
        # script  state  "rank"  N  expression
        if len(fields) < 5 or not fields[0].endswith(".py"):
            continue
        individuals.append(Individual(fields[0], fields[1], int(fields[3]), fields[4]))
    if not individuals:
        raise SystemExit("%s lists no individuals. Run generate_runs.py first." % path)
    return individuals


def output_name(script):
    """run_007.py -> output_007.txt, so the pair is easy to line up."""
    match = re.search(r"(\d+)", script)
    return "output_%s.txt" % match.group(1) if match else "output_%s.txt" % script


def execute(run_dir, individual, timeout):
    """Run one generated script, capture its output, and write it alongside.

    Returns (exit_code, seconds, output_path). An exit code of None means the
    script was still going when the timeout expired.
    """
    script_path = os.path.join(run_dir, individual.script)
    started = time.time()
    try:
        # cwd is run/ so the caches unsloth drops stay in the output folder.
        completed = subprocess.run(
            [sys.executable, script_path],
            cwd=run_dir, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
        captured = completed.stdout + completed.stderr
        code = completed.returncode
    except subprocess.TimeoutExpired as expired:
        captured = (expired.stdout or "") + (expired.stderr or "")
        captured += "\n\n!! killed after %ss (--timeout)\n" % timeout
        code = None

    elapsed = time.time() - started
    output_path = os.path.join(run_dir, output_name(individual.script))
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("# %s\n# %s\n# predicted rank %d, index says %s\n\n"
                     % (individual.script, individual.expression,
                        individual.rank, individual.state))
        handle.write(captured)
    return code, elapsed, output_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run every generated script and collect its output."
    )
    parser.add_argument("--run-dir", default=os.path.join(_HERE, "run"),
                        help="folder holding the generated scripts (default run)")
    parser.add_argument("--limit", type=int, default=0,
                        help="run only the first N individuals (0 = all)")
    parser.add_argument("--include-blocked", action="store_true",
                        help="also run the ones index.txt marks BAD")
    parser.add_argument("--timeout", type=int, default=900,
                        help="seconds to allow each script (default 900)")
    args = parser.parse_args(argv)

    individuals = read_index(args.run_dir)
    selected = [one for one in individuals
                if args.include_blocked or one.state != "BAD"]
    skipped = len(individuals) - len(selected)
    if args.limit:
        selected = selected[:args.limit]

    if not selected:
        raise SystemExit("nothing to run: all %d individuals are marked BAD. "
                         "Pass --include-blocked to run them anyway." % len(individuals))

    print("running %d of %d individuals%s"
          % (len(selected), len(individuals),
             " (%d skipped as BAD)" % skipped if skipped else ""))
    print("each one loads the base model, so this takes a while\n")

    rows = []
    failures = 0
    started = time.time()
    for number, one in enumerate(selected, 1):
        print("[%d/%d] %s  %s" % (number, len(selected), one.script, one.expression))
        code, elapsed, output_path = execute(args.run_dir, one, args.timeout)

        if code == 0:
            verdict = "ok"
        elif code is None:
            verdict = "timeout"
            failures += 1
        else:
            verdict = "exit %d" % code
            failures += 1
        print("        %-9s %6.1fs  -> %s" % (verdict, elapsed, os.path.basename(output_path)))

        if code not in (0, None):
            # Show why, so a systemic problem is obvious without opening files.
            tail = [line for line in open(output_path, encoding="utf-8").read().splitlines()
                    if line.strip()][-1:]
            if tail:
                print("        %s" % tail[0][:100])

        rows.append("%-14s %-5s %-8s %7.1f  %-22s %s"
                    % (one.script, one.state, verdict, elapsed,
                       os.path.basename(output_path), one.expression))

    results_path = os.path.join(args.run_dir, RESULTS_NAME)
    with open(results_path, "w", encoding="utf-8") as handle:
        handle.write("%-14s %-5s %-8s %7s  %-22s %s\n"
                     % ("script", "state", "result", "secs", "output", "expression"))
        handle.write("\n".join(rows) + "\n")

    print("\nran %d in %.1fs, %d failed" % (len(selected), time.time() - started, failures))
    print("results: %s" % results_path)

    # A chromosome that cannot run is a result, not a pipeline failure. Only a
    # sweep where nothing at all worked points at something systemic.
    if failures == len(selected):
        raise SystemExit("every individual failed -- check %s"
                         % os.path.basename(results_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
