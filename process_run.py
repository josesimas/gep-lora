"""
process_run.py - Execute every generated script and collect what it says.

generate_runs.py writes run/run_001.py ... run_NNN.py but does not run them.
This does: it launches each one the way you would by hand, captures everything
it prints, and writes a summary.

    run/output_001.txt          everything run_001.py printed
    run/output_result_001.json  the exchanges, plus the chromosome and weight draw
    run/results.txt             one row per individual: state, exit code, seconds

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

None of that cost applies to scripts generated from template_code_mocked.py:
they load nothing, answer at random, and print their own QUALITY:/REASON: lines,
which land in the transcript so evaluate_run.py has nothing left to score. This
notices which kind it is looking at and drops the venv check accordingly.

Usage:
    python process_run.py                     # every runnable individual
    python process_run.py --limit 3           # just the first few, to smoke test
    python process_run.py --include-blocked   # BAD ones too
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import namedtuple

_HERE = os.path.dirname(os.path.abspath(__file__))

INDEX_NAME = "index.txt"
RESULTS_NAME = "results.txt"

Outcome = namedtuple("Outcome", "code seconds output_path result_path exchanges")


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


def _number(script):
    """run_007.py -> "007", so every file for one individual lines up."""
    match = re.search(r"(\d+)", script)
    return match.group(1) if match else script


def drawn_weights(stdout):
    """The blend weights a run drew for itself, from the line it prints.

    Every generated script redraws w1..w5 at startup, so two runs of the same
    chromosome are scored under different blends. Recording the draw is what
    makes a transcript traceable back to the weights that produced it.

    Read off the "weights: w1=..., w2=..." line rather than recomputed, so this
    is the draw that was actually used. All five are recorded, not just the ones
    this tree happens to reference.
    """
    line = re.search(r"^weights:.*$", stdout, re.MULTILINE)
    if not line:
        return {}
    return {name: float(value)
            for name, value in re.findall(r"(w\d+)=([-+0-9.eE]+)", line.group(0))}


GRADE_PREFIXES = ("QUALITY:", "REASON:")


def _split_grade(lines):
    """Separate a reply from the QUALITY:/REASON: lines that may follow it.

    Only the mocked template (template_code_mocked.py) prints those, so for a
    real run this returns the lines untouched and an empty grade -- the scoring
    still comes from evaluate_run.py. For a mocked run it is what carries the
    made-up score into the transcript, so a dry sweep needs no judge endpoint.

    A reply line that genuinely started with "QUALITY:" would be cut short here.
    That is the price of not needing a separate channel out of the child
    process, and no real reply has ever begun that way.
    """
    answer, grade = [], {}
    for line in lines:
        if line.startswith("QUALITY:"):
            try:
                grade["quality"] = float(line[len("QUALITY:"):].strip())
            except ValueError:
                pass                            # not a number: leave it ungraded
        elif line.startswith("REASON:"):
            grade["reason"] = line[len("REASON:"):].strip()
        elif not grade:                         # still in the reply itself
            answer.append(line)
    return answer, grade


def exchanges(stdout):
    """The YOU/COACH pairs in a run's stdout, as {"question", "answer"} dicts.

    A reply can wrap over several lines, so a question owns everything printed
    after it until the next question. Only stdout is scanned -- the loading bars
    and warnings arrive on stderr, so they cannot leak into a transcript.

    A question whose reply never arrived (a run killed mid-generation) keeps an
    empty answer, rather than being dropped as if it had never been asked.

    An exchange also picks up "quality" and "reason" when the run printed them,
    which only the mocked template does; see _split_grade.
    """
    blocks, current = [], None
    for line in stdout.splitlines():
        if line.startswith("YOU:"):
            if current is not None:
                blocks.append(current)
            current = [line]
        elif current is not None:
            current.append(line)
    if current is not None:
        blocks.append(current)

    transcript = []
    for block in blocks:
        question = block[0][len("YOU:"):].strip()
        answer, grade = [], {}
        for offset, line in enumerate(block[1:], start=1):
            if line.startswith("COACH:"):
                # The reply is the rest of that line plus every line after it.
                rest = [line[len("COACH:"):].strip()] + block[offset + 1:]
                answer, grade = _split_grade(rest)
                break
        while answer and not answer[-1].strip():   # trim the gap before the next question
            answer.pop()
        exchange = {"question": question, "answer": "\n".join(answer)}
        # Same key order a scored transcript ends up with either way.
        exchange.update(grade)
        transcript.append(exchange)
    return transcript


def launch(run_dir, script, timeout):
    """Run one generated script. -> (exit code, seconds, stdout, stderr).

    An exit code of None means the script was still going when the timeout
    expired. stdout is kept apart from stderr so the transcript can be taken
    from it cleanly.

    Kept apart from execute() below because a script has to be launched the same
    way whatever collects the result: execute() files the output under run/,
    while main_sqlite.py puts the very same four values into the database.
    """
    script_path = os.path.join(run_dir, script)
    started = time.time()
    try:
        # cwd is run/ so the caches unsloth drops stay in the output folder.
        completed = subprocess.run(
            [sys.executable, script_path],
            cwd=run_dir, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
        out, err, code = completed.stdout, completed.stderr, completed.returncode
    except subprocess.TimeoutExpired as expired:
        out, err = expired.stdout or "", expired.stderr or ""
        err += "\n\n!! killed after %ss (--timeout)\n" % timeout
        code = None
    return code, time.time() - started, out, err


def verdict_of(code):
    """The one word results.txt uses for an exit code."""
    if code == 0:
        return "ok"
    if code is None:
        return "timeout"
    return "exit %d" % code


def execute(run_dir, individual, timeout):
    """Run one generated script and write both of its output files."""
    code, elapsed, out, err = launch(run_dir, individual.script, timeout)
    number = _number(individual.script)

    # The full capture, for working out why a run went wrong.
    output_path = os.path.join(run_dir, "output_%s.txt" % number)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("# %s\n# %s\n# predicted rank %d, index says %s\n\n"
                     % (individual.script, individual.expression,
                        individual.rank, individual.state))
        handle.write(out + err)

    # The conversation as JSON, with the two things needed to make sense of it
    # later: which tree was built, and which weight draw it was built with. A
    # score means little without both. A run that failed before answering leaves
    # "exchanges": [], which still parses.
    transcript = exchanges(out)
    result_path = os.path.join(run_dir, "output_result_%s.json" % number)
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump({"chromosome": individual.expression,
                   "weights": drawn_weights(out),
                   "exchanges": transcript},
                  handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    return Outcome(code, elapsed, output_path, result_path, len(transcript))


def imports_unsloth(source):
    """Does this generated script load the real thing? The mocked ones do not.

    A test on the source itself, so it can be asked of a script held in the
    sqlite database as easily as of one sitting in run/.
    """
    return "import unsloth" in source or "from unsloth" in source


def needs_unsloth(run_dir, individuals):
    """Do the generated scripts import unsloth? The mocked ones do not.

    Asked of the scripts themselves rather than of a flag, because the answer
    is a property of the template they came from: scripts generated from
    template_code_mocked.py load nothing, and demanding the venv for them would
    put a GPU-less machine out of reach of the very sweep meant to run there.
    They are all generated from one template, so the first is representative.
    """
    for individual in individuals[:1]:
        try:
            with open(os.path.join(run_dir, individual.script), encoding="utf-8") as handle:
                source = handle.read()
        except OSError:
            return True                         # cannot tell: keep the guard
        return imports_unsloth(source)
    return True


def check_interpreter():
    """Fail fast if this interpreter cannot import what the scripts need.

    Each generated script is launched with sys.executable, so running this with
    the wrong python means every individual dies on `import unsloth` -- after
    paying for a process launch each time, and leaving a population's worth of
    empty transcripts behind. find_spec only looks the module up, so the check
    costs nothing next to actually importing it.
    """
    probe = ("import importlib.util, sys; "
             "sys.exit(0 if importlib.util.find_spec('unsloth') else 1)")
    if subprocess.run([sys.executable, "-c", probe]).returncode:
        raise SystemExit(
            "%s cannot import unsloth, and the generated scripts run under this "
            "same interpreter -- every one of them would fail. Re-run with the "
            "project venv's python (see the PATH gotcha in README.md)."
            % sys.executable
        )


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

    # Absolute from here on: each script is launched with cwd set to this
    # folder, so a relative --run-dir would be resolved against itself a second
    # time and the script path would come out doubled.
    args.run_dir = os.path.abspath(args.run_dir)

    individuals = read_index(args.run_dir)
    if needs_unsloth(args.run_dir, individuals):
        check_interpreter()

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
    print("each one loads the base model, so this takes a while\n"
          if needs_unsloth(args.run_dir, individuals)
          else "these are mocked scripts: nothing is loaded, so this is quick\n")

    rows = []
    failures = 0
    started = time.time()
    for number, one in enumerate(selected, 1):
        print("[%d/%d] %s  %s" % (number, len(selected), one.script, one.expression))
        outcome = execute(args.run_dir, one, args.timeout)

        verdict = verdict_of(outcome.code)
        if outcome.code != 0:
            failures += 1
        print("        %-9s %6.1fs  %d exchange(s) -> %s"
              % (verdict, outcome.seconds, outcome.exchanges,
                 os.path.basename(outcome.result_path)))

        if outcome.code not in (0, None):
            # Show why, so a systemic problem is obvious without opening files.
            tail = [line for line
                    in open(outcome.output_path, encoding="utf-8").read().splitlines()
                    if line.strip()][-1:]
            if tail:
                print("        %s" % tail[0][:100])

        rows.append("%-14s %-5s %-8s %7.1f %5d  %-26s %s"
                    % (one.script, one.state, verdict, outcome.seconds, outcome.exchanges,
                       os.path.basename(outcome.result_path), one.expression))

    results_path = os.path.join(args.run_dir, RESULTS_NAME)
    with open(results_path, "w", encoding="utf-8") as handle:
        handle.write("%-14s %-5s %-8s %7s %5s  %-26s %s\n"
                     % ("script", "state", "result", "secs", "qa", "transcript", "expression"))
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
