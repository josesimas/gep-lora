"""
process_run.py - Launch a generated script and make sense of what it printed.

main.py writes each individual's script out of the database, hands it to
launch() as its own process, and files everything it said back into the
database. This module is the part that knows how to do that: how to run one
script, how to read a transcript out of its stdout, and how to check that this
interpreter can run it at all.

Each script is a separate process, because each loads the base model at import
and attaches its own adapters -- they cannot share an interpreter. That also
means process is the expensive step: a model load per individual, so a
population of 100 is a long sweep. Keep COUNT small in settings.py while
iterating, or use --limit.

Separate processes are also what makes them overlappable, and launch_batch()
runs PROCESS_RUN_BATCH_SIZE of them at once -- the model loads are what a sweep
spends its time on, and they are independent. The ceiling is fixed rather than
adaptive: N scripts means N copies of the base model resident together, so the
number belongs to the machine and is set in settings.py, not guessed at here.

Individuals that fail are recorded, not fatal: a chromosome that cannot run is
a result, the same as one that can. Only a sweep where nothing at all ran
returns a failing exit code, since that points at something systemic.

None of that cost applies to scripts generated from template_code_mocked.py:
they load nothing, answer at random, and print their own QUALITY:/REASON: lines,
which land in the transcript so the evaluate step has nothing left to score.
imports_unsloth() is how the pipeline notices which kind it is looking at and
drops the venv check accordingly.
"""

import concurrent.futures
import os
import re
import subprocess
import sys
import time


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

    Nothing is written here: the four values go straight into the database, so
    the only file this step needs is the script itself.
    """
    script_path = os.path.join(run_dir, script)
    started = time.time()
    try:
        # cwd is the run folder, so the caches unsloth drops stay there.
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


def batch_size(value):
    """How many scripts may run at once, from the setting -> at least 1.

    A sweep created before PROCESS_RUN_BATCH_SIZE existed has no value recorded
    for it, and a value below 1 is the same request as 1 stated badly; both mean
    one script at a time rather than an error, since neither says anything about
    the machine this is now running on.
    """
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def batches(items, size):
    """`items` in consecutive groups of at most `size`, in the order given.

    Groups rather than a refilling queue: main.py stores a batch's results
    before it starts the next one, which is what keeps the database written from
    one thread and in the order the individuals were selected in.
    """
    return [items[start:start + size] for start in range(0, len(items), size)]


def launch_batch(run_dir, scripts, timeout):
    """Run these scripts at once. -> one launch() result each, in `scripts` order.

    Results come back in the order asked for, not the order they finished, so a
    caller can pair them with the individuals it passed in without the children
    having to say who they are.

    Threads, not processes: each one only waits on a subprocess, and the work is
    in the children anyway. The timeout is per script, as it is sequentially --
    a batch is not killed because one member of it hung.

    The seconds each result carries are still that script's own wall clock, so
    they overlap and no longer add up to the time the batch took.
    """
    if len(scripts) == 1:                       # the sequential case, unchanged
        return [launch(run_dir, scripts[0], timeout)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(scripts)) as pool:
        running = [pool.submit(launch, run_dir, script, timeout)
                   for script in scripts]
        return [future.result() for future in running]


def verdict_of(code):
    """The one word an execution's verdict column holds for an exit code."""
    if code == 0:
        return "ok"
    if code is None:
        return "timeout"
    return "exit %d" % code


def imports_unsloth(source):
    """Does this generated script load the real thing? The mocked ones do not.

    A test on the source itself, so it can be asked of a script held in the
    database as easily as of one already written out to disk.
    """
    return "import unsloth" in source or "from unsloth" in source


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
