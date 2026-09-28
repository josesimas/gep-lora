"""
testpass.py - A finished job's blends, put in front of its testing split.

The API's half of `python -m gep_lora.core.testing.test_run_with_dataset --from-db`: what a
form needs to offer (`choices`), what a request may ask for (`options_for`)
and the command line the worker runs (`command`). Choosing the individuals,
re-pointing their scripts, running them and grading the answers are that
script's, and are not decided again here.

A testing pass is **the job, queued again** -- task `test` -- the way an
evaluation is: it reads and writes the job's own database, so it takes the
job's place in the queue, its log and its cancel, and the worker settles the
job back where its search is (done) when it ends. It may only be asked of a
**finished** search: a half-finished one's blends are not what the search
found, and the testing pass exists to ask whether what it found holds up.

Two differences from the pass gep_lora/core/pipeline/main.py runs at the end of a search, and both
are the point of asking for one:

  * **every blend, by default.** gep_lora/core/pipeline/main.py's pass tests the individuals above
    TESTING_MIN_QUALITY -- the ones worth testing to decide whether the search
    worked. This one is asked for so a person can *choose* a blend, and a
    blend they were never shown a testing score for is one they cannot weigh
    up. `min_quality` narrows it back down; ALL means every blend that ran
    and was scored (a quality of 0.0 included).
  * **nothing twice.** It always runs with --resume: a blend already tested
    cleanly on this split, as the chromosome its script builds now, keeps the
    row it has, and only the rest run -- so asking again after gep_lora/core/pipeline/main.py's own
    pass costs only the blends that pass left out.
"""

from gep_lora.service import results
from gep_lora.core.storage import store
from gep_lora.core.testing import test_run_with_dataset

SCRIPT = test_run_with_dataset.__name__

# A minimum no scored individual can fail to beat: qualities are 0..1, and the
# pass takes those strictly above it. Individuals with no quality at all (BAD,
# never ran) are not above anything, since they have nothing to test.
ALL = -1.0


class TestPassError(ValueError):
    """What is wrong with a testing request, in words for whoever sent it."""


def choices(db_path, run_id):
    """What a testing form needs, out of the job's own sweep.

    How many testing questions it holds and how many a pass asks by default,
    how many blends could be tested (and how many of those already have been),
    and the testing results so far.
    """
    conn = results.connect(db_path)
    try:
        conf = store.get_settings(conn, run_id)
        held = [entry for entry in store.dataset_summary(conn, run_id)
                if entry["split"] == "testing"]
        candidates = test_run_with_dataset.candidates(conn, run_id, ALL)
        tested = {row["number"] for row in store.test_results(conn, run_id)
                  if row["verdict"] == "ok"}
        return {
            "records": held[0]["records"] if held else 0,
            "count": conf.get("TESTING_COUNT"),
            "blends": len(candidates),
            "tested": len({row["number"] for row in candidates} & tested),
            "defaults": {"min_quality": None, "count": conf.get("TESTING_COUNT")},
            "results": [dict(zip(row.keys(), row)) for row in store.test_quality(conn, run_id)],
        }
    finally:
        conn.close()


def _whole(body, name, low):
    value = body.get(name)
    if value in (None, ""):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or value != int(value) or value < low:
        raise TestPassError("%s must be a whole number of at least %d, or left out"
                            % (name, low))
    return int(value)


def options_for(body, offered):
    """A request, checked against what this sweep offers. -> options.

    {"min_quality": q | None, "count": n | None, "limit": n | None} -- None
    min_quality is every blend (ALL); None count is the sweep's own
    TESTING_COUNT; None limit is no cap on how many blends.
    """
    body = body or {}
    if not isinstance(body, dict):
        raise TestPassError("the body must be an object")
    unknown = sorted(set(body) - {"min_quality", "count", "limit"})
    if unknown:
        raise TestPassError("unknown field(s): %s; a testing pass takes min_quality, "
                            "count and limit" % ", ".join(unknown))
    if not offered["records"]:
        raise TestPassError("this job holds no testing split, so nothing says which "
                            "questions its search never saw")
    minimum = body.get("min_quality")
    if minimum is not None:
        if isinstance(minimum, bool) or not isinstance(minimum, (int, float)) \
                or not 0.0 <= minimum < 1.0:
            raise TestPassError("min_quality must be a number from 0 up to (not "
                                "including) 1, or left out for every blend")
        minimum = float(minimum)
    if not offered["blends"]:
        raise TestPassError("no blend of this job ran and was scored, so there is "
                            "nothing to test")
    return {"min_quality": minimum, "count": _whole(body, "count", 1),
            "limit": _whole(body, "limit", 1)}


def command(python, db_path, run_id, options, timeout=None):
    """The test_run_with_dataset.py command line for one testing pass.

    Run as a module from the repo root, like every CLI below the top level,
    with --from-db: the questions are the job's own stored testing split,
    written beside its database the way gep_lora/core/pipeline/main.py's pass writes them.
    """
    minimum = options.get("min_quality")
    argv = [python, "-u", "-m", SCRIPT, "--from-db",
            "--db", db_path, "--run", str(run_id),
            # One argument, so a negative value is never read as a flag.
            "--min-quality=%r" % (ALL if minimum is None else float(minimum)),
            "--resume"]
    if options.get("count"):
        argv += ["--count", str(options["count"])]
    if options.get("limit"):
        argv += ["--limit", str(options["limit"])]
    if timeout:
        argv += ["--timeout", str(timeout)]
    return argv
