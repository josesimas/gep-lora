"""
worker.py - The background half: queued jobs, one at a time, through main.py.

    python -m async_api.worker              # runs until stopped
    python -m async_api.worker --once       # one job, if there is one, then exit

The worker takes the oldest queued job (registry ids are arrival order), and
runs its database through main.py exactly as a person would:

    python main.py --db <job folder>/job.sqlite3 --run <its sweep>

main.py adopts the prepared sweep, reads its questions out of its rows, and
keeps everything it writes in `job_run<N>/` beside the database -- so a job
never touches run_db/ or another job's folder.

main.py is run as a **subprocess** here, unlike main.py's own drivers, which
call each other as libraries. Two things need it: a cancel has to be able to
stop a search (and the lora servers under it) without stopping the worker, and
each job's console output belongs in its own `job.log`. The interpreter
concern main.py raises is met by using sys.executable -- start the worker with
the venv's python, as you would main.py.

It also runs **verifications** -- one of a finished job's blends beside the
LoRAs it is made of (see verify.py) -- through
`python -m testing.evaluate_chromosome_against_loras`, in the same way and for
the same reasons. Jobs come first: a verification asks a question about a
search that has already finished, and a job is one that has not started. One
card, one queue, whichever kind of work is in it.

One worker per JOBS_DIR. A worker that starts finds any job still marked
running with nothing behind it (the last worker died) and marks it failed:
main.py refuses to adopt a sweep that already holds individuals, so it cannot
simply be queued again. A verification left running is marked failed too, and
that one can simply be asked for again.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time

from async_api import results
from async_api import settings
from async_api import verify
from async_api.registry import CANCELLED, DONE, FAILED, Registry

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = os.path.join(_ROOT, "main.py")


def say(message):
    print("[worker %s] %s" % (time.strftime("%H:%M:%S"), message), flush=True)


def command(registry, job, python=None, main=MAIN):
    """The main.py command line for one job."""
    options = json.loads(job["options"] or "{}")
    argv = [python or sys.executable, "-u", main,
            "--db", registry.database(job), "--run", str(job["run_id"])]
    if options.get("no_test"):
        argv.append("--no-test")
    if options.get("limit"):
        argv += ["--limit", str(options["limit"])]
    if options.get("timeout"):
        argv += ["--timeout", str(options["timeout"])]
    if options.get("test_min_quality") is not None:
        argv += ["--test-min-quality", str(options["test_min_quality"])]
    return argv


def kill_tree(process):
    """Stop main.py and everything it started -- the scripts, the lora servers."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       capture_output=True)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except OSError:
            pass
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


class Stopped(BaseException):
    """The worker itself is stopping while a child runs. Re-raised after tidying."""


def supervise(argv, log_path, what, on_start, cancelled=None, poll=None):
    """Run one child to its end, logging it. -> (exit code, cancelled or reason).

    The supervision a job and a verification share: the child in its own
    process group, so a cancel stops it and everything it started without
    stopping the worker; its console in its own log file; and its pid handed
    back through `on_start`, so whoever queued it can be told. `cancelled()` is
    asked between polls -- a verification passes none, having nothing to cancel.

    -> (code, True/False) once the child is done, or (None, reason) when it
    could never be started. Raises Stopped when the worker is interrupted,
    having killed the child: how that ends is each kind of work's own to record.
    """
    poll = settings.WORKER_POLL_SECONDS if poll is None else poll
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    extra = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
             else {"start_new_session": True})
    folder = os.path.dirname(log_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("=== %s started %s\n$ %s\n" % (what, time.ctime(), " ".join(argv)))
        log.flush()
        try:
            process = subprocess.Popen(argv, cwd=_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       env=env, **extra)
        except OSError as error:
            return None, str(error)
        on_start(process.pid)

        stopped = False
        try:
            while process.poll() is None:
                if cancelled is not None and cancelled():
                    say("%s: cancel requested, stopping it" % what)
                    kill_tree(process)
                    stopped = True
                    break
                time.sleep(poll)
        except BaseException:
            # Ctrl+C on the worker. The child is in its own process group and
            # would otherwise carry on with nobody to record how it ended.
            kill_tree(process)
            log.write("=== %s stopped with the worker at %s\n" % (what, time.ctime()))
            raise Stopped(process.returncode)
        code = process.wait()
        log.write("=== %s %s with exit code %s at %s\n"
                  % (what, "cancelled" if stopped else "finished", code, time.ctime()))
    return code, stopped


def run_job(registry, job, argv=None, poll=None):
    """Run one claimed job to its end. -> the status it finished in."""
    argv = argv or command(registry, job)
    what = "job %d" % job["id"]
    say("%s: %s" % (what, " ".join(argv[2:])))
    try:
        code, cancelled = supervise(
            argv, os.path.join(registry.folder(job), "job.log"), what,
            lambda pid: registry.set_pid(job["id"], pid),
            lambda: registry.cancel_requested(job["id"]), poll)
    except Stopped as stopped:
        registry.finish(job["id"], FAILED, exit_code=stopped.args[0],
                        error="the worker stopped while this job was running")
        raise
    if code is None:
        registry.finish(job["id"], FAILED,
                        error="could not start main.py: %s" % cancelled)
        return FAILED

    if cancelled:
        registry.finish(job["id"], CANCELLED, exit_code=code, error="cancelled")
        return CANCELLED
    if code == 0:
        registry.finish(job["id"], DONE, exit_code=0)
        return DONE
    if search_finished(registry, job):
        # main.py's exit code is the testing pass's when the search succeeded
        # and it did not -- nothing above TESTING_MIN_QUALITY, say. The search's
        # results stand, and can go live; the note says what did not happen.
        registry.finish(job["id"], DONE, exit_code=code,
                        error="the search finished; the testing pass did not "
                              "(exit %s, see the job log)" % code)
        return DONE
    registry.finish(job["id"], FAILED, exit_code=code,
                    error="main.py exited with %s; see the job log" % code)
    return FAILED


def verify_command(registry, row, python=None):
    """The comparison's command line for one verification, from its options."""
    job = registry.job(row["job_id"])
    return verify.command(python or sys.executable, registry.database(job),
                          job["run_id"], row["number"],
                          json.loads(row["options"] or "{}"),
                          registry.verification_folder(job, row["id"]))


def run_verification(registry, row, argv=None, poll=None):
    """Run one claimed verification to its end. -> the status it finished in.

    Its own log beside its own answers, and its own status: a verification that
    fails says so without touching the job it read, whose search has finished
    and whose results still stand. Exit 0 is not enough to call it done -- the
    scores file has to be there, since that is the whole product.
    """
    job = registry.job(row["job_id"])
    folder = registry.verification_folder(job, row["id"])
    argv = argv or verify_command(registry, row)
    what = "verification %d (job %d, individual %d)" % (row["id"], job["id"], row["number"])
    say("%s: %s" % (what, " ".join(argv[3:])))
    try:
        code, why = supervise(argv, os.path.join(folder, "verify.log"), what,
                              lambda pid: registry.set_verification_pid(row["id"], pid),
                              None, poll)
    except Stopped as stopped:
        registry.finish_verification(row["id"], FAILED, exit_code=stopped.args[0],
                                     error="the worker stopped while this "
                                           "verification was running")
        raise
    if code is None:
        registry.finish_verification(row["id"], FAILED,
                                     error="could not start the comparison: %s" % why)
        return FAILED
    if code == 0 and verify.report(folder) is not None:
        registry.finish_verification(row["id"], DONE, exit_code=0)
        return DONE
    registry.finish_verification(
        row["id"], FAILED, exit_code=code,
        error=("the comparison exited with %s; see its log" % code if code
               else "the comparison ran but wrote no scores; see its log"))
    return FAILED


def recover_verifications(registry):
    """Mark the verifications a dead worker left running as failed. -> how many."""
    stale = registry.orphaned_verifications()
    for row in stale:
        registry.finish_verification(row["id"], FAILED,
                                     error="the worker stopped while this "
                                           "verification was running")
        say("verification %d was left running by a previous worker; marked failed"
            % row["id"])
    return len(stale)


def search_finished(registry, job):
    """Did every generation of the job's search get scored, and the sweep end well?"""
    done = results.progress(registry.database(job), job["run_id"])
    if not done or done["generations_scored"] < done["generations_expected"]:
        return False
    return results.run_status(registry.database(job), job["run_id"]) == "done"


def recover(registry):
    """Mark the jobs a dead worker left running as failed. -> how many."""
    stale = registry.orphaned()
    for job in stale:
        registry.finish(job["id"], FAILED,
                        error="the worker stopped while this job was running")
        say("job %d was left running by a previous worker; marked failed" % job["id"])
    return len(stale)


def serve(registry, once=False, poll=None):
    poll = settings.WORKER_POLL_SECONDS if poll is None else poll
    recover(registry)
    recover_verifications(registry)
    say("watching %s" % registry.path)
    while True:
        job = registry.claim_next()
        if job is not None:
            status = run_job(registry, job, poll=poll)
            say("job %d: %s" % (job["id"], status))
            if once:
                return 0
            continue
        # Jobs first, always: a verification asks about a search that has
        # already finished, and a queued job is one that has not started.
        row = registry.claim_next_verification()
        if row is not None:
            status = run_verification(registry, row, poll=poll)
            say("verification %d: %s" % (row["id"], status))
            if once:
                return 0
            continue
        if once:
            return 0
        time.sleep(poll)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run queued API jobs through main.py.")
    parser.add_argument("--jobs-dir", default=None,
                        help="the registry folder (default %s)" % settings.JOBS_DIR)
    parser.add_argument("--once", action="store_true",
                        help="run at most one job, then exit")
    args = parser.parse_args(argv)
    try:
        return serve(Registry(args.jobs_dir), once=args.once)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
