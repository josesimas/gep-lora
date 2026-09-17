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

One worker per JOBS_DIR. A worker that starts finds any job still marked
running with nothing behind it (the last worker died) and marks it failed:
main.py refuses to adopt a sweep that already holds individuals, so it cannot
simply be queued again.
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


def run_job(registry, job, argv=None, poll=None):
    """Run one claimed job to its end. -> the status it finished in."""
    poll = settings.WORKER_POLL_SECONDS if poll is None else poll
    argv = argv or command(registry, job)
    folder = registry.folder(job)
    log_path = os.path.join(folder, "job.log")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    extra = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
             else {"start_new_session": True})

    say("job %d: %s" % (job["id"], " ".join(argv[2:])))
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("=== job %d started %s\n$ %s\n" % (job["id"], time.ctime(), " ".join(argv)))
        log.flush()
        try:
            process = subprocess.Popen(argv, cwd=_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       env=env, **extra)
        except OSError as error:
            registry.finish(job["id"], FAILED, error="could not start main.py: %s" % error)
            return FAILED
        registry.set_pid(job["id"], process.pid)

        cancelled = False
        try:
            while process.poll() is None:
                if registry.cancel_requested(job["id"]):
                    say("job %d: cancel requested, stopping main.py" % job["id"])
                    kill_tree(process)
                    cancelled = True
                    break
                time.sleep(poll)
        except BaseException:
            # The worker itself is stopping (Ctrl+C). main.py is in its own
            # process group and would otherwise carry on with nobody to record
            # how it ended.
            kill_tree(process)
            log.write("=== job %d stopped with the worker at %s\n" % (job["id"], time.ctime()))
            registry.finish(job["id"], FAILED, exit_code=process.returncode,
                            error="the worker stopped while this job was running")
            raise
        code = process.wait()
        log.write("=== job %d %s with exit code %s at %s\n"
                  % (job["id"], "cancelled" if cancelled else "finished", code, time.ctime()))

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
    say("watching %s" % registry.path)
    while True:
        job = registry.claim_next()
        if job is None:
            if once:
                return 0
            time.sleep(poll)
            continue
        status = run_job(registry, job, poll=poll)
        say("job %d: %s" % (job["id"], status))
        if once:
            return 0


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
