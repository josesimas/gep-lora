"""
server_pool.py - Start some lora_server.py processes, hand them out, stop them.

The client side of lora_server.py, and the only part of the pipeline that knows
those processes exist. The process step calls pool_for() before it launches a
batch and stop() when it is done; everything in between is one method:
envs_for(n) hands back the environment each script in a batch is launched with,
which is how a generated client finds its server.

Why a pool rather than one server
---------------------------------
PROCESS_RUN_BATCH_SIZE scripts already run at once, and a server serves one
request at a time, so N scripts want N servers. The number is a setting rather
than a guess because the cost is the same one a batch has always had: every
server holds its own copy of the base model.

Whether it pays is an open question and worth measuring before believing.
Concurrency today buys 2.4-3.0x on four scripts, and the parts that overlap
well are the CPU-bound import and load -- which is exactly what a warm server
removes. What is left is GPU-bound on one card, so N servers may time-share
rather than multiply. LORA_SERVER_COUNT = 1 is the experiment, and if one warm
server is within a few percent of four, this module's whole reason to exist
goes with it.

Assignment is positional, not a queue
-------------------------------------
A batch is capped at the pool's size, and the k-th script of a batch gets the
k-th server. So there is no acquire/release, no lock and no waiting: the same
"fixed ceiling rather than a refilling queue" the batch itself already is. It
also means a batch's servers are the same ones every batch, which is what makes
recycling countable.

Recycling
---------
A server that has built three hundred blends is not the server that built the
first one, whatever care reset() takes. LORA_SERVER_RECYCLE_AFTER is how many
builds one is allowed before the pool restarts it between batches: the model
load is paid again, once, every N individuals, which bounds the drift while
still amortising it N-fold. 0 never recycles, which is the fastest and the
least defensible.

How long a pool lives
---------------------
As long as the driver that started it, not as long as the step. The pool hangs
off start_run.Context, which continue_run.py builds once and reuses for every
generation it turns, so generation 2 finds the servers of generation 1 still up
and loads nothing. Bringing them down at the end of each process step would
mean paying the startup again next generation to reload the same model that had
just been released.

The one thing that overrides that is start_run.wants_the_card(): with
JUDGE_BACKEND = 'unsloth' the evaluate step loads a judge in this same
interpreter, and a pool left standing through it would be N base models on the
card while a judge tries to find room for one more. So on that backend, and
only on it, the pool does come down per step -- which is the rule the local
judge itself already lives under, applied to whoever else is holding VRAM.

A whole search still pays it twice, because main.py runs its first generation
through start_run.py and the rest through continue_run.py, and those are two
Contexts. Two rather than one-per-generation is where this stops; closing the
last gap would mean a pool outliving the driver that made it, which is a worse
thing to own than a second model load. `python -m metrics.report --step process`
gives the startup its own row, "start servers", so it is measured rather than
argued about.
"""

import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

# The repo folder, one above this one: the servers are started as
# `python -m blends.lora_server` from there, so they import this package the
# way every other module does.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The environment variable a generated client reads its server's URL out of.
# It is an environment variable rather than a marker in the script because
# which server an individual ran against is an accident of the batch it landed
# in -- baking it into script_source would make a stored script describe one
# particular run of it. Same name in templates/template_remote_code.py.
SERVER_ENV = "GEP_LORA_SERVER"

# And how long that client waits on one request, so a build that hangs is one
# individual's timeout rather than the batch's.
TIMEOUT_ENV = "GEP_LORA_SERVER_TIMEOUT"

# How often the pool asks a starting server whether it is ready yet.
_POLL = 1.0


class Worker:
    """One lora_server.py process: where it listens, and what it has built."""

    __slots__ = ("index", "port", "host", "process", "log", "log_path",
                 "builds", "total")

    def __init__(self, index, host, port, log_path):
        self.index = index
        self.host = host
        self.port = port
        self.log_path = log_path
        self.process = None
        self.log = None
        # Since this process started, and since the pool started: recycling
        # resets the first (it is what the recycle count is counted against)
        # and never the second, which is what the step reports at the end.
        self.builds = 0
        self.total = 0

    @property
    def url(self):
        return "http://%s:%d" % (self.host, self.port)

    @property
    def alive(self):
        return self.process is not None and self.process.poll() is None


class Pool:
    """`count` servers on consecutive ports, all holding the same base model."""

    def __init__(self, base_model, count, host="127.0.0.1", port=8770,
                 run_dir=".", startup_timeout=900, request_timeout=1800,
                 recycle_after=0, say=print):
        self.base_model = base_model
        self.count = max(1, int(count))
        self.host = host
        self.base_port = int(port)
        self.run_dir = run_dir
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self.recycle_after = max(0, int(recycle_after or 0))
        self.say = say
        self.workers = []

    # --- lifecycle ---------------------------------------------------------

    def start(self):
        """Launch every server and wait for all of them. -> seconds.

        Started together rather than one after another: they are independent,
        and the load is what takes the time. It is also the pool's whole VRAM
        peak arriving at once, which is the same peak a batch of the same size
        has always had.
        """
        started = time.time()
        self.workers = [Worker(index, self.host, self.base_port + index,
                               os.path.join(self.run_dir,
                                            "lora_server_%d.log"
                                            % (self.base_port + index)))
                        for index in range(self.count)]
        for worker in self.workers:
            self._spawn(worker)
        self.say("starting %d lora server(s) on %s, ports %d-%d -- one base "
                 "model load each, and they stay up across generations"
                 % (self.count, self.host, self.base_port,
                    self.base_port + self.count - 1))
        # Said when they start, not only when they fail. A server's own account
        # of a build -- the PEFT warning, the traceback, the OOM -- is in its
        # log and nowhere else: the client only ever sees the one-line message
        # the request came back with, and that is all the execution's stderr
        # can hold. Nobody should have to ask where it went.
        self.say("        their logs: %s"
                 % os.path.join(self.run_dir, "lora_server_<port>.log"))

        failures = []
        threads = [threading.Thread(target=self._await_one, args=(worker, failures))
                   for worker in self.workers]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if failures:
            self.stop()
            raise SystemExit(
                "lora server(s) did not come up: %s\nSee %s"
                % ("; ".join(failures), os.path.join(self.run_dir, "lora_server_*.log")))

        seconds = time.time() - started
        self.say("%d server(s) ready in %.1fs" % (self.count, seconds))
        return seconds

    def _spawn(self, worker):
        """Launch one server process, its output going to its own log file.

        sys.executable, for the reason the generated scripts use it: the server
        imports unsloth, so it needs the same venv the process step already
        demands. Its stdout is a file rather than a pipe because nothing reads
        it as it runs -- it is a log to look at when something went wrong, and
        an unread pipe would eventually block the process that filled it.
        """
        # Appended to, not truncated: a recycled server is a new process on the
        # same port, and overwriting would throw away the log of the one whose
        # last build is the reason you are reading it. The header is what tells
        # one life from the next.
        worker.log = open(worker.log_path, "a", encoding="utf-8", errors="replace")
        worker.log.write("\n=== lora server on port %d, started %s ===\n"
                         % (worker.port,
                            time.strftime("%Y-%m-%dT%H:%M:%S")))
        worker.log.flush()
        worker.builds = 0
        worker.process = subprocess.Popen(
            [sys.executable, "-u", "-m", "blends.lora_server",
             "--base-model", self.base_model,
             "--host", self.host, "--port", str(worker.port)],
            cwd=_ROOT, stdout=worker.log, stderr=subprocess.STDOUT,
        )

    def _await_one(self, worker, failures):
        """Poll one server's /health until it says ready, or give up saying why."""
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if not worker.alive:
                failures.append("port %d exited with %s (see %s)"
                                % (worker.port, worker.process.returncode,
                                   os.path.basename(worker.log_path)))
                return
            health = self.health(worker)
            if health and health.get("ready"):
                if health.get("base_model") != self.base_model:
                    failures.append("port %d holds %s, not %s"
                                    % (worker.port, health.get("base_model"),
                                       self.base_model))
                return
            time.sleep(_POLL)
        failures.append("port %d was still not ready after %ss"
                        % (worker.port, self.startup_timeout))

    def health(self, worker):
        """What one server says about itself, or None while it cannot be reached."""
        try:
            with urllib.request.urlopen(worker.url + "/health", timeout=5) as reply:
                return json.loads(reply.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError):
            return None

    def stop(self):
        """Ask every server to stop, then make sure it did.

        Asked first, because a server that shuts itself down releases the card
        the way it meant to; killed after, because a hung one still has to let
        go before the next step wants the VRAM.
        """
        for worker in self.workers:
            if not worker.alive:
                continue
            try:
                request = urllib.request.Request(worker.url + "/shutdown",
                                                 data=b"", method="POST")
                urllib.request.urlopen(request, timeout=5).read()
            except (urllib.error.URLError, OSError):
                pass
        for worker in self.workers:
            if worker.process is None:
                continue
            try:
                worker.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                worker.process.kill()
                worker.process.wait()
            if worker.log is not None:
                worker.log.close()
                worker.log = None
        built = sum(worker.total for worker in self.workers)
        self.say("stopped %d lora server(s) after %d build(s)"
                 % (len(self.workers), built))
        self.workers = []

    # --- handing them out --------------------------------------------------

    @property
    def size(self):
        return len(self.workers)

    def envs_for(self, count):
        """The environment for each script of a batch of `count`. -> list of dicts.

        Positional: the k-th script of the batch talks to the k-th server. The
        caller caps its batch at self.size, so every script in one gets a server
        to itself; asking for more than there are is a caller bug rather than a
        queue, and says so.
        """
        if count > self.size:
            raise ValueError("a batch of %d wants more servers than the pool's %d"
                             % (count, self.size))
        envs = []
        for worker in self.workers[:count]:
            worker.builds += 1              # one script, one build
            worker.total += 1
            envs.append({SERVER_ENV: worker.url,
                         TIMEOUT_ENV: str(self.request_timeout)})
        return envs

    def maintain(self):
        """Between batches: restart anything dead, and anything worn out.

        Dead first, because a server that fell over would otherwise fail every
        remaining individual assigned to its slot; worn out second, which is
        LORA_SERVER_RECYCLE_AFTER doing its job. Restarting is the same spawn
        and the same wait as at startup, so a recycled server is a fresh one in
        every sense that matters.
        """
        due = [worker for worker in self.workers
               if not worker.alive
               or (self.recycle_after and worker.builds >= self.recycle_after)]
        if not due:
            return 0
        for worker in due:
            why = "died" if not worker.alive else "%d builds" % worker.builds
            self.say("        recycling the server on port %d (%s)"
                     % (worker.port, why))
            if worker.process is not None and worker.alive:
                try:
                    request = urllib.request.Request(worker.url + "/shutdown",
                                                     data=b"", method="POST")
                    urllib.request.urlopen(request, timeout=5).read()
                    worker.process.wait(timeout=30)
                except (urllib.error.URLError, OSError, subprocess.TimeoutExpired):
                    worker.process.kill()
                    worker.process.wait()
            if worker.log is not None:
                worker.log.close()
                worker.log = None
            self._spawn(worker)

        failures = []
        threads = [threading.Thread(target=self._await_one, args=(worker, failures))
                   for worker in due]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if failures:
            self.stop()
            raise SystemExit("a recycled lora server did not come back: %s"
                             % "; ".join(failures))
        return len(due)


def wanted(source):
    """Does this generated script want a server? -- the same shape as
    process_run.imports_unsloth().

    A test on the source itself, so it can be asked of a script held in the
    database as easily as of one on disk, and so the pipeline adapts to which
    template a sweep was generated from rather than to what settings.py says
    today. A sweep generated from template_code.py answers no however many
    servers the settings ask for, which is what makes the switch one line.
    """
    return SERVER_ENV in (source or "")


def pool_for(conf, source, base_model, run_dir, config, say=print):
    """A started Pool if these scripts want one, else None.

    `conf` is the sweep's stored settings and `config` the settings module, read
    in that order for the reason every knob is: a sweep created before these
    existed falls back to the file, and one created with them keeps its own.
    """
    if not wanted(source):
        return None

    def setting(name):
        value = conf.get(name, getattr(config, name, None))
        return getattr(config, name, None) if value is None else value

    count = setting("LORA_SERVER_COUNT")
    try:
        count = int(count)
    except (TypeError, ValueError):
        count = 1
    if count < 1:
        raise SystemExit(
            "these scripts are lora_server clients (TEMPLATE = "
            "'template_remote_code.py') but LORA_SERVER_COUNT is %r, so there "
            "would be nothing for them to talk to." % count)

    pool = Pool(base_model, count,
                host=setting("LORA_SERVER_HOST"),
                port=setting("LORA_SERVER_PORT"),
                run_dir=run_dir,
                startup_timeout=setting("LORA_SERVER_STARTUP_TIMEOUT"),
                request_timeout=setting("LORA_SERVER_TIMEOUT"),
                recycle_after=setting("LORA_SERVER_RECYCLE_AFTER"),
                say=say)
    pool.start()
    return pool
