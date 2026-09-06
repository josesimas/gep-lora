"""record.py - Measure a step while it runs, and write down what it cost.

One Meter per step, made by start_run.run() and handed to the step on its
Context. A step that says nothing still gets a row -- wall time is measured
around it either way -- so instrumenting a step is only ever adding detail to a
row that already exists, and a step added later is timed without being touched.

What a step says:

    context.count(len(selected), "individuals", skipped=len(unchanged))
    context.phase("model_load", 62.1, execution_id=41)
    with context.timer("prepare"):
        prepared = evaluator.prepare(...)

`count` is what makes the row comparable with another step's: seconds alone say
which step is slow, seconds per item say whether the fix is a cheaper item or
fewer of them. `phase` is what makes the row actionable: a step total cannot
tell a fixed cost paid once from a per-item cost paid a hundred times, and
those two want opposite fixes.

Phases accumulate by (name, execution_id), so a step may report the same phase
as many times as it happens and the row that lands holds calls, total and worst
case. Timing is perf_counter throughout -- a monotonic clock, since these are
durations and the wall clock is free to move under a step that takes an hour.
"""

import time

from storage import store


class Meter:
    """The measurements of one step of one pass, until commit() writes them."""

    __slots__ = ("conn", "run_id", "pass_no", "position", "step", "generation",
                 "watermark", "started_at", "items", "unit", "skipped", "note",
                 "_phases")

    def __init__(self, conn, run_id, pass_no, position, step,
                 generation=None, watermark=None):
        self.conn = conn
        self.run_id = run_id
        self.pass_no = pass_no
        self.position = position
        self.step = step
        self.generation = generation
        self.watermark = watermark
        self.started_at = None
        self.items = 0
        self.unit = None
        self.skipped = 0
        self.note = None
        self._phases = {}

    # --- what a step says while it runs ------------------------------------

    def count(self, items, unit=None, skipped=0, note=None):
        """How much work the step actually did, and how much it declined.

        Called rather than inferred: only the step knows whether its unit is an
        individual, an answer or a chromosome, and only it knows how many of
        them it skipped -- individuals that were BAD, unchanged since their last
        run, or already scored cost nothing, and counting them would make the
        per-item cost of every step that skips look better than it is.
        """
        self.items = items
        self.unit = unit or self.unit
        self.skipped = skipped
        if note:
            self.note = note

    def phase(self, name, seconds, calls=1, longest=None, execution_id=None,
              detail=None):
        """Add `seconds` to a named phase of this step.

        Repeats accumulate: calls add up, seconds add up, and `longest` keeps
        the worst single occurrence -- the three numbers that say whether a
        phase is one long thing or many short ones.
        """
        key = (name, execution_id)
        entry = self._phases.get(key)
        if entry is None:
            entry = {"phase": name, "execution_id": execution_id, "calls": 0,
                     "seconds": 0.0, "longest": 0.0, "detail": detail}
            self._phases[key] = entry
        entry["calls"] += calls
        entry["seconds"] += seconds
        entry["longest"] = max(entry["longest"], longest if longest is not None
                               else seconds)
        if detail and not entry["detail"]:
            entry["detail"] = detail
        return entry

    def timer(self, name, execution_id=None, detail=None):
        """`with meter.timer("prepare"):` -- the phase above, timed for you."""
        return _Timer(self, name, execution_id, detail)

    def phases_from(self, records, execution_id=None):
        """Add phases somebody else measured -- process_run.timings(), say,
        reading the TIMING: lines a generated script printed about itself."""
        for entry in records:
            self.phase(entry["phase"], entry["seconds"],
                       calls=entry.get("calls", 1), longest=entry.get("longest"),
                       execution_id=execution_id, detail=entry.get("detail"))

    # --- the row ------------------------------------------------------------

    def commit(self, seconds, status="ok"):
        """Write the step row and its phases. Returns the step row's id.

        Its own transaction, committed here, because a step that failed is
        exactly when the numbers are most worth keeping and the driver is about
        to stop.
        """
        step_id = store.add_step_timing(
            self.conn, self.run_id, self.pass_no, self.position, self.step,
            seconds, generation=self.generation, watermark=self.watermark,
            started_at=self.started_at, items=self.items, unit=self.unit,
            skipped=self.skipped, status=status, note=self.note)
        if self._phases:
            store.add_phase_timings(self.conn, step_id, self._phases.values())
        self.conn.commit()
        return step_id


class _Timer:
    """The context manager Meter.timer() hands back. Times what it wraps and
    records it even when that raised -- an exception is not a reason to forget
    how long the thing took before it threw."""

    __slots__ = ("meter", "name", "execution_id", "detail", "started")

    def __init__(self, meter, name, execution_id, detail):
        self.meter = meter
        self.name = name
        self.execution_id = execution_id
        self.detail = detail
        self.started = None

    def __enter__(self):
        self.started = time.perf_counter()
        return self

    def __exit__(self, kind, error, traceback):
        self.meter.phase(self.name, time.perf_counter() - self.started,
                         execution_id=self.execution_id, detail=self.detail)
        return False


class _Blank(Meter):
    """A Meter that measures nothing and writes nothing.

    What a Context built outside start_run.run() gets -- a step called straight
    from a test or a shell is still a step, and it should not have to know
    whether anybody is keeping time. Everything a step says is accepted and
    dropped, so instrumentation never needs an `if meter:` around it.
    """

    def __init__(self):
        super().__init__(None, None, 0, 0, "")

    def commit(self, seconds, status="ok"):
        return None


def blank():
    return _Blank()
