"""metrics - what each step of the pipeline cost, recorded the way results are.

`record.py` is the write side: the Meter a step is handed while it runs, and
the one place a step_timings row is made. `report.py` is the read side: the
ranked breakdown, printed.

    from metrics import record

    meter = record.Meter(conn, run_id, pass_no, 4, "process")
    meter.count(10, "individuals", skipped=3)
    meter.phase("model_load", 62.1, execution_id=7)
    meter.commit(408.9)

Nothing here decides what a phase is; the step that measured it does. See the
step_timings and phase_timings comments in storage/store.py for why the two
levels are the two levels.
"""

from metrics.record import Meter, blank

__all__ = ["Meter", "blank"]
