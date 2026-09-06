"""report.py - What a sweep spent its time on, ranked.

    python -m metrics.report                 # the latest sweep
    python -m metrics.report --run 3         # that one
    python -m metrics.report --step process  # one step's phases, in full
    python -m metrics.report --csv           # the same rows, for a spreadsheet

Three questions, in the order they are asked:

    which step   -- the step table, ranked by total seconds, with what each
                    step did (items) and what one of those cost
    where inside -- the phase table for the steps worth looking at, with calls
                    beside seconds, because a cost paid once and a cost paid a
                    hundred times want opposite fixes
    is it moving -- the pass table, when a sweep has run more than one, since a
                    cost that grows with the population is a different problem
                    from one that does not

Nothing here decides anything: it prints what start_run.py recorded, and says
which of the two shapes each phase has -- once per pass, or once per item. What
to do about it is a judgement about the search, and belongs to whoever reads it.
"""

import argparse
import os
import sys

from config import settings as config
from storage import store

# Phases that cover other phases, and would be double counting if they were
# ranked beside them. "script" is the process step's own wall time per
# individual and "total" is the same span measured from inside the script;
# "compact" runs inside the combine.svd that called it. They are printed, but
# under their own heading and out of the shares.
WHOLE = ("script", "total")
NESTED = ("compact",)


def _seconds(value):
    """A duration, in the widest unit that still says something."""
    value = value or 0.0
    if value < 1:
        return "%.0fms" % (value * 1000)
    if value < 90:
        return "%.1fs" % value
    if value < 5400:
        return "%.1fm" % (value / 60)
    return "%.1fh" % (value / 3600)


def _share(part, whole):
    return "%5.1f%%" % (100.0 * part / whole) if whole else "    -"


def _shape(calls, over, noun):
    """What kind of cost a phase is: the one thing seconds alone cannot say,
    and the thing that decides which fix would work.

    A phase called once whatever else happens is a fixed price -- only paying
    it less often makes it cheaper. A phase called once per script, or per
    pass, scales with the sweep, so a cheaper one and fewer of them both work.
    More calls than there are scripts is per something smaller still: a prompt,
    a node, a leaf, and the ratio says which.
    """
    if not over or not calls:
        return "once"
    if calls == over:
        return "per %s" % noun
    if calls < over:
        return "%d of %d %ss" % (calls, over, noun)
    return "%.1f per %s" % (float(calls) / over, noun)


def steps(conn, run_id):
    """The step table: which step to look at first.

    Ranked by what each cost in total, with what it did beside it -- a step is
    only expensive relative to how much work it was given, and seconds per item
    is the number that survives a change in population size.
    """
    rows = store.step_costs(conn, run_id)
    if not rows:
        return 0.0
    total = sum(row["seconds"] or 0.0 for row in rows)
    print("where the time went -- %s over %d step(s)"
          % (_seconds(total), len(rows)))
    print("    %-11s %9s %7s %7s %-24s %9s %10s"
          % ("step", "seconds", "share", "passes", "did", "per item", "worst pass"))
    for row in rows:
        did = ("%d %s" % (row["items"] or 0, row["unit"] or "items")
               + (" (+%d skipped)" % row["skipped"] if row["skipped"] else ""))
        per = (_seconds((row["seconds"] or 0.0) / row["items"])
               if row["items"] else "-")
        print("    %-11s %9s %7s %7d %-24s %9s %10s%s"
              % (row["step"], _seconds(row["seconds"]), _share(row["seconds"], total),
                 row["passes"], did, per, _seconds(row["longest"]),
                 "  %d failed" % row["failures"] if row["failures"] else ""))
    return total


def _table(rows, measured, over, noun):
    print("    %-18s %7s %9s %7s %9s %9s  %s"
          % ("phase", "calls", "seconds", "share", "mean", "worst", "shape"))
    for row in rows:
        calls = row["calls"] or 0
        print("    %-18s %7d %9s %7s %9s %9s  %s"
              % (row["phase"], calls, _seconds(row["seconds"]),
                 _share(row["seconds"], measured),
                 _seconds((row["seconds"] or 0.0) / calls) if calls else "-",
                 _seconds(row["longest"]), _shape(calls, over, noun)))


def phases(conn, run_id, step, budget, passes_run):
    """Where inside one step the time went, in the two tiers it really has.

    The step's own phases are one tier -- work the step does itself, once per
    pass -- and they are shares of the step's wall time. The phases a generated
    script reports about itself are another, nested inside the "script" row and
    shares of *that*: process launches PROCESS_RUN_BATCH_SIZE scripts at once,
    so script seconds and wall seconds are different quantities, and ranking a
    model load against a materialise would be comparing a total against a sum
    of overlapping totals.

    Two denominators, said out loud, rather than one that is wrong for half the
    rows -- and both residuals named, since on this pipeline the gap between
    what a script cost from outside and what it says it cost from inside is a
    real thing (a Python interpreter, and unsloth landing in it) rather than a
    rounding error.
    """
    rows = store.phase_costs(conn, run_id, step)
    if not rows:
        return
    own = [row for row in rows
           if not row["executions"] and row["phase"] not in WHOLE + NESTED]
    inner = [row for row in rows
             if row["executions"] and row["phase"] not in WHOLE + NESTED]
    script = ([row for row in rows if row["phase"] == "script"] or [None])[0]
    whole = ([row for row in rows if row["phase"] == "total"] or [None])[0]
    nested = [row for row in rows if row["phase"] in NESTED]

    print()
    print("inside %s -- %s of wall time over %d pass(es)"
          % (step, _seconds(budget), passes_run))

    if own or script:
        print("  what the step itself did (share of its %s):" % _seconds(budget))
        _table(own, budget, passes_run, "pass")
        named = sum(row["seconds"] or 0.0 for row in own)
        rest = budget - named
        if script and budget and rest > 0.01 * budget:
            # The step's wall time minus its own work is what it spent waiting
            # on children. Named rather than left as "not named", because it is
            # not unaccounted for at all -- it is the next table.
            print("    %-18s %7s %9s %7s %9s %9s  %s"
                  % ("waiting on scripts", script["calls"] or 0, _seconds(rest),
                     _share(rest, budget), "", "",
                     "%s of script time, overlapping" % _seconds(script["seconds"])))
        elif budget and rest > 0.05 * budget:
            print("    %-18s %7s %9s %7s"
                  % ("(not named)", "", _seconds(rest), _share(rest, budget)))

    if inner:
        # The scripts' own account of themselves, against their own total.
        over = (script["calls"] if script else
                max(row["executions"] for row in inner))
        measured = ((script["seconds"] if script else None)
                    or sum(row["seconds"] or 0.0 for row in inner))
        print("  inside those %d script(s) -- %s of script time, %s each:"
              % (over, _seconds(measured),
                 _seconds(measured / over) if over else "-"))
        _table(inner, measured, over, "script")

        named = sum(row["seconds"] or 0.0 for row in inner)
        if whole and script:
            # Two clocks on the same span: the step's, which starts when the
            # process is launched, and the script's own, which starts when its
            # first line runs. The difference is the interpreter, and on this
            # pipeline it is not small -- see the import phase, which is only
            # the part of it the script can see.
            outside = (script["seconds"] or 0.0) - (whole["seconds"] or 0.0)
            if outside > 0.01 * measured:
                print("    %-18s %7d %9s %7s %9s %9s  %s"
                      % ("(before it runs)", script["calls"] or 0,
                         _seconds(outside), _share(outside, measured),
                         _seconds(outside / (script["calls"] or 1)), "",
                         "interpreter startup and exit, outside the script's clock"))
            inside = (whole["seconds"] or 0.0) - named
            if inside > 0.01 * measured:
                print("    %-18s %7s %9s %7s %9s %9s  %s"
                      % ("(inside, unnamed)", "", _seconds(inside),
                         _share(inside, measured), "", "",
                         "between the phases: reading prompts, printing, cleanup"))
        elif measured and measured - named > 0.05 * measured:
            print("    %-18s %7s %9s %7s"
                  % ("(not named)", "", _seconds(measured - named),
                     _share(measured - named, measured)))

    for row in nested:
        print("    %-18s %7d %9s %7s %9s %9s  %s"
              % (row["phase"], row["calls"] or 0, _seconds(row["seconds"]), "",
                 _seconds((row["seconds"] or 0.0) / row["calls"])
                 if row["calls"] else "-", _seconds(row["longest"]),
                 "inside one of the phases above, not beside it"))


def passes(conn, run_id):
    """What each pass cost, oldest first. A generation each, under continue_run.py.

    Worth its own table only when there is more than one: a cost that climbs
    generation after generation is the population growing under it, and that is
    a different problem from a step that was always slow.
    """
    rows = store.pass_costs(conn, run_id)
    if len(rows) < 2:
        return
    print()
    print("pass by pass -- %d of them" % len(rows))
    print("    %-6s %-8s %-20s %7s %7s %10s"
          % ("pass", "gen", "started", "steps", "high#", "seconds"))
    for row in rows:
        print("    %-6d %-8s %-20s %7d %7s %10s"
              % (row["pass_no"], row["generation"] or "-", row["started_at"],
                 row["steps"], row["watermark"] if row["watermark"] is not None else "-",
                 _seconds(row["seconds"])))


def first_look(conn, run_id, total):
    """The three biggest pieces of the sweep, phase by phase across every step.

    The step table ranks steps and the phase tables rank within one; this ranks
    across all of them, because "which step first" is only a useful question
    while the answer is not "the one phase inside it that is 60% of everything".
    """
    phases = store.phase_costs(conn, run_id)
    rows = [row for row in phases if row["phase"] not in WHOLE + NESTED]
    if not rows or not total:
        return
    # What each row is a share *of*: a phase inside a generated script is a
    # share of script time, because a batch runs several at once and their
    # seconds overlap both each other and the sweep's wall clock. Sharing them
    # against the sweep would have three phases adding up to 150% of it.
    scripts = {row["step"]: row["seconds"] or 0.0
               for row in phases if row["phase"] == "script"}
    print()
    print("the biggest named pieces of this sweep")
    for place, row in enumerate(rows[:3], 1):
        calls = row["calls"] or 0
        against = scripts.get(row["step"]) if row["executions"] else None
        print("    %d. %s/%s -- %s, %s of %s, over %d call(s) at %s each"
              % (place, row["step"], row["phase"], _seconds(row["seconds"]),
                 _share(row["seconds"], against or total).strip(),
                 "script time" if against else "the sweep", calls,
                 _seconds((row["seconds"] or 0.0) / calls) if calls else "-"))


def as_csv(conn, run_id, out=sys.stdout):
    """The same rows, one line each, for a spreadsheet or a diff between sweeps.

    Deliberately the rows and not the tables: a table is a reading of the
    numbers and this is the numbers.
    """
    out.write("kind,step,phase,passes_or_calls,seconds,items,unit\n")
    for row in store.step_costs(conn, run_id):
        out.write("step,%s,,%d,%.4f,%d,%s\n"
                  % (row["step"], row["passes"], row["seconds"] or 0.0,
                     row["items"] or 0, row["unit"] or ""))
    for row in store.phase_costs(conn, run_id):
        out.write("phase,%s,%s,%d,%.4f,,\n"
                  % (row["step"], row["phase"], row["calls"] or 0,
                     row["seconds"] or 0.0))


def report(conn, run_id, step=None):
    """The whole thing, in the order the questions are asked."""
    run = store.get_run(conn, run_id)
    if run is None:
        raise SystemExit("no run %d in %s" % (run_id, conn.path))
    rows = store.step_timings(conn, run_id)
    if not rows:
        raise SystemExit(
            "run %d recorded no timings. Sweeps run before start_run.py started "
            "keeping them have none, and nothing can be worked out after the "
            "fact -- run a generation of it and the rows will be there."
            % run_id)

    print("run %d -- %s, %s, status %s"
          % (run_id, run["template"], run["created_at"], run["status"]))
    total = steps(conn, run_id)

    if step:
        wanted = [row for row in store.step_costs(conn, run_id)
                  if row["step"] == step]
        if not wanted:
            raise SystemExit("run %d has no step called %r. It ran: %s"
                             % (run_id, step,
                                ", ".join(sorted({row["step"] for row in rows}))))
        phases(conn, run_id, step, wanted[0]["seconds"], wanted[0]["passes"])
    else:
        # Only the steps with phases worth reading: a step that named none has
        # nothing under its wall time, and printing an empty table for each of
        # them buries the two that matter.
        named = {row["step"] for row in store.phase_costs(conn, run_id)}
        for row in store.step_costs(conn, run_id):
            if row["step"] in named:
                phases(conn, run_id, row["step"], row["seconds"], row["passes"])

    first_look(conn, run_id, total)
    passes(conn, run_id)
    print()
    print("one step in full: python -m metrics.report --run %d --step process"
          % run_id)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="What a sweep spent its time on: which step, and where inside it.")
    parser.add_argument("--db", default=config.DB_PATH, metavar="PATH",
                        help="the database (default: %s)" % config.DB_PATH)
    parser.add_argument("--run", type=int, default=0, metavar="N",
                        help="which sweep; 0, the default, is the latest")
    parser.add_argument("--step", metavar="NAME",
                        help="one step's phases in full, instead of every step's")
    parser.add_argument("--csv", action="store_true",
                        help="the rows behind the tables, as csv on stdout")
    args = parser.parse_args(argv)

    path = args.db if os.path.isabs(args.db) else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.db)
    if not os.path.exists(path):
        raise SystemExit("no database at %s" % path)

    conn = store.connect(args.db)
    run_id = store.latest_run(conn) if args.run == 0 else args.run
    if run_id is None:
        raise SystemExit("%s holds no sweeps" % conn.path)
    if args.csv:
        as_csv(conn, run_id)
    else:
        report(conn, run_id, args.step)
    return 0


if __name__ == "__main__":
    sys.exit(main())
