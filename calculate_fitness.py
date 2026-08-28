"""
calculate_fitness.py - Turn the judged answers into one number per individual.

    exchanges.quality  -->  individuals.fitness

The judge scores answers, not chromosomes: a sweep comes out of `evaluate` with
a quality on every exchange and nothing at all on the individual that produced
them. Selection needs the opposite -- one comparable number per individual --
so this step folds a transcript into its mean.

    fitness = the average quality across the exchanges of the individual's
              most recent execution

The most recent execution and not all of them, for the same reason `evaluate`
scores only that one: running a chromosome again under a different weight seed
is a new result, not an amendment to the old one. That choice already lives in
the `individual_quality` view, which is where the average is read from.

An individual with nothing to average -- never run, run and crashed before it
answered, or still unjudged -- gets 0.0 rather than NULL. Fitness is what a
selection step sorts on, and a missing score there would have to be given a
meaning at every call site; giving it one here, once, says the only sensible
thing: an individual that produced no judged answer is worth nothing. That is
also what a BAD individual gets, which is right -- a blend PEFT refuses to
build cannot be selected for.

Partly judged individuals are averaged over the answers that do have a score
(the view's AVG skips NULLs) and reported, since a fitness over half a
transcript is a weaker claim than one over all of it.

The same numbers are also written to `fitness_history`, one row per individual
per generation, stamped with the moment they were worked out. `individuals.
fitness` is a column that only ever holds *now*: the next generation overwrites
it and mutation clears it outright, so a sweep that keeps only that column can
say how fit its population is and nothing at all about whether the search is
getting anywhere. The history is the record of the run as a run -- what a
fitness curve is drawn from.

main.py calls this as a library:

    calculate_fitness.assign(conn, run_id)
"""

from collections import namedtuple

import store

# What one pass of this step came to: which generation it recorded, when it
# recorded it, and the individual_quality rows it worked from, best first.
Snapshot = namedtuple("Snapshot", "generation recorded_at rows")


def fitness_of(row):
    """The fitness of one `individual_quality` row.

    Its mean answer quality, or 0.0 when it has none to speak of.
    """
    quality = row["quality"]
    return 0.0 if quality is None else round(float(quality), 6)


def entry_for(row):
    """One `individual_quality` row as a fitness_history row.

    It carries the chromosome and the state alongside the number, rather than
    leaning on a join back to `individuals`: the individual this describes will
    be mutated into a different chromosome and given a different fitness before
    the sweep is done, and a history read through the population as it stands
    today would report every past generation in terms of the present one.
    """
    return {"number": row["number"],
            "chromosome": row["chromosome"],
            "state": row["state"],
            "fitness": fitness_of(row),
            "answers": row["answers"] or 0,
            "unscored": row["unscored"] or 0}


def assign(conn, run_id):
    """Write individuals.fitness for a whole sweep, and record the generation.

    Two writes of one number: the column selection and elitism read, and a row
    in `fitness_history` so the value survives the next generation overwriting
    the column.

    Returns a Snapshot -- the generation recorded, its timestamp, and the rows
    it worked from, best first -- so a caller can report on the sweep without
    asking the database a second time.

    A sweep with no individuals is not a generation and is not recorded; the
    caller has a better complaint to make about that than this step does.
    """
    rows = store.quality_rows(conn, run_id)
    for row in rows:
        store.set_fitness(conn, run_id, row["number"], fitness_of(row))
    conn.commit()
    if not rows:
        return Snapshot(0, None, rows)

    generation = store.fitness_generation(conn, run_id, len(rows))
    recorded_at = store.record_fitness(conn, run_id, generation,
                                       [entry_for(row) for row in rows])
    return Snapshot(generation, recorded_at, rows)
