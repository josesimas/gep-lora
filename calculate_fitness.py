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

main.py calls this as a library:

    calculate_fitness.assign(conn, run_id)
"""

import store


def fitness_of(row):
    """The fitness of one `individual_quality` row.

    Its mean answer quality, or 0.0 when it has none to speak of.
    """
    quality = row["quality"]
    return 0.0 if quality is None else round(float(quality), 6)


def assign(conn, run_id):
    """Write individuals.fitness for a whole sweep.

    Returns the rows it worked from, best first, so a caller can report on the
    sweep without asking the database a second time.
    """
    rows = store.quality_rows(conn, run_id)
    for row in rows:
        store.set_fitness(conn, run_id, row["number"], fitness_of(row))
    conn.commit()
    return rows
