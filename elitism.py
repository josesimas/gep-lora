"""
elitism.py - Name the individual this generation carries forward.

    individuals.fitness  -->  individuals.is_best

Elitism is the rule that the best individual survives a generation untouched.
Selection, crossover and mutation are all free to lose it otherwise: the best
chromosome of one generation can easily have no descendant in the next, and a
search that can go backwards wastes the generations it spends climbing again.
Marking it is what lets a later step copy it across unchanged.

    is_best = 1 for one individual with the highest fitness, 0 for every other

Exactly one, whatever the fitness values look like. `individuals.fitness` is a
mean of judged answers, so ties are ordinary -- two chromosomes can easily blend
to the same score, and every individual of an all-zero generation ties with the
rest. The tie is broken by the lowest individual number: an arbitrary rule, but
a fixed one, so a sweep re-run over the same database elects the same individual
rather than a different one each time.

The flag is set from the stored `fitness` column and nothing else. Reading the
transcripts again here would be a second, quietly different definition of "best"
-- if the fitness rule changes, it changes in calculate_fitness.py and this step
follows it without knowing that it did.

main.py calls this as a library:

    elitism.elect(conn, run_id)
"""

import store


def best_of(rows):
    """The elite among `rows`: highest fitness, lowest number breaking the tie.

    Returns None when there is no elite to be had -- an empty population, or one
    where nothing scored above 0.0. `fitness` defaults to 0.0, so an all-zero
    population is either a sweep that never reached the fitness step or one
    where every individual failed; neither has a best worth carrying forward,
    and electing one anyway would dress up an arbitrary pick as a result.
    """
    best = max(rows, key=lambda row: (row["fitness"] or 0.0, -row["number"]),
               default=None)
    if best is None or not (best["fitness"] or 0.0) > 0.0:
        return None
    return best


def elect(conn, run_id):
    """Mark one individual as the sweep's elite. -> (the elite row, all rows).

    Writes nothing when there is no elite: a sweep keeps whatever is_best it
    already had rather than gaining a meaningless one. The population comes back
    too, in number order, so a caller can report on the election -- or on the
    lack of one -- without asking the database a second time.
    """
    rows = store.individuals(conn, run_id)
    best = best_of(rows)
    if best is not None:
        store.mark_best(conn, run_id, best["number"])
        conn.commit()
    return best, rows
