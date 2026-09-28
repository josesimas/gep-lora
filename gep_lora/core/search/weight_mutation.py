"""
weight_mutation.py - Mutation of the blend weights alone, minus the elite.

    individuals.chromosome  -->  chromosome, has_changed, fitness

mutation.py offers every symbol a chance to move; this offers only the weights.
A weight is a `w1`..`wn` symbol -- the child of an `L*` node, saying at which
of the n drawn weights (one per slot in the sweep's LORA_SLOTS) that adapter enters the blend -- so a tree of ten
blended adapters carries ten of them. Operators and slots are left exactly as
they are: the blend keeps its shape and its adapters, and only how much of each
it takes is varied.

How many weights move
---------------------
Not a chance per weight but a count over all of them. Every weight of every
non-elite chromosome goes into one pool, and

    round(rate * weights in the pool)

of them are drawn from it, without replacement, and each replaced by a
*different* weight symbol. Ten weights at 0.1 is exactly one change, somewhere
among the ten; a population holding sixty weights at 0.1 is exactly six,
wherever they happen to fall -- two in one individual and none in another is as
likely as any other spread. Rounding is half up, so a pool too small for the
rate to reach one half moves nothing.

A swap from one weight symbol to another is class-local in the sense
mutation.py explains -- arity 0 either side -- so the tree keeps its shape and
every result still decodes; generate_population.check() is run on each one
before it is stored all the same.

The elite does not change
-------------------------
The individual marked `is_best` contributes nothing to the pool, so none of its
weights can be drawn and it is not written at all -- for the reason mutation.py
gives: elitism named it so that it survives the generation intact.

has_changed, and the fitness that goes with it
----------------------------------------------
An individual with a weight moved gets its new chromosome through
store.set_chromosome(), which sets has_changed to 1 and clears its fitness to
NULL: the score belonged to the blend weighted the old way.

Unlike mutation.py this step only ever *raises* has_changed. It runs after
mutation in the same generation, and an individual mutation moved and this step
did not has still changed this round -- writing 0 over it would tell process to
skip a chromosome that has never been run.

gep_lora/core/pipeline/start_run.py calls this as a library:

    weight_mutation.apply(conn, run_id, rate, rng)
"""

from collections import namedtuple

from gep_lora.core.search import generate_population
from gep_lora.core.storage import store

# One individual this round touched: what it was, what it became, and how many
# weights differ between the two.
Change = namedtuple("Change", "number before after weights")


def weight_positions(chromosome):
    """The positions of the weight symbols in a chromosome, in order."""
    return [position for position, symbol in enumerate(chromosome.split("."))
            if symbol in generate_population.VARIABLES]


def draw_count(rate, total):
    """How many of `total` weights a round at `rate` moves, rounded half up."""
    return min(total, int(rate * total + 0.5))


def mutate(chromosomes, rate, rng, slots=generate_population.MAX_SLOTS):
    """A pool of chromosomes through one draw. -> the chromosomes they came out as.

    `chromosomes` is a list; the result is a list of the same length, in the
    same order, each entry the chromosome that one became (unchanged if none of
    its weights was drawn).
    """
    pool = [(index, position)
            for index, chromosome in enumerate(chromosomes)
            for position in weight_positions(chromosome)]
    symbols = [chromosome.split(".") for chromosome in chromosomes]
    for index, position in rng.sample(pool, draw_count(rate, len(pool))):
        current = symbols[index][position]
        symbols[index][position] = rng.choice(
            [other for other in generate_population.weight_symbols(slots)
             if other != current])
    mutated = [".".join(parts) for parts in symbols]
    for chromosome in mutated:
        generate_population.check(chromosome)
    return mutated


def differences(before, after):
    """How many symbols differ between two chromosomes of the same shape."""
    return sum(one != other for one, other in zip(before.split("."), after.split(".")))


def apply(conn, run_id, rate, rng, slots=generate_population.MAX_SLOTS):
    """Mutate the weights of a whole population but its elite. -> (changes, rows).

    Writes the new chromosome, has_changed = 1 and a cleared fitness for every
    individual that moved, and nothing for any other. `changes` holds one Change
    per individual that moved; `rows` is the population as it was before.
    """
    rows = store.individuals(conn, run_id)
    eligible = [row for row in rows if not row["is_best"]]
    mutated = mutate([row["chromosome"] for row in eligible], rate, rng, slots)
    changes = []
    for row, after in zip(eligible, mutated):
        if after == row["chromosome"]:
            continue
        store.set_chromosome(conn, row["id"], after)
        changes.append(Change(row["number"], row["chromosome"], after,
                              differences(row["chromosome"], after)))
    conn.commit()
    return changes, rows


def pool_size(rows):
    """How many weights the non-elite individuals of `rows` hold between them."""
    return sum(len(weight_positions(row["chromosome"]))
               for row in rows if not row["is_best"])
