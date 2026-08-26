"""
mutation.py - Point mutation over the population, minus the elite.

    individuals.chromosome  -->  chromosome, has_changed, fitness

Selection makes copies; mutation is what makes them worth having. Every symbol
of every non-elite chromosome is offered a chance, `rate`, of being replaced by
a different symbol -- so a chromosome of eleven symbols at rate 0.1 expects
about one change, and any individual may come through untouched.

The elite does not change
-------------------------
The individual marked `is_best` is passed over: elitism named it precisely so
that the best result found so far survives a generation intact, and mutating it
would throw away the thing the flag exists to protect. It is the one row this
step reads and does not write.

Only valid changes
------------------
A symbol may only be replaced by one of its own kind:

    CAT SVD LIN     swap freely among themselves       (arity 2, operator children)
    L1 .. L5        swap freely among themselves       (arity 1, variable child)
    w1 .. w5        swap freely among themselves       (arity 0)

and the root is never touched at all, because the grammar fixes it at CAT.

That restriction is the whole trick. A symbol's arity and the alphabet its
children are drawn from are properties of its *class*, not of the symbol, so a
swap inside a class leaves the tree exactly the shape it was: every node still
has the number of children it had, and every child is still legal where it
stands. Reaching across classes would not -- turning a `CAT` into an `L2` would
leave a node with two children where one belongs, and the second of them a
subtree where a bare `w` is required, which is not a chromosome at all. There is
no repair step here and no tail to absorb the damage, so the mutation simply
does not make changes it would have to repair. Every result is put through
generate_population.check() before it is stored, which is the guarantee rather
than a hope.

What may still come out broken is a `LIN` above two different ranks -- swapping
`CAT` for `LIN` can easily produce one. That is a legal chromosome describing a
blend PEFT will not build, which is a property of the search space and is culled
downstream: the runs step marks it `state = 'BAD'` and process skips it.

has_changed, and the fitness that goes with it
----------------------------------------------
has_changed is set to 1 on an individual whose chromosome this step actually
altered, and 0 on every other -- the elite included, and an individual the dice
passed over. It is this round's answer, not a running total, so it is written
for every individual each time rather than only for the ones that moved.

Setting it to 1 clears that individual's fitness to NULL. The score was earned
by the chromosome that has just been replaced, which makes keeping it worse than
stale: it would let a mutant be elected, or win a slice of the roulette wheel,
on the strength of a blend it no longer describes. NULL says what is true --
this chromosome has not been judged yet -- and both elitism and selection
already read a missing fitness as no fitness, so a mutant simply waits its turn
until process and evaluate have given it one of its own.

main.py calls this as a library:

    mutation.apply(conn, run_id, rate, rng)
"""

from collections import namedtuple

import generate_population
import store

# One individual this round touched: what it was, what it became, and how many
# symbols differ between the two.
Change = namedtuple("Change", "number before after symbols")


def alternatives(symbol, position):
    """The symbols `symbol` may legally become at `position`.

    Empty for the root, which the grammar fixes at CAT, and for anything that
    is not in the alphabet at all.
    """
    if position == 0:
        return ()
    for family in (generate_population.BINARY_OPS,
                   generate_population.UNARY_OPS,
                   generate_population.VARIABLES):
        if symbol in family:
            return tuple(other for other in family if other != symbol)
    return ()


def mutate(chromosome, rate, rng):
    """One chromosome through the dice. -> the chromosome it came out as.

    Each symbol independently has probability `rate` of being replaced by one of
    its alternatives, drawn uniformly. The result is checked before it is
    returned: a mutation that could not be read back would be a bug here, not a
    result, and should never reach a caller.
    """
    symbols = chromosome.split(".")
    drawn = []
    for position, symbol in enumerate(symbols):
        choices = alternatives(symbol, position)
        drawn.append(rng.choice(choices)
                     if choices and rng.random() < rate else symbol)
    mutated = ".".join(drawn)
    generate_population.check(mutated)
    return mutated


def differences(before, after):
    """How many symbols differ between two chromosomes of the same shape."""
    return sum(one != other for one, other in zip(before.split("."), after.split(".")))


def apply(conn, run_id, rate, rng):
    """Mutate a whole population but for its elite. -> (changes, rows).

    Writes has_changed for every individual; for the ones that actually moved it
    also writes the new chromosome and clears the fitness the old one earned.
    `changes` holds one Change per individual that moved; `rows` is the
    population as it was before any of it.
    """
    rows = store.individuals(conn, run_id)
    changes = []
    for row in rows:
        if row["is_best"]:
            # The one row that is read and not written: see the module note.
            store.set_changed(conn, row["id"], 0)
            continue
        mutated = mutate(row["chromosome"], rate, rng)
        if mutated == row["chromosome"]:
            store.set_changed(conn, row["id"], 0)
        else:
            store.set_chromosome(conn, row["id"], mutated)
            changes.append(Change(row["number"], row["chromosome"], mutated,
                                  differences(row["chromosome"], mutated)))
    conn.commit()
    return changes, rows
