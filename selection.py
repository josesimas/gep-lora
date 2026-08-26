"""
selection.py - Roulette wheel sampling over the population.

    individuals.fitness  -->  more individuals

Fitness-proportionate selection, the classic one: give every individual a slice
of a wheel as wide as its fitness, spin the wheel once per pick, and take
whoever it lands on. A chromosome twice as fit is twice as likely to be picked,
and no chromosome is picked for certain -- a mediocre individual keeps a real
chance, which is what stops the search collapsing onto the first good blend it
finds.

Picks are drawn **with replacement**: the same individual can come up several
times in one round, and that is the mechanism rather than a flaw in it. That is
how a fit chromosome comes to have several descendants.

Nothing is deleted
------------------
The picks are *appended* to the population. Selection here does not thin a
generation down to its survivors and it never overwrites a row: an individual
already in the sweep keeps its number, its script, its executions and its
transcripts, whether or not the wheel ever landed on it. What the step leaves
behind is a larger population, whose newest members are copies of its fittest,
waiting for whatever comes next to vary them.

A copy is the parent field for field: its tree, its state, its rank, its script,
its weight seed and its fitness, not merely its chromosome. Only the id and the
number are its own, and only because those are what make it a row of its own. It
is therefore a clone in the full sense -- it arrives already carrying the result
its parent earned, rather than as a blank waiting to be built and judged.

The one thing no copy inherits is `is_best`. That flag does not describe an
individual, it picks one out of the population, so it is not the parent's to
hand on: a copy of the elite is not itself the elite, and a sweep that ended a
round with several of them would have lost the only thing the flag says. Every
copy arrives at 0, and the next election decides on the fitness it earns.

That inheritance is meant to be *spent*, not kept. The copies exist for whatever
comes next to vary, and a copy that is varied has a chromosome its inherited
tree, script, rank and fitness no longer describe -- they are the parent's
answers to a question the child no longer asks. Re-deriving them is what
`python main.py trees runs` does, from the chromosome, for every individual;
until then a copy carries its parent's, including the script name, so two rows
can name the same run_NNN.py and the same weight seed while they are still the
same chromosome anyway.

Because it appends, the step is not idempotent: running it twice runs two
rounds of selection and the population grows twice. That is what a second
generation *is*, so it is deliberate -- but it does mean `python main.py
selection` is a thing you do on purpose, not a thing you repeat to be sure it
took.

Zero fitness
------------
A slice as wide as 0.0 can never be landed on, so an individual with no fitness
is never picked -- correct for roulette, and worth saying out loud because
`individuals.fitness` defaults to 0.0. A population where *everything* is 0.0
has no wheel to spin at all: every slice is empty, and picking anyway would just
be a uniform draw wearing a selection step's name. That case selects nobody and
writes nothing.

main.py calls this as a library:

    selection.select(conn, run_id, count, rng)
"""

import bisect
from collections import namedtuple

import store

# What one round of selection came to: the rows the wheel landed on, in the
# order it landed on them; the numbers they were appended under; and the
# population as it was before any of it.
Round = namedtuple("Round", "parents numbers population")


def wheel(rows):
    """The wheel `rows` make: (cumulative edges, total width).

    Edge i is where individual i's slice ends, so the slices are laid end to end
    and a mark anywhere in [0, total) falls in exactly one of them. A negative
    fitness would eat into its neighbour, so it is floored at zero -- quality is
    0..1 and cannot produce one, but the wheel should not depend on that.
    """
    edges, total = [], 0.0
    for row in rows:
        total += max(0.0, row["fitness"] or 0.0)
        edges.append(total)
    return edges, total


def spin(rows, edges, total, rng):
    """One spin: the individual the mark falls on.

    bisect_right and not bisect_left, so that a mark landing exactly on an edge
    goes to the slice that *starts* there rather than the one that ended -- which
    is what skips the zero-width slices of the unfit instead of handing a pick
    to one of them.
    """
    index = bisect.bisect_right(edges, rng.random() * total)
    return rows[min(index, len(rows) - 1)]


def draw(rows, count, rng):
    """Spin the wheel `count` times, with replacement. -> the rows it landed on.

    Empty when there is nothing to spin: no population, nothing asked for, or a
    wheel of nothing but zero-width slices.
    """
    edges, total = wheel(rows)
    if not rows or count <= 0 or total <= 0.0:
        return []
    return [spin(rows, edges, total, rng) for _ in range(count)]


def select(conn, run_id, count, rng):
    """Run one round of selection, appending the picks. -> a Round.

    Writes nothing when the wheel cannot be spun, in which case `parents` and
    `numbers` are both empty and the population is exactly as it was.
    """
    rows = store.individuals(conn, run_id)
    parents = draw(rows, len(rows) if count is None else count, rng)
    numbers = store.append_copies(conn, run_id, parents) if parents else []
    return Round(parents, numbers, rows)
