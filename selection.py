"""
selection.py - Roulette wheel sampling over the population.

    individuals.fitness  -->  n copies of the fit and one complete stranger,
                              in place of the n+1 weakest

Fitness-proportionate selection, the classic one: give every individual a slice
of a wheel as wide as its fitness, spin the wheel once per pick, and take
whoever it lands on. A chromosome twice as fit is twice as likely to be picked,
and no chromosome is picked for certain -- a mediocre individual keeps a real
chance, which is what stops the search collapsing onto the first good blend it
finds.

Picks are drawn **with replacement**: the same individual can come up several
times in one round, and that is the mechanism rather than a flaw in it. That is
how a fit chromosome comes to have several descendants.

What one round does
-------------------
Three things, in this order, and the arithmetic is the point:

1. **Appends `n` copies** of whoever the wheel landed on, `n` being
   `SELECTION_COUNT` (or the size of the population, when that is None).
2. **Culls the `n+1` weakest** individuals of the population as it was before
   the round -- never the elite.
3. **Appends one brand-new individual**, drawn from
   `generate_population.random_tree()` under the sweep's own `MAX_DEPTH`,
   `BRANCH_PROB` and `UNIQUE`: the same draw the first population came from.

`n + 1` in and `n + 1` out, so a generation ends **exactly the size it began**:
a sweep drawn at `COUNT = 10` is still ten individuals after ten generations.
The cull is `n+1` and not `n` because the round appends `n+1` rows -- the
copies *and* the newcomer -- and it is that total, not the number of copies,
that has to come back out. What moves is not the size but the membership: the
fit are duplicated, the weak are gone, and one member of every generation owes
nothing to either.

That makes `SELECTION_COUNT` a knob on the **turnover** rather than on the
size. At 2 over a population of 10, three individuals in three out each
generation; at None it is as many copies as the population holds, which is more
than the cull can take (see below) and the only setting that still grows it.

The newcomer is the reason the cull is safe. Selection can only ever pick
chromosomes the population already holds, and mutation only ever nudges a
symbol into a sibling of its own class, so a search that culls and copies is a
search whose gene pool can only narrow. One fresh draw per generation is a
floor under that: whatever the population has converged on, every generation
still contains one tree that was grown from nothing.

What a cull takes with it
-------------------------
The individual whole -- its executions, its exchanges and its test_results all
cascade (see store.delete_individuals). Its number is retired rather than
reused, and its rows in `fitness_history` stay exactly where they are, because
the history is what each generation *was* and a cull is not allowed to rewrite
that. So a sweep can always say that individual #7 scored 0.04 in generation 3
and was culled at the end of it.

The weakest means the lowest `fitness`, with NULL read as 0.0 the way every
other reader of that column reads it, and ties broken on the **lowest number**
-- so between two equally weak individuals the longer-standing one goes, and
re-running the step culls the same rows rather than a coin-flip's worth of
different ones. `is_best` is never eligible: culling the elite would discard
the one thing elitism exists to protect, and it is the only individual the step
refuses outright. When the population is too small to give up `n+1` non-elite
rows it gives up as many as it has, and grows by the difference -- which is
what `SELECTION_COUNT = None` does every round, since asking for the whole
population as copies asks the cull for one more row than the population has
individuals to spare.

A parent the wheel just landed on can itself be culled -- weak individuals do
occasionally get picked -- and nothing needs doing about that: the copy it
produced carries its chromosome, its script and its fitness forward, which is
all the parent had to pass on.

What a copy is
--------------
A copy is the parent field for field: its tree, its state, its rank, its
script, its weight seed and its fitness, not merely its chromosome. Only the id
and the number are its own, and only because those are what make it a row of
its own. It is therefore a clone in the full sense -- it arrives already
carrying the result its parent earned, rather than as a blank waiting to be
built and judged.

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

The newcomer needs the same two steps for the opposite reason: it arrives with
a chromosome and *nothing else* -- no tree, no script, no seed, no fitness --
exactly as a member of the first population does, so `trees` and `runs` are
what make it runnable and `process` is what earns it a score.

A round replacing what it adds does not make the step idempotent: running it
twice runs two rounds of selection, and the second one culls what the first one
left -- the population comes out the same size and made of different
individuals. That is what a second generation *is*, so it is deliberate
-- but it does mean `python main.py selection` is a thing you do on purpose,
not a thing you repeat to be sure it took.

Zero fitness
------------
A slice as wide as 0.0 can never be landed on, so an individual with no fitness
is never picked -- correct for roulette, and worth saying out loud because
`individuals.fitness` defaults to 0.0. A population where *everything* is 0.0
has no wheel to spin at all: every slice is empty, and picking anyway would just
be a uniform draw wearing a selection step's name. That case selects nobody,
culls nobody, draws no newcomer and writes nothing -- a round that cannot say
which individuals are the fit ones cannot be trusted to say which are the weak
ones either, and one that culled on that basis would empty a population instead
of holding it steady.

main.py calls this as a library:

    selection.select(conn, run_id, count, rng, conf)
"""

import bisect
from collections import namedtuple

import generate_population
import store

# What one round of selection came to: the rows the wheel landed on, in the
# order it landed on them; the numbers their copies were appended under; the
# population as it was before any of it; the rows the cull took; and the
# newcomer, or None when the round did nothing at all.
Round = namedtuple("Round", "parents numbers population culled newcomer")

# The one individual per round that is nobody's copy.
Newcomer = namedtuple("Newcomer", "number chromosome")

# How many draws to spend looking for a chromosome the population does not
# already hold, before settling for one it does. A generous ceiling on a cheap
# draw: the alternative is failing a generation over a duplicate, which would
# be a worse answer than a duplicate.
FRESH_ATTEMPTS = 100


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


def weakest(rows, count):
    """The `count` weakest of `rows`, weakest first. -> rows, never the elite.

    Reads the stored `fitness` column and nothing else, so "weakest" here means
    exactly what "best" means to elitism.py -- one definition of the number,
    living in calculate_fitness.py. NULL is 0.0, as it is to every other reader:
    a mutant whose score was cleared has not been judged since, and an unjudged
    individual is not one the search has any reason to keep.

    Fewer than `count` rows come back when the population cannot spare them,
    which is the one way a round can leave the population bigger than it found
    it.
    """
    eligible = [row for row in rows if not row["is_best"]]
    eligible.sort(key=lambda row: (row["fitness"] or 0.0, row["number"]))
    return eligible[:max(0, count)]


def fresh(rows, rng, conf):
    """A brand-new chromosome, drawn the way the first population was.

    generate_population.build_population() with a count of one, so the newcomer
    is grown by the same random_tree() under the same MAX_DEPTH and BRANCH_PROB
    and validated by the same check() -- there is no second draw and no second
    grammar. Its own `unique` flag is left off because it only dedupes within
    the batch it draws, and a batch of one has nothing to dedupe against.

    The sweep's UNIQUE is honoured here instead, and against something more
    useful: the chromosomes the population held going into this round, which
    includes the ones about to be culled. A newcomer exists to bring the search
    something it does not have, so re-drawing what it just discarded would be
    the one draw that achieves nothing. Exhausting the attempts is not an error
    -- a converged search that can only find chromosomes it already holds is
    telling you something, and a duplicate newcomer says it more usefully than
    a failed generation would.
    """
    held = {row["chromosome"] for row in rows}
    chromosome = None
    for _ in range(FRESH_ATTEMPTS if conf.get("UNIQUE", True) else 1):
        chromosome = generate_population.build_population(
            1, rng, conf["MAX_DEPTH"], conf["BRANCH_PROB"], False)[0]
        if chromosome not in held:
            break
    return chromosome


def select(conn, run_id, count, rng, conf):
    """Run one round of selection: copy, cull, draw one stranger. -> a Round.

    Writes nothing when the wheel cannot be spun, in which case `parents`,
    `numbers` and `culled` are all empty, `newcomer` is None, and the
    population is exactly as it was.

    The order the three writes happen in is the order they have to happen in.
    The copies go in first, so the cull is never asked to take rows the round
    has not yet replaced, and so the numbers it retires are always below the
    ones this round handed out -- which is what keeps store.next_number() from
    ever reissuing one. The cull then reads the population as it was *before*
    the round -- `rows`, not the table -- so it can only ever take individuals
    the search has actually judged, never one of the copies just appended or
    the newcomer about to be. And the newcomer's number comes last because it
    is the newest thing in the sweep.

    The newcomer is drawn from the same `rng` as the spins, after them, so the
    round as a whole stays reproducible from the sweep's stored
    SELECTION_MASTER_SEED. The cull consumes no randomness at all.
    """
    rows = store.individuals(conn, run_id)
    parents = draw(rows, len(rows) if count is None else count, rng)
    if not parents:
        return Round([], [], rows, [], None)

    numbers = store.append_copies(conn, run_id, parents)
    culled = store.delete_individuals(
        conn, run_id, [row["number"] for row in weakest(rows, len(parents) + 1)])
    chromosome = fresh(rows, rng, conf)
    newcomer = Newcomer(store.append_individual(conn, run_id, chromosome), chromosome)
    return Round(parents, numbers, rows, culled, newcomer)
