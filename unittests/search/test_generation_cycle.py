"""
test_generation_cycle.py - The four steps as a search, not as four steps.

The other modules here test each step against its own contract. This one runs
them in the order continue_run.py runs them --

    fitness -> elitism -> selection -> mutation

-- over a sweep whose scores come from a stand-in judge, and asks the questions
that only make sense of the loop as a whole: does the population stay the size
it was drawn at, does the best individual ever go backwards, do the numbers
keep rising, is every chromosome in the table still a chromosome, and does the
search actually climb.

The judge is arithmetic rather than a model: an individual scores the fraction
of its leaves that are `w1`. That is a target mutation can reach (variables
swap freely among themselves) and selection can spread, so a working search
climbs toward it and a broken one does not -- which is the point of scoring on
something with a known answer rather than on a blend.

Nothing here needs a GPU, a base model or a judge endpoint: the executions and
transcripts are written straight into the tables, exactly as a mocked sweep's
would be.
"""

import unittest

from search import (calculate_fitness, elitism, generate_population as gp,
                    mutation, selection)
from storage import store

from unittests.search.support import CONF, SweepTestCase, rng


# The knobs a real sweep would carry, small enough to run in a second.
POPULATION = 12
SELECTION_COUNT = 3
MUTATION_RATE = 0.15
GENERATIONS = 14


def target_score(chromosome):
    """The stand-in judge: how much of this tree's weight comes from w1.

    Between 0.0 and 1.0, deterministic, and reachable -- mutation swaps a
    variable for another variable, so every chromosome has a path to 1.0
    without changing shape.
    """
    leaves = [symbol for symbol in chromosome.split(".")
              if symbol in gp.VARIABLES]
    return round(sum(leaf == "w1" for leaf in leaves) / len(leaves), 6)


class TargetScoreTests(unittest.TestCase):
    """The stand-in judge itself, so a failure below is never its fault."""

    def test_all_w1_scores_one(self):
        self.assertEqual(target_score("CAT.L1.L2.w1.w1"), 1.0)

    def test_no_w1_scores_zero(self):
        self.assertEqual(target_score("CAT.L1.L2.w2.w3"), 0.0)

    def test_it_counts_only_the_leaves(self):
        self.assertEqual(target_score("CAT.SVD.L1.L1.L2.w1.w1.w3"),
                         round(2 / 3, 6))


class GenerationCycleTests(SweepTestCase):
    """A whole search, four steps at a time."""

    def setUp(self):
        super().setUp()
        self.draw_population()

    def draw_population(self):
        """Seed the current sweep, and its two generators, from fixed seeds.

        Called again by the reproducibility test against a second sweep, which
        is why the seeds are written here rather than passed in: the claim is
        that the same seeds give the same search.
        """
        store.add_individuals(self.conn, self.run, self.starting_population())
        # One generator per step, seeded once, the way a sweep derives each of
        # its own from a stored master seed.
        self.selection_rng = rng(555)
        self.mutation_rng = rng(777)

    def starting_population(self):
        """A drawn population, restricted to trees with several leaves.

        The restriction is what makes the climb worth measuring. A two-leaf
        tree scores 1.0 whenever both its variables happen to be w1, which is
        one draw in twenty-five -- so a population of small trees usually
        arrives with the target already in it, and every test of whether the
        search *found* anything would be testing whether the first draw got
        lucky. At four leaves that is one in six hundred, and reaching 1.0
        takes selection spreading a partial answer and mutation finishing it.
        """
        generator = rng(2024)
        population = []
        while len(population) < POPULATION:
            chromosome = gp.build_population(
                1, generator, CONF["MAX_DEPTH"], 0.9, False)[0]
            leaves = [symbol for symbol in chromosome.split(".")
                      if symbol in gp.VARIABLES]
            if len(leaves) >= 4 and chromosome not in population:
                population.append(chromosome)
        return population

    # --- the loop ----------------------------------------------------------

    def judge(self):
        """Stand in for process + evaluate: every individual runs and is graded.

        One execution per individual per generation, carrying a transcript
        whose mean is the chromosome's target score -- which is what
        calculate_fitness folds back into individuals.fitness.
        """
        for row in self.population():
            quality = target_score(row["chromosome"])
            self.run_individual(row["number"], [quality, quality])

    def generation(self):
        """One turn of the crank. -> the fitness snapshot it recorded."""
        self.judge()
        snapshot = calculate_fitness.assign(self.conn, self.run)
        elitism.elect(self.conn, self.run)
        selection.select(self.conn, self.run, SELECTION_COUNT,
                         self.selection_rng, CONF)
        mutation.apply(self.conn, self.run, MUTATION_RATE, self.mutation_rng)
        return snapshot

    def search(self, generations=GENERATIONS):
        return [self.generation() for _ in range(generations)]

    # Both read the snapshot the step handed back rather than querying the
    # history again. A generation that appends nobody is recorded as a
    # restatement of the one before it, so re-reading by generation number
    # later can hand two different snapshots the same row -- and a test that
    # compared a stalled search against itself would call it progress.

    def best_fitness(self, snapshot):
        return max(calculate_fitness.fitness_of(row) for row in snapshot.rows)

    def mean_fitness(self, snapshot):
        scores = [calculate_fitness.fitness_of(row) for row in snapshot.rows]
        return sum(scores) / len(scores)

    # --- what the loop has to keep true ------------------------------------

    def test_the_population_stays_the_size_it_was_drawn_at(self):
        # SELECTION_COUNT sets the turnover, not the size: n+1 in, n+1 out.
        for _generation in range(GENERATIONS):
            self.generation()
            self.assertEqual(len(self.population()), POPULATION)

    def test_every_chromosome_in_the_table_always_decodes(self):
        # There is no repair step anywhere in the search, so a chromosome that
        # cannot be read back is a bug rather than a result.
        for _generation in range(GENERATIONS):
            self.generation()
            for row in self.population():
                gp.check(row["chromosome"])

    def test_the_best_never_goes_backwards(self):
        # What elitism is for: the elite is not culled and not mutated, so it
        # arrives in the next generation with the same chromosome and earns
        # the same score. A search that can go backwards wastes the
        # generations it spent climbing.
        best = [self.best_fitness(snapshot) for snapshot in self.search()]
        for earlier, later in zip(best, best[1:]):
            self.assertGreaterEqual(later, earlier,
                                    "the search lost its best individual")

    def test_the_search_climbs(self):
        snapshots = self.search()
        self.assertGreater(self.best_fitness(snapshots[-1]),
                           self.best_fitness(snapshots[0]),
                           "selection and mutation together found nothing")
        self.assertGreater(self.mean_fitness(snapshots[-1]),
                           self.mean_fitness(snapshots[0]),
                           "the population as a whole did not improve")

    def test_it_reaches_the_target_it_is_pointed_at(self):
        # The judge's answer is 1.0 and it is reachable by variable swaps
        # alone; a search that works gets there.
        snapshots = self.search(20)
        self.assertEqual(self.best_fitness(snapshots[-1]), 1.0)

    def test_one_generation_is_recorded_per_turn(self):
        snapshots = self.search(6)
        self.assertEqual([snapshot.generation for snapshot in snapshots],
                         [1, 2, 3, 4, 5, 6])
        self.assertEqual([row["generation"] for row in
                          store.fitness_by_generation(self.conn, self.run)],
                         [1, 2, 3, 4, 5, 6])

    def test_the_history_holds_the_whole_population_each_generation(self):
        self.search(5)
        for row in store.fitness_by_generation(self.conn, self.run):
            self.assertEqual(row["individuals"], POPULATION)

    def test_a_sweep_holds_at_most_one_elite_throughout(self):
        for _generation in range(GENERATIONS):
            self.generation()
            self.elite()        # asserts at most one itself

    def test_the_elite_survives_the_generation_intact(self):
        # The individual elected in a generation is the one that generation
        # has to carry forward: selection may not cull it and mutation may not
        # touch it. (Last generation's elite has no such protection -- the
        # election is what confers it, and it is re-run every round.)
        for _generation in range(GENERATIONS):
            self.judge()
            calculate_fitness.assign(self.conn, self.run)
            elected, _rows = elitism.elect(self.conn, self.run)
            self.assertIsNotNone(elected, "nothing scored, so this proves nothing")
            number, chromosome = elected["number"], elected["chromosome"]

            selection.select(self.conn, self.run, SELECTION_COUNT,
                             self.selection_rng, CONF)
            mutation.apply(self.conn, self.run, MUTATION_RATE,
                           self.mutation_rng)

            survivor = self.individual(number)
            self.assertIsNotNone(survivor, "the elite was culled")
            self.assertEqual(survivor["chromosome"], chromosome,
                             "the elite was mutated")
            self.assertEqual(survivor["has_changed"], 0)

    def test_the_numbers_keep_rising(self):
        # The sweep's clock: selection appends before it culls and only ever
        # culls rows that were there before the round, so the high-water mark
        # can only go up -- which is what dates a generation.
        watermark = store.high_number(self.conn, self.run)
        for _generation in range(GENERATIONS):
            self.generation()
            now = store.high_number(self.conn, self.run)
            self.assertGreater(now, watermark)
            watermark = now

    def test_a_number_is_never_reused(self):
        seen = set(self.numbers())
        for _generation in range(GENERATIONS):
            self.generation()
            current = set(self.numbers())
            self.assertEqual(current & seen - current, set())
            seen |= current
        self.assertEqual(len(seen), POPULATION + GENERATIONS * (SELECTION_COUNT + 1))

    def test_every_generation_brings_in_one_stranger(self):
        # The floor under a search that culls and copies: without it the gene
        # pool can only narrow.
        before = {row["chromosome"] for row in self.population()}
        self.judge()
        calculate_fitness.assign(self.conn, self.run)
        elitism.elect(self.conn, self.run)
        result = selection.select(self.conn, self.run, SELECTION_COUNT,
                                  self.selection_rng, CONF)
        self.assertIsNotNone(result.newcomer)
        self.assertNotIn(result.newcomer.chromosome, before)

    def test_a_culled_individual_leaves_the_history_alone(self):
        self.search(5)
        history = store.fitness_history(self.conn, self.run, generation=1)
        living = set(self.numbers())
        self.assertEqual(len(history), POPULATION)
        self.assertTrue(set(row["number"] for row in history) - living,
                        "nothing was culled, so this proves nothing")

    def test_a_whole_search_is_reproducible(self):
        # Every generator is seeded from the sweep's own stored master seeds,
        # so the same database run again is the same search.
        first = [self.best_fitness(snapshot) for snapshot in self.search(8)]

        # A second sweep beside the first, in the same database, drawn and
        # driven from the same seeds.
        self.run = store.create_run(self.conn, "template_code_mocked.py")
        self.draw_population()
        second = [self.best_fitness(snapshot) for snapshot in self.search(8)]

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
