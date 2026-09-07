"""
test_selection.py - The wheel, the cull, and the one stranger.

The arithmetic is the thing worth pinning down: a round appends `n` copies and
one newcomer and culls `n + 1`, so a generation ends exactly the size it began
and SELECTION_COUNT is a knob on turnover rather than on size. Around that sit
the rules that keep the population coherent -- the elite is never culled, a
copy is its parent field for field but never inherits is_best, a culled number
is retired rather than reused, and a wheel of nothing but zero-width slices
writes nothing at all.
"""

import unittest
from unittest import mock

from search import generate_population as gp, selection
from storage import store

from unittests.search.support import CONF, VALID, SweepTestCase, rng


def rows(*pairs):
    """(number, fitness) pairs as wheel()/draw() read them."""
    return [{"number": number, "fitness": fitness, "is_best": 0,
             "chromosome": "CAT.L1.L2.w1.w2"}
            for number, fitness in pairs]


class WheelTests(unittest.TestCase):
    """Slices laid end to end."""

    def test_the_edges_are_cumulative(self):
        edges, total = selection.wheel(rows((1, 0.2), (2, 0.3), (3, 0.5)))
        self.assertEqual(edges, [0.2, 0.5, 1.0])
        self.assertEqual(total, 1.0)

    def test_a_zero_fitness_slice_has_no_width(self):
        edges, _total = selection.wheel(rows((1, 0.4), (2, 0.0), (3, 0.6)))
        self.assertEqual(edges[0], edges[1])

    def test_a_null_fitness_reads_as_zero(self):
        _edges, total = selection.wheel(rows((1, None), (2, 0.5)))
        self.assertEqual(total, 0.5)

    def test_a_negative_fitness_does_not_eat_its_neighbour(self):
        # Quality is 0..1 and cannot produce one, but the wheel should not
        # depend on that being true.
        edges, total = selection.wheel(rows((1, -5.0), (2, 0.5)))
        self.assertEqual((edges, total), ([0.0, 0.5], 0.5))

    def test_an_empty_population_has_no_wheel(self):
        self.assertEqual(selection.wheel([]), ([], 0.0))


class DrawTests(unittest.TestCase):
    """Spinning it."""

    def test_it_spins_as_many_times_as_asked(self):
        picks = selection.draw(rows((1, 0.5), (2, 0.5)), 7, rng(61))
        self.assertEqual(len(picks), 7)

    def test_zero_fitness_is_never_picked(self):
        # A slice as wide as 0.0 cannot be landed on. bisect_right is what
        # skips it rather than handing it the mark that lands on its edge.
        population = rows((1, 0.0), (2, 0.5), (3, 0.0), (4, 0.5), (5, 0.0))
        picks = selection.draw(population, 500, rng(62))
        self.assertEqual({pick["number"] for pick in picks}, {2, 4})

    def test_the_fitter_individual_is_picked_more_often(self):
        population = rows((1, 0.1), (2, 0.9))
        picks = selection.draw(population, 2000, rng(63))
        share = sum(pick["number"] == 2 for pick in picks) / len(picks)
        self.assertGreater(share, 0.8)
        self.assertLess(share, 0.98)

    def test_picks_are_with_replacement(self):
        # The mechanism, not a flaw in it: that is how a fit chromosome comes
        # to have several descendants.
        picks = selection.draw(rows((1, 1.0)), 4, rng(64))
        self.assertEqual([pick["number"] for pick in picks], [1, 1, 1, 1])

    def test_an_all_zero_wheel_picks_nobody(self):
        self.assertEqual(selection.draw(rows((1, 0.0), (2, 0.0)), 5, rng(65)), [])

    def test_an_empty_population_picks_nobody(self):
        self.assertEqual(selection.draw([], 5, rng(66)), [])

    def test_asking_for_nothing_picks_nobody(self):
        self.assertEqual(selection.draw(rows((1, 1.0)), 0, rng(67)), [])

    def test_it_is_reproducible(self):
        population = rows((1, 0.3), (2, 0.3), (3, 0.4))
        self.assertEqual([pick["number"] for pick in
                          selection.draw(population, 20, rng(68))],
                         [pick["number"] for pick in
                          selection.draw(population, 20, rng(68))])


class WeakestTests(unittest.TestCase):
    """Who the cull takes."""

    def test_the_lowest_fitness_goes_first(self):
        population = rows((1, 0.5), (2, 0.1), (3, 0.9), (4, 0.2))
        self.assertEqual([row["number"] for row in selection.weakest(population, 2)],
                         [2, 4])

    def test_a_tie_breaks_on_the_lowest_number(self):
        # So the longer-standing of two equally weak individuals goes, and a
        # re-run culls the same rows.
        population = rows((3, 0.0), (1, 0.0), (2, 0.0))
        self.assertEqual([row["number"] for row in selection.weakest(population, 2)],
                         [1, 2])

    def test_null_is_zero_here_too(self):
        population = rows((1, 0.5), (2, None), (3, 0.4))
        self.assertEqual([row["number"] for row in selection.weakest(population, 1)],
                         [2])

    def test_the_elite_is_never_eligible(self):
        population = rows((1, 0.9), (2, 0.0), (3, 0.1))
        population[1]["is_best"] = 1
        self.assertEqual([row["number"] for row in selection.weakest(population, 2)],
                         [3, 1])

    def test_it_gives_up_what_it_has_when_it_cannot_spare_them(self):
        # The one way a round leaves the population bigger than it found it.
        population = rows((1, 0.1), (2, 0.2))
        population[0]["is_best"] = 1
        self.assertEqual(len(selection.weakest(population, 5)), 1)

    def test_asking_for_nothing_takes_nothing(self):
        self.assertEqual(selection.weakest(rows((1, 0.1)), 0), [])
        self.assertEqual(selection.weakest(rows((1, 0.1)), -3), [])


class FreshTests(unittest.TestCase):
    """The one individual per round that is nobody's copy."""

    def test_it_draws_a_legal_chromosome(self):
        gp.check(selection.fresh(rows((1, 0.5)), rng(71), CONF))

    def test_it_avoids_what_the_population_already_holds(self):
        # A newcomer exists to bring the search something it does not have.
        held = [{"number": index, "fitness": 0.5, "is_best": 0,
                 "chromosome": chromosome}
                for index, chromosome in enumerate(VALID, 1)]
        generator = rng(72)
        for _ in range(30):
            self.assertNotIn(selection.fresh(held, generator, CONF),
                             {row["chromosome"] for row in held})

    def test_it_settles_for_a_duplicate_rather_than_failing_a_generation(self):
        # A converged search that can only find chromosomes it already holds
        # is telling you something, and a duplicate says it more usefully than
        # a failed generation would.
        held = rows((1, 0.5))       # holds CAT.L1.L2.w1.w2
        with mock.patch.object(gp, "build_population",
                               return_value=["CAT.L1.L2.w1.w2"]) as drawn:
            chromosome = selection.fresh(held, rng(73), CONF)
        self.assertEqual(chromosome, "CAT.L1.L2.w1.w2")
        self.assertEqual(drawn.call_count, selection.FRESH_ATTEMPTS)

    def test_without_unique_it_draws_once(self):
        with mock.patch.object(gp, "build_population",
                               return_value=["CAT.L1.L2.w1.w2"]) as drawn:
            selection.fresh(rows((1, 0.5)), rng(74),
                            dict(CONF, UNIQUE=False))
        self.assertEqual(drawn.call_count, 1)

    def test_it_is_drawn_under_the_sweep_settings(self):
        with mock.patch.object(gp, "build_population",
                               return_value=["CAT.L1.L2.w1.w2"]) as drawn:
            selection.fresh([], rng(75), CONF)
        _count, _generator, max_depth, branch_prob, unique = drawn.call_args[0]
        self.assertEqual((max_depth, branch_prob, unique),
                         (CONF["MAX_DEPTH"], CONF["BRANCH_PROB"], False))


class SelectTests(SweepTestCase):
    """One whole round against a stored sweep."""

    def setUp(self):
        super().setUp()
        self.populate(count=6)
        self.set_fitness({1: 0.0, 2: 0.1, 3: 0.2, 4: 0.3, 5: 0.4, 6: 0.5})

    def test_a_generation_ends_the_size_it_began(self):
        # n + 1 in, n + 1 out. The cull is n+1 and not n because the round
        # appends the copies *and* the newcomer.
        store.mark_best(self.conn, self.run, 6)
        self.conn.commit()
        for _round in range(4):
            selection.select(self.conn, self.run, 2, rng(81), CONF)
            self.assertEqual(len(self.population()), 6)

    def test_the_turnover_is_what_the_count_sets(self):
        store.mark_best(self.conn, self.run, 6)
        self.conn.commit()
        before = set(self.numbers())
        result = selection.select(self.conn, self.run, 2, rng(82), CONF)
        self.assertEqual(len(result.parents), 2)
        self.assertEqual(len(result.numbers), 2)
        self.assertEqual(len(result.culled), 3)
        self.assertEqual(len(set(self.numbers()) - before), 3)  # 2 copies + 1 newcomer

    def test_the_copies_and_the_newcomer_are_appended_above_everyone(self):
        result = selection.select(self.conn, self.run, 2, rng(83), CONF)
        self.assertEqual(result.numbers, [7, 8])
        self.assertEqual(result.newcomer.number, 9)

    def test_a_culled_number_is_never_handed_out_again(self):
        # Every number an old script, execution, transcript or history row
        # refers to goes on meaning the individual it meant.
        selection.select(self.conn, self.run, 2, rng(84), CONF)
        seen = set(self.numbers())
        selection.select(self.conn, self.run, 2, rng(85), CONF)
        self.assertEqual(seen & set(self.numbers()) - seen, set())
        self.assertEqual(store.next_number(self.conn, self.run),
                         max(self.numbers()) + 1)

    def test_the_weakest_are_the_ones_that_go(self):
        store.mark_best(self.conn, self.run, 6)
        self.conn.commit()
        result = selection.select(self.conn, self.run, 2, rng(86), CONF)
        self.assertEqual([row["number"] for row in result.culled], [1, 2, 3])

    def test_the_elite_is_never_culled(self):
        # Even when it is the weakest thing in the population.
        store.mark_best(self.conn, self.run, 1)     # fitness 0.0
        self.conn.commit()
        result = selection.select(self.conn, self.run, 2, rng(87), CONF)
        self.assertNotIn(1, [row["number"] for row in result.culled])
        self.assertIn(1, self.numbers())
        self.assertEqual(self.elite(), 1)

    def test_a_cull_takes_the_individual_whole(self):
        self.run_individual(1, [0.0, 0.0])
        self.run_individual(6, [0.5, 0.5])
        self.assertEqual((self.execution_count(), self.exchange_count()), (2, 4))

        store.mark_best(self.conn, self.run, 6)
        self.conn.commit()
        selection.select(self.conn, self.run, 2, rng(88), CONF)

        # #1 was culled and took its transcripts with it; #6 kept its own.
        self.assertEqual((self.execution_count(), self.exchange_count()), (1, 2))

    def test_a_copy_is_its_parent_field_for_field(self):
        # Only id and number are its own -- it arrives already carrying the
        # result its parent earned.
        self.set_fitness({1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.9})
        store.mark_best(self.conn, self.run, 6)
        parent = self.individual(6)
        store.set_tree(self.conn, parent["id"], "a drawing")
        store.set_script(self.conn, parent["id"], "ok", 24, "run_006.py",
                         "print('hello')", 555)
        self.conn.commit()
        parent = self.individual(6)

        result = selection.select(self.conn, self.run, 1, rng(89), CONF)
        self.assertEqual([row["number"] for row in result.parents], [6])
        copy = self.individual(result.numbers[0])

        for column in ("chromosome", "tree", "state", "rank", "script_name",
                       "script_source", "weight_seed", "fitness"):
            self.assertEqual(copy[column], parent[column],
                             "a copy did not inherit %s" % column)

    def test_no_copy_inherits_is_best(self):
        # That flag picks an individual out of the population rather than
        # describing one, so it is not the parent's to hand on.
        self.set_fitness({1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.9})
        store.mark_best(self.conn, self.run, 6)
        self.conn.commit()
        selection.select(self.conn, self.run, 3, rng(90), CONF)
        self.assertEqual(self.elite(), 6)       # asserts at most one itself

    def test_the_newcomer_arrives_with_nothing(self):
        # A chromosome and no tree, script, seed or fitness -- exactly as a
        # member of the first population.
        result = selection.select(self.conn, self.run, 2, rng(91), CONF)
        newcomer = self.individual(result.newcomer.number)
        self.assertEqual(newcomer["chromosome"], result.newcomer.chromosome)
        gp.check(newcomer["chromosome"])
        for column in ("tree", "state", "rank", "script_name", "script_source",
                       "weight_seed"):
            self.assertIsNone(newcomer[column],
                              "a newcomer arrived carrying %s" % column)
        self.assertEqual(newcomer["fitness"], 0.0)
        self.assertEqual(newcomer["is_best"], 0)

    def test_the_newcomer_is_not_one_of_the_culled(self):
        # UNIQUE is honoured against the population going into the round,
        # culled ones included -- re-drawing what was just discarded is the
        # one draw that achieves nothing.
        result = selection.select(self.conn, self.run, 2, rng(92), CONF)
        self.assertNotIn(result.newcomer.chromosome,
                         {row["chromosome"] for row in result.population})

    def test_an_all_zero_population_writes_nothing(self):
        # A round that cannot say which individuals are fit cannot be trusted
        # to say which are weak.
        self.set_fitness({number: 0.0 for number in range(1, 7)})
        before = self.chromosomes()
        result = selection.select(self.conn, self.run, 2, rng(93), CONF)
        self.assertEqual((result.parents, result.numbers, result.culled), ([], [], []))
        self.assertIsNone(result.newcomer)
        self.assertEqual(self.chromosomes(), before)
        self.assertEqual(len(result.population), 6)

    def test_an_empty_population_writes_nothing(self):
        store.add_individuals(self.conn, self.run, [])
        result = selection.select(self.conn, self.run, 2, rng(94), CONF)
        self.assertIsNone(result.newcomer)
        self.assertEqual(self.numbers(), [])

    def test_asking_for_no_copies_does_nothing(self):
        result = selection.select(self.conn, self.run, 0, rng(95), CONF)
        self.assertIsNone(result.newcomer)
        self.assertEqual(len(self.population()), 6)

    def test_count_none_with_an_elite_grows_by_two(self):
        # As many copies as the population holds is one more than the cull can
        # take, and the elite is a second row it cannot touch.
        store.mark_best(self.conn, self.run, 6)
        self.conn.commit()
        selection.select(self.conn, self.run, None, rng(96), CONF)
        self.assertEqual(len(self.population()), 8)

    def test_count_none_without_an_elite_grows_by_one(self):
        selection.select(self.conn, self.run, None, rng(97), CONF)
        self.assertEqual(len(self.population()), 7)

    def test_the_round_is_not_idempotent(self):
        # Running it twice is two generations, not one done twice.
        store.mark_best(self.conn, self.run, 6)
        self.conn.commit()
        selection.select(self.conn, self.run, 2, rng(98), CONF)
        after_one = set(self.numbers())
        selection.select(self.conn, self.run, 2, rng(98), CONF)
        self.assertEqual(len(self.population()), 6)
        self.assertNotEqual(set(self.numbers()), after_one)

    def test_the_population_returned_is_the_one_before_the_round(self):
        result = selection.select(self.conn, self.run, 2, rng(99), CONF)
        self.assertEqual([row["number"] for row in result.population],
                         [1, 2, 3, 4, 5, 6])

    def test_the_round_is_reproducible_from_its_seed(self):
        other = store.create_run(self.conn, "template_code_mocked.py")
        store.add_individuals(self.conn, other,
                              [row["chromosome"] for row in self.population()])
        for number, fitness in ((1, 0.0), (2, 0.1), (3, 0.2),
                                (4, 0.3), (5, 0.4), (6, 0.5)):
            store.set_fitness(self.conn, other, number, fitness)
        self.conn.commit()

        first = selection.select(self.conn, self.run, 2, rng(100), CONF)
        second = selection.select(self.conn, other, 2, rng(100), CONF)
        self.assertEqual([row["number"] for row in first.parents],
                         [row["number"] for row in second.parents])
        self.assertEqual(first.newcomer.chromosome, second.newcomer.chromosome)

    def test_only_the_fit_are_copied(self):
        self.set_fitness({1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.9})
        result = selection.select(self.conn, self.run, 4, rng(101), CONF)
        self.assertEqual({row["number"] for row in result.parents}, {6})

    def test_a_parent_may_itself_be_culled(self):
        # Weak individuals do occasionally get picked, and nothing needs doing
        # about it: the copy carries everything the parent had to pass on. A
        # cull deep enough to take the whole population reaches the one row
        # the wheel could land on, which is the sharpest form of the case.
        self.set_fitness({1: 0.01, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0})
        result = selection.select(self.conn, self.run, 5, rng(102), CONF)
        self.assertEqual({row["number"] for row in result.parents}, {1})
        self.assertIn(1, [row["number"] for row in result.culled])
        self.assertNotIn(1, self.numbers())
        # The copies were appended before the cull, so they survive it whole.
        self.assertEqual(self.individual(result.numbers[0])["chromosome"],
                         result.parents[0]["chromosome"])
        self.assertEqual(len(self.population()), 6)

    def test_every_stored_chromosome_still_decodes(self):
        store.mark_best(self.conn, self.run, 6)
        self.conn.commit()
        for _round in range(3):
            selection.select(self.conn, self.run, 2, rng(103), CONF)
        for row in self.population():
            gp.check(row["chromosome"])


if __name__ == "__main__":
    unittest.main()
