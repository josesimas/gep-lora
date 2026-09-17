"""
test_weight_mutation.py - Weight-only mutation, counted over the whole pool.

Three claims are held down here.

Only weights move: every operator and slot comes out where it went in, so the
blend keeps its shape and its adapters.

The count is over all the weights, not a chance per weight: round(rate * pool)
positions move, exactly, wherever they fall -- and the elite's weights are not
in the pool at all.

The bookkeeping: an individual that moved gets has_changed = 1 and a NULL
fitness, and this step never writes has_changed = 0 over what mutation set.
"""

import unittest

from search import generate_population as gp, weight_mutation as wm
from storage import store

from unittests.search.support import VALID, SweepTestCase, rng


class CountTests(unittest.TestCase):

    def test_ten_weights_at_a_tenth_is_one(self):
        self.assertEqual(wm.draw_count(0.1, 10), 1)

    def test_it_rounds_half_up(self):
        self.assertEqual(wm.draw_count(0.1, 15), 2)
        self.assertEqual(wm.draw_count(0.1, 14), 1)
        self.assertEqual(wm.draw_count(0.1, 4), 0)

    def test_it_never_exceeds_the_pool(self):
        self.assertEqual(wm.draw_count(1.0, 7), 7)

    def test_the_positions_are_the_w_symbols(self):
        self.assertEqual(wm.weight_positions("CAT.SVD.LIN.L1.L2.L3.L1.w3.w3.w2.w1"),
                         [7, 8, 9, 10])


class MutateTests(unittest.TestCase):

    def pool(self, chromosomes):
        return sum(len(wm.weight_positions(c)) for c in chromosomes)

    def test_exactly_the_drawn_count_moves(self):
        generator = rng(61)
        chromosomes = list(VALID)
        total = self.pool(chromosomes)
        for rate in (0.0, 0.1, 0.25, 0.5, 1.0):
            after = wm.mutate(chromosomes, rate, generator)
            moved = sum(wm.differences(b, a) for b, a in zip(chromosomes, after))
            self.assertEqual(moved, wm.draw_count(rate, total))

    def test_only_weights_move(self):
        generator = rng(62)
        for _ in range(100):
            after = wm.mutate(list(VALID), 1.0, generator)
            for before, mutated in zip(VALID, after):
                for one, other in zip(before.split("."), mutated.split(".")):
                    if one not in gp.VARIABLES:
                        self.assertEqual(one, other)
                    else:
                        self.assertIn(other, gp.VARIABLES)
                gp.check(mutated)

    def test_the_draw_spans_the_whole_pool(self):
        # One change over ten weights spread across two chromosomes can land
        # in either of them.
        generator = rng(63)
        pair = ["CAT.CAT.L5.L3.L4.w3.w3.w2", "CAT.CAT.L5.L3.L4.w3.w3.w2",
                "CAT.CAT.L5.L3.L4.w3.w3.w2"]
        landed = set()
        for _ in range(100):
            after = wm.mutate(pair, 0.1, generator)
            landed.update(i for i, (b, a) in enumerate(zip(pair, after)) if b != a)
        self.assertEqual(landed, {0, 1, 2})

    def test_it_is_reproducible(self):
        self.assertEqual(wm.mutate(list(VALID), 0.3, rng(64)),
                         wm.mutate(list(VALID), 0.3, rng(64)))


class ApplyTests(SweepTestCase):

    def setUp(self):
        super().setUp()
        self.drawn = self.populate(count=5)
        self.set_fitness({1: 0.1, 2: 0.9, 3: 0.4, 4: 0.0, 5: 0.6})

    def test_the_elite_is_never_touched(self):
        store.mark_best(self.conn, self.run, 2)
        self.conn.commit()
        changes, _rows = wm.apply(self.conn, self.run, 1.0, rng(71))
        self.assertEqual(sorted(change.number for change in changes), [1, 3, 4, 5])
        elite = self.individual(2)
        self.assertEqual(elite["chromosome"], self.drawn[1])
        self.assertEqual(elite["fitness"], 0.9)
        self.assertEqual(elite["has_changed"], 0)

    def test_the_count_ignores_the_elites_weights(self):
        store.mark_best(self.conn, self.run, 2)
        self.conn.commit()
        rows = self.population()
        pool = wm.pool_size(rows)
        self.assertEqual(pool, sum(len(wm.weight_positions(c))
                                   for i, c in enumerate(self.drawn) if i != 1))
        changes, _rows = wm.apply(self.conn, self.run, 0.5, rng(72))
        self.assertEqual(sum(change.weights for change in changes),
                         wm.draw_count(0.5, pool))

    def test_a_moved_individual_is_flagged_and_loses_its_fitness(self):
        changes, _rows = wm.apply(self.conn, self.run, 1.0, rng(73))
        self.assertEqual(len(changes), 5)
        for row in self.population():
            self.assertEqual(row["has_changed"], 1)
            self.assertIsNone(row["fitness"])

    def test_has_changed_is_never_cleared(self):
        # Mutation ran first and moved everyone; a round here that moves
        # nobody must not undo that.
        for row in self.population():
            store.set_changed(self.conn, row["id"], 1)
        self.conn.commit()
        wm.apply(self.conn, self.run, 0.0, rng(74))
        self.assertEqual({row["has_changed"] for row in self.population()}, {1})

    def test_nothing_moves_at_rate_zero(self):
        changes, _rows = wm.apply(self.conn, self.run, 0.0, rng(75))
        self.assertEqual(changes, [])
        self.assertEqual(list(self.chromosomes().values()), self.drawn)
        self.assertEqual(self.fitnesses(), {1: 0.1, 2: 0.9, 3: 0.4, 4: 0.0, 5: 0.6})

    def test_an_empty_population_is_not_an_error(self):
        store.add_individuals(self.conn, self.run, [])
        changes, rows = wm.apply(self.conn, self.run, 1.0, rng(76))
        self.assertEqual((changes, list(rows)), ([], []))


if __name__ == "__main__":
    unittest.main()
