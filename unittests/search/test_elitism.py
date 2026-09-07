"""
test_elitism.py - Naming the individual a generation carries forward.

A small step with a sharp contract: exactly one is_best in a sweep or none at
all, the rule is the stored fitness column and nothing else, ties break on the
lowest number so a re-run elects the same individual, and an all-zero
population elects nobody rather than dressing up an arbitrary pick as a result.
"""

import unittest

from search import elitism
from storage import store

from unittests.search.support import SweepTestCase


def rows(*pairs):
    """(number, fitness) pairs as the mapping-ish rows best_of() reads."""
    return [{"number": number, "fitness": fitness} for number, fitness in pairs]


class BestOfTests(unittest.TestCase):
    """The rule, away from the database."""

    def test_the_highest_fitness_wins(self):
        self.assertEqual(elitism.best_of(rows((1, 0.2), (2, 0.9), (3, 0.4)))["number"],
                         2)

    def test_a_tie_breaks_on_the_lowest_number(self):
        # Arbitrary but fixed, so a sweep re-run over the same database elects
        # the same individual rather than a different one each time.
        self.assertEqual(elitism.best_of(rows((5, 0.7), (2, 0.7), (9, 0.7)))["number"],
                         2)

    def test_an_all_zero_population_has_no_elite(self):
        self.assertIsNone(elitism.best_of(rows((1, 0.0), (2, 0.0))))

    def test_a_null_fitness_reads_as_zero(self):
        self.assertIsNone(elitism.best_of(rows((1, None), (2, None))))

    def test_a_null_fitness_loses_to_a_real_one(self):
        self.assertEqual(elitism.best_of(rows((1, None), (2, 0.1)))["number"], 2)

    def test_an_empty_population_has_no_elite(self):
        self.assertIsNone(elitism.best_of([]))

    def test_a_single_scored_individual_is_the_elite(self):
        self.assertEqual(elitism.best_of(rows((7, 0.05)))["number"], 7)


class ElectTests(SweepTestCase):
    """The step against a stored sweep."""

    def setUp(self):
        super().setUp()
        self.populate(count=5)

    def test_it_marks_exactly_one(self):
        self.set_fitness({1: 0.1, 2: 0.9, 3: 0.4, 4: 0.0, 5: 0.6})
        best, rows_back = elitism.elect(self.conn, self.run)
        self.assertEqual(best["number"], 2)
        self.assertEqual(self.elite(), 2)       # asserts at most one itself
        self.assertEqual(len(rows_back), 5)

    def test_it_clears_last_generation_elite(self):
        # One statement, so a sweep never has a window with two of them.
        self.set_fitness({1: 0.9, 2: 0.1, 3: 0.0, 4: 0.0, 5: 0.0})
        elitism.elect(self.conn, self.run)
        self.assertEqual(self.elite(), 1)

        self.set_fitness({1: 0.1, 2: 0.9, 3: 0.0, 4: 0.0, 5: 0.0})
        elitism.elect(self.conn, self.run)
        self.assertEqual(self.elite(), 2)

    def test_re_electing_is_stable(self):
        self.set_fitness({1: 0.3, 2: 0.3, 3: 0.3, 4: 0.3, 5: 0.3})
        first, _ = elitism.elect(self.conn, self.run)
        second, _ = elitism.elect(self.conn, self.run)
        self.assertEqual(first["number"], second["number"])

    def test_an_all_zero_population_writes_nothing(self):
        # Either the fitness step never ran or nothing scored; neither has a
        # best worth carrying forward.
        self.set_fitness({number: 0.0 for number in range(1, 6)})
        best, _ = elitism.elect(self.conn, self.run)
        self.assertIsNone(best)
        self.assertIsNone(self.elite())

    def test_an_all_zero_population_keeps_the_elite_it_already_had(self):
        self.set_fitness({1: 0.0, 2: 0.5, 3: 0.0, 4: 0.0, 5: 0.0})
        elitism.elect(self.conn, self.run)
        self.set_fitness({number: 0.0 for number in range(1, 6)})
        elitism.elect(self.conn, self.run)
        self.assertEqual(self.elite(), 2,
                         "a sweep keeps whatever is_best it had rather than "
                         "gaining a meaningless one")

    def test_a_cleared_fitness_is_not_electable(self):
        # Mutation clears a mutant's fitness to NULL; until it is judged again
        # it must not be able to win.
        self.set_fitness({1: 0.5, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0})
        store.set_chromosome(self.conn, self.individual(1)["id"],
                             "CAT.L1.L2.w1.w2")
        self.conn.commit()
        best, _ = elitism.elect(self.conn, self.run)
        self.assertIsNone(best)

    def test_it_reads_the_column_and_not_the_transcripts(self):
        # One definition of "best", living in calculate_fitness.py: a
        # transcript that disagrees with the stored column does not win.
        self.run_individual(3, [1.0, 1.0])
        self.set_fitness({1: 0.0, 2: 0.6, 3: 0.0, 4: 0.0, 5: 0.0})
        best, _ = elitism.elect(self.conn, self.run)
        self.assertEqual(best["number"], 2)

    def test_the_population_comes_back_in_number_order(self):
        self.set_fitness({1: 0.1, 2: 0.9, 3: 0.4, 4: 0.2, 5: 0.6})
        _best, rows_back = elitism.elect(self.conn, self.run)
        self.assertEqual([row["number"] for row in rows_back], [1, 2, 3, 4, 5])

    def test_an_empty_population_elects_nobody(self):
        store.add_individuals(self.conn, self.run, [])
        best, rows_back = elitism.elect(self.conn, self.run)
        self.assertIsNone(best)
        self.assertEqual(list(rows_back), [])


if __name__ == "__main__":
    unittest.main()
