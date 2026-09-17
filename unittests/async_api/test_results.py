"""
test_results.py - What a job's results say is best.
"""

import unittest

from async_api import results


def one(number, fitness, is_best=False, state="ok"):
    return {"number": number, "fitness": fitness, "is_best": is_best, "state": state,
            "chromosome": "CAT.L1.L2.w1.w2", "quality": fitness}


class BestTests(unittest.TestCase):

    def test_a_stale_elite_flag_does_not_win(self):
        # A finished sweep: elitism marked #4 a generation ago, fitness ran since.
        population = [one(4, 0.57, is_best=True), one(5, 0.86), one(7, 0.76)]
        self.assertEqual(results.best(population)["number"], 5)

    def test_ties_go_to_the_lowest_number(self):
        self.assertEqual(results.best([one(9, 0.5), one(3, 0.5)])["number"], 3)

    def test_bad_and_unscored_individuals_are_never_best(self):
        self.assertIsNone(results.best([one(1, 0.9, state="BAD"), one(2, None), one(3, 0.0)]))


if __name__ == "__main__":
    unittest.main()
