"""
test_main.py - Where a stopped sweep got to, read back out of its database.

main.where_it_stopped() is what `main.py --resume` and `--evaluate` plan by.
It reads two things a driver writes as it goes -- the fitness snapshots and the
steps that finished -- so these tests write exactly those and nothing else: a
sweep here is the record of how far a search went, not a search.
"""

import unittest

import main
from storage import store

from unittests.search.support import SweepTestCase

GENERATION = ["trees", "runs", "process", "evaluate", "fitness"]
TAIL = ["elitism", "selection", "mutation", "weight_mutation"]


class WhereItStoppedTests(SweepTestCase):

    def ran(self, *steps):
        """Record `steps` as finished, one pass, in order -- taking a fitness
        snapshot wherever `fitness` is among them, as the step would."""
        pass_no = store.next_pass(self.conn, self.run)
        for position, step in enumerate(steps, 1):
            store.add_step_timing(self.conn, self.run, pass_no, position, step, 0.1)
            if step == "fitness":
                generation = len(store.fitness_by_generation(self.conn, self.run)) + 1
                store.record_fitness(self.conn, self.run, generation, [
                    {"number": 1, "chromosome": "CAT.L2.L1.w5.w3", "state": "ok",
                     "fitness": 0.5, "answers": 3, "unscored": 0}])
        self.conn.commit()

    def plan(self, generations=2):
        return main.where_it_stopped(self.conn, self.run, generations)

    def test_a_sweep_with_no_population_is_not_started(self):
        plan = self.plan()
        self.assertFalse(plan.populated)

    def test_stopped_in_the_first_generation_it_starts_that_one_again(self):
        self.populate()
        self.ran("population", "trees", "runs", "process")
        plan = self.plan()
        self.assertEqual(plan.rest, GENERATION + TAIL)
        self.assertEqual((plan.scored, plan.generations, plan.at_rest), (0, 2, False))

    def test_stopped_part_way_through_the_tail_it_finishes_the_tail(self):
        self.populate()
        self.ran("population", *GENERATION)
        self.ran("elitism", "selection")
        plan = self.plan()
        self.assertEqual(plan.rest, ["mutation", "weight_mutation"])
        self.assertEqual(plan.tail_done, ("elitism", "selection"))
        self.assertEqual((plan.scored, plan.generations, plan.at_rest), (1, 2, True))

    def test_resting_on_a_snapshot_it_breeds_the_next_generation(self):
        self.populate()
        self.ran("population", *GENERATION)
        plan = self.plan()
        self.assertEqual(plan.rest, TAIL)
        self.assertEqual(plan.tail_done, ())

    def test_part_way_into_a_later_generation_it_starts_that_one_again(self):
        self.populate()
        self.ran("population", *GENERATION, *TAIL)
        self.ran("trees", "runs", "process")
        plan = self.plan()
        self.assertEqual(plan.rest, GENERATION + TAIL)
        self.assertEqual((plan.scored, plan.generations), (1, 1))

    def test_the_last_generation_stops_after_fitness(self):
        self.populate()
        self.ran("population", *GENERATION, *TAIL)
        self.ran(*GENERATION, *TAIL)
        self.ran("trees")
        plan = self.plan()
        self.assertEqual(plan.rest, GENERATION)
        self.assertEqual(plan.generations, 0)

    def test_a_failed_step_is_not_a_finished_one(self):
        self.populate()
        self.ran("population", *GENERATION, "elitism")
        store.add_step_timing(self.conn, self.run, store.next_pass(self.conn, self.run),
                              1, "selection", 0.1, status="failed")
        self.conn.commit()
        self.assertEqual(self.plan().rest, ["selection", "mutation", "weight_mutation"])

    def test_a_search_with_every_generation_scored_is_complete(self):
        self.populate()
        self.ran("population", *GENERATION, *TAIL)
        self.ran(*GENERATION, *TAIL)
        self.ran(*GENERATION)
        plan = self.plan()
        self.assertTrue(plan.complete)
        self.assertEqual((plan.rest, plan.generations), ([], 0))

    def test_an_older_sweep_that_ended_in_mutation_says_it_was_bred(self):
        # A finished search used to run the tail after its last generation too;
        # an evaluation must not restate fitness over a population bred since.
        self.populate()
        self.ran("population", *GENERATION, *TAIL)
        plan = self.plan(generations=0)
        self.assertTrue(plan.complete)
        self.assertEqual(plan.tail_done, tuple(TAIL))


if __name__ == "__main__":
    unittest.main()
