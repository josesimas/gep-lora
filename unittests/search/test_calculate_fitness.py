"""
test_calculate_fitness.py - A judged transcript folded into one number.

Three things this step promises, and all three matter to the steps after it:

  * fitness is the mean quality over the individual's *most recent* execution,
    and 0.0 -- never NULL -- when there is nothing to average;
  * the same numbers go into fitness_history, carrying the chromosome and state
    as they were then rather than as they are now;
  * which generation that snapshot belongs to is derived from the highest
    individual number the population holds, so re-running the step restates a
    generation instead of inventing one.
"""

import unittest

from search import calculate_fitness
from storage import store

from unittests.search.support import SweepTestCase


class FitnessOfTests(unittest.TestCase):
    """The one row -> one number rule, on its own."""

    def test_a_mean_is_kept(self):
        self.assertEqual(calculate_fitness.fitness_of({"quality": 0.75}), 0.75)

    def test_nothing_to_average_is_zero_and_not_none(self):
        # A missing score would otherwise have to be given a meaning at every
        # call site; it is given one here, once.
        self.assertEqual(calculate_fitness.fitness_of({"quality": None}), 0.0)

    def test_it_rounds_to_six_places(self):
        self.assertEqual(calculate_fitness.fitness_of({"quality": 1 / 3}),
                         0.333333)


class EntryForTests(unittest.TestCase):
    """One history row, built from one view row."""

    def test_it_carries_the_chromosome_and_state_of_the_moment(self):
        entry = calculate_fitness.entry_for(
            {"number": 4, "chromosome": "CAT.L1.L2.w1.w2", "state": "ok",
             "quality": 0.5, "answers": 3, "unscored": 1})
        self.assertEqual(entry, {"number": 4, "chromosome": "CAT.L1.L2.w1.w2",
                                 "state": "ok", "fitness": 0.5,
                                 "answers": 3, "unscored": 1})

    def test_missing_counts_become_zero(self):
        entry = calculate_fitness.entry_for(
            {"number": 1, "chromosome": "CAT.L1.L2.w1.w2", "state": None,
             "quality": None, "answers": None, "unscored": None})
        self.assertEqual((entry["answers"], entry["unscored"], entry["fitness"]),
                         (0, 0, 0.0))


class AssignTests(SweepTestCase):
    """The step over a whole sweep."""

    def setUp(self):
        super().setUp()
        self.populate(count=4)

    def test_fitness_is_the_mean_of_the_transcript(self):
        self.run_individual(1, [1.0, 0.5, 0.0])
        calculate_fitness.assign(self.conn, self.run)
        self.assertEqual(self.individual(1)["fitness"], 0.5)

    def test_an_individual_that_never_ran_scores_zero(self):
        self.run_individual(1, [1.0])
        calculate_fitness.assign(self.conn, self.run)
        for number in (2, 3, 4):
            self.assertEqual(self.individual(number)["fitness"], 0.0,
                             "a missing score has to be a number, not NULL")

    def test_an_execution_with_no_answers_scores_zero(self):
        self.run_individual(1, [], verdict="exit 1", exit_code=1)
        calculate_fitness.assign(self.conn, self.run)
        self.assertEqual(self.individual(1)["fitness"], 0.0)

    def test_an_unjudged_transcript_scores_zero(self):
        self.run_individual(1, [None, None])
        calculate_fitness.assign(self.conn, self.run)
        self.assertEqual(self.individual(1)["fitness"], 0.0)

    def test_a_half_judged_transcript_averages_what_was_scored(self):
        # The view's AVG skips NULLs, so the mean is over the answers that
        # actually carry a quality -- a weaker claim, and reported as one.
        self.run_individual(1, [1.0, None, 0.0, None])
        snapshot = calculate_fitness.assign(self.conn, self.run)
        self.assertEqual(self.individual(1)["fitness"], 0.5)
        row = next(row for row in snapshot.rows if row["number"] == 1)
        self.assertEqual((row["answers"], row["unscored"]), (4, 2))

    def test_only_the_latest_execution_counts(self):
        # Running a chromosome again under a different weight seed is a new
        # result, not an amendment to the old one.
        self.run_individual(1, [0.1, 0.1])
        self.run_individual(1, [0.9, 0.9])
        calculate_fitness.assign(self.conn, self.run)
        self.assertEqual(self.individual(1)["fitness"], 0.9)

    def test_the_snapshot_comes_back_best_first(self):
        self.run_individual(1, [0.2])
        self.run_individual(2, [0.8])
        self.run_individual(3, [0.5])
        snapshot = calculate_fitness.assign(self.conn, self.run)
        self.assertEqual([row["number"] for row in snapshot.rows][:3], [2, 3, 1])

    def test_an_empty_population_is_not_a_generation(self):
        store.add_individuals(self.conn, self.run, [])
        snapshot = calculate_fitness.assign(self.conn, self.run)
        self.assertEqual((snapshot.generation, snapshot.recorded_at), (0, None))
        self.assertEqual(store.fitness_history(self.conn, self.run), [])


class HistoryTests(SweepTestCase):
    """fitness_history: what the sweep used to be."""

    def setUp(self):
        super().setUp()
        self.populate(count=3)
        self.run_individual(1, [0.4])
        self.run_individual(2, [0.8])

    def test_the_first_snapshot_is_generation_one(self):
        snapshot = calculate_fitness.assign(self.conn, self.run)
        self.assertEqual(snapshot.generation, 1)
        self.assertIsNotNone(snapshot.recorded_at)

    def test_it_records_one_row_per_individual(self):
        calculate_fitness.assign(self.conn, self.run)
        history = store.fitness_history(self.conn, self.run)
        self.assertEqual([row["number"] for row in history], [1, 2, 3])
        self.assertEqual({row["population"] for row in history}, {3})

    def test_a_generation_shares_one_timestamp(self):
        snapshot = calculate_fitness.assign(self.conn, self.run)
        stamps = {row["recorded_at"]
                  for row in store.fitness_history(self.conn, self.run)}
        self.assertEqual(stamps, {snapshot.recorded_at})

    def test_re_running_restates_the_same_generation(self):
        # Pure arithmetic over stored transcripts, and a reasonable thing to
        # redo after a re-scored evaluate -- so it must not invent a round.
        calculate_fitness.assign(self.conn, self.run)
        second = calculate_fitness.assign(self.conn, self.run)
        self.assertEqual(second.generation, 1)
        self.assertEqual(len(store.fitness_history(self.conn, self.run)), 3)

    def test_a_restatement_carries_the_newer_scores(self):
        calculate_fitness.assign(self.conn, self.run)
        self.run_individual(1, [1.0])          # re-run, better transcript
        calculate_fitness.assign(self.conn, self.run)
        history = store.fitness_history(self.conn, self.run, generation=1)
        self.assertEqual(next(row["fitness"] for row in history
                              if row["number"] == 1), 1.0)

    def test_a_higher_number_means_a_new_generation(self):
        # The watermark: selection always leaves the population on a higher
        # number than it found it, which is what dates a round.
        calculate_fitness.assign(self.conn, self.run)
        store.append_individual(self.conn, self.run, "CAT.L1.L2.w1.w2")
        second = calculate_fitness.assign(self.conn, self.run)
        self.assertEqual(second.generation, 2)
        self.assertEqual([row["generation"] for row in
                          store.fitness_by_generation(self.conn, self.run)],
                         [1, 2])

    def test_a_static_population_still_advances_the_generation(self):
        # The case that broke dating by size: append one, cull one, and the
        # population is the size it was but on a higher number.
        calculate_fitness.assign(self.conn, self.run)
        store.append_individual(self.conn, self.run, "CAT.L1.L2.w1.w2")
        store.delete_individuals(self.conn, self.run, [3])
        self.assertEqual(len(self.population()), 3)
        self.assertEqual(calculate_fitness.assign(self.conn, self.run).generation, 2)

    def test_a_history_row_keeps_the_chromosome_it_was_scored_under(self):
        # The individual goes on being rewritten; the history must not be
        # read through the population as it stands today.
        calculate_fitness.assign(self.conn, self.run)
        original = self.individual(2)["chromosome"]
        store.set_chromosome(self.conn, self.individual(2)["id"],
                             "CAT.L1.L2.w1.w2")
        self.conn.commit()
        history = store.fitness_history(self.conn, self.run, generation=1)
        self.assertEqual(next(row["chromosome"] for row in history
                              if row["number"] == 2), original)

    def test_a_culled_individual_stays_in_the_history_it_lived_through(self):
        calculate_fitness.assign(self.conn, self.run)
        store.delete_individuals(self.conn, self.run, [2])
        history = store.fitness_history(self.conn, self.run, generation=1)
        self.assertIn(2, [row["number"] for row in history])
        self.assertEqual(next(row["fitness"] for row in history
                              if row["number"] == 2), 0.8)

    def test_the_generation_summary_reads_back(self):
        calculate_fitness.assign(self.conn, self.run)
        row = store.fitness_by_generation(self.conn, self.run)[0]
        self.assertEqual((row["individuals"], row["scored"]), (3, 2))
        self.assertEqual((row["worst"], row["best"]), (0.0, 0.8))

    def test_the_best_of_a_generation_is_elitism_rule_over_the_history(self):
        calculate_fitness.assign(self.conn, self.run)
        best = store.best_of_generation(self.conn, self.run, 1)
        self.assertEqual(best["number"], 2)


if __name__ == "__main__":
    unittest.main()
