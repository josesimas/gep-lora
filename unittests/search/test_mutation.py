"""
test_mutation.py - Point mutation, and the two rules that make it safe.

Two claims are worth holding down here.

The grammatical one: a swap is class-local, so a mutated chromosome has the
same shape as the one it came from and still decodes. There is no repair step
in the search, so a mutation that could produce a broken tree would produce a
broken sweep.

The bookkeeping one: has_changed is written for every individual every round,
the elite is passed over entirely, and an individual that actually moved has
its fitness cleared to NULL -- because that score was earned by the chromosome
that has just been replaced.
"""

import unittest

from search import generate_population as gp, mutation

from unittests.search.support import VALID, SweepTestCase, rng


ALWAYS = 1.0        # every symbol takes its chance
NEVER = 0.0         # none of them do


def classes(chromosome):
    """Each symbol as the family it belongs to, so two chromosomes can be
    compared on shape alone."""
    families = {"binary": gp.BINARY_OPS, "unary": gp.UNARY_OPS,
                "variable": gp.VARIABLES}
    out = []
    for symbol in chromosome.split("."):
        out.append(next(name for name, family in families.items()
                        if symbol in family))
    return out


class AlternativesTests(unittest.TestCase):
    """What a symbol may legally become."""

    def test_the_root_is_never_touched(self):
        self.assertEqual(mutation.alternatives("CAT", 0), ())

    def test_a_binary_op_becomes_another_binary_op(self):
        self.assertEqual(set(mutation.alternatives("CAT", 3)), {"SVD", "LIN"})

    def test_a_unary_op_becomes_another_unary_op(self):
        self.assertEqual(set(mutation.alternatives("L2", 1)),
                         {"L1", "L3", "L4", "L5"})

    def test_a_variable_becomes_another_variable(self):
        self.assertEqual(set(mutation.alternatives("w4", 7)),
                         {"w1", "w2", "w3", "w5"})

    def test_a_symbol_is_never_its_own_alternative(self):
        for symbol in gp.BINARY_OPS + gp.UNARY_OPS + gp.VARIABLES:
            self.assertNotIn(symbol, mutation.alternatives(symbol, 1))

    def test_the_alternatives_are_the_rest_of_the_class(self):
        for family in (gp.BINARY_OPS, gp.UNARY_OPS, gp.VARIABLES):
            for symbol in family:
                self.assertEqual(set(mutation.alternatives(symbol, 1)),
                                 set(family) - {symbol})

    def test_a_symbol_outside_the_alphabet_has_none(self):
        self.assertEqual(mutation.alternatives("L9", 1), ())


class MutateTests(unittest.TestCase):
    """One chromosome through the dice."""

    def test_rate_zero_changes_nothing(self):
        for chromosome in VALID:
            self.assertEqual(mutation.mutate(chromosome, NEVER, rng(41)),
                             chromosome)

    def test_rate_one_changes_every_symbol_but_the_root(self):
        for chromosome in VALID:
            mutated = mutation.mutate(chromosome, ALWAYS, rng(42))
            before, after = chromosome.split("."), mutated.split(".")
            self.assertEqual(after[0], "CAT")
            for position, (one, other) in enumerate(zip(before, after)):
                if position == 0:
                    self.assertEqual(one, other)
                else:
                    self.assertNotEqual(one, other,
                                        "position %d did not move" % position)

    def test_the_root_survives_any_rate(self):
        generator = rng(43)
        for _ in range(200):
            mutated = mutation.mutate(generator.choice(VALID), ALWAYS, generator)
            self.assertTrue(mutated.startswith("CAT."))

    def test_the_result_always_decodes(self):
        # The guarantee, not a hope: mutate() runs check() itself, so this
        # fails loudly rather than storing something unreadable.
        generator = rng(44)
        for _ in range(400):
            chromosome = generator.choice(VALID)
            gp.check(mutation.mutate(chromosome, generator.random(), generator))

    def test_the_shape_is_preserved(self):
        # Class-local swaps are the whole trick: same length, same arity at
        # every position, so the tree comes out the shape it went in.
        generator = rng(45)
        for _ in range(400):
            chromosome = generator.choice(VALID)
            mutated = mutation.mutate(chromosome, generator.random(), generator)
            self.assertEqual(len(mutated.split(".")), len(chromosome.split(".")))
            self.assertEqual(classes(mutated), classes(chromosome))
            self.assertEqual(gp.levels(gp.decode(mutated)[0]).__len__(),
                             gp.levels(gp.decode(chromosome)[0]).__len__())

    def test_a_middling_rate_moves_some_and_not_others(self):
        # A rate of 0.1 over eleven symbols expects about one change, and any
        # individual may come through untouched -- both must be reachable.
        generator = rng(46)
        chromosome = "CAT.SVD.LIN.L1.L2.L3.L1.w3.w3.w2.w1"
        results = [mutation.mutate(chromosome, 0.1, generator)
                   for _ in range(200)]
        self.assertTrue(any(result == chromosome for result in results))
        self.assertTrue(any(result != chromosome for result in results))

    def test_it_is_reproducible(self):
        self.assertEqual(mutation.mutate(VALID[8], 0.4, rng(47)),
                         mutation.mutate(VALID[8], 0.4, rng(47)))

    def test_a_CAT_can_become_a_LIN(self):
        # Expected, and not this step's problem: a LIN over mismatched ranks is
        # a legal chromosome describing a blend PEFT will not build, and the
        # runs step culls it as BAD.
        generator = rng(48)
        results = {mutation.mutate("CAT.CAT.L5.L3.L4.w3.w3.w2", ALWAYS, generator)
                   .split(".")[1] for _ in range(50)}
        self.assertIn("LIN", results)


class DifferencesTests(unittest.TestCase):

    def test_it_counts_the_symbols_that_moved(self):
        self.assertEqual(mutation.differences("CAT.L1.L2.w1.w2",
                                              "CAT.L3.L2.w1.w5"), 2)

    def test_identical_chromosomes_differ_nowhere(self):
        self.assertEqual(mutation.differences(VALID[0], VALID[0]), 0)


class ApplyTests(SweepTestCase):
    """A whole population through one round."""

    def setUp(self):
        super().setUp()
        self.drawn = self.populate(count=5)
        self.set_fitness({1: 0.1, 2: 0.9, 3: 0.4, 4: 0.0, 5: 0.6})

    def test_nothing_moves_at_rate_zero(self):
        changes, rows = mutation.apply(self.conn, self.run, NEVER, rng(51))
        self.assertEqual(changes, [])
        self.assertEqual(len(rows), 5)
        self.assertEqual(list(self.chromosomes().values()), self.drawn)

    def test_has_changed_is_written_for_everyone(self):
        # This round's answer, not a running total: every individual is
        # written each time, including the ones the dice passed over.
        for row in self.population():
            self.conn.execute("UPDATE individuals SET has_changed = 1 WHERE id = ?",
                              (row["id"],))
        self.conn.commit()
        mutation.apply(self.conn, self.run, NEVER, rng(52))
        self.assertEqual({row["has_changed"] for row in self.population()}, {0})

    def test_a_moved_individual_is_flagged_and_loses_its_fitness(self):
        mutation.apply(self.conn, self.run, ALWAYS, rng(53))
        for row in self.population():
            self.assertEqual(row["has_changed"], 1)
            self.assertIsNone(row["fitness"],
                              "a mutant keeps a score its chromosome no longer earned")

    def test_an_unmoved_individual_keeps_its_fitness(self):
        mutation.apply(self.conn, self.run, NEVER, rng(54))
        self.assertEqual(self.fitnesses(),
                         {1: 0.1, 2: 0.9, 3: 0.4, 4: 0.0, 5: 0.6})

    def test_the_elite_is_read_and_not_written(self):
        from storage import store
        store.mark_best(self.conn, self.run, 2)
        self.conn.commit()
        before = self.individual(2)["chromosome"]

        mutation.apply(self.conn, self.run, ALWAYS, rng(55))

        elite = self.individual(2)
        self.assertEqual(elite["chromosome"], before,
                         "mutating the elite throws away what elitism protects")
        self.assertEqual(elite["has_changed"], 0)
        self.assertEqual(elite["fitness"], 0.9, "the elite keeps its score")
        self.assertEqual(elite["is_best"], 1)

    def test_every_other_individual_still_moves_around_the_elite(self):
        from storage import store
        store.mark_best(self.conn, self.run, 2)
        self.conn.commit()
        changes, _rows = mutation.apply(self.conn, self.run, ALWAYS, rng(56))
        self.assertEqual(sorted(change.number for change in changes),
                         [1, 3, 4, 5])

    def test_the_changes_describe_what_happened(self):
        changes, rows = mutation.apply(self.conn, self.run, ALWAYS, rng(57))
        self.assertEqual(len(changes), 5)
        held = self.chromosomes()
        for change in changes:
            self.assertNotEqual(change.before, change.after)
            self.assertEqual(held[change.number], change.after)
            self.assertEqual(change.symbols,
                             mutation.differences(change.before, change.after))
            self.assertEqual(change.symbols, len(change.before.split(".")) - 1)

    def test_the_rows_returned_are_the_population_before_the_round(self):
        _changes, rows = mutation.apply(self.conn, self.run, ALWAYS, rng(58))
        self.assertEqual([row["chromosome"] for row in rows], self.drawn)

    def test_every_stored_mutant_still_decodes(self):
        mutation.apply(self.conn, self.run, ALWAYS, rng(59))
        for row in self.population():
            gp.check(row["chromosome"])

    def test_an_empty_population_is_not_an_error(self):
        from storage import store
        store.add_individuals(self.conn, self.run, [])
        changes, rows = mutation.apply(self.conn, self.run, ALWAYS, rng(60))
        self.assertEqual((changes, list(rows)), ([], []))


if __name__ == "__main__":
    unittest.main()
