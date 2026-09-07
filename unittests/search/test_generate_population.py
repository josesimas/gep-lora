"""
test_generate_population.py - The alphabet, the encoding, and the random draw.

This module is the one the rest of the search is built on: every other step
reads trees through decode() and writes them through encode(), and mutation's
whole safety argument is that a swap inside a symbol class leaves a tree that
still decodes. So the tests here are about the grammar holding, not about any
particular tree.
"""

import unittest

from search import generate_population as gp

from unittests.search.support import rng


# The chromosome plan.txt draws, and the one the docstrings all use.
CANONICAL = "CAT.SVD.LIN.L1.L2.L3.L1.w3.w3.w2.w1"

CANONICAL_LEVELS = [
    ["CAT"],
    ["SVD", "LIN"],
    ["L1", "L2", "L3", "L1"],
    ["w3", "w3", "w2", "w1"],
]


class AlphabetTests(unittest.TestCase):
    """The symbols, their arities, and what may sit under what."""

    def test_the_three_classes_do_not_overlap(self):
        families = (gp.BINARY_OPS, gp.UNARY_OPS, gp.VARIABLES)
        symbols = [symbol for family in families for symbol in family]
        self.assertEqual(len(symbols), len(set(symbols)))

    def test_arity_covers_every_symbol_and_nothing_else(self):
        expected = set(gp.BINARY_OPS) | set(gp.UNARY_OPS) | set(gp.VARIABLES)
        self.assertEqual(set(gp.ARITY), expected)

    def test_arity_is_the_class(self):
        # Mutation swaps inside a class precisely because arity follows the
        # class rather than the symbol; if that stops being true, the shape of
        # a mutated tree stops being guaranteed.
        self.assertEqual({gp.ARITY[symbol] for symbol in gp.BINARY_OPS}, {2})
        self.assertEqual({gp.ARITY[symbol] for symbol in gp.UNARY_OPS}, {1})
        self.assertEqual({gp.ARITY[symbol] for symbol in gp.VARIABLES}, {0})

    def test_the_root_is_a_binary_op(self):
        self.assertIn(gp.ROOT, gp.BINARY_OPS)

    def test_children_alphabet_follows_the_class_too(self):
        for symbol in gp.BINARY_OPS:
            self.assertEqual(gp.children_alphabet(symbol),
                             gp.BINARY_OPS + gp.UNARY_OPS)
        for symbol in gp.UNARY_OPS:
            self.assertEqual(gp.children_alphabet(symbol), gp.VARIABLES)
        for symbol in gp.VARIABLES:
            self.assertEqual(gp.children_alphabet(symbol), ())

    def test_an_unknown_symbol_has_no_children(self):
        self.assertEqual(gp.children_alphabet("L9"), ())


class DecodeTests(unittest.TestCase):
    """Reading a K-expression back."""

    def test_the_canonical_expression_decodes_whole(self):
        root, used = gp.decode(CANONICAL)
        self.assertEqual(root.symbol, "CAT")
        self.assertEqual(used, len(CANONICAL.split(".")))

    def test_levels_are_the_level_order_walk(self):
        root, _ = gp.decode(CANONICAL)
        self.assertEqual(gp.levels(root), CANONICAL_LEVELS)

    def test_encode_is_decode_backwards(self):
        root, _ = gp.decode(CANONICAL)
        self.assertEqual(gp.encode(root), CANONICAL)

    def test_the_shortest_legal_tree(self):
        root, used = gp.decode("CAT.L1.L2.w1.w2")
        self.assertEqual(used, 5)
        self.assertEqual(gp.levels(root), [["CAT"], ["L1", "L2"], ["w1", "w2"]])

    def test_a_trailing_tail_is_reported_not_dropped(self):
        # decode() stops when the tree is complete and says how much it used;
        # draw_trees shows the remainder and check() refuses it.
        root, used = gp.decode("CAT.L1.L2.w1.w2.w3.w4")
        self.assertEqual(used, 5)
        self.assertEqual(gp.encode(root), "CAT.L1.L2.w1.w2")

    def test_the_root_must_be_CAT(self):
        for expression in ("SVD.L1.L2.w1.w2", "L1.w1", "w1"):
            with self.assertRaises(ValueError):
                gp.decode(expression)

    def test_a_binary_op_may_not_take_a_variable(self):
        with self.assertRaises(ValueError):
            gp.decode("CAT.w1.w2")

    def test_a_unary_op_may_not_take_an_operator(self):
        with self.assertRaises(ValueError):
            gp.decode("CAT.L1.L2.L3.w1.w2")

    def test_an_expression_that_runs_out_is_refused(self):
        with self.assertRaises(ValueError):
            gp.decode("CAT.L1.L2.w1")

    def test_an_unknown_symbol_is_refused(self):
        with self.assertRaises(ValueError):
            gp.decode("CAT.L1.L9.w1.w2")


class CheckTests(unittest.TestCase):
    """check() is the guarantee every step leans on before it stores anything."""

    def test_a_whole_expression_passes(self):
        self.assertEqual(gp.check(CANONICAL).symbol, "CAT")

    def test_an_unused_tail_is_refused(self):
        with self.assertRaises(ValueError):
            gp.check("CAT.L1.L2.w1.w2.w3")

    def test_a_broken_expression_is_refused(self):
        with self.assertRaises(ValueError):
            gp.check("CAT.L1.L2.w1")


class RandomTreeTests(unittest.TestCase):
    """The draw: valid by construction, and capped where it says it is."""

    def walk(self, root, depth=0):
        """(node, depth) for every node, so a test can talk about levels."""
        yield root, depth
        for child in root.children:
            yield from self.walk(child, depth + 1)

    def test_every_drawn_tree_is_a_legal_chromosome(self):
        generator = rng(11)
        for _ in range(300):
            tree = gp.random_tree(generator, generator.randint(1, 5), 0.5)
            gp.check(gp.encode(tree))       # raises if it is not

    def test_the_root_is_always_CAT(self):
        generator = rng(12)
        for _ in range(100):
            self.assertEqual(gp.random_tree(generator, 3, 0.5).symbol, "CAT")

    def test_every_node_has_the_children_its_arity_asks_for(self):
        generator = rng(13)
        for _ in range(200):
            tree = gp.random_tree(generator, 4, 0.6)
            for node, _depth in self.walk(tree):
                self.assertEqual(len(node.children), gp.ARITY[node.symbol],
                                 "%s has the wrong number of children" % node.symbol)

    def test_every_child_is_legal_where_it_stands(self):
        generator = rng(14)
        for _ in range(200):
            tree = gp.random_tree(generator, 4, 0.6)
            for node, _depth in self.walk(tree):
                allowed = gp.children_alphabet(node.symbol)
                for child in node.children:
                    self.assertIn(child.symbol, allowed)

    def test_max_depth_caps_the_operators(self):
        # max_depth is the deepest level an *operator* may sit at, so a
        # variable can appear one level below it and no further.
        generator = rng(15)
        for max_depth in (1, 2, 3, 4):
            for _ in range(60):
                tree = gp.random_tree(generator, max_depth, 0.9)
                for node, depth in self.walk(tree):
                    if node.symbol in gp.VARIABLES:
                        self.assertLessEqual(depth, max_depth + 1)
                    else:
                        self.assertLessEqual(depth, max_depth)

    def test_a_branch_only_keeps_growing_above_the_cap(self):
        # Below max_depth the draw is forced to arity 1, which is what closes
        # the tree off; a binary op at or past the cap would run away.
        generator = rng(16)
        for max_depth in (1, 2, 3):
            for _ in range(60):
                tree = gp.random_tree(generator, max_depth, 0.9)
                for node, depth in self.walk(tree):
                    if node.symbol in gp.BINARY_OPS:
                        self.assertLess(depth, max(max_depth, 1))

    def test_branch_prob_zero_gives_the_shallowest_tree(self):
        tree = gp.random_tree(rng(17), 5, 0.0)
        self.assertEqual(len(gp.levels(tree)), 3)   # CAT, two L*, two variables

    def test_the_draw_is_reproducible(self):
        first = gp.encode(gp.random_tree(rng(99), 4, 0.5))
        second = gp.encode(gp.random_tree(rng(99), 4, 0.5))
        self.assertEqual(first, second)


class BuildPopulationTests(unittest.TestCase):
    """A whole population, which is what a sweep starts from."""

    def test_it_returns_the_count_asked_for(self):
        population = gp.build_population(12, rng(21), 4, 0.3, True)
        self.assertEqual(len(population), 12)

    def test_every_member_round_trips(self):
        for chromosome in gp.build_population(30, rng(22), 4, 0.4, False):
            gp.check(chromosome)

    def test_unique_gives_no_duplicates(self):
        population = gp.build_population(25, rng(23), 4, 0.5, True)
        self.assertEqual(len(population), len(set(population)))

    def test_without_unique_duplicates_are_allowed(self):
        # A shallow, never-branching draw has few trees to find, so the same
        # one comes up more than once -- which unique=False is meant to permit.
        population = gp.build_population(40, rng(24), 1, 0.0, False)
        self.assertLess(len(set(population)), len(population))

    def test_the_same_seed_gives_the_same_population(self):
        self.assertEqual(gp.build_population(10, rng(25), 4, 0.3, True),
                         gp.build_population(10, rng(25), 4, 0.3, True))

    def test_a_different_seed_gives_a_different_population(self):
        self.assertNotEqual(gp.build_population(10, rng(25), 4, 0.3, True),
                            gp.build_population(10, rng(26), 4, 0.3, True))

    def test_it_gives_up_rather_than_looping_forever(self):
        # Depth 1 with no branching has 5*5*5*5 trees in it; asking for more
        # unique ones than the budget can find has to raise, not spin.
        with self.assertRaises(RuntimeError) as caught:
            gp.build_population(700, rng(27), 1, 0.0, True)
        self.assertIn("unique", str(caught.exception))

    def test_it_never_exceeds_the_depth_it_was_given(self):
        # build_population draws its own depth per individual, capped by the
        # one it was handed, so no tree may be deeper than that cap allows.
        for chromosome in gp.build_population(40, rng(28), 3, 0.6, False):
            root, _ = gp.decode(chromosome)
            self.assertLessEqual(len(gp.levels(root)), 3 + 2)


if __name__ == "__main__":
    unittest.main()
