"""
test_draw_trees.py - The drawing a sweep stores beside every individual.

draw() is the one place a chromosome becomes something readable, and the
drawing is stored rather than derived on demand, so it wants to keep saying the
same thing: the expression, a blank line, then one row per level. Reading the
rows back top to bottom must give the expression again -- that is the whole
claim the level-order encoding makes.
"""

import unittest

from search import draw_trees, generate_population as gp

from unittests.search.support import VALID, rng


class DrawTests(unittest.TestCase):

    def test_the_canonical_drawing(self):
        self.assertEqual(
            draw_trees.draw("CAT.SVD.LIN.L1.L2.L3.L1.w3.w3.w2.w1"),
            ["CAT.SVD.LIN.L1.L2.L3.L1.w3.w3.w2.w1",
             "",
             "CAT",
             "SVD.LIN",
             "L1.L2.L3.L1",
             "w3.w3.w2.w1"])

    def test_it_leads_with_the_expression_and_a_blank(self):
        for chromosome in VALID:
            lines = draw_trees.draw(chromosome)
            self.assertEqual(lines[0], chromosome)
            self.assertEqual(lines[1], "")

    def test_the_rows_read_back_as_the_expression(self):
        # The point of a level-order encoding: the drawing is the chromosome
        # with newlines in it, and nothing else.
        for chromosome in VALID:
            rows = draw_trees.draw(chromosome)[2:]
            self.assertEqual(".".join(rows), chromosome)

    def test_one_row_per_level(self):
        for chromosome in VALID:
            root, _ = gp.decode(chromosome)
            self.assertEqual(len(draw_trees.draw(chromosome)) - 2,
                             len(gp.levels(root)))

    def test_an_unused_tail_is_shown_rather_than_hidden(self):
        lines = draw_trees.draw("CAT.L1.L2.w1.w2.w3.w4")
        self.assertEqual(lines[0], "CAT.L1.L2.w1.w2.w3.w4")
        self.assertIn("(unused tail: w3.w4)", lines)

    def test_a_whole_expression_has_no_tail_note(self):
        for chromosome in VALID:
            self.assertNotIn("unused tail", "\n".join(draw_trees.draw(chromosome)))

    def test_it_refuses_what_the_decoder_refuses(self):
        with self.assertRaises(ValueError):
            draw_trees.draw("SVD.L1.L2.w1.w2")

    def test_every_drawn_population_member_draws(self):
        for chromosome in gp.build_population(40, rng(31), 4, 0.5, True):
            rows = draw_trees.draw(chromosome)[2:]
            self.assertEqual(".".join(rows), chromosome)


if __name__ == "__main__":
    unittest.main()
