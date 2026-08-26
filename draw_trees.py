"""
draw_trees.py - Draw one individual as a tree.

Rebuilds a chromosome with the same decoder that generated it and lays it out
the way plan.txt draws them: the expression, a blank line, then one row per
level of the tree. main.py stores that drawing on the individual, so a sweep
carries a readable picture of every tree it grew.

    CAT.SVD.LIN.L1.L2.L3.L1.w3.w3.w2.w1

    CAT
    SVD.LIN
    L1.L2.L3.L1
    w3.w3.w2.w1

Reading the rows top to bottom, left to right gives the expression back, since
that level-order walk is exactly what the K-expression encodes.
"""

from generate_population import decode, levels


def draw(expression):
    """One expression -> the block of lines describing it."""
    root, used = decode(expression)
    lines = [expression, ""]
    lines.extend(".".join(row) for row in levels(root))
    tail = expression.split(".")[used:]
    if tail:
        lines.append("")
        lines.append("(unused tail: %s)" % ".".join(tail))
    return lines
