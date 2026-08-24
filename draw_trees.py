"""
draw_trees.py - Draw every individual in population.txt as a tree.

Reads the dotted K-expressions produced by generate_population.py, rebuilds
each tree with the same decoder that generated them, and writes one drawing
per individual to trees.txt, laid out the way plan.txt draws them: the
expression, a blank line, then one row per level of the tree.

    #2
    CAT.SVD.LIN.L1.L2.L3.L1.w3.w3.w2.w1

    CAT
    SVD.LIN
    L1.L2.L3.L1
    w3.w3.w2.w1

Reading the rows top to bottom, left to right gives the expression back, since
that level-order walk is exactly what the K-expression encodes.

Usage:
    python draw_trees.py                       # population.txt -> trees.txt
    python draw_trees.py --input other.txt --output other_trees.txt
"""

import argparse
import os

from generate_population import decode, levels

_HERE = os.path.dirname(os.path.abspath(__file__))


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


def main():
    parser = argparse.ArgumentParser(
        description="Draw each row of population.txt as a tree."
    )
    parser.add_argument("--input", default=os.path.join(_HERE, "tmp/population.txt"),
                        help="file of K-expressions, one per line (default population.txt)")
    parser.add_argument("--output", default=os.path.join(_HERE, "tmp/trees.txt"),
                        help="where to write the drawings (default trees.txt)")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as handle:
        expressions = [line.strip() for line in handle if line.strip()]
    if not expressions:
        raise SystemExit("%s has no expressions in it" % args.input)

    blocks = []
    bad = 0
    for number, expression in enumerate(expressions, 1):
        try:
            body = draw(expression)
        except ValueError as error:
            body = [expression, "", "!! cannot draw: %s" % error]
            bad += 1
        blocks.append("\n".join(["#%d" % number] + body))

    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write("\n\n\n".join(blocks) + "\n")

    print("drew %d trees from %s to %s" % (len(expressions), args.input, args.output))
    if bad:
        print("%d line(s) could not be drawn -- see the !! markers" % bad)


if __name__ == "__main__":
    main()
