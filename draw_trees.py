"""
draw_trees.py - Draw every individual in population.txt as a tree.

Reads the dotted K-expressions produced by generate_population.py, rebuilds
each tree with the same decoder that generated them, and writes one drawing
per individual to trees.txt:

    #11  CAT.LIN.L2.L1.CAT.w3.w4.CAT.L5.L3.L2.w4.w4.w4
    CAT
    |-- LIN
    |   |-- L2
    |   |   `-- w3
    |   `-- L1
    |       `-- w4
    `-- CAT
        ...

Indentation shows parent -> child directly, which the flat level rows of the
K-expression cannot: in "CAT.L1.LIN.w1.SVD.L3" you have to count arities to
see that w1 belongs to L1 and not to LIN.

Usage:
    python draw_trees.py                       # population.txt -> trees.txt
    python draw_trees.py --unicode             # draw with box-drawing glyphs
    python draw_trees.py --input other.txt --output other_trees.txt
"""

import argparse
import os

from generate_population import decode, levels

_HERE = os.path.dirname(os.path.abspath(__file__))

# (branch, last branch, vertical run, empty run)
ASCII_GLYPHS = ("|-- ", "`-- ", "|   ", "    ")
UNICODE_GLYPHS = ("├── ", "└── ", "│   ", "    ")


def render(root, glyphs):
    """Tree -> list of drawn lines, root first."""
    tee, elbow, pipe, gap = glyphs
    lines = [root.symbol]

    def walk(node, prefix):
        last_index = len(node.children) - 1
        for index, child in enumerate(node.children):
            is_last = index == last_index
            lines.append(prefix + (elbow if is_last else tee) + child.symbol)
            walk(child, prefix + (gap if is_last else pipe))

    walk(root, "")
    return lines


def draw(expression, glyphs, with_levels):
    """One expression -> the block of lines describing it."""
    root, used = decode(expression)
    lines = render(root, glyphs)
    tail = expression.split(".")[used:]
    if tail:
        lines.append("(unused tail: %s)" % ".".join(tail))
    if with_levels:
        lines.append("")
        lines.append("levels:")
        for row in levels(root):
            lines.append("    " + ".".join(row))
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Draw each row of population.txt as a tree."
    )
    parser.add_argument("--input", default=os.path.join(_HERE, "population.txt"),
                        help="file of K-expressions, one per line (default population.txt)")
    parser.add_argument("--output", default=os.path.join(_HERE, "trees.txt"),
                        help="where to write the drawings (default trees.txt)")
    parser.add_argument("--unicode", action="store_true",
                        help="draw with box-drawing glyphs instead of ASCII")
    parser.add_argument("--levels", action="store_true",
                        help="also print the level rows under each drawing")
    args = parser.parse_args()

    glyphs = UNICODE_GLYPHS if args.unicode else ASCII_GLYPHS

    with open(args.input, encoding="utf-8") as handle:
        expressions = [line.strip() for line in handle if line.strip()]
    if not expressions:
        raise SystemExit("%s has no expressions in it" % args.input)

    width = len(str(len(expressions)))
    blocks = []
    bad = 0
    for number, expression in enumerate(expressions, 1):
        header = "#%s  %s" % (str(number).rjust(width), expression)
        try:
            body = draw(expression, glyphs, args.levels)
        except ValueError as error:
            body = ["!! cannot draw: %s" % error]
            bad += 1
        blocks.append("\n".join([header] + body))

    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write("\n\n".join(blocks) + "\n")

    print("drew %d trees from %s to %s" % (len(expressions), args.input, args.output))
    if bad:
        print("%d line(s) could not be drawn -- see the !! markers" % bad)


if __name__ == "__main__":
    main()
