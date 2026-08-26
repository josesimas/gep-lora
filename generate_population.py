"""
generate_population.py - The alphabet, the encoding, and the random draw.

This is the root module of the pipeline: it owns the symbols an individual is
made of, the Node type, and the encode/decode pair everything else reads trees
with. main.py calls build_population() for a sweep's population and decode() for
each tree; there is no second parser anywhere.

Encoding
--------
Every individual is a Karva (K-)expression: the tree is written out in
level-order (breadth first, left to right), one symbol per position, joined
with dots. Reading it back is the same walk in reverse -- take the symbols in
order and hand each one out as the next child that is still missing.

    CAT.SVD.LIN.L1.L2.L3.L1.w3.w3.w2.w1

        CAT
        SVD.LIN
        L1.L2.L3.L1
        w3.w3.w2.w1

Grammar
-------
    CAT, SVD, LIN   arity 2, their children must be operators
    L1 .. L5        arity 1, their child must be a variable
    w1 .. w5        variables, the leaves of the tree

The first symbol is always CAT.
"""

from collections import deque


# --- the alphabet ---------------------------------------------------------

BINARY_OPS = ("CAT", "SVD", "LIN")           # arity 2, feed on other operators
UNARY_OPS = ("L1", "L2", "L3", "L4", "L5")   # arity 1, feed on variables
VARIABLES = ("w1", "w2", "w3", "w4", "w5")

ROOT = "CAT"

ARITY = {}
ARITY.update({op: 2 for op in BINARY_OPS})
ARITY.update({op: 1 for op in UNARY_OPS})
ARITY.update({var: 0 for var in VARIABLES})


def children_alphabet(symbol):
    """The symbols that are legal as a child of `symbol`."""
    if symbol in BINARY_OPS:
        return BINARY_OPS + UNARY_OPS
    if symbol in UNARY_OPS:
        return VARIABLES
    return ()


class Node:
    """One tree node: a symbol plus its ordered children."""

    __slots__ = ("symbol", "children")

    def __init__(self, symbol):
        self.symbol = symbol
        self.children = []


# --- growing a random tree ------------------------------------------------


def random_tree(rng, max_depth, branch_prob):
    """Grow a random valid tree rooted at CAT.

    `max_depth` is the deepest level an *operator* may sit at (the root is
    level 0), so a variable can appear at most one level below that. At that
    last operator level only arity-1 operators are drawn, which caps the tree.
    `branch_prob` is the chance that an operator above that level is arity 2,
    i.e. that the branch keeps growing instead of closing off with an L*.
    """
    root = Node(ROOT)
    pending = deque([(root, 0)])
    while pending:
        node, depth = pending.popleft()
        for _ in range(ARITY[node.symbol]):
            if node.symbol in UNARY_OPS:
                child = Node(rng.choice(VARIABLES))
            elif depth + 1 >= max_depth or rng.random() >= branch_prob:
                child = Node(rng.choice(UNARY_OPS))
            else:
                child = Node(rng.choice(BINARY_OPS))
            node.children.append(child)
            if ARITY[child.symbol]:
                pending.append((child, depth + 1))
    return root


# --- encoding / decoding --------------------------------------------------


def levels(root):
    """The tree as a list of levels, each a list of symbols."""
    rows = []
    frontier = [root]
    while frontier:
        rows.append([node.symbol for node in frontier])
        frontier = [child for node in frontier for child in node.children]
    return rows


def encode(root):
    """Tree -> dotted K-expression (level-order walk)."""
    return ".".join(symbol for row in levels(root) for symbol in row)


def decode(expression):
    """Dotted K-expression -> (tree, number of symbols the tree used).

    Raises ValueError on anything that breaks the grammar. Symbols left over
    once the tree is complete are the unused tail, and are reported through
    the returned count rather than silently dropped.
    """
    tokens = expression.split(".")
    if not tokens or tokens[0] != ROOT:
        raise ValueError("expression must start with %s" % ROOT)

    root = Node(tokens[0])
    pending = deque([root])
    used = 1
    while pending:
        node = pending.popleft()
        allowed = children_alphabet(node.symbol)
        for _ in range(ARITY[node.symbol]):
            if used >= len(tokens):
                raise ValueError("expression ends while %s still needs a child" % node.symbol)
            symbol = tokens[used]
            used += 1
            if symbol not in allowed:
                raise ValueError("%s is not a legal child of %s" % (symbol, node.symbol))
            child = Node(symbol)
            node.children.append(child)
            if ARITY[symbol]:
                pending.append(child)
    return root, used


def check(expression):
    """Decode `expression` and make sure it round-trips exactly."""
    total = len(expression.split("."))
    root, used = decode(expression)
    if used != total:
        raise ValueError("expression has %d unused trailing symbols" % (total - used))
    if encode(root) != expression:
        raise ValueError("expression does not round-trip")
    return root


# --- the population --------------------------------------------------------


def build_population(count, rng, max_depth, branch_prob, unique):
    """Generate `count` validated expressions."""
    population = []
    seen = set()
    attempts = 0
    attempt_budget = count * 100
    while len(population) < count:
        attempts += 1
        if attempts > attempt_budget:
            raise RuntimeError(
                "could only find %d unique expressions out of %d requested; "
                "raise MAX_DEPTH or BRANCH_PROB in settings.py, or clear UNIQUE"
                % (len(population), count)
            )
        depth = rng.randint(1, max_depth)
        expression = encode(random_tree(rng, depth, branch_prob))
        check(expression)   # never store something we cannot read back
        if unique:
            if expression in seen:
                continue
            seen.add(expression)
        population.append(expression)
    return population
