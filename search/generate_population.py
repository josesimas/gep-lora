"""
generate_population.py - The alphabet, the encoding, and the random draw.

This is the root module of the pipeline: it owns the symbols an individual is
made of, the Node type, and the encode/decode pair everything else reads trees
with. start_run.py calls build_population() for a sweep's population and decode() for
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
    L1 .. L10       arity 1, their child must be a variable
    w1 .. w10       variables, the leaves of the tree

The first symbol is the root, and may be any operator -- CAT, SVD, LIN, or
an L* on its own (`L3.w2`, one adapter and nothing folded into it). Only a
variable cannot start a chromosome: a bare weight is not a blend.

How many of the L*/w* a sweep may use is how many adapters its LORA_SLOTS
names -- L1..Ln, 1 <= n <= MAX_SLOTS, and w1..wn beside them. The alphabet
above is the ceiling, so decode() reads any sweep's chromosomes; the draws
(random_tree, build_population, and mutation's swaps) take the sweep's own
`slots` and never hand out a symbol it has no adapter for. A sweep stored
when the ceiling was five draws exactly what it drew then.
"""

from collections import deque


# --- the alphabet ---------------------------------------------------------

# The most adapters one blend can draw from. Raising it is this line: every
# other reader takes the alphabet from the tuples below.
MAX_SLOTS = 10

BINARY_OPS = ("CAT", "SVD", "LIN")                                  # arity 2, feed on other operators
UNARY_OPS = tuple("L%d" % n for n in range(1, MAX_SLOTS + 1))        # arity 1, feed on variables
VARIABLES = tuple("w%d" % n for n in range(1, MAX_SLOTS + 1))

# The symbols a chromosome may start with: any operator, never a variable.
ROOTS = BINARY_OPS + UNARY_OPS

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


def slot_symbols(slots=MAX_SLOTS):
    """The L* a sweep of `slots` adapters may draw: L1..L<slots>."""
    return UNARY_OPS[:slots]


def weight_symbols(slots=MAX_SLOTS):
    """The w* a sweep of `slots` adapters may draw: w1..w<slots>, one per slot."""
    return VARIABLES[:slots]


def slot_key(slot):
    """Sort key putting L2 before L10, where a plain sort would not."""
    return (len(slot), slot)


def slot_count(lora_slots):
    """How many adapters a LORA_SLOTS dict gives the search. -> n.

    Its keys must be exactly L1..Ln, 1 <= n <= MAX_SLOTS: a gap would be a slot
    the draw can hand out with no adapter behind it, and past the ceiling a slot
    no chromosome can name. Raises ValueError saying which.
    """
    names = sorted(lora_slots or (), key=slot_key)
    count = len(names)
    if not 1 <= count <= MAX_SLOTS:
        raise ValueError("LORA_SLOTS holds %d slot(s); a blend draws from 1 to %d"
                         % (count, MAX_SLOTS))
    if names != list(UNARY_OPS[:count]):
        raise ValueError("LORA_SLOTS must be named L1..L%d with no gaps, not %s"
                         % (count, ", ".join(names)))
    return count


def slots_of(conf):
    """How many slots a sweep's settings give it -- its stored LORA_SLOTS, or
    settings.py's when it stored none, the fallback generate_runs.lora_slots()
    takes too."""
    slots = (conf or {}).get("LORA_SLOTS")
    if slots is None:
        from config import settings
        slots = settings.LORA_SLOTS
    return slot_count(slots)


class Node:
    """One tree node: a symbol plus its ordered children."""

    __slots__ = ("symbol", "children")

    def __init__(self, symbol):
        self.symbol = symbol
        self.children = []


# --- growing a random tree ------------------------------------------------


def root_leaf_prob(conf):
    """The chance a drawn tree is one adapter alone -- the sweep's stored
    ROOT_LEAF_PROB, or settings.py's when it stored none."""
    value = (conf or {}).get("ROOT_LEAF_PROB")
    if value is None:
        from config import settings
        value = settings.ROOT_LEAF_PROB
    return float(value)


def random_tree(rng, max_depth, branch_prob, slots=MAX_SLOTS, leaf_prob=0.0):
    """Grow a random valid tree over the first `slots` L*/w*.

    The root is an L* on its own with probability `leaf_prob` -- a blend of
    one adapter -- and otherwise one of CAT/SVD/LIN, drawn uniformly, with the
    tree growing beneath it.

    `max_depth` is the deepest level an *operator* may sit at (the root is
    level 0), so a variable can appear at most one level below that. At that
    last operator level only arity-1 operators are drawn, which caps the tree.
    `branch_prob` is the chance that an operator above that level is arity 2,
    i.e. that the branch keeps growing instead of closing off with an L*.
    """
    leaves, weights = slot_symbols(slots), weight_symbols(slots)
    root = Node(rng.choice(leaves) if rng.random() < leaf_prob
                else rng.choice(BINARY_OPS))
    pending = deque([(root, 0)])
    while pending:
        node, depth = pending.popleft()
        for _ in range(ARITY[node.symbol]):
            if node.symbol in UNARY_OPS:
                child = Node(rng.choice(weights))
            elif depth + 1 >= max_depth or rng.random() >= branch_prob:
                child = Node(rng.choice(leaves))
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
    if not tokens or tokens[0] not in ROOTS:
        raise ValueError("expression must start with an operator (%s or an L*), not %s"
                         % ("/".join(BINARY_OPS), tokens[0] or "nothing"))

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


def build_population(count, rng, max_depth, branch_prob, unique, slots=MAX_SLOTS,
                     leaf_prob=0.0):
    """Generate `count` validated expressions over the first `slots` L*/w*.

    `leaf_prob` is random_tree()'s: the chance a member is one adapter alone.
    """
    population = []
    seen = set()
    attempts = 0
    attempt_budget = count * 100
    while len(population) < count:
        attempts += 1
        if attempts > attempt_budget:
            raise RuntimeError(
                "could only find %d unique expressions out of %d requested; "
                "raise MAX_DEPTH, BRANCH_PROB or ROOT_LEAF_PROB in settings.py, "
                "or clear UNIQUE"
                % (len(population), count)
            )
        depth = rng.randint(1, max_depth)
        expression = encode(random_tree(rng, depth, branch_prob, slots, leaf_prob))
        check(expression)   # never store something we cannot read back
        if unique:
            if expression in seen:
                continue
            seen.add(expression)
        population.append(expression)
    return population
