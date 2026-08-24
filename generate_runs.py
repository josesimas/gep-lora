"""
generate_runs.py - Turn every tree in population.txt into a runnable script.

Each individual becomes one self-contained file in run/, written in the style
of combination.py: load the base model once, attach the LoRAs the tree names,
fold them together with PEFT's add_weighted_adapter, then chat through the
resulting adapter.

The script itself comes from template_code.py, which is the generated file with
the varying parts marked. Keeping the shape in a template rather than in string
literals means you can open it, syntax-check it, and see what a generated script
looks like directly -- only what genuinely varies per individual is a marker.

Markers, all of the form @@NAME@@:

    inline          replaced within the line, e.g. EXPRESSION = "@@EXPRESSION@@"
    whole line      a line that is only @@NAME@@ (or "# @@NAME@@") is replaced
                    by a block of lines
    #~ prefix       template-only note, dropped from the output

The tree maps onto PEFT directly:

    L<i>.w<j>   attach LoRA slot i, to be blended at weight w<j>
    CAT(a, b)   add_weighted_adapter(..., combination_type="cat")
    SVD(a, b)   add_weighted_adapter(..., combination_type="svd")
    LIN(a, b)   add_weighted_adapter(..., combination_type="linear")

A combined node is itself a named adapter, so it can feed its parent exactly
like a leaf does. Its children's weights are already baked into it, so it
enters its own parent at weight 1.0.

Rank bookkeeping matters, because PEFT constrains it (peft/tuners/lora/model.py,
_check_add_weighted_adapter):

    cat     -> new rank is the SUM of the children's ranks
    svd     -> new rank is the MAX of the children's ranks (no svd_rank given)
    linear  -> both children MUST have the same rank, else ValueError

So a LIN sitting above a CAT usually cannot run. That is a property of the
search space, not a bug here: this generator computes every node's rank up
front and stamps a warning into the files it affects.

Usage:
    python generate_runs.py                    # population.txt -> run/
    python generate_runs.py --output-dir run2
"""

import argparse
import os

from generate_population import UNARY_OPS, decode, levels

_HERE = os.path.dirname(os.path.abspath(__file__))

# Rank of the adapters on disk (r=16 in both adapter_config.json files).
BASE_RANK = 16

COMBINATION_TYPE = {"CAT": "cat", "SVD": "svd", "LIN": "linear"}

TEMPLATE = os.path.join(_HERE, "template_code.py")

MARKER = "@@%s@@"
TEMPLATE_COMMENT = "#~"


class Step:
    """One line of the build plan: either attach a leaf, or combine two nodes."""

    __slots__ = ("kind", "name", "symbol", "variable", "left", "right", "rank", "broken")

    def __init__(self, kind, name, symbol, rank):
        self.kind = kind          # "leaf" or "combine"
        self.name = name          # adapter name inside the model
        self.symbol = symbol      # L1..L5, or CAT/SVD/LIN
        self.variable = None      # leaf only: which w<j> weights it
        self.left = None          # combine only: (name, weight_expr, rank)
        self.right = None
        self.rank = rank
        self.broken = False       # combine only: LIN with mismatched child ranks


def plan(root):
    """Post-order walk of the tree -> the ordered list of build steps.

    Returns (steps, final_adapter_name). Nodes are numbered in the order they
    have to be built, so a step never references a name defined after it.
    """
    steps = []
    counter = [0]

    def next_name(symbol):
        counter[0] += 1
        return "n%d_%s" % (counter[0], symbol)

    def visit(node):
        """-> (adapter_name, weight expression, rank)."""
        if node.symbol in UNARY_OPS:
            # An L* node is a leaf adapter; its single child names the weight.
            variable = node.children[0].symbol
            step = Step("leaf", next_name(node.symbol), node.symbol, BASE_RANK)
            step.variable = variable
            steps.append(step)
            return step.name, 'WEIGHTS["%s"]' % variable, BASE_RANK

        left = visit(node.children[0])
        right = visit(node.children[1])
        left_rank, right_rank = left[2], right[2]

        if node.symbol == "CAT":
            rank, broken = left_rank + right_rank, False
        elif node.symbol == "SVD":
            rank, broken = max(left_rank, right_rank), False
        else:                                   # LIN -> PEFT "linear"
            rank, broken = left_rank, left_rank != right_rank

        step = Step("combine", next_name(node.symbol), node.symbol, rank)
        step.left, step.right, step.broken = left, right, broken
        steps.append(step)
        # The children's weights are already folded in, so this node enters its
        # own parent at full strength.
        return step.name, "1.0", rank

    visit(root)
    return steps, steps[-1].name


# --- the blocks that fill the template ------------------------------------


def build_order_block(steps):
    """The 'this is what gets built' lines for the docstring."""
    notes = []
    for step in steps:
        if step.kind == "leaf":
            notes.append("    %-10s = %s @ %s" % (step.name, step.symbol, step.variable))
        else:
            notes.append("    %-10s = %s(%s, %s)"
                         % (step.name, step.symbol, step.left[0], step.right[0]))
        notes[-1] = "%-46s rank %d" % (notes[-1], step.rank)
        if step.broken:
            notes[-1] += "   <-- FAILS, see NOTE"
    notes[-1] += "   <-- generation runs through this one"
    return notes


def note_block(steps):
    """The warning for trees PEFT will refuse, or nothing at all."""
    broken = [step for step in steps if step.broken]
    if not broken:
        return []
    lines = ["NOTE - this tree cannot run as written."]
    for step in broken:
        lines.append('    %s is LIN, which PEFT maps to combination_type="linear", and'
                     % step.name)
        lines.append("    linear requires both inputs to have the same rank -- but %s has"
                     % step.left[0])
        lines.append("    rank %d and %s has rank %d. PEFT raises ValueError at that call."
                     % (step.left[2], step.right[0], step.right[2]))
    lines.append("    (cat sums its inputs' ranks, which is what pushes them apart.)")
    lines.append("    Fixes: swap that LIN for SVD, pass svd_rank= to the CAT feeding it,")
    lines.append("    or treat the individual as unfit and drop it from the population.")
    lines.append("")
    return lines


def attach_leaves_block(steps):
    """One PeftModel.from_pretrained, then load_adapter for every other leaf."""
    lines = []
    for index, step in enumerate(step for step in steps if step.kind == "leaf"):
        target = 'LORA_SLOTS["%s"]' % step.symbol
        if index == 0:
            lines.append("model = PeftModel.from_pretrained(")
            lines.append('    model, %s, adapter_name="%s"' % (target, step.name))
            lines.append(")")
        else:
            lines.append('model.load_adapter(%s, adapter_name="%s")' % (target, step.name))
    return lines


def combine_nodes_block(steps):
    """One add_weighted_adapter call per binary node, in post-order."""
    lines = []
    for step in (step for step in steps if step.kind == "combine"):
        if step.broken:
            lines.append("# !! PEFT raises ValueError here: linear needs equal ranks, but")
            lines.append("# !! %s is rank %d and %s is rank %d. See NOTE at the top."
                         % (step.left[0], step.left[2], step.right[0], step.right[2]))
        lines.append("model.add_weighted_adapter(")
        lines.append('    adapters=["%s", "%s"],' % (step.left[0], step.right[0]))
        lines.append("    weights=[%s, %s]," % (step.left[1], step.right[1]))
        lines.append('    adapter_name="%s",' % step.name)
        lines.append('    combination_type="%s",' % COMBINATION_TYPE[step.symbol])
        lines.append(")   # rank %d" % step.rank)
    return lines


# --- filling the template -------------------------------------------------


def load_template(path=TEMPLATE):
    """Read the template, dropping its #~ notes."""
    with open(path, encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle
                if not line.lstrip().startswith(TEMPLATE_COMMENT)]


def fill(template_lines, blocks, values):
    """Substitute every marker.

    A line whose only content is one marker (bare, or commented out so the
    template stays valid Python) is replaced by that marker's block of lines.
    Every other marker is replaced inside the line it sits on.
    """
    block_markers = {MARKER % name: lines for name, lines in blocks.items()}
    out = []
    for line in template_lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            stripped = stripped.lstrip("#").strip()
        if stripped in block_markers:
            out.extend(block_markers[stripped])
            continue
        for name, value in values.items():
            line = line.replace(MARKER % name, value)
        out.append(line)

    leftover = [line for line in out if "@@" in line]
    if leftover:
        raise ValueError("template marker was never filled: %s" % leftover[0].strip())
    return "\n".join(out) + "\n"


def render(expression, steps, final, script_name, provenance, label,
           template_lines=None):
    """The complete text of one runnable script, built from the template.

    Callers pass the planning results from plan(); the template supplies
    everything that does not vary between individuals.
    """
    template_lines = load_template() if template_lines is None else template_lines
    root, _ = decode(expression)
    leaves = [step for step in steps if step.kind == "leaf"]

    blocks = {
        "TREE": ["    %s" % ".".join(row) for row in levels(root)],
        "BUILD_ORDER": build_order_block(steps),
        "NOTE": note_block(steps),
        "ATTACH_LEAVES": attach_leaves_block(steps),
        "COMBINE_NODES": combine_nodes_block(steps),
    }
    values = {
        "SCRIPT_NAME": script_name,
        "PROVENANCE": provenance,
        "LABEL": label,
        "EXPRESSION": expression,
        "LEAF_COUNT": str(len(leaves)),
        "FINAL_ADAPTER": final,
        "FINAL_RANK": str(steps[-1].rank),
    }
    return fill(template_lines, blocks, values)


# --- driver ---------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate one runnable script per tree, from template_code.py."
    )
    parser.add_argument("--input", default=os.path.join(_HERE, "population.txt"),
                        help="file of K-expressions, one per line (default population.txt)")
    parser.add_argument("--output-dir", default=os.path.join(_HERE, "run"),
                        help="folder to write the scripts into (default run)")
    parser.add_argument("--template", default=TEMPLATE,
                        help="the template to fill (default template_code.py)")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as handle:
        expressions = [line.strip() for line in handle if line.strip()]
    if not expressions:
        raise SystemExit("%s has no expressions in it" % args.input)

    # Read the template once; every individual reuses the same lines.
    template_lines = load_template(args.template)

    os.makedirs(args.output_dir, exist_ok=True)

    index_lines = []
    runnable = 0
    for number, expression in enumerate(expressions, 1):
        steps, final = plan(decode(expression)[0])
        name = "run_%03d.py" % number
        text = render(
            expression, steps, final,
            script_name=name,
            provenance="Generated by generate_runs.py from line %d of population.txt." % number,
            label="Individual %d" % number,
            template_lines=template_lines,
        )
        with open(os.path.join(args.output_dir, name), "w", encoding="utf-8") as handle:
            handle.write(text)

        broken = any(step.broken for step in steps)
        runnable += not broken
        index_lines.append("%s  %-4s rank %-4d %s"
                           % (name, "BAD" if broken else "ok", steps[-1].rank, expression))

    index_path = os.path.join(args.output_dir, "index.txt")
    with open(index_path, "w", encoding="utf-8") as handle:
        handle.write("script      state rank  expression\n")
        handle.write("\n".join(index_lines) + "\n")

    print("wrote %d scripts to %s (from %s)"
          % (len(expressions), args.output_dir, os.path.basename(args.template)))
    print("%d runnable, %d blocked by PEFT's equal-rank rule for linear"
          % (runnable, len(expressions) - runnable))
    print("index: %s" % index_path)


if __name__ == "__main__":
    main()
