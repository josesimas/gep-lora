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
import ast
import json
import os

from generate_population import UNARY_OPS, decode, levels

_HERE = os.path.dirname(os.path.abspath(__file__))


COMBINATION_TYPE = {"CAT": "cat", "SVD": "svd", "LIN": "linear"}

TEMPLATE = os.path.join(_HERE, "template_code.py")

MARKER = "@@%s@@"
TEMPLATE_COMMENT = "#~"


def resolve_from_template(names, template_path, output_dir):
    """Evaluate module-level settings out of the template.

    Paths like LORA_SLOTS and TRAINING_SET live in template_code.py so the
    generated scripts stay self-contained, but this generator needs them too --
    for the rank analysis, and to check the eval set is readable. Rather than
    keep a second copy that could drift, execute those assignments straight from
    the template, with the same _HERE/_PROJECT the generated script will compute.
    """
    wanted = set(names)
    with open(template_path, encoding="utf-8") as handle:
        source = handle.read()

    here = os.path.abspath(output_dir)
    namespace = {"os": os, "_HERE": here, "_PROJECT": os.path.dirname(here)}
    found = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign):
            continue
        # Plain `NAME = ...` only; this skips things like os.environ[...] = ...
        hit = wanted.intersection(getattr(target, "id", "") for target in node.targets)
        if hit:
            exec(compile(ast.Module([node], []), template_path, "exec"), namespace)
            found.update({name: namespace[name] for name in hit})

    missing = wanted - set(found)
    if missing:
        raise SystemExit("no %s assignment found in %s"
                         % (", ".join(sorted(missing)), template_path))
    return found


def eval_prompt_count(output_dir, template_path=TEMPLATE):
    """How many prompts the generated scripts will find, or a clear failure.

    Counts non-blank lines rather than re-parsing: the template owns the quote
    handling, and every generated script would fail at startup on a missing or
    empty file, so it is worth catching here instead.
    """
    path = resolve_from_template(["TRAINING_SET"], template_path, output_dir)["TRAINING_SET"]
    try:
        with open(path, encoding="utf-8") as handle:
            count = sum(1 for line in handle if line.strip())
    except OSError as error:
        raise SystemExit("cannot read the eval prompts from %s (%s). Every generated "
                         "script reads that file at startup." % (path, error.strerror))
    if not count:
        raise SystemExit("%s has no prompts in it" % path)
    return path, count


def slot_ranks(output_dir, template_path=TEMPLATE):
    """{slot: rank} read from each adapter's own adapter_config.json.

    Ranks are not assumed equal: the five slots may point at LoRAs trained with
    different r values, and cat/svd/linear each treat that differently.
    """
    ranks = {}
    slots = resolve_from_template(["LORA_SLOTS"], template_path, output_dir)["LORA_SLOTS"]
    for slot, path in sorted(slots.items()):
        config_path = os.path.join(path, "adapter_config.json")
        try:
            with open(config_path, encoding="utf-8") as handle:
                config = json.load(handle)
        except OSError as error:
            raise SystemExit(
                "slot %s: cannot read %s (%s). Point LORA_SLOTS[%r] in %s at a "
                "real adapter folder, or fix the path."
                % (slot, config_path, error.strerror, slot, os.path.basename(template_path))
            )
        # Same rule PEFT uses: rank_pattern can raise the rank above r.
        ranks[slot] = max([config["r"]] + list((config.get("rank_pattern") or {}).values()))
    return ranks


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


def plan(root, ranks):
    """Post-order walk of the tree -> the ordered list of build steps.

    `ranks` maps each slot (L1..L5) to the rank read from its adapter_config.json.
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
            rank = ranks[node.symbol]
            step = Step("leaf", next_name(node.symbol), node.symbol, rank)
            step.variable = variable
            steps.append(step)
            return step.name, 'WEIGHTS["%s"]' % variable, rank

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
        lines.append("    rank %d and %s has rank %d, so the run stops there."
                     % (step.left[2], step.right[0], step.right[2]))
    lines.append("    (cat sums its inputs' ranks, which is what pushes them apart.)")
    lines.append("    Fixes: swap that LIN for SVD, pass svd_rank= to the CAT feeding it,")
    lines.append("    or treat the individual as unfit and drop it from the population.")
    lines.append("")
    return lines


def attach_leaves_block(steps):
    """One attach() per leaf, in post-order."""
    return ['attach("%s", "%s")' % (step.name, step.symbol)
            for step in steps if step.kind == "leaf"]


def combine_nodes_block(steps):
    """One combine() per binary node, in post-order."""
    lines = []
    for step in (step for step in steps if step.kind == "combine"):
        if step.broken:
            lines.append("# !! Stops here: linear needs equal ranks, but %s is rank %d"
                         % (step.left[0], step.left[2]))
            lines.append("# !! and %s is rank %d. See NOTE at the top."
                         % (step.right[0], step.right[2]))
        lines.append('combine("%s", "%s", ("%s", %s), ("%s", %s))'
                     % (step.name, COMBINATION_TYPE[step.symbol],
                        step.left[0], step.left[1], step.right[0], step.right[1]))
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
    }
    return fill(template_lines, blocks, values)


# --- driver ---------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate one runnable script per tree, from template_code.py."
    )
    parser.add_argument("--input", default=os.path.join(_HERE, "run", "population.txt"),
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

    # Each slot's rank comes from its own adapter_config.json -- they may differ.
    ranks = slot_ranks(args.output_dir, args.template)

    # The generated scripts read their prompts at startup; fail here if they cannot.
    prompts_path, prompt_count = eval_prompt_count(args.output_dir, args.template)

    os.makedirs(args.output_dir, exist_ok=True)

    index_lines = []
    runnable = 0
    for number, expression in enumerate(expressions, 1):
        steps, final = plan(decode(expression)[0], ranks)
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
    print("slot ranks: %s" % ", ".join("%s=%d" % pair for pair in sorted(ranks.items())))
    print("eval prompts: %d from %s" % (prompt_count, os.path.basename(prompts_path)))
    print("%d runnable, %d blocked by PEFT's equal-rank rule for linear"
          % (runnable, len(expressions) - runnable))
    print("index: %s" % index_path)


if __name__ == "__main__":
    main()
