"""
generate_runs.py - Turn every tree in population.txt into a runnable script.

Each individual becomes one self-contained file in run/, written in the style
of combination.py: load the base model once, attach the LoRAs the tree names,
fold them together with PEFT's add_weighted_adapter, then chat through the
resulting adapter.

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

from generate_population import BINARY_OPS, UNARY_OPS, decode, levels

_HERE = os.path.dirname(os.path.abspath(__file__))

# Rank of the adapters on disk (r=16 in both adapter_config.json files).
BASE_RANK = 16

COMBINATION_TYPE = {"CAT": "cat", "SVD": "svd", "LIN": "linear"}


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


# --- code emission --------------------------------------------------------


def build_notes(steps):
    """The human-readable 'this is what gets built' lines for the docstring."""
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


def render(expression, steps, final, script_name, provenance, label):
    """The complete text of one runnable script.

    `script_name` is what the file will be called on disk, `provenance` the
    sentence saying where the chromosome came from, and `label` the prefix the
    script prints at startup. They are parameters so that both generate_runs.py
    (a whole population) and test.py (one hand-set chromosome) emit the same
    code with an honest header.
    """
    root, _ = decode(expression)
    broken = [step for step in steps if step.broken]
    out = []

    # --- docstring ---
    out.append('"""')
    out.append("%s - Combine LoRAs the way one GEP tree says to, then chat." % script_name)
    out.append("")
    out.append(provenance)
    out.append("Written in the style of combination.py, which stacks two adapters with a")
    out.append("fixed combination_type; here the tree decides both the shape and the weights.")
    out.append("")
    out.append("Expression")
    out.append("    %s" % expression)
    out.append("")
    out.append("Tree")
    for row in levels(root):
        out.append("    %s" % ".".join(row))
    out.append("")
    out.append("How to read it")
    out.append("    L<i>.w<j>   attach LoRA slot i, blended at weight w<j>")
    out.append('    CAT(a, b)   add_weighted_adapter(..., combination_type="cat")')
    out.append('    SVD(a, b)   add_weighted_adapter(..., combination_type="svd")')
    out.append('    LIN(a, b)   add_weighted_adapter(..., combination_type="linear")')
    out.append("")
    out.append("A combined node is itself an adapter, so it feeds its parent like a leaf.")
    out.append("Its children's weights are already folded in, so it enters its parent at")
    out.append("weight 1.0.")
    out.append("")
    out.append("Build order (deepest first)")
    out.extend(build_notes(steps))

    if broken:
        out.append("")
        out.append("NOTE - this tree cannot run as written.")
        for step in broken:
            out.append("    %s is LIN, which PEFT maps to combination_type=\"linear\", and"
                       % step.name)
            out.append("    linear requires both inputs to have the same rank -- but %s has"
                       % step.left[0])
            out.append("    rank %d and %s has rank %d. PEFT raises ValueError at that call."
                       % (step.left[2], step.right[0], step.right[2]))
        out.append("    (cat sums its inputs' ranks, which is what pushes them apart.)")
        out.append("    Fixes: swap that LIN for SVD, pass svd_rank= to the CAT feeding it,")
        out.append("    or treat the individual as unfit and drop it from the population.")

    out.append("")
    out.append("Usage")
    out.append("    python %s                          # demo prompts" % script_name)
    out.append('    python %s "Help me plan my week."   # your own question' % script_name)
    out.append('"""')
    out.append("")

    # --- imports and constants ---
    out.append("import os")
    out.append("import sys")
    out.append("")
    out.append("# Match the training/inference environment: disable Xet download acceleration.")
    out.append('os.environ["HF_HUB_DISABLE_XET"] = "1"')
    out.append("")
    out.append("# Unsloth patches transformers and peft as it loads, so it has to be")
    out.append("# imported before them or the optimizations are silently skipped.")
    out.append("from unsloth import FastLanguageModel")
    out.append("from unsloth.chat_templates import get_chat_template")
    out.append("")
    out.append("import torch")
    out.append("from peft import PeftModel")
    out.append("")
    out.append("_HERE = os.path.dirname(os.path.abspath(__file__))")
    out.append('_ROOT = os.path.dirname(os.path.dirname(_HERE))   # run/ -> project/ -> loras/')
    out.append("")
    out.append("# The base model every adapter was trained on (from adapter_config.json).")
    out.append('BASE_MODEL = "unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit"')
    out.append("")
    out.append('ADAPTER1_DIR = os.path.join(_ROOT, "test001", "my_planning_coach-lora_adapter")')
    out.append('ADAPTER2_DIR = os.path.join(_ROOT, "test002", "my_planning_coach-lora_adapter")')
    out.append("")
    out.append("# The 5 LoRAs the trees refer to. Only two real adapter folders exist on disk,")
    out.append("# so the slots reuse them under different names -- same trick lora_simul_01.py")
    out.append("# uses. PEFT keeps two loads of one folder separate, so this is safe.")
    out.append("LORA_SLOTS = {")
    out.append('    "L1": ADAPTER1_DIR,')
    out.append('    "L2": ADAPTER2_DIR,')
    out.append('    "L3": ADAPTER1_DIR,')
    out.append('    "L4": ADAPTER2_DIR,')
    out.append('    "L5": ADAPTER1_DIR,')
    out.append("}")
    out.append("")
    out.append("# What w1..w5 are worth. Placeholder spread -- retune for your search.")
    out.append("WEIGHTS = {")
    out.append('    "w1": 0.1,')
    out.append('    "w2": 0.3,')
    out.append('    "w3": 0.5,')
    out.append('    "w4": 0.7,')
    out.append('    "w5": 0.9,')
    out.append("}")
    out.append("")
    out.append("MAX_SEQ = 2048")
    out.append("")
    out.append("# The prompts this individual is judged on.")
    out.append("EVAL_PROMPTS = [")
    out.append('    "Help me organize my desktop.",')
    out.append('    "I have three assignments due this week. Just tell me what to do.",')
    out.append('    "Make me a full study schedule for my exams.",')
    out.append("]")
    out.append("")
    out.append('EXPRESSION = "%s"' % expression)
    out.append("")
    out.append('print(f"GPU available: {torch.cuda.is_available()}")')
    out.append('print(f"%s: {EXPRESSION}")' % label)
    out.append("")

    # --- load base ---
    out.append("# ---------------------------------------------------------------------------")
    out.append("# Load the base model once, with no adapter attached yet.")
    out.append("# ---------------------------------------------------------------------------")
    out.append("model, tokenizer = FastLanguageModel.from_pretrained(")
    out.append("    model_name=BASE_MODEL,")
    out.append("    max_seq_length=MAX_SEQ,")
    out.append("    dtype=None,")
    out.append("    load_in_4bit=True,")
    out.append(")")
    out.append("")

    # --- attach leaves ---
    leaves = [step for step in steps if step.kind == "leaf"]
    out.append("# ---------------------------------------------------------------------------")
    out.append("# Attach the %d leaf adapter(s) the tree names, each under its own name so" % len(leaves))
    out.append("# the same slot can appear more than once at different weights.")
    out.append("# ---------------------------------------------------------------------------")
    for index, step in enumerate(leaves):
        target = 'LORA_SLOTS["%s"]' % step.symbol
        if index == 0:
            out.append("model = PeftModel.from_pretrained(")
            out.append("    model, %s, adapter_name=\"%s\"" % (target, step.name))
            out.append(")")
        else:
            out.append('model.load_adapter(%s, adapter_name="%s")' % (target, step.name))
    out.append("")

    # --- combine ---
    combines = [step for step in steps if step.kind == "combine"]
    out.append("# ---------------------------------------------------------------------------")
    out.append("# Fold the tree together, deepest node first. Each call leaves behind a new")
    out.append("# adapter that later calls can use as an input.")
    out.append("# ---------------------------------------------------------------------------")
    for step in combines:
        if step.broken:
            out.append("# !! PEFT raises ValueError here: linear needs equal ranks, but")
            out.append("# !! %s is rank %d and %s is rank %d. See NOTE at the top."
                       % (step.left[0], step.left[2], step.right[0], step.right[2]))
        out.append("model.add_weighted_adapter(")
        out.append('    adapters=["%s", "%s"],' % (step.left[0], step.right[0]))
        out.append("    weights=[%s, %s]," % (step.left[1], step.right[1]))
        out.append('    adapter_name="%s",' % step.name)
        out.append('    combination_type="%s",' % COMBINATION_TYPE[step.symbol])
        out.append(")   # rank %d" % step.rank)
    out.append("")
    out.append('FINAL_ADAPTER = "%s"' % final)
    out.append("model.set_adapter(FINAL_ADAPTER)")
    out.append('print(f"Active adapter: {model.active_adapters} (rank %d)")' % steps[-1].rank)
    out.append("")

    # --- inference ---
    out.append("# Same chat template used during training, so inputs are formatted identically.")
    out.append('tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")')
    out.append("")
    out.append("# Switch to Unsloth's fast inference path (~2x faster generation).")
    out.append("FastLanguageModel.for_inference(model)")
    out.append("")
    out.append("# Qwen ships max_length=32768 in its generation_config.json, and transformers")
    out.append("# warns whenever that and max_new_tokens are both set. Clear it so the cap in")
    out.append("# ask() is the only one in play -- max_new_tokens was winning anyway.")
    out.append("model.generation_config.max_length = None")
    out.append("")
    out.append("")
    out.append("def ask(question, max_new_tokens=250):")
    out.append('    """Send one user turn through this tree\'s combined adapter."""')
    out.append('    msgs = [{"role": "user", "content": question}]')
    out.append("    inputs = tokenizer.apply_chat_template(")
    out.append('        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True')
    out.append("    ).to(model.device)")
    out.append("    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)")
    out.append("    # Slice off the prompt tokens so we only decode the newly generated reply.")
    out.append('    return tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:],')
    out.append("                            skip_special_tokens=True).strip()")
    out.append("")
    out.append("")
    out.append('if __name__ == "__main__":')
    out.append("    if len(sys.argv) > 1:")
    out.append("        # Everything after the script name is treated as one question.")
    out.append('        question = " ".join(sys.argv[1:])')
    out.append('        print(f"\\nYOU: {question}")')
    out.append('        print(f"COACH: {ask(question)}")')
    out.append("    else:")
    out.append("        # Score this individual by eyeballing its answers to the eval prompts.")
    out.append("        for q in EVAL_PROMPTS:")
    out.append('            print(f"\\nYOU: {q}")')
    out.append('            print(f"COACH: {ask(q)}")')

    return "\n".join(out) + "\n"


# --- driver ---------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate one runnable script per tree in population.txt."
    )
    parser.add_argument("--input", default=os.path.join(_HERE, "population.txt"),
                        help="file of K-expressions, one per line (default population.txt)")
    parser.add_argument("--output-dir", default=os.path.join(_HERE, "run"),
                        help="folder to write the scripts into (default run)")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as handle:
        expressions = [line.strip() for line in handle if line.strip()]
    if not expressions:
        raise SystemExit("%s has no expressions in it" % args.input)

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

    print("wrote %d scripts to %s" % (len(expressions), args.output_dir))
    print("%d runnable, %d blocked by PEFT's equal-rank rule for linear"
          % (runnable, len(expressions) - runnable))
    print("index: %s" % index_path)


if __name__ == "__main__":
    main()
