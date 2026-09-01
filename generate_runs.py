"""
generate_runs.py - Turn a tree into a runnable script.

Each individual becomes one self-contained script, written in the style of
combination.py: load the base model once, attach the LoRAs the tree names, fold
them together with PEFT's add_weighted_adapter, then chat through the resulting
adapter. main.py calls plan() and render() here for every individual in a sweep
and stores what comes back; the script only reaches disk long enough to be run.

The script itself comes from a template -- template_code.py by default -- which
is the generated file with the varying parts marked. Keeping the shape in a
template rather than in string literals means you can open it, syntax-check it,
and see what a generated script looks like directly; only what genuinely varies
per individual is a marker.

Which template to fill is an argument, so the same generator produces different
kinds of script from the same population -- it comes from TEMPLATE in
settings.py. template_code_mocked.py is the other one in the repo: same markers
and same rank arithmetic, but no model load and random answers, for exercising
the pipeline without a GPU.

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
front, marks the individual BAD, and stamps a warning into the script it makes.
"""

import json
import os

import settings

from generate_population import UNARY_OPS, decode, levels

_HERE = os.path.dirname(os.path.abspath(__file__))


COMBINATION_TYPE = {"CAT": "cat", "SVD": "svd", "LIN": "linear"}

TEMPLATE = os.path.join(_HERE, "template_code.py")

MARKER = "@@%s@@"
TEMPLATE_COMMENT = "#~"


def training_set_path(value=None):
    """Absolute path of the eval prompts file, from a settings value or None.

    TRAINING_SET is a setting rather than a line in the templates, so the eval
    set can be repointed without editing generated-script code and a sweep
    records which prompts it was scored against. A relative value is taken from
    this file's folder, the way main.py resolves DB_RUN_DIR, so it never depends
    on the cwd a driver happened to be started from; an absolute one is used as
    it stands. None means whatever settings.py currently says -- callers holding
    a sweep's stored settings should pass that instead, or a resumed sweep would
    silently change eval sets.
    """
    value = value or settings.TRAINING_SET
    if not os.path.isabs(value):
        value = os.path.join(_HERE, value)
    return os.path.abspath(value)


def training_count(value=None):
    """The eval-set cap as the templates want it: a positive int, or None.

    Same contract as training_set_path(): None means whatever settings.py
    currently says, and a caller holding a sweep's stored settings passes that
    instead, so a resumed sweep keeps being judged on as many prompts as it was
    created with. Settings.TRAINING_COUNT of None is no cap.

    Validated here rather than in the templates because a bad value should stop
    the generation step, not turn up inside every generated script.
    """
    value = settings.TRAINING_COUNT if value is None else value
    if value is None:
        return None
    try:
        count = int(value)
    except (TypeError, ValueError):
        raise SystemExit("TRAINING_COUNT must be a whole number or None, got %r" % value)
    # int() would quietly turn 2.5 into 2, and a fractional count of prompts is
    # a typo rather than an intention worth honouring.
    if count != float(value):
        raise SystemExit("TRAINING_COUNT must be a whole number or None, got %r" % value)
    if count < 1:
        raise SystemExit(
            "TRAINING_COUNT must be at least 1, got %d. None is how the cap is "
            "turned off; 0 would leave nothing to score." % count
        )
    return count


def eval_prompt_count(training_set=None, count=None):
    """(path, used, total) for the eval set, or a clear failure.

    Counts non-blank lines rather than re-parsing: the template owns reading a
    line, and every generated script would fail at startup on a missing or empty
    file, so it is worth catching here instead. One line is one prompt in both
    shapes the templates accept -- a JSON record per line, or plain text -- so
    the count holds without this having to know which it is looking at.

    `used` is what TRAINING_COUNT leaves of `total` -- the same min() the
    templates apply, computed here so the runs step can report what a sweep will
    actually be scored on rather than what the file happens to hold.
    """
    path = training_set_path(training_set)
    try:
        with open(path, encoding="utf-8") as handle:
            lines = [line for line in handle if line.strip()]
    except OSError as error:
        raise SystemExit("cannot read the eval prompts from %s (%s). Every generated "
                         "script reads that file at startup." % (path, error.strerror))
    total = len(lines)
    if not total:
        raise SystemExit("%s has no prompts in it" % path)
    # The one shape that would slip past a line count: a whole-file JSON array
    # is one record spread over many lines, or many records on one. Caught here
    # so it costs one clear failure rather than one per generated script.
    if lines[0].lstrip().startswith("["):
        raise SystemExit(
            "%s looks like one big JSON array. The generated scripts read the "
            "eval file a line at a time -- write it as one JSON record per line "
            "(JSON Lines), or as plain one-prompt-per-line text." % path
        )
    cap = training_count(count)
    return path, (total if cap is None else min(cap, total)), total


def eval_records(training_set=None, count=None):
    """The eval set as records the evaluate step can score against.

    -> [{"position": 1, "question": "...", "reference": "..." or None}, ...]

    Same file, same order and same cap the generated scripts ask under, read
    here for the other half of the job: an evaluator that compares an answer
    with the answer the dataset already carries needs that reference, and the
    scripts deliberately never see it -- handing a model the reference and then
    scoring its reply would be marking its own homework.

    Parsed the way template_code.py parses a line, because it is the same file
    in the same two shapes: a JSON record with a "messages" list -- first user
    turn is the question, first assistant turn after it is the reference -- or
    a plain line, which is a question with no reference at all. `position` is
    1-based and lines up with exchanges.position, since both count non-blank
    lines of this file from the top.

    Kept next to eval_prompt_count() rather than in evaluate_run.py so the eval
    file has one reader per concern and neither has to know where the file is:
    training_set_path() already answers that, for a stored sweep's value as
    much as for settings.py's.
    """
    path = training_set_path(training_set)
    try:
        with open(path, encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]
    except OSError as error:
        raise SystemExit("cannot read the eval prompts from %s (%s)."
                         % (path, error.strerror))
    cap = training_count(count)
    records = []
    for position, line in enumerate(lines[:cap] if cap else lines, start=1):
        records.append({"position": position,
                        "question": _question_of(line, path, position),
                        "reference": _reference_of(line)})
    return records


def _question_of(line, path, number):
    """The prompt in one eval line -- template_code.py's _prompt_of, once more.

    Duplicated rather than shared because the templates are standalone files
    that must run with nothing of this repo importable; a change to one of these
    two belongs in the other.
    """
    if line[0] == "{":
        try:
            record = json.loads(line)
        except ValueError as error:
            raise SystemExit("%s line %d: starts like a JSON record but will not "
                             "parse (%s)." % (path, number, error))
        messages = record.get("messages") if isinstance(record, dict) else None
        if not isinstance(messages, list):
            raise SystemExit("%s line %d: a JSON eval record needs a 'messages' "
                             "list, the shape datasets/*.json use." % (path, number))
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
        raise SystemExit("%s line %d: no user turn with any text in it, so there "
                         "is nothing to ask." % (path, number))

    if len(line) > 1 and line[0] == line[-1] and line[0] in "\"'":
        line = line[1:-1]
    return line


def _reference_of(line):
    """The dataset's own answer to that line, or None when it carries none.

    A plain prompt file has no references, and neither does a JSON record whose
    messages stop at the user turn. That is not an error here -- it is only an
    error for the evaluators that need one, and they say so themselves.
    """
    if not line.startswith("{"):
        return None
    try:
        record = json.loads(line)
    except ValueError:
        return None
    messages = record.get("messages") if isinstance(record, dict) else None
    if not isinstance(messages, list):
        return None
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "assistant":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return None


def lora_slots(slots=None):
    """{slot: location} with the ones that name a local folder made absolute.

    LORA_SLOTS is a setting rather than a block in the templates, so a slot is
    repointed in one place and both templates follow, and a sweep records which
    five adapters it was scored on. A relative entry is taken from this file's
    folder, the same rule training_set_path() uses -- but only when it really is
    a folder here; anything else is passed through as written, which is what
    leaves the door open for an absolute path or a Hub repo id. None means
    whatever settings.py currently says; callers holding a sweep's stored
    settings should pass those instead, or a resumed sweep would quietly blend
    different adapters.
    """
    resolved = {}
    for slot, where in (slots or settings.LORA_SLOTS).items():
        if not os.path.isabs(where):
            local = os.path.abspath(os.path.join(_HERE, where))
            if os.path.isdir(local):
                where = local
        resolved[slot] = where
    return resolved


def slot_ranks(slots=None):
    """{slot: rank} read from each adapter's own adapter_config.json.

    Ranks are not assumed equal: the five slots may point at LoRAs trained with
    different r values, and cat/svd/linear each treat that differently.
    """
    ranks = {}
    for slot, path in sorted(lora_slots(slots).items()):
        config_path = os.path.join(path, "adapter_config.json")
        try:
            with open(config_path, encoding="utf-8") as handle:
                config = json.load(handle)
        except OSError as error:
            raise SystemExit(
                "slot %s: cannot read %s (%s). Point LORA_SLOTS[%r] in settings.py "
                "at a real adapter folder, or fix the path."
                % (slot, config_path, error.strerror, slot)
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
           template_lines=None, template_path=TEMPLATE, weight_seed=None,
           training_set=None, slots=None, count=None):
    """The complete text of one runnable script, built from the template.

    Callers pass the planning results from plan(); the template supplies
    everything that does not vary between individuals.

    Which template that is comes from `template_path`, or from `template_lines`
    if the caller has already read one -- a batch job reads it once and reuses
    the lines, a one-off caller just names the file.

    `weight_seed` is what the script's WEIGHT_SEED becomes. None leaves the
    script redrawing its blend weights from the OS every execution, so the same
    chromosome is judged under different weights each time. An int pins the draw,
    which is how main.py makes a stored sweep repeatable -- it records the seed
    it stamped in here.

    `training_set` is what the script's TRAINING_SET becomes, resolved to an
    absolute path; None takes it from settings.py. `slots` is the same for
    LORA_SLOTS, and `count` for TRAINING_COUNT. main.py passes the sweep's
    stored values for all three, so a resumed sweep keeps reading the prompts,
    reading as many of them, and blending the adapters it was created with even
    if settings.py has since moved on.
    """
    template_lines = (load_template(template_path) if template_lines is None
                      else template_lines)
    root, _ = decode(expression)
    leaves = [step for step in steps if step.kind == "leaf"]
    # Resolved once: the block below wants it, and it validates, so a bad
    # TRAINING_COUNT fails here rather than in every script rendered after it.
    cap = training_count(count)

    blocks = {
        "TREE": ["    %s" % ".".join(row) for row in levels(root)],
        "BUILD_ORDER": build_order_block(steps),
        "NOTE": note_block(steps),
        "ATTACH_LEAVES": attach_leaves_block(steps),
        "COMBINE_NODES": combine_nodes_block(steps),
        # One line, but a block rather than an inline marker: it stands in for
        # the assignment itself, so the generated file gets a plain literal.
        "WEIGHT_SEED": ["WEIGHT_SEED = %s"
                        % ("None" if weight_seed is None else int(weight_seed))],
        # Likewise a block: the generated script gets the resolved path as a
        # literal, rather than working it out from where it happens to sit.
        "TRAINING_SET": ["TRAINING_SET = %r" % training_set_path(training_set)],
        # And the cap on it, as a literal for the same reason: the script slices
        # the prompts it read, so it carries the number rather than looking it
        # up in a settings.py that may have moved on since.
        "TRAINING_COUNT": ["TRAINING_COUNT = %s" % cap],
        # And likewise the adapters: one resolved entry per slot, in slot order
        # so two scripts built from the same settings are byte-identical here.
        "LORA_SLOTS": (["LORA_SLOTS = {"]
                       + ["    %r: %r," % pair
                          for pair in sorted(lora_slots(slots).items())]
                       + ["}"]),
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
