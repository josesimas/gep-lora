#~ TEMPLATE, not a script to run. It is the stand-in for template_code.py:
#~ same markers, same generated shape, but nothing is loaded and nothing is
#~ generated. Fill it the same way, with the template as an argument:
#~
#~     python generate_runs.py --template template_code_mocked.py
#~
#~ What it is for: exercising the pipeline -- generate_runs -> process_run ->
#~ evaluate_run, or `python main.py` end to end -- in seconds instead of hours,
#~ with no GPU, no base-model load and no judge endpoint. Use it when what you
#~ are testing is the plumbing, never when you are testing a blend.
#~
#~ What it keeps faithful to the real template:
#~   - the weight draw, and the "weights:" line process_run.py reads it from
#~   - the ranks, read from each adapter's own adapter_config.json
#~   - the attach/combine call order, and PEFT's equal-rank rule for linear,
#~     so a BAD individual stops in the same place with the same message
#~   - the YOU:/COACH: transcript shape
#~
#~ What it fakes: the answers, and their scores. It prints QUALITY:/REASON:
#~ lines after each answer, which process_run.py folds into the transcript --
#~ so mocked transcripts arrive pre-scored and evaluate_run.py skips them
#~ (an exchange that already has a quality is left alone without --force).
#~
#~ Markers work exactly as in template_code.py:
#~   @@NAME@@            inline, replaced inside the line it sits on
#~   a line that is just @@NAME@@ (or "# @@NAME@@") is replaced by a block
#~ Lines starting with "#~" never reach the output, and this file is kept as
#~ valid Python so the linter still works on it.
"""
@@SCRIPT_NAME@@ - MOCK of the combination script for one GEP tree.

@@PROVENANCE@@
Generated from template_code_mocked.py, so no model is loaded and no text is
generated: the answers, qualities and reasons below are random. The tree is
still walked and the ranks are still real, so this individual is blocked or
runnable for exactly the same reasons the real script would be.

Expression
    @@EXPRESSION@@

Tree
@@TREE@@

How to read it
    L<i>.w<j>   attach LoRA slot i, blended at weight w<j>
    CAT(a, b)   add_weighted_adapter(..., combination_type="cat")
    SVD(a, b)   add_weighted_adapter(..., combination_type="svd")
    LIN(a, b)   add_weighted_adapter(..., combination_type="linear")

Build order (deepest first)
@@BUILD_ORDER@@

@@NOTE@@
Usage
    python @@SCRIPT_NAME@@                          # mock the eval prompts
    python @@SCRIPT_NAME@@ "Help me plan my week."   # mock one question
"""

import json
import os
import random
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)                  # run/ -> project/

# Named for the record only; nothing here loads it.
BASE_MODEL = "unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit"

# The same slots as the real template, read the same way: the ranks decide
# which trees are blocked, and a mock that disagreed about that would be
# testing a different population than the one you are about to run for real.
#~ Whole-line marker filled by generate_runs.py from LORA_SLOTS in settings.py,
#~ exactly as in template_code.py -- which is what keeps the two agreeing about
#~ the ranks without either of them holding a copy of the paths.
# @@LORA_SLOTS@@

# What w1..w5 are worth: a fresh random draw every run, strictly between 0 and
# 1, exactly as in the real template. Set WEIGHT_SEED to an int to repeat one
# particular draw.
#~ Filled by the same whole-line marker as in template_code.py -- see the note
#~ there. The weight draw is one of the parts this mock does NOT fake, so the
#~ seed has to arrive here in exactly the same way.
# @@WEIGHT_SEED@@

_rng = random.Random(WEIGHT_SEED)


def _weight():
    """A blend weight in (0, 1), both ends excluded."""
    value = 0.0
    while value == 0.0:
        value = _rng.random()
    return value


WEIGHTS = {name: _weight() for name in ("w1", "w2", "w3", "w4", "w5")}

# --- the mock --------------------------------------------------------------

# Seeds the fake answers and scores. None gives a fresh set every execution,
# which is the honest default: a mocked score means nothing, and one that
# looked stable would be easy to mistake for a real signal.
MOCK_SEED = None

# Seconds to pretend the model load takes, and each answer. Left at 0 so a
# mocked sweep is instant; raise them to exercise process_run.py's --timeout.
MOCK_LOAD_DELAY = 0.0
MOCK_ANSWER_DELAY = 0.0

_mock = random.Random(MOCK_SEED)

_OPENERS = (
    "Let us start with the smallest piece you can finish today.",
    "Before anything else, write down what actually has a deadline.",
    "Here is a plan you can run without rearranging your whole week.",
    "Take the pressure off first, then we will sequence the work.",
    "One thing at a time -- here is the order I would go in.",
)

_STEPS = (
    "- Block 25 minutes for the piece you have been avoiding.",
    "- Put every deadline on one page, nearest first.",
    "- Decide what you are deliberately not doing this week.",
    "- Clear a single surface, not the whole room.",
    "- Pick a stopping point, so finishing is defined.",
    "- Schedule the hardest task when your energy is highest.",
)

_CLOSERS = (
    "Check back tomorrow and we will adjust from there.",
    "That is enough of a plan to start; the rest can wait.",
    "If that slips, shrink the step rather than the schedule.",
    "Tell me which part feels heaviest and we will rework it.",
)

# Banded by score, so the reason a mocked exchange carries at least agrees with
# the number next to it. A reader skimming mocked output should not have to
# work out which of the two fields is the fiction.
_REASONS = (
    (0.8, ("concrete steps, well matched to the question",
           "clear structure, actionable, appropriate tone")),
    (0.5, ("useful but generic; no specific schedule",
           "coherent and on topic, a little shallow")),
    (0.0, ("rambling, and never answers what was asked",
           "repeats the question back without adding anything")),
)


def _rank(adapter_dir):
    """The rank PEFT would allocate, from the adapter's own adapter_config.json.

    Read for real even here: the point of a mocked sweep is to find out which
    individuals are blocked, and that answer comes from the ranks.
    """
    with open(os.path.join(adapter_dir, "adapter_config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    return max([config["r"]] + list((config.get("rank_pattern") or {}).values()))


MAX_SEQ = 2048

# The prompts this individual is judged on, read from the file exactly as the
# real script reads them.
#~ Whole-line marker filled by generate_runs.py from TRAINING_SET in
#~ settings.py, exactly as in template_code.py -- the mock reads the same
#~ prompts, it only invents the answers to them.
# @@TRAINING_SET@@


def _prompts(path):
    """One prompt per line. Surrounding quotes are optional, blanks are skipped."""
    prompts = []
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError as error:
        raise SystemExit(f"cannot read the eval prompts from {path}: {error.strerror}")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if len(line) > 1 and line[0] == line[-1] and line[0] in "\"'":
            line = line[1:-1]
        prompts.append(line)
    if not prompts:
        raise SystemExit(f"{path} has no prompts in it")
    return prompts


EVAL_PROMPTS = _prompts(TRAINING_SET)

EXPRESSION = "@@EXPRESSION@@"

print("GPU available: False (MOCK -- template_code_mocked.py, nothing is loaded)")
print(f"@@LABEL@@: {EXPRESSION}")
# Printed in the real format, because process_run.py reads the weights off this
# line rather than recomputing them.
print("weights: " + ", ".join(f"{k}={v:.4f}" for k, v in WEIGHTS.items()))

# ---------------------------------------------------------------------------
# "Load" the base model. This is the step that costs minutes for real.
# ---------------------------------------------------------------------------
time.sleep(MOCK_LOAD_DELAY)
model = None

# ---------------------------------------------------------------------------
# Attach the @@LEAF_COUNT@@ leaf adapter(s) the tree names, each under its own name so
# the same slot can appear more than once at different weights.
# ---------------------------------------------------------------------------
RANKS = {}


def attach(name, slot):
    """Record what loading LORA_SLOTS[slot] under `name` would have produced."""
    RANKS[name] = _rank(LORA_SLOTS[slot])
    return name


#~ One attach() per leaf, in post-order.
# @@ATTACH_LEAVES@@

# ---------------------------------------------------------------------------
# Fold the tree together, deepest node first -- on paper. No adapter is built,
# but every rank is, so the arithmetic that decides a tree's fate is the real
# arithmetic.
# ---------------------------------------------------------------------------
def combine(name, combination_type, left, right):
    """Fold two adapters into one under `name`, tracking the resulting rank.

    PEFT's rules (peft/tuners/lora/model.py, _check_add_weighted_adapter):
    cat sums the input ranks, svd takes the max, linear demands they match.
    The linear check is kept, and kept identical, so a BAD individual fails
    here in a mocked sweep exactly as it would in a real one.
    """
    (left_name, left_weight), (right_name, right_weight) = left, right
    left_rank, right_rank = RANKS[left_name], RANKS[right_name]

    if combination_type == "linear" and left_rank != right_rank:
        raise SystemExit(
            f"{name}: combination_type='linear' needs both inputs at the same rank, "
            f"but {left_name} is rank {left_rank} and {right_name} is rank {right_rank}. "
            f"cat sums its inputs' ranks, which is usually what pushes them apart."
        )

    if combination_type == "cat":
        RANKS[name] = left_rank + right_rank
    elif combination_type == "svd":
        RANKS[name] = max(left_rank, right_rank)
    else:
        RANKS[name] = left_rank
    return name


#~ One combine() per binary node, in post-order.
# @@COMBINE_NODES@@

FINAL_ADAPTER = "@@FINAL_ADAPTER@@"
print(f"Active adapter: ['{FINAL_ADAPTER}'] (rank {RANKS[FINAL_ADAPTER]})")


def ask(question, max_new_tokens=250):
    """A plausible-looking reply, assembled at random. Ignores the question.

    Deliberately multi-line: the real replies wrap, and a transcript reader
    that only ever saw one-liners would not be tested by this.
    """
    time.sleep(MOCK_ANSWER_DELAY)
    body = _mock.sample(_STEPS, _mock.randint(2, 3))
    return "\n".join([_mock.choice(_OPENERS)] + body + [_mock.choice(_CLOSERS)])


def grade():
    """A random score and a reason to match, in the shape evaluate_run.py writes.

    0.0 is worst and 1.0 is best, rounded to 3 places like a real judge's. It
    is noise: useful for checking that the plumbing carries the field and that
    selection reads it, useless for judging a blend.
    """
    quality = round(_mock.random(), 3)
    for floor, phrases in _REASONS:
        if quality >= floor:
            return quality, _mock.choice(phrases)
    return quality, ""


def say(question):
    """Print one exchange in the form process_run.py parses."""
    quality, reason = grade()
    print(f"\nYOU: {question}")
    print(f"COACH: {ask(question)}")
    # Folded into this exchange by process_run.py, so a mocked sweep comes out
    # already scored and never needs the judge endpoint.
    print(f"QUALITY: {quality}")
    print(f"REASON: {reason}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Everything after the script name is treated as one question.
        say(" ".join(sys.argv[1:]))
    else:
        for q in EVAL_PROMPTS:
            say(q)
