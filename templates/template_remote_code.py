#~ TEMPLATE, not a script to run. generate_runs.py reads this file and fills in
#~ the @@MARKERS@@ to produce one run_NNN.py per individual -- the same markers
#~ template_code.py fills, from the same render() call, so switching between
#~ them is one line in settings.py and nothing in the generator.
#~
#~ This is the thin-client half of blends/lora_server.py. What a script built
#~ from template_code.py does itself -- import unsloth, load the base model,
#~ attach, combine -- this one asks a server that already has the model open.
#~ Everything the pipeline reads off a script is unchanged: the YOU:/COACH:
#~ transcript, the weights: line, the TIMING: lines, the exit code, the fact
#~ that one script is one individual and one execution.
#~
#~ The trick that keeps the generator out of it: ATTACH_LEAVES and
#~ COMBINE_NODES are filled with the same attach()/combine() call lines as
#~ ever, and it is the definitions below that differ -- here they record a
#~ build plan instead of touching a model. The rank rule is therefore checked
#~ in exactly the same place, from each adapter's own config on disk, before
#~ any server is asked for anything.
#~
#~ Lines starting with "#~" are template-only notes and never reach the output.
"""
@@SCRIPT_NAME@@ - Combine LoRAs the way one GEP tree says to, then chat.

@@PROVENANCE@@
The remote form: the blend is built on a lora_server.py process that already
holds the base model open, instead of in this process. Same tree, same weights,
same prompts, same transcript -- what it saves is the import and the model load,
which on this machine are about 54% of what a script costs.

Expression
    @@EXPRESSION@@

Tree
@@TREE@@

How to read it
    L<i>.w<j>   attach LoRA slot i, blended at weight w<j>
    CAT(a, b)   add_weighted_adapter(..., combination_type="cat")
    SVD(a, b)   add_weighted_adapter(..., combination_type="svd")
    LIN(a, b)   add_weighted_adapter(..., combination_type="linear")

A combined node is itself an adapter, so it feeds its parent like a leaf.
Its children's weights are already folded in, so it enters its parent at
weight 1.0.

Build order (deepest first)
@@BUILD_ORDER@@

@@NOTE@@
Usage
    python @@SCRIPT_NAME@@                          # demo prompts
    python @@SCRIPT_NAME@@ "Help me plan my week."   # your own question

    The server it talks to comes from $GEP_LORA_SERVER, which the process step
    sets per script when it hands out its pool. Run one by hand against a
    server you started yourself:

        python -m blends.lora_server --base-model <id> --port 8770
        GEP_LORA_SERVER=http://127.0.0.1:8770 python @@SCRIPT_NAME@@
"""

import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

#~ Nothing heavy is imported here, which is the entire point -- but the phase is
#~ still printed, so a sweep's report shows what it used to cost next to what it
#~ costs now rather than showing a phase that has silently gone missing.
_STARTED = time.perf_counter()


def _timing(phase, seconds, label=""):
    """Say what this script spent on one thing, on the channel everything else
    it says comes out on -- see process_run.timings(), which reads these back.

    Identical to template_code.py's, including for the phases that happened on
    the server: those come back from /build as numbers and are re-printed here,
    so phase_timings is written exactly as it always was. model_load simply
    stops appearing among them, which is what this template is for.
    """
    print(f"TIMING: {phase} {seconds:.4f}{' ' + label if label else ''}", flush=True)


_timing("import", time.perf_counter() - _STARTED)

# Which server builds this blend. An environment variable rather than something
# filled into the script, because which server an individual ran against is an
# accident of the batch it landed in -- baking it in would make a stored script
# describe one particular run of itself. blends/server_pool.py sets it; the
# default is only for running one of these by hand.
DEFAULT_SERVER = "http://127.0.0.1:8770"
SERVER_URL = (os.environ.get("GEP_LORA_SERVER") or DEFAULT_SERVER).rstrip("/")
try:
    REQUEST_TIMEOUT = float(os.environ.get("GEP_LORA_SERVER_TIMEOUT") or 1800)
except ValueError:
    REQUEST_TIMEOUT = 1800.0


def _post(path, payload):
    """One request to the server. -> the decoded reply, or a clear SystemExit.

    Every failure here is this individual's failure and nobody else's: a
    SystemExit is a non-zero exit code, which the process step records as a
    failed execution the same way it records a script that ran out of memory.
    The server's own message is passed through, because it is the one that
    knows what went wrong.
    """
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        SERVER_URL + path, data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as reply:
            return json.loads(reply.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        try:
            detail = json.loads(detail).get("error", detail)
        except ValueError:
            pass
        raise SystemExit(f"lora server {SERVER_URL}{path}: {detail}")
    except urllib.error.URLError as error:
        raise SystemExit(
            f"cannot reach the lora server at {SERVER_URL} ({error.reason}). "
            f"The process step starts a pool of them when TEMPLATE is "
            f"template_remote_code.py; on its own, start one with "
            f"python -m blends.lora_server --base-model <id>."
        )
    except OSError as error:
        raise SystemExit(f"lora server {SERVER_URL}{path}: {error}")


# The base model every adapter was trained on. Sent to the server on every
# build, which checks it against the one it holds -- adapters folded onto the
# wrong base is the one mismatch this script could not otherwise see.
# @@BASE_MODEL@@

# Where each of the 5 LoRAs the trees refer to lives, already resolved. Read
# here (for the ranks) and sent to the server (to load), so the server needs no
# settings of its own: an individual carries its own adapters.
# @@LORA_SLOTS@@

# What w1..w5 are worth. Drawn here, exactly as in template_code.py -- the seed
# is the individual's own, so a stored sweep replays weight for weight, and the
# weights go to the server as part of the build plan.
# @@WEIGHT_SEED@@

_rng = random.Random(WEIGHT_SEED)


def _weight():
    """A blend weight in (0, 1), both ends excluded."""
    value = 0.0
    while value == 0.0:
        value = _rng.random()
    return value


WEIGHTS = {name: _weight() for name in ("w1", "w2", "w3", "w4", "w5")}


def _rank(adapter_dir):
    """The rank PEFT will allocate for this adapter, from its adapter_config.json.

    Read here rather than on the server, and before anything is asked of it: the
    equal-rank rule for linear is what makes a tree BAD, and a tree that cannot
    run should not cost a round trip to find out. The server checks it again for
    the same reason the generated script used to -- see combine() there.
    """
    with open(os.path.join(adapter_dir, "adapter_config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    return max([config["r"]] + list((config.get("rank_pattern") or {}).values()))


# The prompts this individual is judged on, read from a file at startup.
# @@TRAINING_SET@@

# How many of those records this individual is judged on.
# @@TRAINING_COUNT@@


def _prompt_of(line, path, number):
    """One eval prompt, from one non-blank line of the eval file.

    template_code.py's _prompt_of, unchanged -- the two shapes, told apart by
    the line itself: a JSON record carrying a "messages" list, whose first user
    turn is the prompt (the assistant turn beside it is somebody else's answer
    to the same question and must never reach the model), or a plain line.
    """
    if line[0] == "{":
        try:
            record = json.loads(line)
        except ValueError as error:
            raise SystemExit(
                f"{path} line {number}: starts like a JSON record but will not "
                f"parse ({error})."
            )
        messages = record.get("messages") if isinstance(record, dict) else None
        if not isinstance(messages, list):
            raise SystemExit(
                f"{path} line {number}: a JSON eval record needs a 'messages' "
                f"list, the shape datasets/*.json use."
            )
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
        raise SystemExit(
            f"{path} line {number}: no user turn with any text in it, so there "
            f"is nothing to ask."
        )

    if len(line) > 1 and line[0] == line[-1] and line[0] in "\"'":
        line = line[1:-1]
    return line


def _prompts(path, count=None):
    """One prompt per non-blank line, capped at `count` -- the first `count` of
    them, because every individual has to answer the same questions for their
    scores to mean anything next to each other."""
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError as error:
        raise SystemExit(f"cannot read the eval prompts from {path}: {error.strerror}")

    prompts = []
    for number, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        if not prompts and line[0] == "[":
            raise SystemExit(
                f"{path} looks like one big JSON array. The eval file is read a "
                f"line at a time -- write it as one JSON record per line (JSON "
                f"Lines), or as plain one-prompt-per-line text."
            )
        prompts.append(_prompt_of(line, path, number))
    if not prompts:
        raise SystemExit(f"{path} has no prompts in it")
    return prompts if count is None else prompts[:count]


EVAL_PROMPTS = _prompts(TRAINING_SET, TRAINING_COUNT)

EXPRESSION = "@@EXPRESSION@@"

print(f"@@LABEL@@: {EXPRESSION}")
# Print the draw, so a run can be traced back to the weights that produced it.
print("weights: " + ", ".join(f"{k}={v:.4f}" for k, v in WEIGHTS.items()))
print(f"SERVER: {SERVER_URL}")

# ---------------------------------------------------------------------------
# Describe the blend. The @@LEAF_COUNT@@ leaf adapter(s) and the nodes above
# them are the same attach()/combine() lines a local script gets -- here they
# record a plan for the server rather than building anything.
# ---------------------------------------------------------------------------
PLAN = []
RANKS = {}


def attach(name, slot):
    """Record "load LORA_SLOTS[slot] under `name`", and its rank."""
    RANKS[name] = _rank(LORA_SLOTS[slot])
    PLAN.append({"op": "attach", "name": name, "slot": slot})
    return name


#~ One attach() per leaf, in post-order -- the same block template_code.py gets.
# @@ATTACH_LEAVES@@


def combine(name, combination_type, left, right):
    """Record "fold these two into one under `name`", tracking the rank.

    PEFT's rules (peft/tuners/lora/model.py, _check_add_weighted_adapter): cat
    sums the input ranks, svd takes the max, linear demands they match. The
    linear case is checked here, before the server is asked for anything, so a
    tree PEFT would refuse costs nothing and fails with the message it always
    failed with.
    """
    (left_name, left_weight), (right_name, right_weight) = left, right
    left_rank, right_rank = RANKS[left_name], RANKS[right_name]

    if combination_type == "linear" and left_rank != right_rank:
        raise SystemExit(
            f"{name}: combination_type='linear' needs both inputs at the same rank, "
            f"but {left_name} is rank {left_rank} and {right_name} is rank {right_rank}. "
            f"cat sums its inputs' ranks, which is usually what pushes them apart."
        )

    PLAN.append({"op": "combine", "name": name, "type": combination_type,
                 "left": [left_name, left_weight],
                 "right": [right_name, right_weight]})

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

# ---------------------------------------------------------------------------
# Hand the plan over. This is where a local script would be a minute into its
# model load; here the model is already open and only the adapters are new.
# ---------------------------------------------------------------------------
_started = time.perf_counter()
BUILD = _post("/build", {"base_model": BASE_MODEL, "slots": LORA_SLOTS,
                         "plan": PLAN, "final": FINAL_ADAPTER})
# The round trip as one phase, and then the server's own account of what went
# on inside it. The two overlap on purpose -- "build" is a NESTED phase in
# metrics/report.py, the way "compact" sits inside "combine.svd".
_timing("build", time.perf_counter() - _started)
for entry in BUILD.get("timings", []):
    _timing(entry["phase"], float(entry["seconds"]), entry.get("label", ""))

print(f"GPU available: {BUILD.get('cuda')} (on the server)")
# Which server, and how many blends it had built before this one. A warm server
# is a weaker guarantee than a cold process, so the number that bounds it is
# written down where every other fact about an execution is: its stdout.
print(f"served by {SERVER_URL}, build #{BUILD.get('builds')}")
# The line process_run.Progress watches for, in the shape it watches for.
print(f"Active adapter: ['{FINAL_ADAPTER}'] (rank {BUILD['rank']})")


# How many eval prompts go through the model in one generate() call. The
# weights are read once per call whatever it is answering, so five prompts in
# one call cost far less than five calls of one; what grows with the batch is
# the KV cache. Same constant, and the same reason for it, as in
# template_code.py -- keep the two the same or two sweeps stop comparing.
ANSWER_BATCH = 8


def answer(questions, max_new_tokens=250):
    """Answer several questions in one generate() call. -> replies, in order."""
    reply = _post("/generate", {"prompts": list(questions),
                                "max_new_tokens": max_new_tokens})
    replies = reply.get("replies") or []
    if len(replies) != len(questions):
        raise SystemExit("the lora server answered %d of %d prompts"
                         % (len(replies), len(questions)))
    return replies


def ask(question, max_new_tokens=250):
    """Send one user turn through this tree's combined adapter."""
    return answer([question], max_new_tokens)[0]


def _say(questions):
    """Answer a batch, print the exchanges, then say what the batch cost.

    Printed batch by batch rather than after the last one: a script that hits
    its timeout still leaves the exchanges it had finished. The timing line goes
    last, after the final reply -- a line between "COACH:" and the rest of a
    wrapped answer would be read as part of the answer.

    The seconds are measured here rather than taken from the server's reply, so
    "generate" goes on meaning what the script paid, network and all.
    """
    started = time.perf_counter()
    replies = answer(questions)
    seconds = time.perf_counter() - started
    for question, reply in zip(questions, replies):
        print(f"\nYOU: {question}")
        print(f"COACH: {reply}")
    _timing("generate", seconds, "%d answer(s)" % len(questions))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Everything after the script name is treated as one question.
        _say([" ".join(sys.argv[1:])])
    else:
        for start in range(0, len(EVAL_PROMPTS), ANSWER_BATCH):
            _say(EVAL_PROMPTS[start:start + ANSWER_BATCH])
        # The whole script, from its first line to its last answer: what the
        # phases above should add up to, and the number the process step
        # measures from outside as well.
        _timing("total", time.perf_counter() - _STARTED)
