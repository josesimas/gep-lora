#~ TEMPLATE, not a script to run. It is to template_baseline.py what
#~ template_code_mocked.py is to template_code.py: same markers, same generated
#~ shape, same YOU:/COACH: transcript, but no model is loaded and the answers
#~ are invented.
#~
#~ baseline_run.py fills this one whenever the sweep itself was generated from
#~ template_code_mocked.py, so the "llm_judge_baseline" plumbing -- generating
#~ the control, caching it, reading it back, handing it to a judge -- can be
#~ exercised on a machine with no GPU. What it produces is cached under
#~ "mock:<model>", never under the real model's name, so an invented baseline
#~ can never end up in front of a real judge.
#~
#~ A mocked baseline is noise, exactly like a mocked quality: it says the
#~ pipeline works, never that a blend does.
#~
#~ Lines starting with "#~" never reach the output, and this file is kept as
#~ valid Python so the linter still works on it.
"""
@@SCRIPT_NAME@@ - MOCK of the base-model baseline.

@@PROVENANCE@@
Generated from template_baseline_mocked.py, so nothing is loaded and nothing is
generated: the answers below are random. They stand in for what the base model
would have said, and they are cached apart from the real ones.
"""

import json
import os
import random as _random
import sys
import time

# How long the mock pretends a base-model load and a generate() take, in
# seconds. Small but not zero, so progress reporting has something to report.
MOCK_LOAD_DELAY = 1.0
MOCK_ANSWER_DELAY = 0.05

#~ Whole-line marker, as in template_baseline.py: the model these answers are
#~ pretending to come from. Only its name reaches the output -- nothing loads it.
# @@BASE_MODEL@@

#~ Whole-line marker: the eval prompts file, resolved and stamped in.
# @@TRAINING_SET@@

#~ And the cap on it.
# @@TRAINING_COUNT@@

# Seeded off the model name, so the same mocked baseline comes back every time
# rather than a fresh invention per run -- a cache that changed under the sweep
# it was serving would be no test of a cache.
_mock = _random.Random("baseline:" + str(BASE_MODEL))

_OPENERS = (
    "Sure. Here is a straightforward way to look at it.",
    "That depends on a few things, but broadly:",
    "Good question. The short version:",
)
_STEPS = (
    "Start by writing down what you already know.",
    "Break the work into pieces you can finish in a sitting.",
    "Decide what done looks like before you begin.",
    "Keep a list of what you are deliberately not doing.",
    "Check in on it at the end of the week.",
)
_CLOSERS = (
    "Let me know which part you would like to go into.",
    "Happy to go deeper on any of these.",
    "That should be enough to get moving.",
)


def _prompt_of(line, path, number):
    """One eval prompt, from one non-blank line of the eval file."""
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
    """One prompt per non-blank line, capped at `count`."""
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

print("GPU available: False (MOCK -- template_baseline_mocked.py, nothing is loaded)")
print(f"Baseline: {BASE_MODEL} with no adapter attached")

# The load that costs minutes for real.
time.sleep(MOCK_LOAD_DELAY)

# Said in the words template_baseline.py says it in, so process_run.Progress
# reads a mocked baseline exactly as it reads a real one.
print("Active adapter: none (base model, rank 0)")


def ask(question, max_new_tokens=250):
    """A plausible-looking base-model reply, assembled at random."""
    time.sleep(MOCK_ANSWER_DELAY)
    body = _mock.sample(_STEPS, _mock.randint(2, 3))
    return "\n".join([_mock.choice(_OPENERS)] + body + [_mock.choice(_CLOSERS)])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"\nYOU: {question}")
        print(f"COACH: {ask(question)}")
    else:
        for q in EVAL_PROMPTS:
            print(f"\nYOU: {q}")
            print(f"COACH: {ask(q)}")
