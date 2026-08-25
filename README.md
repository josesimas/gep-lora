# GEP LoRA combination search

Tooling that turns Gene Expression Programming trees into runnable LoRA-blending
scripts. A chromosome describes *how to fold several LoRA adapters into one*, and
each chromosome becomes a standalone Python file that builds that blend and chats
through it.

The spec these tools implement is [plan.txt](plan.txt).

---

## The chromosome

An individual is a **K-expression** (Karva notation): the tree written out in
level-order — breadth first, left to right — one symbol per position, joined with
dots.

```
CAT.SVD.LIN.L1.L2.L3.L1.w3.w3.w2.w1
```

Reading it back is the same walk in reverse: take the symbols in order and hand
each one out as the next child that is still missing. That gives the tree:

```
CAT
SVD.LIN
L1.L2.L3.L1
w3.w3.w2.w1
```

No brackets are needed, because every symbol's arity is fixed.

### Alphabet

| Symbol | Arity | Children must be | Meaning |
|---|---|---|---|
| `CAT` `SVD` `LIN` | 2 | operators | combine two blends |
| `L1`–`L5` | 1 | a variable | one LoRA adapter |
| `w1`–`w5` | 0 | — | a blend weight |

The first symbol is always `CAT`.

Because `L*` only accepts variables and `CAT`/`SVD`/`LIN` only accept operators,
every leaf `w` sits under an `L`, and every `L` sits under a binary operator.

### How it maps onto PEFT

The three binary operators are exactly PEFT's `add_weighted_adapter`
combination types, which is what makes the whole scheme run:

| Tree | PEFT call |
|---|---|
| `CAT(a, b)` | `add_weighted_adapter(..., combination_type="cat")` |
| `SVD(a, b)` | `add_weighted_adapter(..., combination_type="svd")` |
| `LIN(a, b)` | `add_weighted_adapter(..., combination_type="linear")` |
| `L<i>.w<j>` | attach LoRA slot *i*, to be blended at weight `w<j>` |

A combined node is itself a named adapter, so it feeds its parent exactly like a
leaf does. Its children's weights are already folded into it, so **it enters its
own parent at weight 1.0**.

This is the same idea as [combination.py](combination.py), which stacks two
adapters with a hardcoded `combination_type="cat"`. Here the tree decides both
the shape and the weights.

---

## The rank rule (why 27 of 100 individuals can't run)

PEFT constrains the rank of the adapter each call produces
(`peft/tuners/lora/model.py`, `_check_add_weighted_adapter`):

| Operator | Resulting rank |
|---|---|
| `cat` | **sum** of the two inputs' ranks |
| `svd` | **max** of the two inputs' ranks (when no `svd_rank` is passed) |
| `linear` | both inputs **must have the same rank**, else `ValueError` |

The five LoRAs were trained at **different ranks** — `L1`=16, `L2`=16, `L3`=8,
`L4`=4, `L5`=32 — so leaves do not start out matched, and `CAT` pushes them
further apart as you nest it. Therefore **a `LIN` whose two inputs came from
different adapters, or from a `CAT`, usually cannot run**.

Nothing assumes a shared rank: each slot's is read from its own
`adapter_config.json`, both when the scripts are generated and again when they
run.

This is a property of the search space, not a bug. Every node's rank is computed
statically at generation time, so you find out before running anything rather
than crashing mid-script. In the current population, **73 of 100 run and 27 are
blocked**. Blocked scripts are still written, with a `NOTE` naming the offending
node and both ranks, and are marked `BAD` in `run/index.txt`; at runtime they
stop themselves with the same message rather than letting a bare `ValueError`
surface from inside PEFT.

Final ranks across the population now span 8 to 92.

If you want the blocked shapes to survive, the options are: map `LIN` to `svd`
when ranks diverge, pass `svd_rank=` to the `CAT` feeding it, or treat those
individuals as unfit and let selection drop them.

---

## Scripts

Run everything from `project/`.

### 0. `main.py` — the whole pipeline

```bash
python main.py
```

Runs every step in order — population, trees, runs, process, evaluate —
stopping at the first failure, since each step builds on the one before it.

The last two steps are the slow ones: `process` executes the generated scripts,
one base-model load per individual, and `evaluate` then makes one grading call
per answer. A full `python main.py` is therefore a long operation, and
`python main.py population trees runs` stops short of both.

Run it with the **venv's python** — `process` launches each generated script
with `sys.executable`, so the wrong interpreter fails every individual. It now
checks this up front rather than discovering it once per individual.

```bash
python main.py runs
```

Runs a subset. Handy after editing `template_code.py`, when the population is
still good. Steps always execute in pipeline order regardless of how you type
them, and `python main.py --list` shows them without running anything.

Population settings for a complete run live at the top of the file:

```python
COUNT = 100
SEED = 42      # an int repeats the same population; None grows a fresh one
UNIQUE = True
```

`SEED` controls the *chromosomes* only. Each generated script still draws its own
LoRA blend weights at runtime — see `WEIGHT_SEED` for those.

**Adding a step.** Append a `Step` to `STEPS`. It names a callable and the
arguments to pass it, and the callable is just another script's `main(argv)`, so
a step behaves exactly as if you ran that script yourself:

```python
Step("score", score_runs.main, ["--input", "run/index.txt"],
     "score every runnable individual -> run/scores.txt"),
```

Any plain function taking a list of arguments works too.

### 1. `generate_population.py` → `run/population.txt`

Grows random valid trees and writes one K-expression per line.

```bash
python generate_population.py --count 100 --seed 42 --unique
```

| Flag | Default | Meaning |
|---|---|---|
| `--count` | 100 | how many individuals |
| `--output` | `run/population.txt` | output file |
| `--seed` | none | RNG seed, for a reproducible population |
| `--max-depth` | 4 | deepest level an *operator* may sit at (root is level 0) |
| `--branch-prob` | 0.6 | chance an operator is arity 2 and keeps the branch growing |
| `--unique` | off | reject duplicate expressions |
| `--preview` | 0 | also print the first N as level rows |

Size varies per individual: a max operator depth is drawn from `1..--max-depth`,
then `--branch-prob` decides whether each operator keeps growing (arity 2) or
closes the branch off with an `L*`. Every expression is decoded and re-encoded
before being written, so nothing lands in the file that cannot be read back.

The committed file was generated with `--seed 42 --unique`: 100 individuals,
5–32 symbols each (mean 9.5), tree depths 2–5.

### 2. `draw_trees.py` → `run/trees.txt`

Draws every row of `population.txt` in the layout `plan.txt` uses — index,
expression, blank line, then one row per tree level.

```bash
python draw_trees.py
```

| Flag | Default | Meaning |
|---|---|---|
| `--input` | `run/population.txt` | file of K-expressions, one per line |
| `--output` | `run/trees.txt` | where to write the drawings |

Blocks are separated by two blank lines, so the blank line inside each block
stays unambiguous. Trailing symbols that the tree does not consume are reported
as `(unused tail: ...)` rather than dropped silently — `plan.txt`'s first example
has two such symbols.

### 3. `generate_runs.py` + `template_code.py` → `run/`

Turns every individual into a self-contained runnable script by filling in
`template_code.py`.

```bash
python generate_runs.py
```

| Flag | Default | Meaning |
|---|---|---|
| `--input` | `run/population.txt` | file of K-expressions, one per line |
| `--output-dir` | `run` | folder to write the scripts into |
| `--template` | `template_code.py` | the template to fill; `template_code_mocked.py` for a dry run |

Produces `run/run_001.py` … `run/run_100.py` plus `run/index.txt`:

```
script      state rank  expression
run_001.py  ok   rank 32   CAT.L1.L3.w2.w2
...
run_073.py  BAD  rank 48   CAT.L1.LIN.w5.CAT.L3.L5.L1.w2.w4.w5
```

Each generated script carries its tree and build plan in its docstring, then:
loads the base model once, attaches each leaf adapter under its own name, folds
the tree deepest-node-first with `add_weighted_adapter`, activates the final
adapter, and answers the eval prompts.

Each *occurrence* of an `L*` gets its own adapter name (`n1_L2`, `n4_L1`, …), so
one slot can appear several times at different weights. PEFT keeps repeated
loads of one folder separate, so this is safe.

#### The template

`template_code.py` is the generated script with the varying parts marked. It is
deliberately kept as valid Python, so your editor, linter and
`python -m compileall` all still work on it — what you see there is what gets
written, minus the markers.

| Marker form | Meaning |
|---|---|
| `@@NAME@@` | replaced inside the line it sits on |
| a line that is only `@@NAME@@` or `# @@NAME@@` | replaced by a whole block of lines |
| a line starting with `#~` | template-only note, never reaches the output |

Blocks are `TREE`, `BUILD_ORDER`, `NOTE`, `ATTACH_LEAVES`, `COMBINE_NODES`;
inline values are `SCRIPT_NAME`, `PROVENANCE`, `LABEL`, `EXPRESSION`,
`LEAF_COUNT`, `FINAL_ADAPTER`, `FINAL_RANK`. Any marker left unfilled raises
rather than being written into a generated file.

To change what every generated script looks like, edit `template_code.py` and
re-run `generate_runs.py`. Only add code to the generator itself when the new
part varies per individual.

#### `template_code_mocked.py` — the dry run

Which template gets filled is `--template`, so the same generator produces a
different kind of script from the same population:

```bash
python generate_runs.py --template template_code_mocked.py
```

The mocked template has the same markers and produces the same shaped script,
but loads nothing and generates nothing: `ask()` assembles a reply out of canned
fragments and `grade()` draws a quality with a reason to match. A whole sweep
then takes **seconds instead of hours, with no GPU and no judge**, which is what
you want when the thing under test is the pipeline rather than a blend.

Set `TEMPLATE = "template_code_mocked.py"` at the top of `main.py` to run the
whole pipeline that way.

What it keeps real, so a dry run tells you something true about a population:

- the weight draw, and the `weights:` line `process_run.py` reads it from
- the ranks, read from each slot's own `adapter_config.json`
- the `attach`/`combine` order, and PEFT's equal-rank rule for `linear` — a
  `BAD` individual stops at the same node with the same message, so the `ok`/
  `BAD` split in `index.txt` is the split you will get for real

What it fakes is the answers and the scores. Mocked scripts print `QUALITY:` and
`REASON:` lines after each reply; `process_run.py` folds those into the
transcript, so mocked transcripts arrive already scored and `evaluate_run.py`
skips them without contacting a judge at all. **A mocked quality is noise** —
never read one as a result.

Two things adjust themselves rather than needing a flag: `process_run.py` drops
its venv check when the scripts it is about to run do not import unsloth, so a
mocked sweep runs under any Python 3; and `evaluate_run.py` only reaches for the
judge when some answer actually lacks a quality.

`MOCK_SEED` fixes the fake answers and scores, and `MOCK_LOAD_DELAY` /
`MOCK_ANSWER_DELAY` buy back some fake slowness — useful for exercising
`process_run.py --timeout`.

### 4. `process_run.py` → `run/output_NNN.txt`, `run/results.txt`

`generate_runs.py` writes the scripts; this one runs them.

```bash
python process_run.py
```

| Flag | Default | Meaning |
|---|---|---|
| `--run-dir` | `run` | folder holding the generated scripts |
| `--limit` | 0 (all) | run only the first N individuals |
| `--include-blocked` | off | also run the ones `index.txt` marks `BAD` |
| `--timeout` | 900 | seconds to allow each script |

Each individual runs as a **separate process** — every script loads the base
model at import and attaches its own adapters, so they cannot share an
interpreter. That makes this the expensive step: one model load per individual.
`--limit 3` is the way to smoke-test before committing to a full sweep.

Output goes next to the scripts, two files per individual:

```
run/output_007.txt           everything run_007.py printed, warnings and all
run/output_result_007.json   the exchanges plus the weight draw that produced them
run/results.txt              script, state, result, seconds, exchanges, expression
```

The transcript is JSON: the conversation, plus the two things needed to make
sense of it later — which tree was built, and which weight draw built it.

```json
{
  "chromosome": "CAT.L1.L3.w2.w2",
  "weights": { "w1": 0.3529, "w2": 0.2882, "w3": 0.8712, "w4": 0.8846, "w5": 0.5110 },
  "exchanges": [
    {
      "question": "Help me organize my desktop.",
      "answer": "Before we lay anything out, let's call the one thing..."
    }
  ]
}
```

Each file is self-contained: a scorer never has to cross-reference `index.txt`
to know what produced a given set of answers, and the files stay meaningful if
they are moved or collected from several runs.

The weights matter as much as the tree. Every script redraws `w1`–`w5` at
startup, so the same chromosome run twice is judged under two different blends —
and in practice that changes the answers markedly. Recording the draw is what
lets a score be attributed to a specific blend rather than to the tree alone, and
what lets repeated runs of one individual be averaged. All five are recorded, not
just the ones the tree references. They are read off the `weights:` line the run
printed, so they are the values actually used; if that line is ever missing,
`weights` comes back `{}` rather than a guess.

The exchanges are taken from stdout only, so the loading bars and warnings that
arrive on stderr cannot leak in, and a reply wrapping over several lines is kept
whole (the newlines survive as `
` in the JSON string). A question whose reply
never arrived — a run killed mid-generation — keeps an empty `answer` rather than
vanishing, and a run that failed before answering at all leaves `"exchanges": []`,
which still parses. The `qa` column in `results.txt` shows the count without
opening anything.

`BAD` individuals are skipped by default. They stop at their bad combine step,
but only *after* paying for a full model load, so running them costs the same as
a real evaluation and tells you what `index.txt` already said.
`--include-blocked` runs them anyway and captures the error.

**Individual failures are results, not pipeline failures.** A chromosome that
crashes is recorded in `results.txt` and the sweep carries on; the last line of
its output is echoed so a systemic problem is obvious. Only a sweep where
*nothing* ran returns a failing exit code.

Children are launched with `sys.executable`, so they inherit whichever
interpreter you started this with — run it with the venv's python (see the PATH
gotcha below) or every child will fail on `import unsloth`.

### 5. `evaluate_run.py` → `quality` in each transcript

Scores every answer with a judge model — a *different* model from the blended
one that produced them.

```bash
python evaluate_run.py
```

| Flag | Default | Meaning |
|---|---|---|
| `--run-dir` | `run` | folder holding the transcripts |
| `--base-url` | `http://172.22.208.1:1234/v1` | OpenAI-compatible endpoint |
| `--api-key` | `$JUDGE_API_KEY` | bearer token; LMStudio ignores it |
| `--model` | endpoint's first chat model | judge model id |
| `--timeout` | 120 | seconds per grading call |
| `--limit` | 0 (all) | score only the first N transcripts |
| `--force` | off | re-score answers that already have a quality |

The score and the judge's reason land in the exchange they grade:

```json
{
  "question": "...",
  "answer": "...",
  "quality": 0.4,
  "reason": "generic advice, no concrete schedule"
}
```

`quality` is what selection reads; `reason` is what tells you whether the judge
is grading the way you intended — worth reading when a whole individual scores
0.0, or when scores cluster and you suspect the rubric rather than the answers.

`0.0` is worst, `1.0` is best — that is the number a fitness function selects
on. Every judge setting lives in one block at the top of the file, including
`SYSTEM_PROMPT`, which **is** the fitness criterion: it grades relevance,
usefulness, specificity, coherence and appropriateness, with anchors at 1.0 /
0.7 / 0.5 / 0.3 / 0.0. Tune it deliberately — the whole search optimises toward
whatever it rewards.

**Local by default, cloud by swap.** The judge speaks the OpenAI-compatible
`/v1/chat/completions` API, so a hosted model is a URL change:

```bash
python evaluate_run.py --model gpt-4o-mini --base-url https://api.openai.com/v1
```

Claude is *not* OpenAI-compatible — using a Claude model as the judge needs a
separate backend via the `anthropic` SDK.

**Resumable.** An exchange that already has a `quality` is skipped unless
`--force`, and each file is saved as it completes, so an interrupted sweep keeps
its work. An empty answer scores `0.0` without spending a call.

Reply parsing is deliberately tolerant — bare JSON, code-fenced JSON, JSON
wrapped in prose, and a bare number all work. Two traps worth knowing about,
both hit on the first real run: the score is requested **before** the reason so
a long reason cannot truncate it away, and `MAX_TOKENS` is generous because a
reasoning judge spends its budget thinking and returns an empty message if it
runs out mid-thought.

### 6. `test.py` → `run/test_*`

Try one chromosome by hand without touching `population.txt`. Set the variable
at the top of the file and run it:

```python
CHROMOSOME = "CAT.L1.L2.w5.w2.w2.w1"
```

```bash
python test.py
```

Or pass one straight in:

```bash
python test.py CAT.SVD.LIN.L1.L2.L3.L1.w3.w3.w2.w1
```

It prints the tree, the build plan and a verdict, then writes
`run/test_tree.txt` (same format as `trees.txt`) and `run/test_run.py` (same
script as those in `run/`). The `test_` prefix keeps them apart from the
population's `run_NNN.py` and `trees.txt`:

```
chromosome: CAT.SVD.LIN.L1.L2.L3.L1.w3.w3.w2.w1

tree
    CAT
    SVD.LIN
    L1.L2.L3.L1
    w3.w3.w2.w1

build order (deepest first)
    n1_L1      = L1 @ w3                       rank 16
    n2_L2      = L2 @ w3                       rank 16
    n3_SVD     = SVD(n1_L1, n2_L2)             rank 16
    n4_L3      = L3 @ w2                       rank 16
    n5_L1      = L1 @ w1                       rank 16
    n6_LIN     = LIN(n4_L3, n5_L1)             rank 16
    n7_CAT     = CAT(n3_SVD, n6_LIN)           rank 32   <-- generation runs through this one

verdict: ok -- 7 adapters, final rank 32
```

`test.py` calls the same builders the batch tools use (`draw_trees.draw`,
`generate_runs.plan/render`), so a chromosome tested here produces byte-identical
output to what it would get as a line in `population.txt`.

Bad input is reported rather than half-processed:

| Input | Result |
|---|---|
| `CAT.w1.L2.w5` | `not a valid chromosome: w1 is not a legal child of CAT` |
| `SVD.L1.L2.w1.w2` | `not a valid chromosome: expression must start with CAT` |
| a `LIN` above a `CAT` | `verdict: BLOCKED`, naming the node and both ranks |
| `CAT.L1.L2.w5.w2.w2.w1` | builds the tree, reports the 2 unused trailing symbols |

---

## Running a generated script

The generated scripts need the project venv, which lives one level up at
`D:\sage-is\loras\.venv` (Python 3.11.9, torch 2.11.0+cu128, unsloth, peft 0.20.0).

```bash
D:\sage-is\loras\.venv\Scripts\python.exe run\run_004.py
```

Then either the eval prompts, or your own question:

```bash
D:\sage-is\loras\.venv\Scripts\python.exe run\run_004.py "Help me plan my week."
```

### PATH gotcha

This machine has Python 3.13 first on PATH, and that one has no torch. A
`(.venv)` prompt only proves `PROMPT` was set at some point — it does not prove
`.venv\Scripts` is on *this* window's PATH, and the two drift apart across a new
`cmd`, a `cd /d`, or a window opened from elsewhere. Symptom:

```
ModuleNotFoundError: No module named 'unsloth'
```

Check what is actually resolving, and fix it:

```bash
where python
```

```bash
call D:\sage-is\loras\.venv\Scripts\activate.bat
```

Using the venv's full interpreter path always works regardless of PATH.

Note that the repo-root `activate.bat` ends with `cmd /k`, which spawns a
*nested* shell — if you run that one, use the new prompt it gives you rather than
the original window.

---

## Things to tune

Both live as tables at the top of every generated script, so they are easy to
change per individual or globally in `generate_runs.py`.

**`WEIGHTS`** — nothing in the repo defines what `w1`–`w5` are worth, so each
run draws them fresh, strictly between 0 and 1:

```python
WEIGHTS = {name: _weight() for name in ("w1", "w2", "w3", "w4", "w5")}
```

`_weight()` calls `random.random()`, which yields `[0.0, 1.0)`, and rejects an
exact `0.0` — leaving the open interval `(0, 1)`. The draw is printed at startup.

Because the default `WEIGHT_SEED = None` redraws every execution, **the same tree
scores differently each time it runs**. Set `WEIGHT_SEED` to an int in
`template_code.py` (or per script) when you need a comparison you can repeat.

**`LORA_SLOTS`** — one independent entry per slot, so any single line can be
repointed at a different adapter without touching the others:

```python
LORA_SLOTS = {
    "L1": os.path.join(_PROJECT, "Lora001", "my_planning_coach-lora_adapter"),
    "L2": os.path.join(_PROJECT, "Lora002", "my_planning_coach-lora_adapter"),
    "L3": os.path.join(_PROJECT, "Lora003", "my_planning_coach-lora_adapter"),
    "L4": os.path.join(_PROJECT, "Lora004", "my_planning_coach-lora_adapter"),
    "L5": os.path.join(_PROJECT, "Lora005", "my_planning_coach-lora_adapter"),
}
```

Five genuinely distinct adapters, one per slot, all trained on the same base
model (`unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit` — they must share a base
for PEFT to combine them). Their ranks differ:

| Slot | Folder | `r` |
|---|---|---|
| `L1` | `Lora001` | 16 |
| `L2` | `Lora002` | 16 |
| `L3` | `Lora003` | 8 |
| `L4` | `Lora004` | 4 |
| `L5` | `Lora005` | 32 |

`_PROJECT` is the folder holding this README, resolved from the generated
script's own location, so `run/` and `test/` both find the adapters. Any entry
may equally be an absolute path or a Hub repo id.

Ranks are **not** assumed equal. Each slot's rank is read from its own
`adapter_config.json` — at generation time for the docstring and `index.txt`
verdicts, and again at runtime by the generated script, which tracks the rank of
every intermediate adapter in `RANKS` and refuses a `linear` step whose two
inputs disagree. Point a slot at an `r=8` LoRA and `CAT(r16, r8)` reports rank
24, with more trees turning up `BAD` because their `LIN` nodes no longer match.

**`EVAL_PROMPTS`** — the questions each individual answers, read from
`training_set.txt` at startup rather than baked into the scripts, so editing that
file changes the eval set without regenerating anything:

```python
TRAINING_SET = os.path.join(_PROJECT, "training_set.txt")
EVAL_PROMPTS = _prompts(TRAINING_SET)
```

One prompt per line. Surrounding double or single quotes are optional (the file
currently uses them), blank lines are skipped, and a missing or empty file fails
with a clear message — at generation time as well as at runtime, since otherwise
every script would die at startup.

There is no fitness *score* yet; the scripts print replies for you to judge.
Scoring is the natural next piece, and it plugs in where `EVAL_PROMPTS` is
consumed.

**Reply length** — `ask(question, max_new_tokens=250)` caps each reply. Qwen ships
`max_length=32768` in its `generation_config.json`, and transformers warns when
both that and `max_new_tokens` are set, so the generated scripts clear it right
after `for_inference`:

```python
model.generation_config.max_length = None
```

`max_new_tokens` was taking precedence regardless, so this only silences the
warning — the effective cap is unchanged.

---

## Files

| Path | What it is |
|---|---|
| `plan.txt` | the original spec |
| `main.py` | runs the whole pipeline; add future steps to its `STEPS` list |
| `generate_population.py` | grows random trees → `run/population.txt` |
| `run/population.txt` | 100 K-expressions, one per line |
| `draw_trees.py` | draws every individual → `run/trees.txt` |
| `run/trees.txt` | 100 tree drawings in level-row layout |
| `generate_runs.py` | fills `template_code.py`, one runnable script per individual → `run/` |
| `template_code.py` | the generated script with `@@MARKERS@@` for the varying parts |
| `template_code_mocked.py` | the same, mocked: no model load, random answers and scores |
| `run/run_001.py` … `run_100.py` | the generated combination scripts |
| `run/index.txt` | script → state, final rank, expression |
| `training_set.txt` | the eval prompts, one per line, read by every generated script |
| `process_run.py` | runs every generated script → `run/output_NNN.txt`, `run/results.txt` |
| `evaluate_run.py` | scores every answer with a judge model → `quality` in the transcripts |
| `test.py` | try a single chromosome → `run/test_*` |
| `run/test_tree.txt`, `run/test_run.py` | output for the chromosome currently set in `test.py` |
| `combination.py` | the original two-adapter script the generated code is modelled on |

### Pipeline

```
main.py                       runs all of this in order
   |
plan.txt                      the rules
   |
generate_population.py  -->   run/population.txt  (the chromosomes)
   |                              |
   |                          draw_trees.py  -->  run/trees.txt    (readable trees)
   |                              |
   +--------------------->    generate_runs.py -> run/run_NNN.py   (runnable blends)
                                  +               run/index.txt
                              template_code.py    (the shape of those scripts)

                                  |
                              process_run.py -> run/output_result_NNN.json (replies + weights)
                                  evaluate_run.py -> + quality per answer
                                                run/results.txt

test.py  -->  run/test_tree.txt + run/test_run.py  (one chromosome, same builders)
```

Regenerating from scratch:

```bash
python main.py
```

Or one stage at a time, which is what `main.py` does for you:

```bash
python generate_population.py --count 100 --seed 42 --unique
```

```bash
python draw_trees.py
```

```bash
python generate_runs.py
```
