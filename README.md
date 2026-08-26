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
blocked**. Blocked scripts are still generated, with a `NOTE` naming the
offending node and both ranks, and their individual is recorded with
`state = 'BAD'`; at runtime they stop themselves with the same message rather
than letting a bare `ValueError` surface from inside PEFT.

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

A run that includes `population` starts a new sweep; one that does not resumes
the most recent one, or the one `--run` names. See
[Repeating a sweep](#repeating-a-sweep).

#### Where a sweep lives

Everything a sweep produces goes into one database, `run_db/gep.sqlite3`: the
population, every setting it ran under, every seed, every generated script, every
transcript and every score. The sweep itself is a row, so sweeps accumulate
instead of replacing each other, can be queried across, and — because the seeds
are stored rather than only the values they produced — can be *repeated*. See
[store.py](#7-storepy--the-sweep-database) below.

The only thing that reaches the disk is the generated `run_NNN.py` scripts, in
`run_db/`, and only until they have run; `process` deletes each one it has
processed. They are a cache of what the database already holds.

| Option | Default | Meaning |
|---|---|---|
| `--db` | `run_db/gep.sqlite3` | which database |
| `--run` | new, or the latest | the sweep to work on (`0` = the latest) |
| `--label` | none | a note stored with the sweep, to find it again later |
| `--run-dir` | `run_db` | where the generated scripts go |
| `--limit` | 0 (all) | process only the first N individuals |
| `--include-blocked` | off | also run the ones marked `BAD` |
| `--keep-scripts` | off | leave the generated scripts on disk after processing |
| `--timeout` | 900 | seconds to allow each script |
| `--force` | off | re-score answers that already have a quality |

#### Settings

Settings for a complete run live in `settings.py`:

```python
COUNT = 10
SEED = 42      # an int repeats the same population; None grows a fresh one
UNIQUE = True
TEMPLATE = "template_code_mocked.py"
```

`SEED` controls the *chromosomes* only. Each individual's LoRA blend weights come
from its own seed — see `WEIGHT_MASTER_SEED`.

Every upper-case name in `settings.py` is snapshotted into the sweep when it
starts, so a knob added there is a knob recorded — and a resumed sweep reads the
settings **it** was created with, not whatever the file says now.

**Adding a step.** Append a `Step(name, callable, description)` to `main.py`'s
`STEPS`. The callable takes the `Context` — the connection, the run id, the
settings that sweep was created with, the run folder and the parsed options:

```python
Step("select", step_select,
     "pick the survivors of this sweep -> individuals.selected"),
```

### 1. `generate_population.py` → the `individuals` rows

The root module: it owns the alphabet, the `Node` type, and the
`encode`/`decode` pair every other module reads trees with — there is no second
parser anywhere. The `population` step calls `build_population()` and stores one
row per chromosome.

| Setting | Default | Meaning |
|---|---|---|
| `COUNT` | 10 | how many individuals |
| `SEED` | 42 | RNG seed; `None` draws one and records it |
| `MAX_DEPTH` | 4 | deepest level an *operator* may sit at (root is level 0) |
| `BRANCH_PROB` | 0.6 | chance an operator is arity 2 and keeps the branch growing |
| `UNIQUE` | on | reject duplicate expressions |

Size varies per individual: a max operator depth is drawn from `1..MAX_DEPTH`,
then `BRANCH_PROB` decides whether each operator keeps growing (arity 2) or
closes the branch off with an `L*`. Every expression is decoded and re-encoded
before being stored, so nothing lands in the database that cannot be read back.

A population of 100 drawn with `SEED = 42` runs 5–32 symbols per individual
(mean 9.5) at tree depths 2–5.

### 2. `draw_trees.py` → `individuals.tree`

Draws each chromosome in the layout `plan.txt` uses — the expression, a blank
line, then one row per tree level — and the `trees` step stores that drawing on
the individual, so a sweep carries a readable picture of every tree it grew.

Trailing symbols that the tree does not consume are reported as
`(unused tail: ...)` rather than dropped silently — `plan.txt`'s first example
has two such symbols. A chromosome that cannot be drawn at all is stored with
its complaint under a `!!` marker rather than being skipped.

### 3. `generate_runs.py` + `template_code.py` → `individuals.script_source`

Turns every individual into a self-contained runnable script by filling in
`template_code.py`. Which template gets filled is `TEMPLATE` in `settings.py`.

The `runs` step stores each script in full, alongside the verdict the rank
arithmetic reached and the rank of the final adapter, then writes the scripts out
to `run_db/` ready for `process`:

```
number  state  rank  chromosome
1       ok     32    CAT.L1.L3.w2.w2
...
73      BAD    48    CAT.L1.LIN.w5.CAT.L3.L5.L1.w2.w4.w5
```

`python store.py --show 0` prints that table for a stored sweep.

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
re-run `python main.py runs`. Only add code to the generator itself when the new
part varies per individual.

#### `template_code_mocked.py` — the dry run

Which template gets filled is `TEMPLATE` in `settings.py`, so the same generator
produces a different kind of script from the same population:

```python
TEMPLATE = "template_code_mocked.py"
```

The mocked template has the same markers and produces the same shaped script,
but loads nothing and generates nothing: `ask()` assembles a reply out of canned
fragments and `grade()` draws a quality with a reason to match. A whole sweep
then takes **seconds instead of hours, with no GPU and no judge**, which is what
you want when the thing under test is the pipeline rather than a blend.

What it keeps real, so a dry run tells you something true about a population:

- the weight draw, and the `weights:` line the pipeline reads it from
- the ranks, read from each slot's own `adapter_config.json`
- the `attach`/`combine` order, and PEFT's equal-rank rule for `linear` — a
  `BAD` individual stops at the same node with the same message, so the `ok`/
  `BAD` split is the split you will get for real

What it fakes is the answers and the scores. Mocked scripts print `QUALITY:` and
`REASON:` lines after each reply; `process_run.exchanges()` folds those into the
transcript, so a mocked sweep arrives already scored and `evaluate` skips it
without contacting a judge at all. **A mocked quality is noise** — never read one
as a result.

Two things adjust themselves rather than needing a flag: `process` drops its venv
check when the scripts it is about to run do not import unsloth, so a mocked
sweep runs under any Python 3; and `evaluate` only reaches for the judge when
some answer actually lacks a quality.

`MOCK_SEED` fixes the fake answers and scores, and `MOCK_LOAD_DELAY` /
`MOCK_ANSWER_DELAY` buy back some fake slowness — useful for exercising
`--timeout`.

### 4. `process_run.py` → `executions`, `exchanges`

The `runs` step writes the scripts; this one runs them. `main.py` hands each
script to `process_run.launch()` and files what it said back into the database.

```bash
python main.py process --limit 3
```

| Option | Default | Meaning |
|---|---|---|
| `--limit` | 0 (all) | run only the first N individuals |
| `--include-blocked` | off | also run the ones marked `BAD` |
| `--keep-scripts` | off | leave the generated scripts on disk afterwards |
| `--timeout` | 900 | seconds to allow each script |

Each individual runs as a **separate process** — every script loads the base
model at import and attaches its own adapters, so they cannot share an
interpreter. That makes this the expensive step: one model load per individual.
`--limit 3` is the way to smoke-test before committing to a full sweep.

Each run becomes an `executions` row — exit code, verdict, seconds, the weight
seed it was stamped with, the weights it drew, and the whole of stdout and stderr
— with one `exchanges` row per question:

```sql
SELECT x.position, x.question, x.answer, x.quality
  FROM exchanges x
  JOIN executions e ON e.id = x.execution_id
 WHERE e.individual_id = 7
 ORDER BY x.position;
```

`executions` is a table rather than a column because the same chromosome run
again is a second result, not a correction of the first — so nothing is
overwritten and two runs of one individual can be compared or averaged.

The weights matter as much as the tree. Two individuals with the same tree and
different weights answer differently, so a score belongs to a *blend*, not to a
tree alone. All five are recorded, not just the ones the tree references. They
are read off the `weights:` line the run printed, so they are the values actually
used; if that line is ever missing, `weights` comes back `{}` rather than a
guess.

The exchanges are taken from stdout only, so the loading bars and warnings that
arrive on stderr cannot leak in, and a reply wrapping over several lines is kept
whole. A question whose reply never arrived — a run killed mid-generation — keeps
an empty `answer` rather than vanishing, and a run that failed before answering
at all simply has no exchanges. The `answers` column of the `individual_quality`
view shows the count without a join.

`BAD` individuals are skipped by default. They stop at their bad combine step,
but only *after* paying for a full model load, so running them costs the same as
a real evaluation and tells you what the `runs` step already worked out.
`--include-blocked` runs them anyway and captures the error.

**Individual failures are results, not pipeline failures.** A chromosome that
crashes is recorded as an execution with its exit code and the sweep carries on;
the last line of its output is echoed so a systemic problem is obvious. Only a
sweep where *nothing* ran returns a failing exit code. The commit is per
individual, so an interrupted sweep keeps everything it had already done.

Children are launched with `sys.executable`, so they inherit whichever
interpreter you started this with — run it with the venv's python (see the PATH
gotcha below) or every child will fail on `import unsloth`.

### 5. `evaluate_run.py` → `exchanges.quality`

Scores every answer with a judge model — a *different* model from the blended
one that produced them. Only the most recent execution of each individual is
scored; older ones keep the scores they were given.

```bash
python main.py evaluate
```

| Where | Setting | Default | Meaning |
|---|---|---|---|
| `evaluate_run.py` | `BASE_URL` | `http://172.22.208.1:1234/v1` | OpenAI-compatible endpoint |
| `evaluate_run.py` | `API_KEY` | `$JUDGE_API_KEY` | bearer token; LMStudio ignores it |
| `evaluate_run.py` | `MODEL` | endpoint's first chat model | judge model id |
| `evaluate_run.py` | `TIMEOUT` | 300 | seconds per grading call |
| `main.py` | `--force` | off | re-score answers that already have a quality |

Every one of those is snapshotted into the sweep when it starts — `SYSTEM_PROMPT`
included — so a stored sweep can say what it was graded by.

The score and the judge's reason land on the exchange they grade, with the model
that gave them and when:

```
position  quality  reason                              judge_model
1         0.4      generic advice, no concrete schedule qwen2.5-7b-instruct
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
`/v1/chat/completions` API, so a hosted model is a URL change at the top of
`evaluate_run.py`:

```python
BASE_URL = "https://api.openai.com/v1"
MODEL = "gpt-4o-mini"
```

Claude is *not* OpenAI-compatible — using a Claude model as the judge needs a
separate backend via the `anthropic` SDK.

**Resumable.** An exchange that already has a `quality` is skipped unless
`--force`, and each score is committed as it arrives, so an interrupted sweep
keeps its work. An empty answer scores `0.0` without spending a call, and a sweep
where nothing needs grading never contacts the judge at all.

Reply parsing is deliberately tolerant — bare JSON, code-fenced JSON, JSON
wrapped in prose, and a bare number all work. Two traps worth knowing about,
both hit on the first real run: the score is requested **before** the reason so
a long reason cannot truncate it away, and `MAX_TOKENS` is generous because a
reasoning judge spends its budget thinking and returns an empty message if it
runs out mid-thought.

### 6. `test.py` → `run/test_*`

Try one chromosome by hand without starting a sweep. Set the variable at the top
of the file and run it:

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
`run/test_tree.txt` (the same drawing a sweep stores on an individual) and
`run/test_run.py` (the same script a sweep generates). They go in `run/` — a
folder of their own, beside `run_db/` — and the `test_` prefix keeps them apart
from a sweep's `run_NNN.py`:

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

`test.py` calls the same builders the pipeline uses (`draw_trees.draw`,
`generate_runs.plan/render`), so a chromosome tested here produces byte-identical
output to what it would get as an individual in a sweep.

Bad input is reported rather than half-processed:

| Input | Result |
|---|---|
| `CAT.w1.L2.w5` | `not a valid chromosome: w1 is not a legal child of CAT` |
| `SVD.L1.L2.w1.w2` | `not a valid chromosome: expression must start with CAT` |
| a `LIN` above a `CAT` | `verdict: BLOCKED`, naming the node and both ranks |
| `CAT.L1.L2.w5.w2.w2.w1` | builds the tree, reports the 2 unused trailing symbols |

### 7. `store.py` → the sweep database

The schema, and the only module that imports `sqlite3`. Everything above is a
library of pure functions — `build_population`, `draw`, `plan`/`render`,
`launch`, `judge` — and `main.py` is what calls them and puts the results here.

```
runs          one sweep: when, which template, which interpreter, which commit
  settings    every knob it ran under, including the seeds
  individuals the population: chromosome, tree, rank, verdict, and the
              generated script in full
    executions  one per time that individual was run: exit code, seconds, the
                weight seed and the weights it drew, stdout, stderr
      exchanges the questions and answers, and the judge's score for each
```

`evaluate` scores the most recent execution of each individual; older ones keep
the scores they were given.

```bash
python store.py --list
```

```bash
python store.py --show 0
```

`--show` takes a run id, or `0` for the most recent, and prints the settings the
sweep ran under alongside every individual and its mean quality. There is also a
view for the query you actually want:

```sql
SELECT number, chromosome, quality, weights
  FROM individual_quality
 WHERE run_id = 1 AND state = 'ok'
 ORDER BY quality DESC
 LIMIT 5;
```

```bash
python store.py --export 0 --into export
```

Writes a stored sweep back out as a folder of text files — `population.txt`,
`trees.txt`, `index.txt`, the scripts, `output_NNN.txt`,
`output_result_NNN.json`, `results.txt` — for the times you want to diff two
populations, grep a transcript or hand someone a folder. It is a *view* of a
sweep, derived from the database; the database stays the store.

The exported transcript carries everything needed to make sense of it on its
own — which tree was built, and which weights built it:

```json
{
  "chromosome": "CAT.L1.L3.w2.w2",
  "weights": { "w1": 0.3529, "w2": 0.2882, "w3": 0.8712, "w4": 0.8846, "w5": 0.5110 },
  "exchanges": [
    {
      "question": "Help me organize my desktop.",
      "answer": "Before we lay anything out, let's call the one thing...",
      "quality": 0.65,
      "reason": "asks a useful clarifying question but gives no concrete step"
    }
  ]
}
```

#### Repeating a sweep

This is what the database is for. Each individual gets its own weight seed,
derived from the sweep's `WEIGHT_MASTER_SEED` and the individual's number,
stamped into its generated script and stored beside it. Re-running `process`
produces the identical draw — and because the seed is per individual rather than
per sweep, no two individuals share a blend.

Seeds left as `None` in `settings.py` are drawn when the sweep is created and
stored as the number that was drawn, so a sweep is repeatable even when it was
never asked to be — whatever it used is written down.

Every step but `population` can be re-run against a sweep already in the
database, and reads the settings **that sweep** was created with rather than
whatever `settings.py` says now. That is what makes a resumed sweep still be the
same sweep.

```bash
python main.py process evaluate
```

Resumes the most recent sweep; `--run 3` names one instead.

#### What still touches the disk

The generated `run_NNN.py` scripts, and only those — `process` launches them as
subprocesses, so they have to be real files. They land in `run_db/`, beside the
database, because a generated script finds the LoRA folders and
`training_set.txt` by going up **one** level from itself. A folder one level
below the project works; a folder inside one would not.

They are a cache of `individuals.script_source`, not a second copy of the truth,
which is what makes both halves of their life cycle safe: `process` writes any
that are missing or stale before it runs, and **deletes each one it has
processed** once the sweep is through. So a finished sweep leaves `run_db/`
holding the database and nothing else — no spent scripts piling up, and no stale
script for someone to run by hand a week later.

```bash
python main.py process --keep-scripts
```

keeps them when you want to read or re-run one. Otherwise they come back from
the database on demand:

```bash
python main.py runs
```

Only the scripts that actually ran are removed. Ones skipped as `BAD`, or left
out by `--limit`, are still waiting and stay where they are.

---

## Running a generated script

The generated scripts need the project venv, which lives one level up at
`D:\sage-is\loras\.venv` (Python 3.11.9, torch 2.11.0+cu128, unsloth, peft 0.20.0).

```bash
D:\sage-is\loras\.venv\Scripts\python.exe run_db\run_004.py
```

Then either the eval prompts, or your own question:

```bash
D:\sage-is\loras\.venv\Scripts\python.exe run_db\run_004.py "Help me plan my week."
```

`process` deletes each script it has run, so bring one back with
`python main.py runs` (or keep them with `--keep-scripts`) before running it by
hand.

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
change per individual or globally in `template_code.py`.

**`WEIGHTS`** — nothing in the repo defines what `w1`–`w5` are worth, so each
run draws them fresh, strictly between 0 and 1:

```python
WEIGHTS = {name: _weight() for name in ("w1", "w2", "w3", "w4", "w5")}
```

`_weight()` calls `random.random()`, which yields `[0.0, 1.0)`, and rejects an
exact `0.0` — leaving the open interval `(0, 1)`. The draw is printed at startup.

A script whose `WEIGHT_SEED` is `None` redraws every execution, so **the same
tree scores differently each time it runs**. The pipeline never leaves it that
way: the `runs` step stamps each individual with its own seed, derived from the
sweep's `WEIGHT_MASTER_SEED` and the individual's number, so a sweep repeats
weight for weight without every individual sharing one blend. Set
`WEIGHT_MASTER_SEED` to an int to fix that from the start; left `None`, one is
drawn when the sweep is created and stored as the number drawn, which is just as
repeatable after the fact. To try one particular draw by hand, edit the
`WEIGHT_SEED` line of a generated script directly.

`WEIGHT_SEED` is a template marker, so it is the one setting a generated script
carries as a literal rather than inheriting from the template.

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
script's own location, so scripts in `run_db/` and `test.py`'s output in `run/`
both find the adapters. Any entry may equally be an absolute path or a Hub repo
id.

Ranks are **not** assumed equal. Each slot's rank is read from its own
`adapter_config.json` — at generation time for the docstring and the `state` and
`rank` recorded on the individual, and again at runtime by the script, which
tracks the rank of
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
| `main.py` | the entry point and the driver; add future steps to its `STEPS` list |
| `settings.py` | COUNT, SEED, TEMPLATE and the rest — every knob, in one place |
| `store.py` | the database: schema, helpers, `--list/--show/--export` |
| `run_db/gep.sqlite3` | every sweep ever run, with its settings, seeds, transcripts and scores |
| `run_db/run_001.py` … | the generated combination scripts, until `process` has run them |
| `generate_population.py` | the alphabet, `encode`/`decode`, and the random draw |
| `draw_trees.py` | draws one chromosome as a tree → `individuals.tree` |
| `generate_runs.py` | fills `template_code.py`, one runnable script per individual |
| `template_code.py` | the generated script with `@@MARKERS@@` for the varying parts |
| `template_code_mocked.py` | the same, mocked: no model load, random answers and scores |
| `training_set.txt` | the eval prompts, one per line, read by every generated script |
| `process_run.py` | launches a generated script and reads its transcript back |
| `evaluate_run.py` | the judge: its settings, its rubric, and one grading call |
| `test.py` | try a single chromosome → `run/test_*` |
| `run/test_tree.txt`, `run/test_run.py` | output for the chromosome currently set in `test.py` |
| `combination.py` | the original two-adapter script the generated code is modelled on |

### Pipeline

`main.py` runs all of this in order. Every module below is a library it calls;
the arrows end in tables, not files.

```
plan.txt                              the rules
   |
generate_population.build_population  -->  individuals (chromosome)
draw_trees.draw                       -->  individuals.tree
generate_runs.plan/render             -->  individuals.script_source
   +                                       + run_db/run_NNN.py  (must be files)
template_code.py                           (the shape of those scripts)

process_run.launch/exchanges          -->  executions, exchanges
evaluate_run.judge                    -->  exchanges.quality

store.py --show / --export                 reads any of it back out

test.py  -->  run/test_tree.txt + run/test_run.py  (one chromosome, same builders)
```

Running the whole thing:

```bash
python main.py
```

Or one stage at a time — the same steps, named:

```bash
python main.py population trees runs
```

```bash
python main.py process --limit 3
```

```bash
python main.py evaluate
```
