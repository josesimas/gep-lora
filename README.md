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
| `L1`–`L10` | 1 | a variable | one LoRA adapter |
| `w1`–`w10` | 0 | — | a blend weight |

The first symbol is the root, and may be any operator: `CAT`, `SVD`, `LIN`,
or an `L*` on its own — `L3.w2` is one adapter and nothing folded into it.
Only a variable cannot start a chromosome. A lone adapter runs at full
strength: weights are applied by the fold above a leaf, and it has none, so its
`w` is carried by the grammar but not used.

The alphabet is a ceiling (`MAX_SLOTS = 10` in `generate_population.py`). A
sweep draws from as many slots as its own `LORA_SLOTS` names -- `L1`..`Ln`, one
to ten of them with no gap, and `w1`..`wn` beside them -- so a sweep of five
adapters only ever holds `L1`–`L5` and `w1`–`w5`, exactly as before the
ceiling was raised, and adding `"L6": ...` .. `"L10": ...` to `LORA_SLOTS`
widens the search. `start_run.freeze()` refuses slots that are not `L1..Ln`.

Because `L*` only accepts variables and `CAT`/`SVD`/`LIN` only accept operators,
every leaf `w` sits under an `L`, and every `L` sits under a binary operator
or is the root.

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

This is the same idea as [combination.py](gep_lora/tools/combination.py), which stacks two
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

## What an `SVD` node costs

`svd` is the one operator that does real work at build time, and PEFT leaves a
trap in it. `_svd_generalized_task_arithmetic_weighted_adapter` runs
`torch.linalg.svd` on each module's delta weight and returns `Vh[:new_rank, :]`,
which `add_weighted_adapter` assigns straight to that module's `lora_A`. That
slice is a **view**: it pins the whole `V` matrix for as long as the adapter
lives, and `V` is sized by the delta weight rather than by the rank. It also
comes back contiguous, so `.contiguous()` is a no-op on it — only a copy lets
the buffer go.

On this base model, measured on an A6000:

| module | delta shape | `lora_A` | buffer pinned behind it |
|---|---|---|---|
| `down_proj` | 1536 × 8960 | 0.55 MB | **306 MB** |
| `gate_proj`, `up_proj` | 8960 × 1536 | 0.09 MB | 9 MB each |
| `q_proj`, `o_proj` | 1536 × 1536 | 0.09 MB | 9 MB each |
| `k_proj`, `v_proj` | 256 × 1536 | 0.09 MB | 9 MB each |

360 MB a layer across 28 layers — a single `SVD` node used to add **~10 GB of
resident VRAM** to hold about 18 MB of weights, seven times the model it
attaches to. [template_code.py](gep_lora/core/templates/template_code.py) closes it in `combine()`:
`svd_full_matrices=False` asks for the thin SVD (the first `new_rank` singular
vectors are identical either way, and the full n × n `V` is only a bigger buffer
to compute and throw away), and `_compact()` then clones every weight still
sitting on storage larger than itself.

One `CAT.SVD.LIN` individual, start to first answer:

| stage | before | after |
|---|---|---|
| base model, 4-bit | 1.47 GB | 1.47 GB |
| four leaf adapters attached | 1.64 GB | 1.64 GB |
| **after the `SVD` node** | **11.58 GB** | **1.71 GB** |
| generating | 11.68 GB | 1.81 GB |
| what `nvidia-smi` shows | **14.0 GB** | **3.6 GB** |

The transient peak during the node is still ~4.7 GB, because the views pile up
across the module loop and are only released once it ends — that peak, not the
resting figure, is what a batch has to fit.

The mocked template builds no adapters, so it has none of this and says nothing
about VRAM; the note where `_compact()` would go says so.

---

## Scripts

Run everything from `project/`.

### 0. `gep_lora/core/pipeline/start_run.py` — the whole pipeline

```bash
python start_run.py
```

Runs the steps in order — population, trees, runs, process, evaluate,
fitness — stopping at the first failure, since each step builds on the one
before it.

**It stops after `fitness`**, and the three steps after it are the reason:
`elitism`, `selection` and `mutation` do not describe this generation, they
build the *next* one, and `gep_lora/core/pipeline/start_run.py` on its own is a run of one generation with
no next one. Stopping there leaves the sweep holding the individuals that were
actually scored, each still described by the script that earned its transcript
— which is what `store.py --export`, the HTML report and
`test_run_with_dataset.py` all want to read.

```bash
python start_run.py --next-generation
```

runs them anyway, leaving a population bred and varied for another generation.
That is what `gep_lora/core/pipeline/main.py` passes, and what you want before carrying a sweep on
by hand. Naming steps explicitly always runs exactly those: `python start_run.py
selection` selects, flag or no flag.

`process` and `evaluate` are the slow ones: `process` executes the generated
scripts, one base-model load per individual — `PROCESS_RUN_BATCH_SIZE` of them
at a time — and `evaluate` then makes one grading call per answer. A full `python start_run.py` is therefore a long operation,
and `python start_run.py population trees runs` stops short of both. `fitness`,
and `elitism`, `selection` and `mutation` behind it, work over what they stored
and cost nothing.

Run it with the **venv's python** — `process` launches each generated script
with `sys.executable`, so the wrong interpreter fails every individual. It now
checks this up front rather than discovering it once per individual.

```bash
python start_run.py runs
```

Runs a subset. Handy after editing `template_code.py`, when the population is
still good. Steps always execute in pipeline order regardless of how you type
them, and `python start_run.py --list` shows them without running anything.

A run that includes `population` starts a new sweep; one that does not resumes
the most recent one, or the one `--run` names. See
[Repeating a sweep](#repeating-a-sweep).

#### Where a sweep lives

Everything a sweep produces goes into one database, `run_db/gep.sqlite3`: the
population, every setting it ran under, every seed, every generated script, every
transcript and every score. The sweep itself is a row, so sweeps accumulate
instead of replacing each other, can be queried across, and — because the seeds
are stored rather than only the values they produced — can be *repeated*. See
[store.py](#13-storagestorepy--the-sweep-database) below.

The only thing that reaches the disk is the generated `run_NNN.py` scripts, in
`run_db/`, and only until they have run; `process` deletes each one it has
processed. They are a cache of what the database already holds.

| Option | Default | Meaning |
|---|---|---|
| `--db` | `run_db/gep.sqlite3` | which database |
| `--run` | new, or the latest | the sweep to work on (`0` = the latest) |
| `--label` | none | a note stored with the sweep, to find it again later |
| `--from-db` | off | read the eval prompts from the sweep's own stored dataset rows rather than the files its settings name (needs `--run`) |
| `--run-dir` | `run_db` | where the generated scripts go |
| `--limit` | 0 (all) | process only the first N individuals |
| `--include-blocked` | off | also run the ones marked `BAD` |
| `--include-unchanged` | off | also run individuals whose chromosome has not changed |
| `--keep-scripts` | off | leave the generated scripts on disk after processing |
| `--timeout` | 900 | seconds to allow each script |
| `--force` | off | re-score answers that already have a quality |

#### Settings

Settings for a complete run live in `settings.py`:

```python
COUNT = 10
SEED = None    # None grows a fresh one (and records it); an int repeats it
UNIQUE = True
TEMPLATE = "template_code_mocked.py"
```

`SEED` controls the *chromosomes* only. Each individual's LoRA blend weights come
from its own seed — see `WEIGHT_MASTER_SEED`. All five of the search's seeds
(`SEED`, `WEIGHT_MASTER_SEED`, `SELECTION_MASTER_SEED`, `MUTATION_MASTER_SEED`,
`WEIGHT_MUTATION_MASTER_SEED`) default to `None`: each is drawn when a sweep is
created and stored in its settings, so every sweep searches differently and any
one of them can still be repeated from what it recorded. Set an int to make
sweeps repeat from the start.

Every upper-case name in `settings.py` is snapshotted into the sweep when it
starts, so a knob added there is a knob recorded — and a resumed sweep reads the
settings **it** was created with, not whatever the file says now.

#### The dataset, saved with the sweep

Creating a sweep does one more thing before the first step runs: it calls
`add_dataset.save_all()`, which reads the dataset and writes it into the
**`datasets`** table, whole.

```
new run 7 in run_db/gep.sqlite3
training    60 record(s), 60 with a reference answer, from D:\...\medical_training_lora_dataset.json
validation  20 record(s), 20 with a reference answer, from D:\...\medical_validation_lora_dataset.json
testing     20 record(s), 20 with a reference answer, from D:\...\medical_testing_lora_dataset.json
```

Alongside the settings, and for the same reason. A fitness number means *this
blend, under these knobs, on these questions*, and a settings table that records
`TRAINING_SET` records only **where** the questions were — the files themselves
go on being edited, repointed and regenerated. A sweep read back in a month can
now say what it was actually asked.

The table holds the three splits — `training`, `validation`, `testing` — one row
per record, each with the question (the user turn), the reference answer (the
assistant turn, where the record has one) and the line it was read from, so a
later reader can disagree with this repo's parsing of it. `position` is 1-based
over non-blank lines from the top, the same numbering `exchanges.position` uses.

Which files those are is three settings:

```python
TRAINING_SET   = "datasets/medical_training_lora_dataset.json"
VALIDATION_SET = None      # e.g. "datasets/medical_validation_lora_dataset.json"
TESTING_SET    = None      # e.g. "datasets/medical_testing_lora_dataset.json"
```

Only the training split is read by the search: it is the eval set every
generated script asks and every fitness number is earned on. The other two are
stored when they are named and read by nothing yet — the point of storing them
now is that a later validation or test pass gets the questions this sweep was
built beside rather than whatever those files hold by then. A split left `None`
is simply not part of the sweep and leaves no rows; a split naming a file that
cannot be read stops the sweep at its first second rather than an hour in.

Every record is stored, uncapped. `TRAINING_COUNT` says how many of them an
*individual* is judged on, which is a fact about the sweep and already in the
settings table, not a fact about the dataset.

`python -m gep_lora.core.storage.store --show` prints a line per split, and `--export` writes each one
back out as `dataset_<split>.txt` — the lines as they were read, so the export
can be diffed against the file on disk today to see whether the dataset has
moved under the sweep.

**A split the sweep was never given.** A validation or testing set decided on
after the sweep started has nowhere to go — `save_all()` runs once, as the sweep
is created, and `gep_lora/core/pipeline/continue_run.py` resumes a sweep that already recorded its
dataset. The same module is that act performed afterwards, by hand:

```bash
python -m gep_lora.core.storage.add_dataset datasets/medical_validation_lora_dataset.json --split validation
```

```
validation  20 record(s), 20 with a reference answer, from D:\...\medical_validation_lora_dataset.json
run 7 in run_db/gep.sqlite3
```

`--db` names a database (default `settings.DB_PATH`), `--run` a sweep (`0`, the
default, is the most recent). It parses nothing and stores nothing itself:
`generate_runs.dataset_records()` reads the file — the same two shapes, the same
1-based positions, uncapped — and `store.save_dataset()` writes it.

A split added by hand is therefore not *like* one the driver stored, it is
stored by the same function: `save_all()` is a loop over `add()`, once per split
its settings name, and the command line is `add()` once. One reader of a dataset
file, one INSERT, one printed line — which is why `--show` and `--export` pick
either up without knowing which it was.

It will **not** overwrite a split the sweep already holds unless `--replace`
says so. Those rows are what that sweep was built on, and a dataset changing
under a finished sweep is the thing the table exists to prevent; adding the
split that was missing is a different act from rewriting the one that was there.

**Adding a step.** Append a `Step(name, callable, description)` to `gep_lora/core/pipeline/start_run.py`'s
`STEPS`. The callable takes the `Context` — the connection, the run id, the
settings that sweep was created with, the run folder and the parsed options:

```python
Step("select", step_select,
     "pick the survivors of this sweep -> individuals.selected"),
```

### 1. `gep_lora/core/search/generate_population.py` → the `individuals` rows

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
| `ROOT_LEAF_PROB` | 0.1 | chance the whole tree is one adapter alone; otherwise the root is `CAT`, `SVD` or `LIN`, uniformly |
| `UNIQUE` | on | reject duplicate expressions |

Size varies per individual: a max operator depth is drawn from `1..MAX_DEPTH`,
then `BRANCH_PROB` decides whether each operator keeps growing (arity 2) or
closes the branch off with an `L*`. Every expression is decoded and re-encoded
before being stored, so nothing lands in the database that cannot be read back.

A population of 100 drawn with `SEED = 42` runs 5–32 symbols per individual
(mean 9.5) at tree depths 2–5.

### 2. `gep_lora/core/search/draw_trees.py` → `individuals.tree`

Draws each chromosome in the layout `plan.txt` uses — the expression, a blank
line, then one row per tree level — and the `trees` step stores that drawing on
the individual, so a sweep carries a readable picture of every tree it grew.

Trailing symbols that the tree does not consume are reported as
`(unused tail: ...)` rather than dropped silently — `plan.txt`'s first example
has two such symbols. A chromosome that cannot be drawn at all is stored with
its complaint under a `!!` marker rather than being skipped.

### 3. `gep_lora/core/blends/generate_runs.py` + `gep_lora/core/templates/template_code.py` → `individuals.script_source`

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

`python -m gep_lora.core.storage.store --show 0` prints that table for a stored sweep.

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

Blocks are `TREE`, `BUILD_ORDER`, `NOTE`, `ATTACH_LEAVES`, `COMBINE_NODES`,
`WEIGHT_SEED`, `BASE_MODEL`, `CHAT_TEMPLATE`, `TRAINING_SET`, `TRAINING_COUNT`,
`LORA_SLOTS`;
inline values are `SCRIPT_NAME`, `PROVENANCE`, `LABEL`, `EXPRESSION`,
`LEAF_COUNT`, `FINAL_ADAPTER`, `FINAL_RANK`. Any marker left unfilled raises
rather than being written into a generated file.

`WEIGHT_SEED`, `BASE_MODEL`, `TRAINING_SET`, `TRAINING_COUNT` and `LORA_SLOTS`
are blocks rather than inline values because each stands in for the
*assignment* — so a generated script carries `WEIGHT_SEED = 12345`,
`BASE_MODEL = '...'`, `TRAINING_SET = '...'`, `TRAINING_COUNT = 10` and the whole
`LORA_SLOTS = {...}` dict as plain literals. They are also the names a linter
calls undefined in the templates themselves, and finds defined in every file
generated from them. `BASE_MODEL` is a setting rather than a line in the
templates so that one name reaches all three of them — the two individual
templates and `template_baseline.py`, the control `llm_judge_baseline` grades
against.

`CHAT_TEMPLATE` travels the same way, to the same three and to the lora servers:
it is the words every prompt is written in. `None`, the default, is the base
model's own template, from its repo; a name (`"qwen-2.5"`, `"llama-3.1"`, ...)
swaps in unsloth's template of that name. It has to be the template the adapters
were trained under -- `create_lora.py --chat-template` says which, default the
model's own -- because an adapter answers in the format it learned, and a base
model prompted in a format it was not built for stops behaving as itself:
Qwen3.5 under `"qwen-2.5"` loses the empty `<think>` block its own template
writes and reasons out loud until the length cap. On the Qwen2.5 repos the two
render byte-for-byte the same. Every template used to hardcode `"qwen-2.5"`, so
a sweep stored before the setting existed resolves to that
(`generate_runs.chat_template_name()`) rather than to `settings.py`'s. A
remote client sends its template with `/build` and the server refuses one it was
not started under, as it already did for `BASE_MODEL`.

To change what every generated script looks like, edit `template_code.py` and
re-run `python start_run.py runs`. Only add code to the generator itself when the new
part varies per individual.

#### `gep_lora/core/templates/template_code_mocked.py` — the dry run

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
sweep runs under any Python 3; and `evaluate` only reaches for a judge when some
answer actually lacks a quality -- whichever `EVALUATOR` is set, and whichever
`JUDGE_BACKEND` it would have asked through, so a mocked sweep neither contacts
an endpoint nor loads a judge model. The local scorers never reach for one at
all.

`MOCK_SEED` fixes the fake answers and scores, and `MOCK_LOAD_DELAY` /
`MOCK_ANSWER_DELAY` buy back some fake slowness — useful for exercising
`--timeout`.

### 4. `gep_lora/core/blends/process_run.py` → `executions`, `exchanges`

The `runs` step writes the scripts; this one runs them. `gep_lora/core/pipeline/start_run.py` hands each
script to `process_run.launch()` and files what it said back into the database.

```bash
python start_run.py process --limit 3
```

| Option | Default | Meaning |
|---|---|---|
| `--limit` | 0 (all) | run only the first N individuals |
| `--include-blocked` | off | also run the ones marked `BAD` |
| `--include-unchanged` | off | also run individuals whose chromosome has not changed since their last execution |
| `--keep-scripts` | off | leave the generated scripts on disk afterwards |
| `--timeout` | 900 | seconds to allow each script |

Each individual runs as a **separate process** — every script loads the base
model at import and attaches its own adapters, so they cannot share an
interpreter. That makes this the expensive step: one model load per individual.
`--limit 3` is the way to smoke-test before committing to a full sweep.

Separate processes are also what lets them overlap. `PROCESS_RUN_BATCH_SIZE` in
`settings.py` is how many run **at once**: the selected individuals are cut into
consecutive batches of that size, and a batch is waited out before the next one
starts — a fixed ceiling on concurrency, not a queue that refills, so a batch
takes as long as its slowest member. `1` is one at a time, and the output then
reads exactly as it did before batching existed.

The ceiling is fixed because the cost is fixed: a batch of N is N copies of the
base model resident at the same time. Set it to what the GPU can actually hold —
reckon on ~3.6 GB resting and ~4.7 GB peak per script on this base model, the
peak being the `SVD` node (see [What an `SVD` node costs](#what-an-svd-node-costs)).
An individual that runs out of memory is recorded as a failed execution like any
other, so too large a batch does not stop a sweep — it quietly fills it with
failures. A mocked sweep loads nothing and can go far higher.

Results are stored in the order the batch was asked for rather than the order
its members finished, and only from the driver's own thread, so the database is
written exactly as it was one at a time; the commit is still per individual, so
an interrupted batch keeps whatever already came back. The `seconds` on an
execution is that script's own wall clock, so within a batch they overlap and no
longer add up to the time the step took. The scripts in a batch share the run
folder as their working directory, and so share the caches unsloth drops there.

### Paying for the model load once — `gep_lora/core/blends/lora_server.py`

"One model load per individual" is a fixed price, and on this machine it is most
of the bill: `import` at 13.3s plus `model_load` at 18.6s is 31.9s, about **54%**
of an average script, and no cheaper tree gets out of it.

`TEMPLATE = "template_remote_code.py"` in `settings.py` moves it. The scripts
become clients of a [`gep_lora/core/blends/lora_server.py`](gep_lora/core/blends/lora_server.py) process
that already holds the base model open: the client sends the build plan its tree
describes, the server attaches and folds the adapters, and then the client asks
its eval prompts through the same HTTP connection. That single line is the whole
switch, and switching back is the same line.

```bash
# by hand, one server, and one generated script against it
python -m gep_lora.core.blends.lora_server --base-model unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit --port 8770
GEP_LORA_SERVER=http://127.0.0.1:8770 python run_db/run_001.py
```

Measured here, on this base model: a `CAT` of two leaves went from ~35s to
**3.0s** — 1.0s to build the blend, 1.9s to answer. An `SVD` tree still costs
its ~70s fold, because that is arithmetic the server has to do too; what
vanishes is only the part that was the same for everybody.

**Nothing about what an individual is changes.** One script is still one
individual and one execution, still stored in `script_source`, still printing
`YOU:`/`COACH:` and `TIMING:` lines on stdout, still keeping a partial
transcript when it is killed, still turning an exit code into a verdict. The
server sits *behind* the scripts rather than in place of them, so `executions`,
`exchanges`, `phase_timings`, the mocked path and the testing pass all work
untouched. `model_load` simply stops appearing among the phases —
`python -m gep_lora.core.metrics.report` is where you watch it go.

[`gep_lora/core/blends/server_pool.py`](gep_lora/core/blends/server_pool.py) is the lifecycle. It starts
`LORA_SERVER_COUNT` servers on consecutive ports at the top of the `process`
step and stops them at the end, and it does so **only when the sweep's own
scripts are clients** — a test on `script_source`, the same shape as the
`imports_unsloth()` test that decides whether a sweep needs the venv. So a sweep
generated from either other template ignores every `LORA_SERVER_*` setting.
`test_run_with_dataset.py` starts a pool the same way, since the scripts a
testing pass runs are that sweep's own.

| Setting | Default | Meaning |
|---|---|---|
| `LORA_SERVER_COUNT` | 2 | how many servers, each holding its own copy of the model. Also caps the batch |
| `LORA_SERVER_HOST` / `LORA_SERVER_PORT` | `127.0.0.1` / 8770 | where they listen — consecutive ports from there |
| `LORA_SERVER_STARTUP_TIMEOUT` | 900 | seconds to wait for a server to finish loading |
| `LORA_SERVER_TIMEOUT` | 1800 | seconds one script waits on one request |
| `LORA_SERVER_RECYCLE_AFTER` | 0 | builds before a server is restarted between batches; 0 never |

**Where a server's own errors go.** A client only ever sees the one line its
request came back with, and that one line is all an execution's `stderr` can
honestly hold about somebody else's process. Everything else — the traceback
inside `add_weighted_adapter`, an OOM, PEFT's warnings, the model load itself —
is in that server's log, written beside the scripts and named for its port:

```
run_db/lora_server_8770.log
run_db/lora_server_8771.log
```

The pool prints the path when it starts. The files are **appended** to, with a
header per server life, so a recycled server does not overwrite the log of the
one whose last build is why you are reading it.

**The pool lives as long as the driver, not as long as the step.** It hangs off
the `Context`, which `gep_lora/core/pipeline/continue_run.py` builds once and reuses for every
generation it turns, so generation 2 finds generation 1's servers still up and
loads nothing:

```
# generation 2 of 3 -- population 4
reusing the 2 lora server(s) already up -- the base model has not been reloaded since they started
```

There is one override, `start_run.wants_the_card()`: with
`JUDGE_BACKEND = "unsloth"` the evaluate step loads a judge in this same
interpreter, and a pool left standing through it would be N base models on the
card while a judge looks for room for one more. On that backend, and only on it,
the pool comes down at the end of each process step and says so.

A whole `gep_lora/core/pipeline/main.py` search pays the startup **twice** — the first generation runs
through `gep_lora/core/pipeline/start_run.py` and the rest through `gep_lora/core/pipeline/continue_run.py`, and those are two
Contexts. That is where it stops: closing the last gap would mean a pool
outliving the driver that made it, which is a worse thing to own than one extra
model load.

Either way it is measured rather than assumed, as its own `start servers` row:

```
inside process -- 31.9s of wall time over 1 pass(es)
  what the step itself did (share of its 31.9s):
    phase                calls   seconds   share      mean     worst  shape
    start servers            1     22.4s   70.2%     22.4s     22.4s  per pass
    waiting on scripts       4      9.5s   29.7%             15.4s of script time, overlapping
  inside those 4 script(s) -- 15.4s of script time, 3.9s each:
    generate                 4      8.9s   57.3%      2.2s      2.5s  per script
    attach                   8      3.7s   24.0%     464ms     854ms  2.0 per script
    combine.cat              4      2.0s   12.8%     492ms     649ms  per script
    inference_setup          4     143ms    0.9%      36ms      39ms  per script
    reset                    2      73ms    0.5%      36ms      37ms  2 of 4 scripts
    import                   4       0ms    0.0%       0ms       0ms  per script
```

`model_load` is not in that list, and `import` has collapsed to nothing;
`generate` — the only part that was ever the point — has gone from a tenth of a
script to well over half of one. That is the whole measurement.

A server serves one request at a time, so the pool **caps the batch**:
`PROCESS_RUN_BATCH_SIZE` is lowered to `LORA_SERVER_COUNT`, and the k-th script
of a batch talks to the k-th server through `$GEP_LORA_SERVER` in its
environment. Positional, not a queue — the same fixed ceiling the batch already
was. It is an environment variable rather than something filled into the script
because which server an individual ran against is an accident of the batch it
landed in; baking it into `script_source` would make a stored script describe one
particular run of itself.

**What it costs is a guarantee.** A cold process per individual meant every
result came from a model that had never seen another blend; a warm server means
it had. That guarantee is recorded rather than quietly dropped: every client
prints `served by http://... , build #N`, which lands in the execution's stdout,
so a stored result says where in a server's life it happened.
`LORA_SERVER_RECYCLE_AFTER` puts a bound on it — one model load back every N
individuals, which keeps most of the saving. `GET /health` reports `allocated`,
`reserved` and `peak` from the server's own `torch`, because Windows' WDDM
driver gives `nvidia-smi` no per-process figure and that is the number the
question turns on.

**What has been measured so far**, at 3 eval prompts and `WEIGHT_SEED = 4242`:

| | `CAT.L1.L2.w1.w2` | `CAT.L3.L5.w3.w5` |
|---|---|---|
| cold process, `template_code.py` | `37ab08e5b39c`, 27.7s | `ac2516e8b599`, 26.5s |
| build #1 on a fresh server | `37ab08e5b39c`, 4.1s | `ac2516e8b599`, 3.8s |
| builds #5–#14, after two `SVD` blends had been built and torn down | `37ab08e5b39c`, 3.1–3.6s | `ac2516e8b599`, 3.7–4.0s |

Byte-identical transcripts cold and warm, and identical on every repeat.
Resting `allocated` over those ten builds stayed in 1.64–1.92 GB with no upward
trend, `reserved` in 1.77–1.98 GB, and `peak` was 4.73 GB — the `SVD` node, and
the same figure [What an `SVD` node costs](#what-an-svd-node-costs) gives.

That is fourteen builds, not three hundred, and it is one base model. The
standing acceptance test is still the long one: the same chromosome as the 1st
and as the 300th build must match a cold-process run, with resting VRAM flat.
The `build #N` line in every execution's stdout is what makes a failure of it
findable after the fact.

**Two servers pay; four barely do.** A warm server removes the CPU-bound import
and load that used to overlap well under concurrency, and what is left is
GPU-bound on one card. Measured with `python -m gep_lora.tools.compare_servers` on paired
sweeps differing only in `LORA_SERVER_COUNT`: 1 → 2 servers gave 1.64×
throughput on the passes with no straggler, −16% on the `process` step and +6%
per individual; 2 → 4 gave 1.29×, −4% and +23% per individual. Past two the
servers mostly time-share, which is why `LORA_SERVER_COUNT = 2`.

The one piece of design debt this introduces is that the blend arithmetic —
attach, combine, the rank rule, `_compact()` — now exists in three places:
`template_code.py`, `template_code_mocked.py`, and `Blend` in `lora_server.py`.
It has to: a generated script is standalone by design and can import nothing
from this repo. So they live under the rule the two templates already lived
under, and both files say so — **a change to the blend arithmetic in one belongs
in all of them.**

### Watching it run

A generated script says nothing at all while it loads the base model, then
prints one question-and-answer pair per eval prompt. Collected and shown at the
end, that is a console that hangs for minutes and then scrolls a transcript
past. So `process` reads each child as it runs and reports where it has got to:

```
[1/6] run_001.py  CAT.SVD.L1.L2.L4.w3.w3.w2
[2/6] run_002.py  CAT.L5.SVD.w1.L1.L1.w2.w2
        gen 2/5, batch 1/3: 2 running at once
        run_001.py  still loading the base model, 30s
        run_002.py  still loading the base model, 30s
        run_001.py  model ready, blend rank 24, 1m12s
        run_002.py  model ready, blend rank 16, 1m20s
        run_001.py  prompt 7/50, 1m44s
        run_002.py  prompt 6/50, 1m52s
        run_001.py  prompt 50/50, 8m03s
        run_001.py  ok         8m11s  50 exchange(s) -> execution 31
```

Two milestones are always said, being one line each: the model coming ready
(with the rank of the blend it built) and the last prompt starting. Between
them the running count is throttled — a script says where it is only once
`PROCESS_RUN_PROGRESS_SECONDS` (default 30) have passed since it last said
anything, so a fifty-prompt run speaks a handful of times rather than fifty.
The same throttle breaks a silence: a script that has printed nothing for that
long is reported as still loading, or still on the prompt it was on. `0` leaves
the milestones and drops the running count.

The transcript itself is never echoed — it goes to the database, and
`store.py --show` reads it back. What is on screen is where things are, not what
was said.

The step banner carries the generation while `gep_lora/core/pipeline/continue_run.py` is working
through them (`[3/8] gen 2/5 process -- ...`), and so does each batch line,
because the generation header scrolls away during the part that takes the hours.
It is display only: no step behaves differently in one generation than another,
and nothing about it is stored.

Streaming the children changed one stored result for the better: a script killed
by `--timeout` now keeps what it had already printed, so a timeout stores the
exchanges that did come back instead of an empty transcript.

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

**Unchanged individuals are skipped too.** An individual whose chromosome has
not moved since it last ran would produce the same execution again at the cost
of another model load, and its result is already in the database.
`has_changed` — written by
[mutation](#9-mutationpy--individualschromosome-individualshas_changed) for
every individual, every round — is exactly that question, so it is exactly what
decides.

An individual that has **never run** is not "unchanged": there is nothing for it
to have changed from. A fresh population therefore runs in full, and so does
every copy `selection` appends, since a copy has no executions of its own
however its flags read.

**Nor is one this generation has already run.** `has_changed` stays set from
mutation until the next round, so on its own it would still call a mutant
"changed" after process had run it -- and a generation interrupted half way
through process would run those again on [resuming](#stopping-and-carrying-on----resume-and---evaluate).
So an individual whose latest execution finished cleanly *and* built exactly the
blend it holds now is skipped as well: the chromosome its script printed on its
first line (`Individual 7: CAT.L1.L2.w1.w3`, read back by
`process_run.expression()`) is the one it holds, and the weight seed it drew
under is its own. Only a clean one -- a crash or a timeout in an interrupted
pass may *be* the interruption, so those run again. In an uninterrupted run
this changes nothing: a mutant's latest execution is always of the chromosome it
had before mutation.

That is what keeps a long run affordable. Over four generations from a
population of 3:

```
generation 1  population 6   running 3 of 6   (3 unchanged)
generation 2  population 12  running 8 of 12  (4 unchanged)
generation 3  population 24  running 15 of 24 (9 unchanged)
generation 4  population 48  running 32 of 48 (16 unchanged)
```

58 model loads rather than 90. `--include-unchanged` runs them all regardless.
A generation where *nothing* needs running is reported and passed over, not
treated as a failure.

**Individual failures are results, not pipeline failures.** A chromosome that
crashes is recorded as an execution with its exit code and the sweep carries on;
the last line of its output is echoed so a systemic problem is obvious. Only a
sweep where *nothing* ran returns a failing exit code. The commit is per
individual, so an interrupted sweep keeps everything it had already done.

Children are launched with `sys.executable`, so they inherit whichever
interpreter you started this with — run it with the venv's python (see the PATH
gotcha below) or every child will fail on `import unsloth`.

### 5. `gep_lora/core/evaluators/` → `exchanges.quality`

Scores every answer. Only the most recent execution of each individual is
scored; older ones keep the scores they were given.

```bash
python start_run.py evaluate
```

**How an answer is scored is a setting.** `gep_lora/core/evaluators/` is a registry —
**one module per evaluator**, plus `gep_lora/core/evaluators/common.py` for what they share —
and `EVALUATOR` in `settings.py` names the one a sweep uses:

```bash
python start_run.py --evaluators     # what is registered, and which one is current
```

| `EVALUATOR` | What it does | Needs |
|---|---|---|
| `llm_judge` | a judge model grades the answer on its own merits | a judge |
| `llm_judge_reference` | the same judge, shown the dataset's own answer to that question as well | a judge, a dataset with assistant turns |
| `llm_judge_answers` | the same two answers, **without the question**: the dataset's answer and the blend's | a judge, a dataset with assistant turns |
| `llm_judge_baseline` | the same judge, shown what the **base model** answered, scoring the improvement | a judge, one cached run of the base model |
| `jev_judge_reference` | `llm_judge_reference`'s question, graded by **Jev** (typesafe.ai) instead of a judge model | a Jev API key, a dataset with assistant turns |
| `similarity` | token or character overlap with the dataset's answer | a dataset with assistant turns |
| `heuristic` | local checks: length, repetition, a required and a forbidden pattern | nothing |
| `panel` | several judge models, aggregated | a judge per member |
| `composite` | several of the evaluators above, their scores combined (mean, median, min, max, geometric, harmonic, trimmed mean) | whatever its members need |

"A judge" is a model, and `JUDGE_BACKEND` says where it runs: an
OpenAI-compatible **endpoint**, or one loaded **here with unsloth**, the way the
generated scripts load the model they blend. See
[Where the judge runs](#where-the-judge-runs) — the four `llm_judge*`
evaluators and `panel` are indifferent to it. `jev_judge_reference` is the one
exception: it never touches `JUDGE_BACKEND` at all, because Jev is not a judge
*model* — see below.

All seven produce the same thing — a `quality` in 0..1 and a short `reason` on
each exchange — so every step downstream is unchanged. `0.0` is worst, `1.0` is
best; that is the number a fitness function selects on.

Like every other setting, `EVALUATOR` is frozen into a sweep when it starts.
Changing `settings.py` does nothing to a sweep already running, which is the
point: two individuals graded under different rubrics are not comparable, and
comparing them is the whole reason a fitness number exists. To change it
mid-search, write it into the sweep instead:

```bash
python continue_run.py --set EVALUATOR='"similarity"'
```

A sweep created before `EVALUATOR` existed has none stored and is read back as
`llm_judge`, which is what it ran under.

#### `llm_judge` — the default

A *different* model from the blended one that produced the answers grades each
of them on its own merits. Every knob is in `settings.py`:

| Setting | Default | Meaning |
|---|---|---|
| `JUDGE_BACKEND` | `"endpoint"` | `"endpoint"` or `"unsloth"` — where the judge runs |
| `JUDGE_BASE_URL` | `http://172.22.208.1:1234/v1` | OpenAI-compatible endpoint |
| `JUDGE_MODEL` | `None` — ask the endpoint | judge model id; **required** on the unsloth backend |
| `JUDGE_TEMPERATURE` | `0.0` | grading should repeat |
| `JUDGE_MAX_TOKENS` | `2000` | headroom for a reasoning judge |
| `JUDGE_TIMEOUT` / `JUDGE_RETRIES` / `JUDGE_RETRY_WAIT` | 300 / 2 / 3 | one call's patience |
| `JUDGE_RESPONSE_FORMAT` | `{"type": "json_object"}` | `None` for an endpoint that rejects it |
| `JUDGE_LOCAL_MAX_SEQ_LENGTH` | `4096` | the context a local judge is loaded with |
| `JUDGE_LOCAL_LOAD_IN_4BIT` | `True` | load a local judge quantised |
| `JUDGE_LOCAL_CHAT_TEMPLATE` | `None` — the model's own | an unsloth template name, for a model that ships none |
| `JUDGE_ABANDON_FRACTION` | `0.1` | give up on an individual whose first 10% of graded answers all score 0 |
| `gep_lora/core/pipeline/start_run.py --force` | off | re-score answers that already have a quality |

The API key is the one judge setting that is *not* in `settings.py`: a sweep
writes its settings into the database, so the key is read from the
`JUDGE_API_KEY` environment variable by `gep_lora/core/evaluators/common.py` instead.

`JUDGE_SYSTEM_PROMPT` **is** the fitness criterion: it grades relevance,
usefulness, specificity, coherence and appropriateness, with anchors at 1.0 /
0.7 / 0.5 / 0.3 / 0.0. Tune it deliberately — the whole search optimises toward
whatever it rewards.

It is **not a setting**. It is a constant in
[`gep_lora/core/evaluators/llm_judge.py`](gep_lora/core/evaluators/llm_judge.py), beside the code that
sends it, and `gep_lora/core/evaluators/panel.py` keeps its own copy — so tuning a panel's
rubric does not move what the single judge selects on. The two start out
identical, which is what makes a difference between them worth reading.

The trade-off is worth knowing: `settings.py` is snapshotted into every sweep,
and this is not, so a new sweep no longer records the rubric it was judged
under — the module as it stood is the only record, and editing it changes what
a later `--force` re-score grades by. A sweep created while it *was* a setting
still holds its own copy in the `settings` table, and that copy still wins, so
those sweeps stay reproducible.

The score and the reason land on the exchange they grade, with what gave them
and when:

```
position  quality  reason                              judge_model
1         0.4      generic advice, no concrete schedule qwen2.5-7b-instruct
```

`quality` is what selection reads; `reason` is what tells you whether the judge
is grading the way you intended — worth reading when a whole individual scores
0.0, or when scores cluster and you suspect the rubric rather than the answers.
`judge_model` records *what* scored it, which for a local evaluator is the
method: `similarity:token_f1`, `heuristic`, `panel:a,b`.

**Local by default, cloud by swap.** The judge speaks the OpenAI-compatible
`/v1/chat/completions` API, so a hosted model is two lines in `settings.py`:

```python
JUDGE_BASE_URL = "https://api.openai.com/v1"
JUDGE_MODEL = "gpt-4o-mini"
```

Claude is *not* OpenAI-compatible — using a Claude model as the judge needs a
separate backend via the `anthropic` SDK.

#### Where the judge runs

The blends are built and prompted with **unsloth**, in a process per individual.
The judge need not be: `JUDGE_BACKEND` picks between two transports, and the
five judging evaluators are indifferent to which one a sweep chose.

| `JUDGE_BACKEND` | Where the tokens come from | Needs |
|---|---|---|
| `"endpoint"` (default) | `POST JUDGE_BASE_URL/chat/completions` | a server up — LMStudio, vLLM, OpenAI, OpenRouter |
| `"unsloth"` | `JUDGE_MODEL`, loaded in the evaluate step's own process | the venv one level up, and VRAM |

```python
JUDGE_BACKEND = "unsloth"
JUDGE_MODEL = "unsloth/qwen2.5-7b-instruct-unsloth-bnb-4bit"
```

That is the whole change. A sweep then runs end to end on this repo and a GPU,
with nothing listening on any port — the same `python main.py`, and the same
`run_db/gep.sqlite3` at the end of it.

**One instrument, two transports.** The rubrics, the retries, the abandon rule,
`judge_model` on every exchange and the way a score is read out of a reply are
all above the split, in `common.ask_judge()`; the backend decides only where the
tokens are produced. So a sweep graded locally is comparable with one graded
over an API in every respect except the judge model itself — which is what
`judge_model` records either way.

**Three things about the local backend are deliberate.**

*It loads on the first answer it grades, not while preparing.*
`llm_judge_baseline` runs the **base** model while preparing, to fill its cache
of control answers; loading the judge before that would put two models on the
card for no reason.

*It is released the moment the step is done* — the evaluate step and the testing
pass both wrap their work so it happens however the step ends. `gep_lora/core/pipeline/main.py` runs a
whole search in one interpreter, so the next generation spawns scripts that each
load the base model, and a judge still resident here would be VRAM taken from
every one of them.

*A model that will not load stops the step; a generation that fails costs one
answer.* A missing `JUDGE_MODEL`, a model this machine cannot load, or a
tokeniser with no chat template are configuration failures and say so once. An
out-of-memory or an over-long prompt during grading fails that one answer, keeps
its NULL quality and is picked up by the next run, exactly as an endpoint's 500
is. And at `JUDGE_TEMPERATURE = 0` an unparseable reply is **not** retried:
greedy decoding returns the same reply, so the retry would only spend another
`generate()` to fail in the same way.

`JUDGE_MODEL` has to be named on this backend. There is nothing to ask what it
has loaded the way an endpoint can be asked, and falling back to `BASE_MODEL`
would leave the model under test grading its own descendants.

[`gep_lora/core/evaluators/local_model.py`](gep_lora/core/evaluators/local_model.py) is the whole of it —
the second module in that package that is not an evaluator, beside `common.py`.

#### `llm_judge_reference` — grading against the dataset's own answer

The eval file is the training data's own format: a JSON record per line with a
`user` turn and an `assistant` turn. The generated scripts deliberately only
ever ask the user turn — handing a model the answer and then scoring its reply
would be marking its own homework — but the *judge* is allowed to see it.

This is the evaluator that can see **style**. A judge grading on merit alone
happily rewards a helpful prose answer from a blend that was supposed to rhyme;
shown `datasets/poem_lora_dataset.json`'s own answer to the same question, it
grades on whether the blend answered in the manner it was fine-tuned to. Its
rubric is `JUDGE_REFERENCE_SYSTEM_PROMPT`, which weighs manner first, then
substance and coherence, and explicitly does not reward copying — a blend that
reproduced the reference word for word would have learned that one answer and
nothing else. Like the other two rubrics it is a constant in the evaluator that
sends it, [`gep_lora/core/evaluators/llm_judge_reference.py`](gep_lora/core/evaluators/llm_judge_reference.py),
not a setting; `gep_lora/core/evaluators/panel.py` keeps its own copy. An item with *no*
reference falls through to `llm_judge.score()` and is graded on merit instead.
Everything else — endpoint, model, timeouts — comes from the same `JUDGE_*`
settings.

A prompt with no reference is graded on merit instead of being dropped: a
shrunken eval set for one individual would make its fitness incomparable with
the rest.

#### `jev_judge_reference` — the same question, graded by Jev instead of a judge

`llm_judge_reference`'s question — does the answer match the dataset's own
answer in manner, substance and coherence — asked of
[Jev](https://typesafe.ai), typesafe.ai's "System One" model, instead of an
LLM. Jev is not a chat model asked to write a paragraph and have a score
extracted from it: it takes a `state` and a typed `question` and returns a
constrained decision — here, a rating against five described levels, as a
probability distribution plus a confidence — so there is no free-text judge
reply to parse, and no explanation in the judge's own words.

The five levels are `llm_judge_reference`'s own rubric anchors (`useless` /
`poor` / `mixed` / `good` / `excellent` → `0.0` / `0.3` / `0.5` / `0.7` /
`1.0`), so a sweep graded by the two evaluators means the same thing by a
`0.7`. The `quality` recorded is the probability-weighted mean of those
anchors, not Jev's own `score` field, whose indexing the API does not pin
down — see
[`gep_lora/core/evaluators/jev_judge_reference.py`](gep_lora/core/evaluators/jev_judge_reference.py). An
item with no reference **fails that exchange**, the way `llm_judge_answers`
does, rather than falling back to a merit-only judge: there is no
merit-only Jev evaluator to fall back to.

Its own settings, all in `settings.py`:

| Setting | Default | Meaning |
|---|---|---|
| `JEV_BASE_URL` | `None` — the public API | the TypeSafe evaluation endpoint |
| `JEV_MODEL` | `"jev-latest"` | which Jev release grades |
| `JEV_TIMEOUT` / `JEV_RETRIES` / `JEV_RETRY_WAIT` | 30 / 2 / 3 | one call's patience |

The API key is read from the `TYPESAFE_API_KEY` environment variable, never
from `settings.py`, for the same reason `JUDGE_API_KEY` is: a sweep writes its
settings into the database, and a bearer token has no business there.

#### `llm_judge_answers` — the two answers, and not the question

The same comparison as `llm_judge_reference` with one thing taken away: the
judge is handed the **reference answer** and the **blend's answer**, and never
sees the question either of them is of.

That is the whole difference, and it is the point. With the question in front of
it a judge quietly grades merit as well as manner — a blend that answers
helpfully but nothing like its training data still reads as a good answer to the
question, and picks up score for it. Without the question there is no such
credit to give: the reference is the only standard in the prompt, so the score
measures agreement with it and nothing else. The prompt is also a question
shorter per call, which on a local judge with `JUDGE_LOCAL_MAX_SEQ_LENGTH` to
spend is not nothing.

The cost is symmetrical, and worth knowing before selecting on it: the judge
cannot tell a blend that misread the question from one that answered a
neighbouring question well, because it never sees which question either answer
is of. It is a *match* score, not a quality score — `llm_judge` grades merit,
`llm_judge_reference` does both at once.

The question is still what **finds** the reference (`common.reference_for()`
keys on it, so an answer cannot be paired with somebody else's reference) and is
then dropped. An exchange with no reference **fails that exchange** rather than
falling back to merit grading the way `llm_judge_reference` does: the fallback
would send the question — the one thing this evaluator exists not to do — and
score a different quantity from the answers around it. Its rubric is
`JUDGE_ANSWERS_SYSTEM_PROMPT` in
[`gep_lora/core/evaluators/llm_judge_answers.py`](gep_lora/core/evaluators/llm_judge_answers.py), and it
opens by telling the judge the question is withheld deliberately, because a
judge that has not been told looks for it and complains instead of grading.
Like `llm_judge_reference`'s, it does not reward copying the reference: a blend
that reproduced it word for word would have learned that one answer and nothing
else. That line arrived after the evaluator did, so a sweep graded by it before
then is not strictly comparable with one graded after.

#### `llm_judge_baseline` — grading the improvement on the base model

The other evaluators ask "is this answer good". This one asks the question the
search is actually for: **did folding these adapters in make the model better
than it was without them.**

The judge is shown three things — the question, what the bare base model
replied, and what the blend replied — and scores the difference on a **centred**
scale:

| Score | Means |
|---|---|
| `1.0` | transformed; far better in every way that matters |
| `0.8` | clearly better |
| `0.6` | slightly better |
| `0.5` | **no meaningful difference** — or a difference not worth having |
| `0.3` | slightly worse |
| `0.0` | ruined: empty, incoherent, repetitive or off topic where the base answer was not |

That centre is the point of it. A fitness of 0.5 says an individual is the base
model with extra steps; above it the blend earned its keep, below it the blend
did harm. Merit-only grading cannot say any of that — a blend that ruins nothing
scores well on merit because the base model was already competent, and the search
has nothing to climb. The rubric is `JUDGE_BASELINE_SYSTEM_PROMPT`, a constant
in [`gep_lora/core/evaluators/llm_judge_baseline.py`](gep_lora/core/evaluators/llm_judge_baseline.py) rather
than a setting — and the one rubric of the three that `panel` does not also keep
a copy of, since nothing else grades this way. Everything else — endpoint,
model, timeouts, retries — comes from the same `JUDGE_*` settings as
`llm_judge`.

**Where the base answers come from, and why you only pay once.**
`baseline_run.py` fills `template_baseline.py` — the same base model, the same
eval file, the same cap, the same chat template and the same `max_new_tokens` as
the individuals, with **nothing attached** — runs it once, and files what it said
in the `baselines` table:

```
model                                          question_key            answer
unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit  help me plan my week.  Here is one way ...
```

That table hangs off no run, on purpose: a base-model answer belongs to the model
and the question and to nothing else. So the first sweep that wants a control
pays one base-model load for it and every sweep afterwards reads the same rows —
and adding prompts to the eval set costs only the new ones, because the step asks
for what is missing rather than for all of it. `BASELINE_TIMEOUT` is how long the
one script gets.

Two guards worth knowing about:

* **A mocked sweep gets a mocked baseline.** `TEMPLATE = "template_code_mocked.py"`
  makes `baseline_run.py` fill `template_baseline_mocked.py` instead, and cache
  what it invents under `mock:<model>` rather than under the model's own name —
  so the whole path (generate → cache → read back → judge) can be exercised on a
  machine with no GPU, and an invented control can never end up in front of a
  real judge. `BASELINE_TEMPLATE` overrides the choice.
* **A missing control fails that one exchange**, rather than quietly falling back
  to merit grading. A merit score and an improvement score are not the same
  number, and averaging the two into one fitness would reward whichever
  individuals happened to lose their baseline.

`BASE_MODEL` in `settings.py` is what ties the two halves together: it is stamped
into every generated script *and* into the baseline, so the control is the model
the blends were actually built on, and repointing it asks for that model's own
baseline instead of reusing the old one's. `CHAT_TEMPLATE` is the other half of
the key: a named template caches under `<model> [chat_template=<name>]`, because
the same question asked in other words is another control. The model's own
template keeps the bare name, which is where every row cached before the setting
existed already sits -- those were `"qwen-2.5"` on the Qwen2.5 repos, whose own
template renders the same.

#### `similarity` — no judge at all

Token or character overlap with the dataset's answer. `SIMILARITY_METRIC` picks
how:

| Metric | What it measures |
|---|---|
| `token_f1` | bag-of-words F1, repeats counted — padding costs precision, missing the reference costs recall |
| `containment` | how much of the reference's vocabulary turns up at all; forgiving about length |
| `sequence` | character-level overlap (`difflib`), so word order and phrasing count |

`SIMILARITY_CASE_SENSITIVE` is off by default. The whole eval half of a sweep
then costs nothing, runs offline and repeats **exactly** — which is worth having
next to a judge whose scores wobble. The trade is real, though: this measures
agreement with one particular answer, not quality, so an answer better than the
dataset's scores badly.

#### `heuristic` — checkable properties

No model either, and no reference. Four equally weighted checks, averaged over
the ones that apply: a length band (`HEURISTIC_MIN_WORDS`, `HEURISTIC_MAX_WORDS`),
a distinct-word ratio that catches the collapsed blend saying the same thing over
and over, and optionally a pattern the answer must match (`HEURISTIC_REQUIRE`)
and one it must not (`HEURISTIC_FORBID`).

This measures whether an answer is **malformed**, not whether it is good — which
is the half of quality a judge charges the most to notice. Use it to smoke-test
the pipeline, or for a task whose rule genuinely is checkable: an all-uppercase
adapter is `HEURISTIC_REQUIRE = r"^[^a-z]*$"`.

#### `panel` — several judges

`PANEL_MODELS` names the members, all reached the same way — `PANEL_BASE_URL`
(or `JUDGE_BASE_URL`), or this machine when `JUDGE_BACKEND` is `"unsloth"`;
`PANEL_AGGREGATE` is `mean`, `median`, `min` or `max`; and
`PANEL_USE_REFERENCE` switches the panel onto the reference rubric. Everything
else about a member comes from the `JUDGE_*` settings, so a panel is several
models grading identically rather than several differently configured judges.

Less noise per score, N times the cost. A member that fails is dropped rather
than fatal — a panel that loses one model still has a score — and only a panel
where nobody answered fails the exchange. An empty `PANEL_MODELS` asks the
endpoint what it has loaded and sits a panel of one on it, which is `llm_judge`
with extra steps; on the unsloth backend there is nothing to ask, so the panel
has to be named.

**On the unsloth backend a panel is N models resident at once.** Each member is
asked about every answer in turn, so unloading between them would reload the
whole panel per answer — the same bargain `PROCESS_RUN_BATCH_SIZE` strikes with
base models. The step's note prints the count before any of it reaches the card:

```
panel: judge-a, judge-b, loaded here with unsloth -- 2 model(s) resident at once, aggregated by mean
```

#### `composite` — several evaluators

`panel` is several *models* under one rubric; `composite` is several
*evaluators*, each grading exactly as it would as `EVALUATOR`, their scores
folded into one. A judge and a checkable property fail in different ways, and a
composite selects on both:

```python
EVALUATOR = "composite"
COMPOSITE_EVALUATORS = ["llm_judge_reference", ["heuristic", 0.5]]   # name, [name, weight] or {"name", "weight"}
COMPOSITE_AGGREGATE = "geometric"
COMPOSITE_FLOOR = 0.2
COMPOSITE_ON_FAILURE = "fail"
```

| `COMPOSITE_AGGREGATE` | What it does |
|---|---|
| `mean` | weighted mean — the default |
| `median` | weighted median; ignores one member out on its own |
| `min` / `max` | the worst / best member; weights ignored |
| `geometric` | weighted geometric mean; a low member costs far more than in the mean, and a 0 from anyone is 0 |
| `harmonic` | weighted harmonic mean — the F1 of the family, harsher on imbalance still |
| `trimmed_mean` | drop the highest and lowest, mean of the rest (needs 3+ members) |

On a 0..1 scale `min ≤ harmonic ≤ geometric ≤ mean ≤ max` always, so moving along
that list is choosing how much one bad member should cost. `COMPOSITE_FLOOR` is a
**veto** applied first: any member below it makes the answer `0.0` — how
`heuristic` becomes a gate rather than one vote. A member that fails fails the
whole exchange under `"fail"` (a mean over whichever members answered is a
different number); `"skip"` combines the rest, as `panel` does.

Members read the sweep's own settings (`JUDGE_*`, `HEURISTIC_*`, ...), are
prepared in order with the step's context — so `llm_judge_baseline` still fills
its cache — and each one's score and reason is kept in the combined reason. A
member may appear once, and `composite` not at all. The abandon rule and the
lora server teardown follow the members: a composite asks a judge only if one of
its members does (`Evaluator.asks_judge(conf)`).

#### Giving up early on a hopeless individual

A judge call is the expensive part of a sweep, and most of the calls an
individual costs are spent confirming what its first few answers already said.
`JUDGE_ABANDON_FRACTION` is how much of one individual's pending answers has to
be graded, and to come back unanimously `0.0`, before the evaluate step stops
asking about that individual: `0.1` is "the first 10%", rounded up, never fewer
than one answer, and `0` or `None` turns the rule off.

The answers it never asks about are stored as `0.0` with the reason
`abandoned: the first N graded answer(s) all scored 0`, rather than left NULL.
That is what makes the individual's **fitness** zero rather than merely the part
of it that was graded — fitness is the mean over an individual's exchanges — and
it is also what keeps a re-run of `evaluate` from asking about them after all.
The console says so as it happens, and again in the step's summary:

```
individual 7  CAT.SVD.LIN.L1.L2.L3.L1.w3.w3.w2.w1
    [1] 0.00  refuses the question, no plan given
    [2] 0.00  repeats the prompt back
    giving up on individual 7 -- abandoned: the first 2 graded answer(s) all scored 0, so its remaining 18 answer(s) are scored 0 unasked
```

The testing pass applies the same rule, from the same setting, over its own
rows — see [`test_run_with_dataset.py`](#14-test_run_with_datasetpy--test_results).
`evaluators.abandon_after()` is the one piece of arithmetic behind both.

Three things it deliberately does not do. It never applies to `similarity` or
`heuristic`: a local *scorer* costs nothing to finish, so stopping it short would
only lose detail, and the rule is on for the evaluators that ask a model (a
`composite` counts when any member does) —
whichever backend they ask it through, since a call it does not make is a
request not sent or a `generate()` not run. An **empty** answer scores `0.0`
without a call, and is not an
evaluation, so it neither counts toward the first 10% nor condemns an individual
by itself — a script that missed one question still gets its other answers
graded. And an answer the evaluator **failed** to score is not a zero either, so
an endpoint having a bad minute cannot abandon anybody.

The cost is a real one: this gives up on an individual that opens badly and
recovers later. On a small eval set it is sharper than it sounds — with
`TRAINING_COUNT = 10` the first 10% is a single answer, so one zero is enough.
Lower the fraction, raise `TRAINING_COUNT`, or set it to `0` if the search
should grade everything whatever the early answers say.

#### Resumable, whichever one is chosen

An exchange that already has a `quality` is skipped unless `--force`, and each
score is committed as it arrives, so an interrupted sweep keeps its work. An
empty answer scores `0.0` without asking anyone. A sweep where nothing needs
grading never contacts an endpoint and never loads a judge — which is what lets
a mocked sweep, already scored by its own template, run on a machine with
nothing up and no GPU.

An evaluator that cannot score one answer fails **that answer** and no more: it
keeps its NULL quality, the step counts it, and a re-run picks it up again. Only
a step where nothing at all could be scored is a failure.

Reply parsing is deliberately tolerant — bare JSON, code-fenced JSON, JSON
wrapped in prose, and a bare number all work. Two traps worth knowing about,
both hit on the first real run: the score is requested **before** the reason so
a long reason cannot truncate it away, and `JUDGE_MAX_TOKENS` is generous
because a reasoning judge spends its budget thinking and returns an empty
message if it runs out mid-thought.

#### Adding one of your own

An evaluator is a name, a description and two functions. **One file per
evaluator**, in `gep_lora/core/evaluators/`, ending in the call that registers it — say
`gep_lora/core/evaluators/mine.py`:

```python
from evaluators import common

def prepare(conf, pending, context=None):   # once per evaluate step
    return common.Prepared(conf, "mine", notes=["mine: ..."])

def score(item, prepared):                  # once per answer -> (quality, reason)
    return 0.5, "why"

common.register(common.Evaluator("mine", "what it does", prepare, score))
```

Then add one line to `gep_lora/core/evaluators/__init__.py` — `from evaluators import mine` —
because importing the module is what runs its `register()`. Nothing else in the
pipeline changes: every step reaches an evaluator through `get(EVALUATOR)`.

Every module here calls its two functions `prepare` and `score`, since the file
name already says which evaluator they belong to — and that is what lets one
build on another. `llm_judge_reference.py`, `llm_judge_answers.py` and
`llm_judge_baseline.py` are `llm_judge.prepare()` and a prompt of their own.

`prepare()` is where the once-per-step work goes — discovering a model, loading
the eval set's answers with `common.load_references()`, filling the base-answer
cache, validating knobs — and what it returns is handed to every `score()` call.
Its `label` is what lands in `exchanges.judge_model`. `context` is the step's own
`Context`, there for an evaluator that needs the database or the run folder;
ignore it otherwise. Knobs go in `settings.py` under a prefix of their own and
reach `score()` as `prepared.conf`, which is the sweep's *stored* settings, never
`settings.py` as it stands now. Anything a second evaluator would also want
belongs in `gep_lora/core/evaluators/common.py`. An evaluator that asks a model should ask it
through `common.ask_judge()` and build its block with `common.judge_settings()`,
which is what makes it work on both backends for free, and should be registered
with `needs_judge=True` so the abandon rule applies to it. `jev_judge_reference`
is the one exception: Jev is not a `JUDGE_BACKEND` model, so it keeps its own
transport and settings and registers with `via_judge_backend=False` as well —
do the same for an evaluator that asks a judge of its own rather than one named
by `JUDGE_BACKEND`/`JUDGE_MODEL`, so `wants_the_card()` and `--evaluators`
don't claim it runs through an endpoint or local model it never touches.

### 6. `gep_lora/core/search/calculate_fitness.py` → `individuals.fitness`, `fitness_history`

The evaluate step scores answers, not chromosomes. Selection needs the opposite — one
comparable number per individual — so this step folds each transcript into its
mean:

```bash
python start_run.py fitness
```

> fitness = the average `quality` across the exchanges of the individual's most
> recent execution

The most recent execution and not all of them, for the same reason `evaluate`
scores only that one: the same chromosome run again under a different weight
seed is a new result, not an amendment to the old one. That choice already
lives in the `individual_quality` view, which is where the average comes from —
this step writes it back onto the row so a selection step can sort on a column
instead of recomputing a join.

```
#    state answers fitness chromosome
1    ok    3       0.595   CAT.L1.L3.w2.w2
2    ok    3       0.350   CAT.L5.SVD.w1.L1.L1.w2.w2
3    ok    0       0.000   CAT.L5.L2.w5.w4
```

An individual with nothing to average — never run, crashed before it answered,
`BAD`, or still unjudged — gets **0.0**, not NULL. Fitness is what selection
sorts on, and a missing score there would have to be given a meaning at every
call site; giving it one here, once, says the only sensible thing: an individual
that produced no judged answer is worth nothing, and a blend PEFT refuses to
build cannot be selected for.

A partly judged individual is averaged over the answers that *do* have a score
and named in the output, since a fitness over half a transcript is a weaker
claim than one over all of it. The step is pure arithmetic over what `evaluate`
stored — no model, no judge, no endpoint — so it is cheap to re-run, and it
re-runs cleanly: it overwrites, never accumulates.

#### The generation it was, not just the generation it is

`individuals.fitness` is a column that only ever holds *now*. The next
generation overwrites it, and mutation clears it outright the moment the
chromosome that earned it is replaced — so a sweep keeping only that column can
say how fit its population is today and nothing whatever about whether the
search got anywhere. The same numbers therefore go into **`fitness_history`**,
one row per individual per generation:

| column | |
|---|---|
| `generation` | 1-based, in the order the `fitness` step ran |
| `population` | how many individuals that generation held |
| `recorded_at` | when the step worked the numbers out |
| `number` | the individual's own number |
| `chromosome`, `state` | as they were **then**, not as they are now |
| `fitness` | what `individuals.fitness` was given |
| `answers`, `unscored` | how much transcript the mean was taken over |

A row is self-contained for the same reason an execution is. The individual it
names will be mutated into a different chromosome and given a different fitness
before the sweep is done, so a history read through a join back to
`individuals` would report every past generation in terms of the present one.

**Nothing counts generations for you**, because until now nothing needed to: a
sweep is a population that keeps being rewritten in place, and neither
`gep_lora/core/pipeline/start_run.py` nor `gep_lora/core/pipeline/continue_run.py` stores a generation number. The count is
derived in `store.fitness_generation()` from the one thing that dates a round
— the **highest individual number** the population holds, which selection
always leaves higher than it found it. Numbers are handed out from the top and
never reused, and a round appends before it culls, so it always ends on a
higher number than it began on. The same watermark `selection` and `mutation`
seed their generators from.

It used to be the *size* of the population, which worked only while selection
grew it. Now that a round culls as many individuals as it appends the size sits
at `COUNT` forever, and dating a round by it would file every generation as a
restatement of the first — one history row, overwritten once a generation, and
no curve at all. The first snapshot is generation 1; a later one is a **new** generation
if the population has grown since the last, and the **same** generation
restated if it has not. That is what makes `python start_run.py fitness` — cheap,
and a reasonable thing to redo after a re-scored `evaluate` — rewrite the
current generation instead of inventing another one.

The blind spot is a generation that selects nobody: `SELECTION_COUNT = 0`, or a
wheel with nothing to spin — selection writes nothing at all in either case,
culls and newcomer included. Such a round is indistinguishable from the one
before it and is recorded as a restatement of it. Both are already dead ends
for the search — an all-zero population stops at the `elitism` step — so
nothing a running sweep wanted is lost there.

The step prints the generation it recorded, and the curve so far once there is
more than one:

```
recorded generation 4 in fitness_history at 2026-08-28T11:12:06
    gen   recorded             pop    best    mean    fittest chromosome
    1     2026-08-28T11:12:05  6      0.717   0.298   CAT.L5.L5.w3.w3
    2     2026-08-28T11:12:05  8      0.717   0.328   CAT.L5.L5.w3.w3
    3     2026-08-28T11:12:06  10     0.717   0.400   CAT.L5.L5.w3.w3
    4     2026-08-28T11:12:06  12     0.834   0.478   CAT.SVD.L3.L1.SVD.w2.w3.L2.L4.w2.w3
```

`python -m gep_lora.core.storage.store --show` prints the same table for a stored sweep (with
`worst` as well), and `--export` writes it out as `fitness_history.txt`
alongside the per-individual rows — the one exported file that is not a view of
the population as it stands now. Note that a rising `mean` with a flat `best`
is often the *turnover*, not improvement: selection appends copies of the fit
and culls the weakest, so the average climbs on its own — the population it is
averaged over is the same size as last generation's but several weak ones
lighter.

```sql
-- the curve, straight out of the database
SELECT generation, recorded_at, population, MAX(fitness) AS best, AVG(fitness) AS mean
  FROM fitness_history WHERE run_id = 3 GROUP BY generation ORDER BY generation;
```

### 7. `gep_lora/core/search/elitism.py` → `individuals.is_best`

Names the one individual this generation carries forward:

```bash
python start_run.py elitism
```

> `is_best = 1` for one individual with the highest fitness, `0` for every other

Elitism is the rule that the best individual survives a generation untouched.
Selection, crossover and mutation are all free to lose it otherwise — the best
chromosome of one generation can easily have no descendant in the next, and a
search that can go backwards wastes the generations it spends climbing again.
This step marks it; a later one copies it across unchanged.

```
elite: individual 3, fitness 0.759
    CAT.L5.L2.w5.w4
cleared is_best on the other 2 individual(s)
```

**Exactly one, and only one.** The flag is set and every other cleared in a
single statement, since they are one fact: a sweep carrying two elites — or
still carrying the last generation's — says nothing at all.

**Ties are ordinary**, not an edge case: `fitness` is a mean of judged answers,
so two blends can easily land on the same score. The lowest individual number
takes it — an arbitrary rule, but a fixed one, so re-running the step over the
same database elects the same individual rather than a different one each time.
A tie is named in the output so the pick doesn't read as a ranking.

**An all-zero population elects nobody.** `fitness` defaults to `0.0`, so that
is either a sweep that never reached the `fitness` step or one where every
individual failed. The step writes nothing and stops, naming both possibilities
— a sweep keeps whatever `is_best` it already had rather than gaining a
meaningless one.

The flag comes from the stored `fitness` column and nothing else. Reading the
transcripts again here would be a second, quietly different definition of
"best": when the fitness rule changes it changes in `calculate_fitness.py`, and
this step follows it without knowing that it did.

### 8. `gep_lora/core/search/selection.py` → copies in, the weakest out, one stranger

Fitness-proportionate selection, the classic roulette wheel:

```bash
python start_run.py selection
```

Every individual gets a slice of the wheel as wide as its fitness; the wheel is
spun once per pick, and whoever it lands on is taken. A chromosome twice as fit
is twice as likely to be picked, and nobody is picked for certain — a mediocre
individual keeps a real chance, which is what stops the search collapsing onto
the first good blend it finds.

```
spun the wheel 3 time(s) over 5 individual(s)
    parent    fitness picked chromosome
    3         0.745   1      CAT.L4.CAT.w4.L3.L4.w5.w1
    4         0.536   1      CAT.CAT.CAT.CAT.SVD.CAT.L1.L2.L2.L1.L4.L4.L4.w4.w1.w5.w2.w3.w3.w1
    2         0.296   1      CAT.L5.SVD.w3.L4.L5.w2.w5
2 individual(s) the wheel never landed on: 1, 5 -- being missed is not what gets an individual culled; being weakest is
appended 3 copy/copies as #6-#8
culled the 2 weakest individual(s), transcripts and all:
    1         0.000   CAT.L4.LIN.w2.LIN.LIN.L1.CAT.LIN.L5.w4.L4.L5.L2.L5.w1.w5.w1.w1.w1
    5         0.000   CAT.SVD.L1.LIN.CAT.w4.L4.L2.L5.L2.w4.w5.w2.w2
    and 2 of their script file(s) from run_db
drew #9 fresh from the same rules the first population came from: CAT.CAT.CAT.L1.SVD.L5.L3.w2.L3.L4.w1.w5.w4.w2
the population is now 7, up from 5
```

**Picks are drawn with replacement** — the same individual can come up several
times in one round. That is the mechanism, not a flaw in it: it is how a fit
chromosome comes to have several descendants.

**A round is three writes**, and the arithmetic is the point:

| | |
|---|---|
| appends | `n` copies of whoever the wheel landed on, `n` being `SELECTION_COUNT` |
| culls | the `n+1` weakest individuals of the population as it was before the round |
| appends | one brand-new individual, drawn the way the first population was |

`n + 1` in and `n + 1` out, so a generation ends **exactly the size it began**:
a sweep drawn at `COUNT = 10` is still ten individuals after ten generations.
The cull is `n+1` and not `n` because a round appends `n+1` rows — the copies
*and* the newcomer — and it is that total that has to come back out. What moves
is not the size but the membership: the fit are duplicated, the weak are gone,
and one member of every generation owes nothing to either.

`SELECTION_COUNT` is therefore a knob on the **turnover**, not on the size — at
2 over a population of 10, three individuals in and three out each generation.
The one exception is `None`, which asks for as many copies as the population
holds: that is one more row than the cull can take, so it is the only setting
that still grows the population, by two a generation.

**The newcomer is what makes the cull safe.** Selection can only pick
chromosomes the population already holds, and mutation only nudges a symbol
into a sibling of its own class, so a search that culls and copies is a search
whose gene pool can only narrow. One fresh draw per generation is a floor under
that: however far the population has converged, every generation still contains
one tree grown from nothing. It comes from
`generate_population.build_population()` under the sweep’s own `MAX_DEPTH`,
`BRANCH_PROB` and `UNIQUE` — the same draw, the same grammar, the same
`check()`, no second copy of any of it. `UNIQUE` is honoured against the
chromosomes the population held going into the round, culled ones included,
since re-drawing what the search just discarded is the one draw that achieves
nothing; a converged sweep that cannot find a new one after a hundred tries
gets a duplicate rather than a failed generation.

**A cull takes the individual whole** — its executions, its exchanges and its
`test_results` all cascade away with the row, because they are the record of a
blend the search has finished with. Two things survive it: the number, which is
retired rather than reused, so no old script, execution or history row is ever
re-pointed at somebody else; and its rows in `fitness_history`, which hang off
the *run* and keep the chromosome, state and fitness as they were in each
generation. A sweep can still say that #7 scored 0.04 in generation 3 and was
culled at the end of it.

**Weakest means the lowest `fitness`**, NULL read as 0.0 the way every other
reader of that column reads it, ties broken on the **lowest number** — so
between two equally weak individuals the longer-standing one goes, and
re-running the step culls the same rows rather than a coin-flip’s worth of
different ones. The elite is never eligible: culling it would discard the one
thing [elitism](#7-elitismpy--individualsis_best) exists to protect. A
population too small to spare `n+1` non-elite rows gives up as many as it has,
grows by the difference, and says so. And a parent the wheel just landed on can itself be culled — weak
individuals do occasionally get picked — which needs no special case: the copy
it produced carries its chromosome, its script and its fitness forward, which
is all the parent had to pass on.

Nothing is *overwritten*, though. An individual that survives the round keeps
its number, its script, its executions and its transcripts, whether or not the
wheel ever landed on it.

A copy is **the parent field for field** — its tree, its state, its rank, its
script, its weight seed and its fitness, not merely its chromosome. Only the
`id` and the `number` are its own, and only because those are what make it a row
of its own. So a copy is a clone in the full sense: it arrives already carrying
the result its parent earned, rather than as a blank waiting to be built and
judged. The column list is read from the table, so a field added to
`individuals` is copied too instead of being quietly dropped from every copy the
search makes.

**`is_best` is the one exception**, and every copy arrives with it at `0`. That
flag does not describe an individual, it picks one out of the population — the
single one this generation carries forward — so it is not the parent's to hand
on. A copy of the elite is not itself the elite; the next election decides that,
on the fitness the copy earns.

That inheritance is meant to be **spent, not kept**. The copies exist for
whatever comes next to vary, and a varied copy has a chromosome its inherited
tree, script, rank and fitness no longer describe — they are the parent's
answers to a question the child no longer asks. Re-deriving them from the
chromosome, for every individual, is exactly what this does:

```bash
python start_run.py trees runs
```

Two consequences of inheriting the rest, both of which the next run of the
relevant step clears up, and neither of which is a reason to hold the copy back:

- **A copy shares its parent's `script_name` and `weight_seed`** until `runs`
  re-stamps them from its number. Harmless while the chromosome is still the
  parent's — the two rows describe the same blend — but it does mean two
  individuals can name the same `run_NNN.py` in that window.
- **A copy inherits `fitness`**, so it is immediately selectable at its parent's
  score rather than sitting out the next round at `0.0` — right up until
  [mutation](#9-mutationpy--individualschromosome-individualshas_changed)
  changes its chromosome, which clears the inherited score.

**Because it appends more than it removes, the step is not idempotent.**
Running it twice runs two rounds, and the second culls what the first left.
That is what a second generation *is*, so it is deliberate — but `python
gep_lora/core/pipeline/start_run.py selection` is something you do on purpose, not something you repeat to
be sure it took.

An individual whose fitness is `0.0` has a slice of width zero and can never be
picked — correct for roulette, and worth saying out loud since `fitness`
defaults to `0.0`, so an individual that has never been through `process` and
`evaluate` sits out until it has. A population where *everything* is `0.0` has
no wheel at all; that case selects nobody, **culls nobody**, draws no newcomer,
writes nothing and stops. A round that cannot say which individuals are the fit
ones cannot be trusted to say which are the weak ones either.

| Where | Setting | Default | Meaning |
|---|---|---|---|
| `settings.py` | `SELECTION_MASTER_SEED` | `None` | where the spins come from; `None` draws one and records it |
| `settings.py` | `SELECTION_COUNT` | `None` | how many picks per round — and so how many are culled, one fewer; `None` means as many as the population holds |

Each round derives its own generator from the master seed and the size of the
population it is spinning over — the same idiom as `WEIGHT_MASTER_SEED` and an
individual's number. One recorded seed, every draw repeatable, and a second
round still draws its own parents rather than the first round's again.

### 9. `gep_lora/core/search/mutation.py` → `individuals.chromosome`, `individuals.has_changed`

Selection makes copies; mutation is what makes them worth having.

```bash
python start_run.py mutation
```

Every symbol of every non-elite chromosome is offered a chance, `MUTATION_RATE`,
of being replaced by a different symbol — **per symbol, not per chromosome**, so
an eleven-symbol individual at `0.1` expects about one change and may well come
through untouched.

```
rate 0.100 per symbol, over 5 of 6 individual(s) (#1 is the elite and is left alone)
    #    symbols chromosome
    2    2       CAT.L5.SVD.w1.L1.L1.w2.w2
         ->      CAT.L5.CAT.w1.L2.L1.w2.w2
mutated 3, left 3 unchanged; has_changed is set on 3 individual(s)
```

**The elite does not change.** The individual marked `is_best` is passed over —
[elitism](#7-elitismpy--individualsis_best) named it precisely so the best
result found so far survives a generation intact, and mutating it would throw
away the thing the flag exists to protect. It is the one row this step reads and
does not write.

#### Only valid changes

A symbol may only be replaced by one of its own kind, wherever it stands —
the root included:

| Class | Swaps with | Why it stays valid |
|---|---|---|
| `CAT` `SVD` `LIN` | each other | arity 2, children still operators |
| `L1`–`Ln` | each other | arity 1, child still a variable |
| `w1`–`wn` | each other | arity 0 |

So a fold at the root stays a fold (`CAT` may become `SVD`) and a lone adapter
stays a lone adapter (`L3.w2` may become `L1.w2`); neither becomes the other,
since that would change the tree's shape rather than a symbol of it.

That restriction is the whole trick. A symbol's arity, and the alphabet its
children are drawn from, are properties of its **class** rather than of the
symbol — so a swap inside a class leaves the tree exactly the shape it was:
every node still has the number of children it had, and every child is still
legal where it stands.

Reaching across classes would not. Turning a `CAT` into an `L2` leaves a node
with two children where one belongs, and the second of them a subtree where a
bare `w` is required — not a chromosome at all. There is no repair step here and
no tail to absorb the damage, so the mutation simply does not make changes it
would have to repair. Every result goes through `generate_population.check()`
before it is stored, which makes that a guarantee rather than a hope.

What *can* still come out unbuildable is a `LIN` above two different ranks —
swapping `CAT` for `LIN` produces one easily. That is a legal chromosome
describing a blend PEFT will not build, so it is culled downstream exactly like
any other: `runs` marks it `state = 'BAD'` and `process` skips it. See
[the rank rule](#the-rank-rule-why-27-of-100-individuals-cant-run).

#### `has_changed`, and the fitness that goes with it

`has_changed` is set to `1` on an individual whose chromosome this step actually
altered, and `0` on every other — the elite included, and an individual the dice
passed over. It is **this round's answer, not a running total**, so it is written
for every individual each time rather than only for the ones that moved.

**Setting it to `1` clears that individual's `fitness` to NULL.** The score was
earned by the chromosome that has just been replaced, which makes keeping it
worse than stale — it would let a mutant be elected, or win a slice of the
roulette wheel, on the strength of a blend it no longer describes. NULL says
what is true: this chromosome has not been judged yet. Both `elitism` and
`selection` already read a missing fitness as no fitness, so a mutant is passed
over by each until `process` and `evaluate` have given it a score of its own.

Its `tree`, `script_source` and `rank` are stale too, but those are only
descriptions and are re-derived wholesale:

```bash
python start_run.py trees runs
```

Note that `fitness` reads the individual's **most recent execution**, so it must
be `process` that runs next, not `fitness` — running `fitness` before the mutant
has been executed again would compute it from the old chromosome's transcript
and put the stale score back.

| Where | Setting | Default | Meaning |
|---|---|---|---|
| `settings.py` | `MUTATION_RATE` | `0.1` | chance per symbol; `0.0` turns mutation off without removing the step |
| `settings.py` | `MUTATION_MASTER_SEED` | `None` | where the dice come from; `None` draws one and records it |

### 9a. `gep_lora/core/search/weight_mutation.py` → the weights alone

```bash
python start_run.py weight_mutation
```

The step after `mutation`, and a narrower one: it only ever swaps a weight
symbol (`w1`–`w5`, the child of each `L*` leaf) for a different one. Operators
and slots are never touched, so the blend keeps its shape and its adapters and
only how much of each it takes moves.

**The rate is a share of all the weights, not a chance per weight.** Every
weight of every non-elite individual goes into one pool, and
`round(WEIGHT_MUTATION_RATE × pool)` of them (half up) are drawn from it without
replacement — so ten weights at `0.1` is exactly one change, and sixty is
exactly six, falling wherever the draw puts them.

```
rate 0.100 of 19 weight(s) across 8 of 8 individual(s): 2 drawn
    #    weights chromosome
    1    1       CAT.L1.L3.w2.w2
         ->      CAT.L1.L3.w3.w2
```

**The elite's weights are not in the pool**, so it is never re-weighted, for
the reason mutation leaves it alone.

An individual that moved gets its new chromosome through
`store.set_chromosome()` — `has_changed = 1`, fitness back to NULL — exactly as
a mutant does. **This step never writes `has_changed = 0`**: it runs after
`mutation` in the same generation, and clearing the flag on an individual
mutation moved would have `process` skip a chromosome that has never run.

| Where | Setting | Default | Meaning |
|---|---|---|---|
| `settings.py` | `WEIGHT_MUTATION_RATE` | `0.1` | share of all non-elite weights moved per round; `0.0` turns it off |
| `settings.py` | `WEIGHT_MUTATION_MASTER_SEED` | `None` | where the draw comes from; `None` draws one and records it |

### 10. `gep_lora/core/pipeline/continue_run.py` → generation after generation

`gep_lora/core/pipeline/start_run.py` runs a sweep from a fresh population through **one** generation.
`gep_lora/core/pipeline/continue_run.py` carries that sweep on:

```bash
python continue_run.py
```

One generation is every step but `population`, in pipeline order — a complete
turn of the crank:

```
trees -> runs -> process -> evaluate -> fitness -> elitism -> selection -> mutation
    -> weight_mutation
```

describe the chromosomes, build them, run them, judge them, score them, keep the
best, breed from the fit, vary the offspring. The population that comes out is
the one the next generation goes in with.

**The last generation stops after `fitness`.** The three steps behind it build
the next generation, and the last one has none, so the sweep comes to rest on
the population that was just scored rather than on one selection has half
replaced and mutation has rewritten. The banner says which generation that is
and the driver says so again when it gets there:

```
# generation 3 of 3 -- population 10, the last: it stops after fitness
...
# stopped after fitness: elitism, selection, mutation, weight_mutation build the next generation
# and there is none, so the population is the one that was just scored
```

**Everything comes out of the database.** There is no `population` step here and
no new sweep: this continues one that already exists, reads the settings *that
sweep* was created with, and writes back into it. Nothing is taken from
`settings.py` as it stands today — `GENERATIONS` included: `generation_count()`
reads `--generations` first, then the sweep's own stored `GENERATIONS`, and only
then `settings.py`'s, which is the fallback for a sweep stored before the setting
existed. A sweep continued a week later therefore runs the search it was set up
to run, not the one whoever last edited `settings.py` had in mind; `--set
GENERATIONS=N` changes it in writing, like any other knob.

`--from-db` goes one step further and takes the *questions* out of the database
too, rather than out of the files the sweep's settings name —
[see §11](#--from-db-and-how-the-rows-become-questions).

Which also means **editing `settings.py` does nothing to a sweep already under
way** — a sweep created before you changed a knob stored the old value and goes
on using it. That is the point for the seeds and the template; it is merely in
the way for a knob you only discover you wanted after the first generation. So
`--set` changes one *on* the sweep:

```bash
python continue_run.py --set SELECTION_COUNT=3
```

The value is written into the sweep's settings table — so the sweep still
records what it ran under, rather than being read past — and takes effect from
that generation on. The name has to be one the sweep already holds, so a typo is
caught rather than quietly stored. Values are read as JSON, so `3`, `0.25` and
`null` all mean what they look like. It is repeatable.

```bash
python continue_run.py --db run_real/gep.sqlite3 --run 3 --generations 5
```

| Option | Default | Meaning |
|---|---|---|
| `--db` | `run_db/gep.sqlite3` | the database holding the sweep |
| `--run` | `0` (the latest) | which sweep to continue |
| `--generations` | the sweep's `GENERATIONS`, then `settings.py`'s | how many turns to run |
| `--from-db` | off | read the eval prompts from the sweep's stored dataset rows |
| `--set NAME=VALUE` | none | change one of the sweep's stored settings, e.g. `--set SELECTION_COUNT=3` |

`--limit`, `--include-blocked`, `--include-unchanged`, `--keep-scripts`,
`--timeout` and `--force` are there too, and mean what they mean in `gep_lora/core/pipeline/start_run.py`
— the steps read them off the options either way.

#### Watch the size

**Selection appends `n` copies plus one newcomer and culls that many again**,
so the population stays the size it was drawn at and the cost per generation is
flat. `process` still loads the base model once per individual, every
generation, though only for the part of the population that actually changed,
since `process` skips individuals whose chromosome has not moved since their
last execution. Ten generations from a population of 10:

```
population by generation: 10 -> 10 -> 10 -> ... -> 10
individuals through process in total: 100
```

`SELECTION_COUNT` is a knob on the **turnover**, not on the size. The exception
is `None`, which asks for as many copies as the population holds — one more
than the cull can take, so that setting alone still grows it, by two a
generation.

The driver prints that projection, and the total, **before it starts anything**,
so the cost is on screen rather than discovered three hours in. It prints and
carries on — it never stops to ask, so it stays usable from a script or a
scheduled job. Fixing `SELECTION_COUNT` to a number makes the growth linear
instead — with `--set SELECTION_COUNT=3` for the sweep in front of you, since
editing `settings.py` only reaches the next one:

```
population by generation: 6 -> 9 -> 12 -> 15 -> 18
individuals through process in total: 42
```

A generation that fails stops the loop: the sweep is marked `failed` and what
the earlier generations did stays in the database.

The `_run` suffix is not decoration — `continue` is a Python keyword, so a
`continue.py` could be run but never imported.

### 11. `gep_lora/core/pipeline/main.py` → the whole search, one command

`gep_lora/core/pipeline/start_run.py` and `gep_lora/core/pipeline/continue_run.py`, in that order, against the same sweep:

```bash
python main.py
```

which is

```bash
python start_run.py
python continue_run.py --run <the sweep gep_lora/core/pipeline/start_run.py just made>
```

so a full run is **1 + `GENERATIONS`** generations — `gep_lora/core/pipeline/start_run.py`'s own turn of the
crank, then the ones `gep_lora/core/pipeline/continue_run.py` adds. `--generations` controls the second
half; there is no way to have fewer than the one `gep_lora/core/pipeline/start_run.py` runs, because drawing
a population and leaving it unjudged would not be a generation. (Use `gep_lora/core/pipeline/start_run.py`
on its own for that.)

```bash
python main.py --generations 3
python main.py --db run_real/gep.sqlite3 --label "overnight"
python main.py --limit 2 --generations 1     # a smoke test of the lot
```

It calls the two drivers **as libraries, in this interpreter**. That matters:
`process` launches every generated script with `sys.executable`, so a subprocess
would be one more chance to run the search under the wrong Python. Whatever you
start this with is what the whole sweep uses — so it still wants the venv's
python, for the same reason `gep_lora/core/pipeline/start_run.py` does.

The sweep is handed on **by id, not by "the latest one"**: `gep_lora/core/pipeline/start_run.py`'s new sweep
is looked up once it exists and named explicitly, so a database that gains a
sweep from somewhere else in between cannot be picked up by mistake. Running
`gep_lora/core/pipeline/main.py` twice into one database leaves two sweeps side by side, each
continued only by its own half of the run.

Options go to whichever driver understands them — `--label` to `gep_lora/core/pipeline/start_run.py`,
`--generations` and `--set` to `gep_lora/core/pipeline/continue_run.py`, and `--db`, `--run-dir`,
`--limit`, `--include-blocked`, `--include-unchanged`, `--keep-scripts`,
`--timeout` and `--force` to both, meaning there what they mean there.

A failing half stops the run: if `gep_lora/core/pipeline/start_run.py` cannot produce a sweep there is
nothing to continue, and its exit code comes straight back out.

#### And then the testing pass

When the sweep names a `TESTING_SET`, the search is followed by
[§14's testing pass](#14-test_run_with_datasetpy--test_results) against it —
same interpreter, same sweep, by id:

```
######################################################################
# the search is done -- testing run 7 against datasets/medical_testing_lora_dataset.json
######################################################################
```

`TESTING_SET` is the trigger because it is the only statement anyone has made
about which questions the search was *not* judged on. Leave it `None` and the
run says so and stops there; the pass can still be run by hand later.

It is deliberately the last thing. A finished search ends in `mutation`, so the
individuals worth testing are the ones that were actually scored — which is what
their stored scripts still describe, and what the pass runs.

Mind what it costs: one base-model load per individual above
`TESTING_MIN_QUALITY`, on top of the search. `--test-min-quality` moves that bar
for one run and `--no-test` skips the pass entirely; `--db`, `--limit`,
`--keep-scripts`, `--timeout` and `--force` reach it too.

```bash
python main.py --no-test                  # the search alone, as before
python main.py --test-min-quality 0.7     # test fewer of them
```

The pass runs only if the search itself finished — a half-finished sweep's best
individual is not what the search found. If the search finishes and the pass
does not, that is said in as many words and the exit code is the pass's.

#### Stopping and carrying on — `--resume` and `--evaluate`

```bash
python main.py --db <sweep.sqlite3> --run 1 --resume
python main.py --db <sweep.sqlite3> --run 1 --evaluate [--force] [--set JUDGE_BASE_URL='"http://..."']
```

**Any sweep can be carried on from wherever it stopped** -- a Ctrl+C, a crash, a
cancelled API job, a machine switched off. `--resume` reads how far it got out
of the database (`where_it_stopped()`), finishes the generation it stopped in,
runs the generations it has left through `gep_lora/core/pipeline/continue_run.py`, and then the testing
pass. Two records say how far, and both are written as the search goes, so a
kill at any moment leaves them true:

- `fitness_history` -- how many generations were **scored**;
- `step_timings` -- which steps **finished** since the last snapshot, which is
  the only way to tell a population about to be bred from one that already has
  been.

The second matters because the tail (`elitism`, `selection`, `mutation`,
`weight_mutation`) is the half that is not safe to run twice: selection is
another round every time, and mutation mutates again. So a sweep resting on a
snapshot with part of the tail done runs only the rest of the tail. Everything
before `fitness` *is* safe to repeat, so a sweep stopped inside a generation
starts that generation again from `trees`: `trees` and `runs` re-derive,
`process` skips whatever already ran as it is (see the process step above),
`evaluate` skips what is scored, and `fitness` records the generation once. A
failed step is not a finished one; a sweep with no population yet is simply run,
the way a prepared one is; one whose search is complete goes straight to the
testing pass, which is resumed too (`test_run_with_dataset.py --resume`:
individuals already tested cleanly are not tested again).

The questions come from the sweep's stored rows when it holds them, as an
adopted sweep's do, so a resumed sweep asks what it asked before whatever the
files say now. `--set` is written into the sweep before anything is planned, so
`--set GENERATIONS=5 --resume` extends a search.

**`--evaluate` grades the answers a sweep already holds** -- the ungraded ones,
or all of them with `--force` -- without moving the search along. It is for a
judge that was down, has moved (`--set JUDGE_BASE_URL=...`, `JUDGE_MODEL`,
`JUDGE_BACKEND`, written into the sweep so it still says what graded it), or is
worth asking again. The rubric is the sweep's own `EVALUATOR`: a fitness is only
comparable with the others when one rubric earned it, so another rubric belongs
to a [verification](#verification--is-the-blend-better-than-the-loras-in-it),
which leaves the sweep alone.

Whether fitness follows is decided by where the sweep stopped, because a
fitness taken at the wrong moment is a generation in the history that never
happened:

| Where it stopped | `--evaluate` runs |
|---|---|
| search complete, or resting on a snapshot before selection | `evaluate`, `fitness` (restating that generation), and `elitism` if it had already run |
| inside a generation that has been through `process` | `evaluate`, `fitness` -- exactly what came next |
| anywhere else (bred since the snapshot, or not all run) | `evaluate` alone; the resume takes fitness from there |

Then the testing pass's stored answers, `--score-only`, if it has any. A resume
after an evaluation carries on from the steps the evaluation recorded.

#### Running a database that is already a sweep

```bash
python main.py --db dbtemplates/test_new_run.sqlite3
```

A database can be **prepared** rather than produced: a run row, its settings, its
dataset, and no individuals. Everything a search needs, and none of the search.
`dbtemplates/test_new_run.sqlite3` is one of those — 49 settings and a 60-record
`training` split, waiting for a population.

Handed one of those, `gep_lora/core/pipeline/main.py` **runs it** rather than starting a second sweep
beside it. `--db <a prepared database>` means "run this"; the alternative would be
a stranger's sweep turning up in somebody's experiment file. `--run 0` (or
`--run N`) says the same thing out loud and takes any sweep by id.

The rule is **the latest sweep having no individuals**. Anything else is a
database to start a new sweep in, exactly as before — one that does not exist
yet, one holding no sweeps, one whose latest sweep has a population — so
`gep_lora/core/pipeline/main.py --db run_real/gep.sqlite3` goes on meaning what it meant. `--label`
suppresses it too, since only a sweep being created can be given one.

From there the database is the only thing the run reads itself out of:

* **the settings are the sweep's stored ones**, which is what resuming already
  meant — so the evaluator, the base model, the adapters, the seeds, the batch
  size and `GENERATIONS` are the ones written down beside the questions;
* **the questions are the sweep's stored dataset rows**, not the files those
  settings name.

`settings.py` and the files under `datasets/` are never opened, and the results
go back into the same file, because `--db` is where the sweep lives. So a
prepared database is a whole experiment in one file: hand it to another machine
and the search it runs there is the one it describes, rather than the one that
machine's `settings.py` happens to say.

```
======================================================================
full run: gep_lora/core/pipeline/start_run.py, then gep_lora/core/pipeline/continue_run.py for 5 more generation(s)
======================================================================

run 1 in dbtemplates/test_new_run.sqlite3 is prepared -- settings and a dataset, no individuals -- so this runs it
    rather than starting a second sweep beside it. Its settings and its questions, not settings.py's.
    49 stored setting(s); evaluator similarity, base model unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit
    training    60 record(s), 60 with a reference answer

resuming run 1 in dbtemplates/test_new_run.sqlite3

datasets from the database (run 1 in test_new_run.sqlite3):
  training    60 record(s) -> dbtemplates\test_new_run_run1\training.jsonl
```

##### It is isolated on disk too

Everything a run of this kind writes goes into **one folder beside the database**,
named for it and the sweep:

```
dbtemplates/
    test_new_run.sqlite3
    test_new_run_run1/
        training.jsonl        the questions, out of the rows
        run_001.py ...        the generated scripts, until they have run
        testing_scripts/      the testing pass's re-pointed copies
```

`run_db/`, `run_testing/` and the rest of the repo are untouched; two prepared
databases cannot tread on each other; and the caches the child processes drop in
their working directory land there as well, since each script is launched with
the run folder as its cwd. The folder is chosen in `start_run.context_for()` — the one
place both drivers build a `Context` — so whichever of them is turning the crank
keeps to it. `--run-dir` (and `--into` for the testing pass) still overrides.

All of it is derived: delete the folder and the sweep is still whole, because the
scripts, the transcripts and the questions are all in the database.

Two things are refused rather than guessed past. **A sweep that already holds
individuals**: `population` appends, so joining a started search would draw a
second population beside the first — the message points at `gep_lora/core/pipeline/main.py
--resume`, which carries one of those on from wherever it got to. Only `--run` can reach that
refusal, since a sweep with a population is never adopted on its own. And
**`--run` together with `--label`**, which names a sweep as it is created; that
one already exists, so label it when you prepare the database. A `--db` that
`--run` names and that is not already a database is refused too, since
`store.connect()` creates what it cannot open and a typo would otherwise become
an empty sweep.

The testing pass is then gated on the sweep **holding a `testing` split** rather
than on `TESTING_SET` naming a file — the same question, asked of the rows.

#### `--from-db`, and how the rows become questions

`--from-db` is the flag underneath all of this, and the three drivers take it on
their own:

```bash
python start_run.py --db <prepared> --run 1 --from-db
python continue_run.py --db <prepared> --run 1 --from-db
python -m gep_lora.core.testing.test_run_with_dataset --db <prepared> --run 1 --from-db
```

It needs a sweep (`--run`), because a *new* sweep is the moment those rows are
read out of the files and stored — there is nothing to read back yet.
`test_run_with_dataset.py` takes no dataset argument alongside it, and records
nothing: the rows it reads already are the stored split.

[`db_datasets.py`](gep_lora/core/storage/db_datasets.py) is the mechanism, and the counterpart of
[`add_dataset.py`](gep_lora/core/storage/add_dataset.py): that one is the only way
*into* the `datasets` table, this is the way back *out*.
`repoint(conn, run_id, conf)` writes each split the sweep holds into the run
folder above — `training.jsonl`, with `.jsonl` or `.txt` chosen from the records
themselves — and hands back the settings with the three `*_SET` values pointing at
what it wrote.

Every existing reader then works unchanged: `generate_runs.eval_records()`, the
evaluators that grade against a reference, the control
`llm_judge_baseline` loads, and above all the generated scripts, which are handed
a path as a literal and open it at startup. Nothing had to learn about sqlite to
be fed from it.

Three things about that are deliberate:

* It writes **from the `content` column** — the line as it was read — so what
  comes out is what went in, parsed by the same readers into the same questions
  and the same references. Rewritten from the rows every time: a cache of them,
  never read back into them.
* It repoints the settings **in memory only**. The `settings` table goes on
  saying where the questions originally came from, which is the provenance those
  rows exist to keep; a sweep that rewrote its own `TRAINING_SET` to name a cache
  would lose the one record of what it was built on.
* A split the sweep does **not** hold has its setting **cleared** rather than left
  alone. The point of the mode is that local files are not consulted, so a
  `TESTING_SET` naming a file no testing rows ever came from names nothing, and
  saying so beats quietly reading it. A sweep with no `training` rows at all is
  refused outright, before a population is drawn — that is a sweep from before
  the `datasets` table existed, and `add_dataset.py` is how it gets some.

### 12. `gep_lora/tools/test.py` → `run/test_*`

Try one chromosome by hand without starting a sweep. Set the variable at the top
of the file and run it:

```python
CHROMOSOME = "CAT.L1.L2.w5.w2.w2.w1"
```

```bash
python -m gep_lora.tools.test
```

Or pass one straight in:

```bash
python -m gep_lora.tools.test CAT.SVD.LIN.L1.L2.L3.L1.w3.w3.w2.w1
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
| `w1.L1.w2` | `not a valid chromosome: expression must start with an operator (CAT/SVD/LIN or an L*), not w1` |
| a `LIN` above a `CAT` | `verdict: BLOCKED`, naming the node and both ranks |
| `CAT.L1.L2.w5.w2.w2.w1` | builds the tree, reports the 2 unused trailing symbols |

### 13. `gep_lora/core/storage/store.py` → the sweep database

The schema, and the only module that imports `sqlite3`. Everything above is a
library of pure functions — `build_population`, `draw`, `plan`/`render`,
`launch`, `score` — and `gep_lora/core/pipeline/start_run.py` is what calls them and puts the results here.

```
runs          one sweep: when, which template, which interpreter, which commit
  settings    every knob it ran under, including the seeds
  datasets    the records it was given -- training, validation and testing --
              saved whole at the moment the sweep was created
  individuals the population: chromosome, tree, rank, verdict, and the
              generated script in full
    executions  one per time that individual was run: exit code, seconds, the
                weight seed and the weights it drew, stdout, stderr
      exchanges the questions and answers, and the judge's score for each
  fitness_history  what each individual's fitness was at the end of each
                generation, and when it was worked out

  test_results  what an individual said on a dataset it was never scored on,
                one row per individual per testing pass, transcript, scores
                and the mean they come to

baselines     what the base model itself answered, per model and question --
              the one table outside that tree, because a base-model answer
              belongs to the model rather than to any one sweep
```

`evaluate` scores the most recent execution of each individual; older ones keep
the scores they were given.

```bash
python -m gep_lora.core.storage.store --list
```

```bash
python -m gep_lora.core.storage.store --show 0
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
python -m gep_lora.core.storage.store --export 0 --into export
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
python start_run.py process evaluate
```

Resumes the most recent sweep; `--run 3` names one instead.

#### What still touches the disk

The generated `run_NNN.py` scripts, and only those — `process` launches them as
subprocesses, so they have to be real files. They land in `run_db/`, beside the
database, because a generated script finds the LoRA folders by going up **one**
level from itself. A folder one level below the project works; a folder inside
one would not. (The eval prompts are exempt: `TRAINING_SET` is resolved at
generation time and stamped into each script as an absolute path.)

They are a cache of `individuals.script_source`, not a second copy of the truth,
which is what makes both halves of their life cycle safe: `process` writes any
that are missing or stale before it runs, and **deletes each one it has
processed** once the sweep is through. So a finished sweep leaves `run_db/`
holding the database and nothing else — no spent scripts piling up, and no stale
script for someone to run by hand a week later.

```bash
python start_run.py process --keep-scripts
```

keeps them when you want to read or re-run one. Otherwise they come back from
the database on demand:

```bash
python start_run.py runs
```

Only the scripts that actually ran are removed. Ones skipped as `BAD`, or left
out by `--limit`, are still waiting and stay where they are.

### 14. `gep_lora/core/testing/test_run_with_dataset.py` → `test_results`

Everything above happens on the training split. Every fitness number, every
election, every roulette slice is earned on those questions, so an individual
that scores well there has been *selected for that file* — and the file stopped
being unseen the moment the first generation was scored on it. This asks the
other question: does the blend hold up on questions it was never picked for?

```bash
python -m gep_lora.core.testing.test_run_with_dataset datasets/medical_testing_lora_dataset.json
```

```
sweep 7 in run_db/gep.sqlite3
testing     20 record(s), 20 with a reference answer, from .../medical_testing_lora_dataset.json

testing 3 individual(s) scoring above 0.500, on 20 question(s)
wrote 3 script(s) to .../run_testing, each re-pointed at the testing set
each one loads the base model, so this takes a while

[1/3] run_005.py  CAT.L5.L5.w3.w3  (training quality 0.577)
[2/3] run_001.py  CAT.L4.L1.w5.w1  (training quality 0.529)
[3/3] run_004.py  CAT.SVD.L3.L1.SVD.w2.w3.L2.L4.w4.w1  (training quality 0.512)
        batch 1/1: 3 running at once
        run_005.py  ok        41.2s  20 answer(s) -> test result 1
```

Four things happen, in order:

1. **The dataset is recorded** as this sweep's `testing` split, through
   `add_dataset.add()` — the same rows a sweep saves for itself, so what the
   pass asked is stored beside what the search asked. Running the same file
   again is not a conflict; a *different* file over a split already stored
   needs `--replace`.

   Unless it came *from* there. `--from-db` takes the dataset argument's place
   and tests on the sweep's own stored `testing` split, written out beside the
   database by `db_datasets.repoint()`
   ([§11](#--from-db-and-how-the-rows-become-questions)), into `testing_scripts/`
   under the sweep's own folder. Nothing is recorded in
   that case: those rows already are the split, and re-storing them from a copy
   of themselves would only overwrite the `source` column with the name of the
   cache. Passing both a file and `--from-db` is refused rather than resolved.
2. **The individuals are chosen** by mean quality on the training split, above
   `TESTING_MIN_QUALITY` (0.5; `--min-quality` for one pass, `--limit` for the
   best few). The bar is the sweep's own stored `TESTING_MIN_QUALITY` before
   `settings.py`'s, the way every other knob is read — those individuals were
   selected under it. How many of the testing questions each one is asked is
   `TESTING_COUNT` (the first N, `None` for all of them, the default; `--count`
   for one pass), read in the same order. The quality itself comes from the `individual_quality`
   view rather than `individuals.fitness`, so a sweep whose fitness step never
   ran still has an answer.
3. **Each one's own script is re-pointed**, into `run_testing/`. Not
   re-rendered — the stored `script_source` is taken as it is and exactly two
   assignments are rewritten, `TRAINING_SET` and `TRAINING_COUNT`. Same
   chromosome, same weight seed, same adapters, same base model, same template
   code of the day it ran; different questions. Re-rendering would rebuild all
   of that from whatever the template says today, which would make the pass a
   comparison of two template versions as much as of a blend.
4. **They run** the way `process` runs a generation — batches of
   `PROCESS_RUN_BATCH_SIZE`, streaming progress, one base-model load each, a
   crash stored as a row rather than stopping the pass — and every answer lands
   in `test_results`.
5. **The answers are graded**, by the sweep's own evaluator unless
   `--evaluator` says otherwise, and the scores go back into the same rows.

The scripts go in a folder of their own (`TESTING_RUN_DIR`, one level below the
project folder like `run_db/`, because a generated script still finds the LoRA
folders by going up one) and are deleted once they have run, `--keep-scripts`
aside. A `run_007.py` that asks the testing questions has no business sitting in
`run_db/` next to the one that asks the training questions.

**Then it grades them**, with the sweep's own `EVALUATOR` — the same registry,
the same `prepare()`/`score()` contract the evaluate step uses, the same rule
that an evaluator which cannot score one answer fails that answer and nothing
else. The sweep's evaluator is the default because a testing quality graded by a
different rubric than the training quality it sits beside would make the
comparison meaningless; `--evaluator NAME` overrides it when the point *is* to
grade this dataset differently.

```
evaluator: similarity -- token or character overlap with the dataset's own answer
grading against .../medical_testing_lora_dataset.json
scoring 60 answer(s)

individual 5  CAT.L5.L5.w3.w3
    [1] 0.62  token f1 0.62 (p 0.58, r 0.67)
    ...
    -> 0.548 over 20 answer(s)

#    training testing  delta   chromosome
5    0.577    0.548    -0.029  CAT.L5.L5.w3.w3
1    0.529    0.402    -0.127  CAT.L4.L1.w5.w1
mean over 2 individual(s): training 0.553, testing 0.475 (-0.078)
```

That last table is the answer to the question the pass exists to ask. An
individual that holds its score answered questions it was never selected on; one
that drops was selected for the training file rather than for the job.

Each score goes into the transcript beside the answer it belongs to, and the row
keeps the mean, which evaluator produced it and what that evaluator's label was
(a judge's model id, or a local method's name). The mean is the same average
over the same kind of answers that `individual_quality` takes for a training
run, which is what makes the two numbers comparable at all.

**The scoring half reads the sweep's settings with one substitution**: the eval
set is the testing dataset (`testing_conf()`) — the same swap `repoint()` makes
to the scripts. Four of the seven evaluators grade against the eval set's own
answers (`llm_judge_reference`, `llm_judge_answers` and `similarity` read the
reference beside each question; `llm_judge_baseline` asks the base model those
questions), and left
pointing at `TRAINING_SET` every one of them would quietly compare a testing
answer with a training question's reference.

**The two halves are separable**, because only the first costs a base-model
load:

```bash
python -m gep_lora.core.testing.test_run_with_dataset testing.json --no-score      # answers only
python -m gep_lora.core.testing.test_run_with_dataset testing.json --score-only    # grade them later
python -m gep_lora.core.testing.test_run_with_dataset testing.json --score-only --force --evaluator similarity
python -m gep_lora.core.testing.test_run_with_dataset testing.json --resume        # carry an interrupted pass on
```

`--resume` is for a pass that was stopped part way: `test_results` is appended
to, so running it again whole would store a second row for every individual it
had already reached. With it, an individual already tested cleanly on that
dataset, as the chromosome its script builds now, is left alone; the rest run,
and the scoring grades every row still ungraded. `gep_lora/core/pipeline/main.py --resume` passes it.

Scoring is resumable the way the evaluate step is — an answer that already has a
quality is left alone unless `--force` — so an interrupted judge, a judge that
was not up yet, or a change of mind about the rubric all cost nothing but the
grading. A mocked pass arrives pre-scored (the mocked template grades its own
answers) and never reaches an evaluator; those rows still take their mean,
recorded as `generated`.

It gives up early on the same terms the evaluate step does. A row here *is* one
individual, so `JUDGE_ABANDON_FRACTION` is a fraction of the answers this pass
still has to grade for it — the ones an interrupted pass already scored are left
alone and do not count — and once that many graded answers have all come back
`0.0`, the rest of that individual's answers are written as `0.0` with the
reason saying so. The row's mean is then the mean of the whole individual, which
is what makes a testing quality comparable with the training quality it is
printed beside. As in the evaluate step, only the evaluators that ask a model
abandon anything, and neither an empty answer nor a failed grading call counts
toward it.

`test_answers` reads the graded transcripts back one answer at a time:

```sql
SELECT number, chromosome, position, question, answer, quality, reason
  FROM test_answers WHERE run_id = 7;
```

**Results live in their own table, on purpose.** `fitness`, `elitism` and
`selection` all read an individual's *latest execution*; a testing pass stored
in `executions` would be picked up as that individual's current result and would
decide the next generation on questions the search is not judged on. Its own
table cannot be read by mistake. Each row keeps the number, the chromosome, the
weight seed, the weights and the training quality that picked it, as they were
when it ran, so mutation rewriting the population afterwards does not rewrite
what the pass found.

One consequence worth knowing: a sweep that has just finished ends in
`mutation`, so most of its individuals hold a chromosome their stored script no
longer describes. The pass runs the script, records the chromosome the *script*
builds, and says how many rows that applies to. Run `trees runs process
evaluate` again first if you want the current population tested.

`python -m gep_lora.core.storage.store --show` lists the passes a sweep has been through.

---

### 14a. `gep_lora/core/testing/evaluate_chromosome_against_loras.py` → is the blend better than one LoRA?

A sweep only ever scores a blend against the bare base model or against the
dataset's own answers, so it cannot say whether a single adapter would have
done as well. This script puts the sweep's best individual next to each adapter
in its `LORA_SLOTS`, with each adapter attached **alone, at full strength**, on
the same questions:

```bash
python -m gep_lora.core.testing.evaluate_chromosome_against_loras --db api_jobs/user1/job6/job.sqlite3
python -m gep_lora.core.testing.evaluate_chromosome_against_loras --db run_db/gep.sqlite3     --evaluator llm_judge_reference --judge-model qwen3-32b --count 20
```

- **What it runs:** the best individual (`is_best`, else the highest fitness;
  `--individual N` for another) and one script per slot (`--slots L1,L3` to
  narrow). All of them are rendered from the sweep's own template and settings,
  and the blend keeps its weight seed, so it is the blend the search scored:
  on job6 its answers matched the stored testing pass word for word.
  `--stored-script` runs the individual's stored script instead, re-pointed.
- **Which questions:** the sweep's stored testing split if it has one, else
  training (`--split`, or `--dataset FILE`), capped by that split's own
  `TESTING_COUNT`/`TRAINING_COUNT` unless `--count` says otherwise.
- **Which judge:** the sweep's `EVALUATOR` and judge, unless `--evaluator`,
  `--judge-model`, `--judge-backend` or `--judge-base-url` name others. It
  never abandons a contestant early, so every mean covers every question.
- **What it says:** each contestant's mean, and against each adapter how many
  questions the blend won, tied and lost, the mean difference, and an exact
  two-sided sign test on the non-ties.
- **What it writes:** nothing to the sweep. Everything goes in
  `<db>_run<N>/lora_comparison/` beside the database (`--into` moves it):
  `answers.json`, keyed on a hash of each script, and a
  `scores_<evaluator>_<judge>.json` and `.md` per grading. Running it again
  with another judge reuses the answers and only grades; `--rerun` runs the
  scripts again, `--no-score` stops after running them.
- **Mocked sweeps:** answers arrive pre-scored and are used as they are unless
  an evaluator or judge is named.

---

### 15. `gep_lora/core/reporting/generate_html_db_stats.py` → an HTML page beside the database

`store.py --show` prints a sweep. This writes the same sweep out as one
self-contained HTML file, with the things a column of numbers cannot draw: the
best blend as a tree, fitness generation by generation, the population's spread,
what the judge actually gave.

```bash
python -m gep_lora.core.reporting.generate_html_db_stats run_db/gep.sqlite3
python -m gep_lora.core.reporting.generate_html_db_stats run_db/gep.sqlite3 --run 2 --open
```

The page lands **beside the database** — `gep_run1_stats.html` next to
`gep.sqlite3` — because that is where the thing it describes lives; `--out`
puts it elsewhere. `--run 0` (the default) is the most recent sweep.

It leads with an individual: the score, the chromosome, the blend drawn as a
tree, the Karva rows, the weight draw, the adapters it attaches, a bar per
question, the whole transcript with the judge's reason under each answer, and
the script that earned it. Then the search's own history, the population, the
distribution of scores, the testing pass, **where the time went**, the dataset
and every setting the sweep was created with.

### The cost section

The same rows `python -m gep_lora.core.metrics.report` prints, drawn. Six tiles first --
total time, the costliest step, what one individual and one answer came to, the
biggest single phase, and how much script time the sweep spent -- then:

- **every step, ranked**, with what it did beside what it cost, so a step is
  read against the work it was given rather than on its own;
- **inside the generated scripts**, as one stacked bar with a legend: the model
  load, the answers, the folds, the unsloth import. Shares of *script* time, not
  of the step's wall time -- a batch runs `PROCESS_RUN_BATCH_SIZE` scripts at
  once, so those seconds overlap each other, and whatever the bar does not cover
  is the interpreter starting up before the script's own clock does;
- **work the steps did themselves** -- materialising scripts, the judge calls,
  the evaluator's `prepare` -- against each other, since beside the waiting it is
  rounding error;
- **every phase**, with `calls` next to the seconds. That column is the one that
  decides what a fix is worth: a phase paid once per pass gets cheaper only by
  running fewer passes, one paid per script or per prompt gets cheaper by being
  cheaper. The `shape` column says which it is;
- **what each individual cost**, one stacked bar and one row per execution, the
  costliest first. Two blends of the same population differ by what their trees
  make the script do -- the leaves it attaches, the nodes it folds -- on top of a
  base-model load every one of them pays;
- **pass by pass**, one column per pass stacked by step. A total that climbs is
  the population growing under it; one that is flat is the price of the search
  rather than of its size.

A sweep with no timing rows -- one run before the tables existed -- simply has no
cost section, the way a sweep with no testing pass has no testing one.

Each leaf of the tree carries its slot's **weight and rank** — `w5 = 0.8400 ·
r16` — because the rank is what decides whether the folds above it can run at
all: `CAT` sums the ranks it meets, `SVD` takes the larger, `LIN` refuses two
that differ. Reading the bottom row is reading the constraint the whole tree was
built under, and a `BAD` individual usually shows its reason right there.

Those ranks come out of the sweep's **own stored scripts**, not off the adapters
on disk. Every generated script's build order names each leaf's rank —

```
    n1_L4      = L4 @ w5                       rank 4
    n2_L1      = L1 @ w5                       rank 16
    n3_CAT     = CAT(n1_L4, n2_L1)             rank 20
```

— which is the rank that individual was *built with*, for the same reason every
step reads its sweep's stored settings rather than `settings.py`: `LORA_SLOTS`
may have been repointed since, and the adapters may not be on this machine at
all. A slot no script mentions simply has no rank shown; an invented one would
be worse than none.

**Which individual is a combobox**, and the best one is what it starts on. Every
individual in the population gets a complete panel — the whole panel above,
built for each of them — and choosing one from the box swaps which is on screen.
There is a second copy of the same box on **The search**, kept in step with the
first, because the selection reaches that chart too: the fitness plot draws the
selected individual's own line across the generations beside the best/mean/worst
band. A line that stops is an individual whose fitness mutation cleared; one
that starts late is a copy selection appended; no line at all is an individual
that was appended after the last fitness snapshot, and the caption says so
rather than leaving you looking at an empty chart.

**Two buttons replay the sweep.**

- *Replay the search*, on the fitness chart, wipes the plot in from the left
  with a playhead, naming each generation as it passes — its best, its mean, its
  size and its fittest chromosome.
- *Play the evolution*, under the population, walks the population forward one
  generation at a time: bars grow and shrink to each generation's scores, the
  chromosome beside each bar changes as mutation rewrites it, new bars arrive
  as selection appends its copies and its newcomer, and bars disappear where
  the cull took them. A slider scrubs to any generation.

Both replay `fitness_history`, which is the only place a sweep says what it
*used to be* — every row keeps the chromosome, the state and the fitness **as
they were then** — so the animation is the recorded past rather than the present
population rearranged. Neither is a second chart drawn in JavaScript: the
drawings are rendered whole by Python, and the script only widens a clip
rectangle over one and moves widths and labels on the other. With no script at
all you get the finished picture of each, which is what printing gets too.

Three things it deliberately does:

- **Read-only, and it refuses to create.** `store.connect()` makes a database it
  cannot open, which is right for a driver and wrong for a reader — a mistyped
  path would leave you with an empty sweep and a report about nothing. So the
  file has to exist first.
- **Nothing external.** No CDN, no fonts, no JSON beside it; the charts are
  hand-drawn SVG that takes its colours from the page's own CSS variables, so
  light and dark are one drawing. The page still works mailed, archived or
  opened offline.
- **It says what the numbers do not.** A best individual that is `BAD`, that has
  never run, or that mutation has rewritten since it was scored gets a note
  saying so. A run that finishes normally no longer produces the last of those —
  its final generation stops after `fitness` — but a sweep stopped
  mid-generation, or driven a step at a time, still can.

`quality` and `fitness` are shown side by side in the population table for the
same reason: the first is the mean over the latest execution's answers, the
second is the column the search reads, and mutation clears the second while
leaving the first standing.

One economy worth knowing about, since a panel per individual is not free:
selection copies `script_source` **verbatim**, so a copy that has not been
through `runs` again carries its parent's script byte for byte. Rather than
repeat eighteen kilobytes of identical Python per copy, those panels say whose
script it is and link to that individual — which is the truer statement anyway.

---

## The async API

`gep_lora/service/` and `gep_lora/apps/web/` put the search behind a web service: a user submits a job (settings
and datasets), a background worker runs it, the user reads the results, puts an
individual live and streams answers from it. Standard library only, like
`lora_server.py`.

```bash
python -m gep_lora.service.users add alice          # prints alice's API key, once
python -m gep_lora.apps.web.server                   # http://127.0.0.1:8780
python -m gep_lora.service.worker                   # the background half
```

Then open **http://127.0.0.1:8780/** -- it lands on the guide. The server has
five pages, and every one of them carries the same top bar, in this order:

| Page | Address | What it is for |
|---|---|---|
| **Guide** | `/guide.html` | start here: an assistant that trains LoRAs and blends them (below) |
| **Visual guide** | `/visual_guide.html` | draw a blend of your LoRAs as a tree, and test it (below) |
| **Compare** | `/blend_comparison.html` | two blends side by side -- opened from any job or drawn -- edited and tested on the same questions (below) |
| **Runs** | `/runs.html` | every search you have run, and what came of it |
| **Settings** | `/settings.html` | appearance, and what a new guide conversation starts from |
| **Console** *(advanced)* | `/console.html` | every endpoint, by hand |

A page's address ends in `.html` and an endpoint's never does (`GET /runs` is the
JSON the Runs page reads). The bar is `gep_lora/apps/web/static/nav.js`, the one file to edit
to add a page or a link; the current page is highlighted, the API key is asked
for once and shared by all of them, and a search open in the guide or the console
(`?job=N`) goes with you when you switch between those two. The old addresses
(`/agent`, `/demo`, `/guide_defaults`) redirect to the new ones.

**The console** is a test page, served by the API itself,
that drives every endpoint -- paste the key, submit a job from a settings form
built from `GET /settings` (the server's own values, with only what you change
sent; *Quick demo* is a one-generation mocked sweep, done in seconds; training and
testing questions come from a shared file, a file you upload, or pasted lines),
watch it run, download its database, read its population,
fitness chart, transcripts and log, **verify a blend against the LoRAs it is made
of** (see below), set an individual live, stream answers in either
format, unset and delete. It is `gep_lora/apps/web/static/console.html`, one file with no dependencies
but the shared bar.

Start the worker with the venv's python, as you would `gep_lora/core/pipeline/main.py` -- it runs
`gep_lora/core/pipeline/main.py` under its own interpreter. Start the server with it too if a real
(non-mocked) sweep is going to go live, since inference loads the model in the
server's process. Knobs are in `gep_lora/service/settings.py` (not `gep_lora/core/config/settings.py`,
whose `snapshot()` would freeze them into every sweep), each overridable as
`GEP_API_<NAME>` in the environment.

**A job is a prepared database.** `POST /jobs` writes exactly what
`python main.py --db <it>` already adopts: a run row, its settings and its
dataset, no individuals, in `api_jobs/user<N>/job<M>/job.sqlite3`. The settings
are `gep_lora/core/config/settings.py` plus the submission's overrides, checked and seeded by
`start_run.freeze()` -- the same function `new_sweep()` uses. The worker takes
jobs in arrival order (registry ids) and runs
`gep_lora/core/pipeline/main.py --db <job.sqlite3> --run <id>` as a subprocess, so a cancel can stop the
search and its lora servers, and the console goes to `job.log`. Everything the
run writes stays in `job_run1/` beside the database.

```json
POST /jobs
{
  "label": "overnight",
  "settings": {"GENERATIONS": 3, "COUNT": 10},
  "datasets": {
    "training": [{"messages": [{"role": "user", "content": "..."},
                               {"role": "assistant", "content": "..."}]}],
    "testing": {"file": "medical_validation_lora_dataset.json"}
  },
  "options": {"no_test": false, "timeout": 900}
}
```

A dataset is a list of records, the text of a JSON Lines / plain file, or
`{"file": name}` naming a file under `datasets/`. Unknown settings, the path and
dataset settings (`LOCKED_SETTINGS`), an unknown evaluator and missing adapters
are refused with a 400 before anything is queued.

| Endpoint | What it does |
|---|---|
| `POST /jobs` | submit; 201 with the queued job |
| `GET /jobs[?status=]` | the user's jobs, each with its best individual |
| `GET /jobs/{id}` | the job and its results: population, best, fitness history, testing, costs |
| `GET /jobs/{id}/status` | status, queue position, generations scored so far |
| `GET /runs` | every search of the user's, newest first, each with the LoRAs it blended (by the user's own catalogue names), how many of its blends were tested, its verifications and what of it is live -- one read for the runs page |
| `GET /jobs/{id}/log` | the tail of `job.log` |
| `GET /jobs/{id}/database` | the job's `job.sqlite3`, as a consistent snapshot (sqlite's backup), even while it runs |
| `GET /jobs/{id}/individuals/{n}` | one individual and its transcript |
| `POST /jobs/{id}/cancel` | a queued job at once; a running one is **stopped** within a poll, and can be resumed |
| `POST /jobs/{id}/label` | `{"label"}` -- rename the job (an empty label clears it); the sweep's own stored label is left as it was |
| `POST /jobs/{id}/resume` | queue a `stopped`, `cancelled` or `failed` job again, to carry on from where it got to |
| `GET /jobs/{id}/evaluate` | what grading its answers again would do: the evaluator, the judge, how many answers and how many ungraded |
| `POST /jobs/{id}/evaluate` | queue it to grade the answers it holds: `{"force"?, "judge_backend"?, "judge_model"?, "judge_base_url"?}` |
| `GET /jobs/{id}/test` | what a testing pass would do: the testing questions, how many blends could be tested and how many have been, the results so far |
| `POST /jobs/{id}/test` | queue a finished job to test its blends on its testing split -- every one that ran, by default: `{"min_quality"?, "count"?, "limit"?}` |
| `DELETE /jobs/{id}/run` | delete what the run produced, keep the job listed as `deleted` |
| `DELETE /jobs/{id}` | delete the job and its folder |
| `GET /jobs/{id}/verify` | what a verification may ask for (the blends, their slots, the splits, the evaluators) and the job's verifications so far |
| `POST /jobs/{id}/verify` | queue one: `{"individual": n?, "slots": "blend"\|"all"\|[...], "split"?, "dataset"?, "count"?, "evaluator"?, "judge_model"?, "judge_backend"?, "judge_base_url"?}` -- `dataset` is `{"file": a shared dataset}` or `{"lora": id}`, that LoRA's training data, or `{"text", "name"?}`, an uploaded file, instead of a split |
| `GET /verifications/{id}` | one verification: its status, and its report once it has one |
| `GET /verifications/{id}/log` | the tail of that verification's own console output |
| `POST /jobs/{id}/live` | `{"individual": n?, "target": "local"?}` -> a token (shown once) |
| `GET /jobs/{id}/live`, `GET /live` | live deployments |
| `DELETE /jobs/{id}/live`, `DELETE /live/{id}` | unset for inference |
| `POST /infer` | `{"token", "prompt", "max_new_tokens"?}` -> the answer, streamed |

### Stopping, resuming, and grading later

**Every job can be stopped and carried on.** Cancelling a running job *stops*
it: the worker kills `gep_lora/core/pipeline/main.py` and marks the job `stopped` rather than
`cancelled`, because what it did is in its database and
[`gep_lora/core/pipeline/main.py --resume`](#stopping-and-carrying-on----resume-and---evaluate) carries
it on from exactly there. `POST /jobs/{id}/resume` puts a `stopped`, `cancelled`
or `failed` job back in the queue with `task: "resume"`; the worker then runs
`gep_lora/core/pipeline/main.py --db <job.sqlite3> --run <id> --resume`, appending to the same
`job.log`. A job cancelled before it ever started is resumed from the top; a
job left running by a worker that died is marked `failed` when the next worker
starts, and is resumable like any other.

**Its answers can be graded later.** `POST /jobs/{id}/evaluate` puts a job that
has stopped with answers in it (`done`, or resumable) back in the queue with
`task: "evaluate"`, and the worker runs `gep_lora/core/pipeline/main.py --evaluate` over it --
`--force` to grade every answer again, and `--set` for a judge that has moved.
The rubric stays the sweep's own `EVALUATOR`. Grading does not move a search,
so the job ends where its search is: `done` if every generation was scored,
`stopped` (and resumable) if not, with any failure of the grading in `error`.
`gep_lora/service/evaluate.py` owns the API's half, the way `verify.py` does for
verifications.

A requeued job keeps its id, and with it its place in the queue. Cancelling a
resume or an evaluation before the worker gets to it puts the job back as it
was (a `done` job stays `done` and can still go live); cancelling one that is
running stops it -- a resume ends `stopped` again, an evaluation where its
search is.

**Its blends can be tested on their own.** `POST /jobs/{id}/test` puts a
*finished* job back in the queue with `task: "test"`, and the worker runs the
testing pass alone -- `python -m gep_lora.core.testing.test_run_with_dataset --from-db --db
<job.sqlite3> --run <id> --min-quality=-1.0 --resume` -- rather than through
`gep_lora/core/pipeline/main.py`, which would only hand it on. Two things set it apart from the pass
`gep_lora/core/pipeline/main.py` runs at the end of a search: it tests **every blend that ran**, not
only those above `TESTING_MIN_QUALITY` (`min_quality` narrows it back down),
because it exists so a person can *choose* a blend and one with no testing
score cannot be weighed up; and it always resumes, so a blend already tested
cleanly on the split keeps its row and only the rest run. Like an evaluation
it does not move the search, so the job ends `done` again, with a failure of
the pass in `error`. A job that is not `done` is a 409 -- a half-finished
search's blends are not what it found -- and one with no testing split a 400.
`gep_lora/service/testpass.py` owns the API's half.

On the page, the job's actions carry **Stop** (a running job) or **Cancel** (a
queued one), **Resume**, and **Evaluate…**, which opens a small form: the
evaluator it will grade with, how many answers are still ungraded, where the
judge is, and whether to grade everything again. The tiles show answers graded
so far, which is the number that says an evaluation is worth asking for.

### Verification — is the blend better than the LoRAs in it?

A search reports fitness, and fitness says nothing about the obvious control: one
of those adapters used on its own. So the page's **Verification** section runs
that comparison for any blend of a finished job -- the blend, and each LoRA its
chromosome names, attached alone at full strength, on the same questions, all
graded by one judge. It is
[`gep_lora/core/testing/evaluate_chromosome_against_loras.py`](#14a-testingevaluate_chromosome_against_loraspy--is-the-blend-better-than-one-lora),
queued.

**It is work, not a request.** Every contestant costs a base-model load, so a
verification goes into the registry's own `verifications` table and the *same*
worker takes it -- after any queued job, since a job is a search that has not
started and a verification is a question about one that has finished. One card,
one queue. `gep_lora/service/verify.py` owns only the API's half: what the form may
offer, what a request may ask, the command line, and the report read back.

On the page, pick a blend (the elite is selected first), what to compare it
against (**the LoRAs in this blend**, or every slot), which split and how many of
its questions, and the evaluator and judge -- defaulting to the sweep's own, since
a verification graded by another rubric would not be comparable with the fitness
the search produced. Anything the sweep cannot offer is a 400 before the task is
queued: an unknown evaluator or backend, a split it does not hold, a slot it has
no adapter for, a `BAD` individual.

**The questions may come from elsewhere.** Instead of a split, `dataset` names
another file: `{"file": name}`, one of the shared datasets, or
`{"lora": id}`, the data one of the user's *own* LoRAs was trained on (another
user's is refused as a missing one is), or `{"text": ..., "name": ...}`, a file
the page read and sent -- JSON Lines, a JSON array (turned into JSON Lines) or
one prompt per line -- stored by `verify.upload()` in the job's own
`verify_datasets/` folder, so deleting the run takes it. The console's
*Questions from* box offers all four. The server resolves it to a path and
the script asks it with `--dataset`; the verification's options say which by
`dataset_label`, never by where the file sits on the server. The evaluator is
passed only when the request names one: left out, the script grades by the
sweep's own `EVALUATOR` as before -- except for a mocked sweep, whose answers
it scores with the numbers its scripts printed rather than asking a judge about
invented answers.

When it finishes, the page draws the report the script left behind:

- **mean quality per contestant**, the blend picked out from the adapters;
- **win / tie / loss per adapter**, question by question, with the exact
  two-sided sign test's *p* beside each bar;
- **blend minus one adapter, per question**, for whichever adapter is picked,
  each bar carrying its question and both scores;
- **the numbers**, and one sentence saying which differences hold up at
  *p* < 0.05 and which are inside the judge's own noise.

Everything a verification writes lives in `verify<id>/` inside the job's folder,
so `DELETE /jobs/{id}/run` takes its answers and scores with it: a reading of a
sweep is worth nothing without the sweep.

### LoRAs — the catalogue, and training new ones

The page's **LoRAs** view lists the user's own adapters and trains new ones.
Both halves go through the API, with the same key as every other screen.

**One database holds everything the API knows**: `api_jobs/api.sqlite` --
users, jobs, verifications, trainings, deployments and the LoRAs. (It was
`jobs.sqlite3` plus a separate catalogue; the first registry to open a folder
holding a `jobs.sqlite3` and no `api.sqlite` copies it across once, through
sqlite's backup, and renames the old file `jobs.sqlite3.merged`. The LoRA rows
came back with `catalog scan`, since their folders are their truth.)

**The catalogue** is its `loras` table ([`gep_lora/core/adapters/catalog.py`](gep_lora/core/adapters/catalog.py)),
one row per adapter folder: base model, rank, alpha, target modules, chat
template, where its data came from, steps, final loss, and a status (`queued`,
`training`, `ready`, `failed`, `cancelled`, `missing`). The folder is the truth
and the row an index -- `python -m gep_lora.core.adapters.catalog scan` rebuilds it from
`loras/`, reading `adapter_config.json`, the `training.json` that
`create_lora.py` now writes beside the weights, or, for an adapter trained
before that existed, its newest checkpoint's `trainer_state.json`. The three
sets under `loras/Lora001..005` (Qwen3.5 0.8B, Qwen2.5 0.5B, Qwen2.5 1.5B
medical) are its first fifteen rows, and belong to `ze`. `create_lora.py` keeps
its own row up as it trains, whether it was started by the API or by hand;
`--no-catalog` opts out.

**A LoRA is seen by its owner and nobody else.** Every row has an `owner`, a
user's name: a LoRA trained through the API is its trainer's from the moment it
is queued, and one found by `scan` or trained by hand is nobody's -- and so
nobody's to see -- until the server gives it to someone:

```bash
python -m gep_lora.core.adapters.catalog own ze --unowned      # every row nobody owns
python -m gep_lora.core.adapters.catalog own ze 3 4 5          # or by id / name; 'nobody' takes them back
python -m gep_lora.core.adapters.catalog scan --owner ze       # new folders go straight to ze
python -m gep_lora.core.adapters.catalog scan | list [--owner ze] | show <id or name> | forget <id> | models
```

`own` refuses a name the database's `users` table does not hold. Another user's
LoRA is a 404 on every `/loras` endpoint, exactly as another user's job is;
names are unique per owner rather than across the catalogue, so choosing one can
never reveal another user's; each user's trainings go to their own
`loras/trained/user<N>/`; and **a job blends its owner's LoRAs and nobody
else's**. A submission must give `LORA_SLOTS` as L1..Ln (one to ten, no gap), each one of
the user's own *ready* catalogue rows -- named by id, by name or by folder, the
same LoRA in as many slots as wanted -- and each trained on the job's
`BASE_MODEL` (`submit.own_slots()`, `check_base_model()`). There is no
default: `gep_lora/core/config/settings.py`'s `LORA_SLOTS` (the `loras/Lora00N` set) is the
command line's, and `GET /settings` offers `{}` for it. A folder the user was
not given -- another user's, one nobody owns, the command line's own set -- is
refused in the same words as one that does not exist, and before anything
reads the disk, so a slot cannot be used to learn what is on the server. The
server can still hand a `Lora00N` folder to someone with `catalog own`; it is
then theirs like any other.

**Training one is queued work, like a job.** `POST /loras` takes a name, a
dataset (the same shapes a job's does, but every record must be a conversation
with an assistant turn) and any of `create_lora.py`'s options; the worker runs
`python -m gep_lora.core.adapters.create_lora loras/trained/user<N>/<name> ...` after any queued job and
before any verification, with its console in `api_jobs/user<N>/lora<id>/train.log`.
The adapter lands in `TRAINED_LORAS_DIR` (`loras/trained/`) and in the
catalogue, so a search's `LORA_SLOTS` can name it like any other. Defaults are
`create_lora.defaults()` -- the command line's own -- except the base model and
chat template, which are the search's. [`gep_lora/service/train.py`](gep_lora/service/train.py)
owns only the API's half (form, checks, command line), the way `verify.py` does.

`create_lora.py` gained the knobs the form offers -- `--dropout`,
`--target-modules`, `--max-steps`, `--scheduler`, `--warmup-steps`,
`--weight-decay`, `--optim` -- and writes, beside the weights, `training.json`
(the recipe, the dataset's source, sha-256 and size, the loss at every logged
step, how it ended) and `dataset.jsonl` (the data itself), so an adapter folder
describes itself. Progress goes to stdout as
`TRAIN: {"step", "of", "epoch", "loss", "lr"}` lines. `--mock` trains nothing: a
fake loss curve and an `adapter_config.json` with no weights, in seconds, for
checking the plumbing -- a mocked LoRA fits a mocked search and is refused
(400) by a submission whose template loads weights.

**Base models** for the form come from three places: the catalogue (models there
are LoRAs for), the Hugging Face cache (models downloaded here with a
`ForCausalLM`/`ForConditionalGeneration` head), and the judge endpoint's list
(served names, labelled as such -- usually a GGUF build, not something unsloth
can train). The box takes any id.

On the page: pick a base model, the data (shared file, upload, paste), rank
(with 4/8/16/32/64 chips), alpha, dropout, chat template, target modules, the
optimisation settings and the smoke-test prompt, and queue it. The detail view
follows it live -- a step meter, the loss curve drawn as it trains, the log --
then shows the recipe, the adapter, the smoke test's answer and the data it
learned. **Use in a new job** opens the job form on its base model with it in a
slot; **Train again…** fills the form from its recipe; **Stop** and **Delete**
are the trainer's own. The job form's base model and five **LoRA slots** now
come from the catalogue too -- each slot a select of that model's ready LoRAs,
with their ranks, and a warning for a slot on another model or a mocked LoRA
under a real template.

| Endpoint | What it does |
|---|---|
| `GET /loras[?base_model=]` | the user's LoRAs, each with its training (if it was trained here) and what may be done with it |
| `GET /loras/form` | defaults, choices, and the base models the user has LoRAs for or that are downloaded |
| `POST /loras` | `{"name", "dataset", "settings"?, "mock"?}` -> 201, queued |
| `POST /loras/scan` | re-read `loras/`; reports on the user's own rows, and how many belong to nobody |
| `GET /loras/{id}` | one LoRA: its record, loss history and live progress |
| `GET /loras/{id}/log` | the tail of its training's console |
| `GET /loras/{id}/dataset[?limit=]` | the conversations it learned |
| `POST /loras/{id}/cancel` | a queued training at once; a running one within a poll |
| `DELETE /loras/{id}` | a LoRA you trained here: folder, scratch, log and rows; never a server default slot |
| `GET /health` | no key needed; what is loaded |
| `GET /settings` | settings.py's values and the choices (templates, evaluators, backends), for a form |
| `GET /datasets` | the shared dataset files a submission may name |
| `GET /judge/models[?base_url=]` | the chat models a judge endpoint lists (default `JUDGE_BASE_URL`), asked by the server |

**Every judge-model box on the page has a list of the endpoint's models under
it** -- the submission form's, the verification's and the evaluation's. The box
is the truth: it is what is sent, and it takes any model id, listed or not. The
list only helps fill it in: picking a model writes it into the box, typing in
the box moves the list to match (or to *not listed*), ↻ asks the endpoint
again, and the list is asked afresh whenever the endpoint URL beside it
changes. It comes from `GET /judge/models`, which asks the endpoint's
`/models` from the server -- LM Studio sends no CORS headers, and the endpoint
that matters is the one the worker's machine can reach -- with
`$JUDGE_API_KEY` if it is set, leaving out embedding models the way
`discover_model()` does, and giving up after `JUDGE_MODELS_TIMEOUT` (5s). An
endpoint that cannot be reached says so under the box, which still works. On
the `unsloth` backend there is no list to ask for, and none is shown.
| `GET /console.html` | no key needed; the test page (`/` opens the guide) |

Every endpoint but `/health` and `/infer` takes `Authorization: Bearer <key>`,
and another user's job is a 404. Only hashes of keys and tokens are stored.

**A job whose search finished but whose testing pass did not is `done`**, with
the reason in `error`: `gep_lora/core/pipeline/main.py` returns the testing pass's exit code in that
case, but the search's results stand and can go live.

**Going live stores a blend spec** with the deployment: base model, chat
template, adapter folders, and the lora-server `/build` plan with every weight
drawn from the individual's own weight seed. It is byte-for-byte the plan that
individual's `template_remote_code.py` script sends (a unit test runs one to
check), so a live blend is built by `lora_server.Blend`, the code that built it
in the search. A target is where it goes: `local` is the one there is --
`inference.ModelCache` loads the model on the first `/infer`, keeps it for
`MODEL_TTL` (300s) after the last request, and rebuilds only the blend when
another deployment on the same base model is asked. Another target is a class
with `publish`/`stream`/`retire` added to `golive.TARGETS`. A deployment of a
mocked sweep streams a mock answer and loads nothing.

Measured on this machine (Qwen3.5-0.8B, one `SVD` node): the first `/infer`
took 53s to its first token (24.5s of it the model load), the next 0.3s.

`/infer` streams chunked `text/plain`, or server-sent events
(`data: {"text": ...}`, then `event: done`) with `Accept: text/event-stream`.

### The LoRA guide — an agent that trains LoRAs, then blends them

**http://127.0.0.1:8780/guide.html** is the page the server opens on, for
someone who has never trained a model. The screen is split in two. On the
left, a guide -- a chat model -- asks for one thing at a time: a friendly
summary of the process and *Yes, let's start*; a dataset (pasted, uploaded, or
one of the shared ones under `datasets/`), which it reads and summarises; how
long they are happy to wait (three options with the time each takes *on this
dataset*, or typed: "about an hour", "5 epochs"); then the plan read back and
*Start training*; once the LoRAs are trained, *Blend them* -- the second
half, below; and once the search is done, *Test all blends*, *Verify it* and
*Go live* -- the third. At any point the chat box takes questions **and requests the
guide carries out** (below). On the right, what is
happening: where the process is, the dataset's numbers (answer lengths, topic
words, samples), the plan, and during training each LoRA's progress and ETA,
the loss curves, the worker's log and every API call the page makes -- redrawn
every second and a half. It is `gep_lora/apps/web/static/guide.html`, one file with no
dependencies but the shared bar, and it uses the same key as the console (and remembers the
conversation in the browser, so a reload mid-training carries on watching).

The Python behind it is `gep_lora/assistant/`, mounted into the web app as the
`/agent/*` endpoints (`gep_lora/apps/web/agent_routes.py`):

| Module | What it owns |
|---|---|
| `settings.py` | which provider and model the guide talks through, the providers' URLs and key variables, the ranks it trains (`LORA_RANKS`), the wait options, the estimate's fallback pace |
| `prompts.py` | **every system prompt**, one per step behind a shared persona, and the wording each step falls back to when no model answers |
| `providers.py` | one chat call, stdlib only: OpenAI-compatible (`/chat/completions`) or Anthropic (`/messages`) |
| `analysis.py` | a dataset in any shape -- JSON Lines, a JSON array, prompt/answer pairs, CSV -- normalised to `{"messages": [...]}` lines and measured |
| `planner.py` | the session (what the chat has set up), the time estimate, and the `POST /loras` bodies |
| `blending.py` | the second half: which of the user's own LoRAs a search combines, its size, where its questions come from, the `POST /jobs` body, and what the search found |
| `release.py` | the third: the search's blends beside their tested scores, the blend picked, the `POST /jobs/{id}/verify` body for it, and what the verification found |
| `selection.py` | which part of a dataset is trained on: first/last N, a range, a percent, a random sample, words to keep or drop, answer lengths, no duplicates |
| `tools.py` | what the chat can *do*: each tool's parameters, the steps it may run in, and what it hands back to the page |
| `commands.py` | the most common requests read without a model, as the same tool calls |
| `agent.py` | each step: the facts, then the model's words for them, or the fallback's; the chat's tool loop |
| `facade.py` | one function per `/agent/*` feature, which `apps/web/agent_routes.py` puts at its address |

**The agent proposes and the API does.** Nothing in `gep_lora/assistant/`
trains, queues or stores anything. `/agent/plan` hands the page one
`POST /loras` body per rank in `LORA_RANKS` (8 and 16 by default -- more than
one because a search blends adapters of different ranks), named for the
dataset, the base model, the day and the rank (`LORA_NAME`:
`poem-qwen3.5-0.8b-20260923-r8`) and made free in the user's catalogue --
each name editable on the page before the training starts, and kept if the
plan is read back again -- with the dataset's first
question as the smoke-test prompt so a finished LoRA's sample answer is on its
own subject. The page sends them to the ordinary `POST /loras`, and watches
them through `GET /loras/{id}` and its log -- the same endpoints and checks as
the console's LoRA form, so a LoRA the guide trained is an ordinary catalogue
row. *Practice run* (`MOCK`) sends `mock: true`: `create_lora.py --mock`, no
GPU, seconds.

**Facts are computed; only the words are the model's.** Record counts,
lengths, duplicates, the estimate and the plan come from `analysis.py` and
`planner.py` and are handed to the model as JSON with a step's prompt. The
estimate's pace is fitted to the LoRAs already trained on the same base model
here (steps against seconds, so loading the model is not read as a slow step),
and is a stated guess until there are some. So **the page works with no model
at all**: a provider that has no key, cannot be reached, or is set to
`scripted` gets the step's fallback wording from `prompts.py`, with a note
under the message saying why.

**The chat can act, not just answer.** Typed messages go to `/agent/chat`,
where the model is given tools (`tools.py`, described to it in
`prompts.TOOLS`) and a few rounds to use them:

| Tool | What it does |
|---|---|
| `select_records`, `use_all_records` | train on part of the dataset: "only the first 20", "records 5 to 30", "half", "a random 25", "drop the ones about fever", "no duplicates", "answers under 40 words" -- or all of it again |
| `show_records`, `dataset_facts` | look before choosing: "show me record 7", "the longest answers" |
| `set_loras` | how many LoRAs and their ranks: "one LoRA at rank 32", "ranks 4, 8 and 16" -- up to `MAX_LORAS` (10, a search's most slots) |
| `set_epochs`, `estimate_time` | how long: "10 epochs", "I can wait 20 minutes" |
| `set_training_options` | learning rate, alpha, dropout, sequence length, batch, warmup, max steps, scheduler, optimiser, the LoRAs' name, the test prompt |
| `set_practice_run` | a practice run on or off |
| `list_demo_datasets`, `use_demo_dataset`, `choose_another_dataset` | switch dataset: "use the poem dataset" |
| `list_my_loras`, `show_plan` | what exists, what is planned |
| `start_training`, `stop_training` | only when asked |
| `open_blending` | on to combining the LoRAs: "blend them" |
| `choose_blend_loras` | which of the user's own ready LoRAs a search blends: "only poem-r8 and poem-r16" |
| `set_blend_search`, `set_blend_questions` | its size -- "5 rounds", "12 blends each", "judge on 20 questions" -- and where the questions come from |
| `show_blend_plan`, `start_blend`, `stop_blend` | what is planned; start and stop, only when asked |
| `start_testing`, `stop_testing` | test every blend of the finished search: "test the blends" |
| `choose_best_blend` | the blend the person thinks is best, by number: "#7 is the best" |
| `set_verify_questions` | what the verification asks: "on the testing questions", "use the poem dataset", "poem-r8's data", "20 questions" |
| `start_verification`, `go_live` | verify the picked blend, or put a blend live -- "verify it", "put #3 live" -- only when asked |

Everything a tool changes lives in the **session**: the part of the dataset in
use, the ranks, epochs, options, name and test prompt. The page keeps it and
sends it with every step, and `planner.session_of()` checks it with
`train.py`'s own limits, so the plan the chat builds is one `POST /loras`
takes. A selection is a few keys applied on the way, never a copy of the data
-- the dataset's card on the right then describes the part in use, and part of
a shared file is sent as its lines, since the whole file is not what it
trained on. **A tool changes the plan, never the world**: starting, stopping
and switching dataset come back as *actions* the page carries out with the
same calls as its buttons, and a changed plan is always read back before a
start. The stage decides what may be asked -- nothing changes while a training
runs, nothing is chosen before there is a dataset -- and a tool asked at the
wrong moment refuses with a reason the model passes on. Each tool run shows
in the transcript as a small line (red when refused). With no model that can
take tools, `commands.py` reads the common requests into the same calls, so
"first 20, rank 32, 5 epochs" works on the scripted provider too.

**Providers.** `lmstudio` (the default) is the judges' endpoint,
`JUDGE_BASE_URL`, with the first model it lists unless one is named; `ollama`,
`openai`, `anthropic`, `gemini`, `mistral`, `openrouter` and `opencode-go` are
the others, and `scripted` asks nobody. OpenCode Go is one key and one URL in
front of three wire formats, so its entry carries `wires` -- model-id prefixes
per format: MiniMax and Qwen go to `/messages` (Anthropic's shape, without
Claude's effort knob), Grok, GPT and Muse to `/responses` (OpenAI's Responses
API), everything else to `/chat/completions`. A family added on another
endpoint needs its prefix there. Go also refuses a request that does not name
its conversation, so the page gives each conversation an id (new on *Start
over*), sends it as `agent.conversation`, and a provider with a
`session_header` gets it in that header (`x-opencode-session`); every request
carries a `User-Agent` of its own, since Cloudflare turns urllib's away. The page's *model* chip switches provider and
model for its own conversation and lists what the provider serves
(`GET /agent/models`); `GEP_AGENT_PROVIDER` / `GEP_AGENT_MODEL` set the
default for everyone. **Keys are environment variables of the server** --
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `MISTRAL_API_KEY`,
`OPENROUTER_API_KEY`, `OPENCODE_API_KEY`, `$JUDGE_API_KEY` for LM Studio -- never settings, never
sent to the browser. Only a local provider may be pointed at another URL from
the page, and then without its key, so no page can send a key anywhere. A
Claude model is asked without `temperature` (Opus 5 and Sonnet 5 refuse it),
at `ANTHROPIC_EFFORT` (low), and only its text blocks are read; a reasoning
model's `<think>` block is stripped from an OpenAI-compatible reply.

| Endpoint | What it does |
|---|---|
| `GET /guide.html` | the page (no key needed to load it) |
| `GET /agent/config` | providers (and whether the server has each key), the default, the ranks, the wait options, the shared datasets described, and the page's own wording |
| `GET /agent/models?provider=[&base_url=]` | the chat models a provider lists |
| `POST /agent/intro`, `/analyse`, `/wait`, `/plan`, `/started`, `/debrief` | one step of the conversation each; `agent: {provider, model, base_url}` picks who phrases it, `session` what has been set up |
| `POST /agent/chat` | a typed message: the reply, the tools run (`steps`), the new `session`, the dataset's facts again if the part in use changed, records asked for (`preview`), and the `actions` for the page |
| `POST /agent/blend/intro` | the user's ready LoRAs, the ones picked (`prefer`: the ones just trained), the search's size and estimate |
| `POST /agent/blend/plan` | the `POST /jobs` body that blends them, read back with exact numbers |
| `POST /agent/blend/started`, `/agent/blend/debrief` | what the guide says once the search is queued, and what it found -- the latter reads only a job of the user's own |
| `POST /agent/test/started`, `/agent/test/debrief` | what the guide says once the testing is queued; then every blend of the search, tested or not, ranked, and the one recommended |
| `POST /agent/verify/plan` | the `POST /jobs/{id}/verify` body for the picked blend, read back with exact numbers |
| `POST /agent/verify/debrief` | the blend beside each of its LoRAs, from the verification's report |
| `POST /agent/live/started` | what the guide says once a blend is live -- sent the deployment's id, never its token |
| `GET /agent/help` | every block of the page that can be explained (`ui_help.py`), `{key: {where, title}}` -- the page draws a question mark on these and no others |
| `POST /agent/help` | `{agent, block, title, shown, stage, history}`: what that block is and what it shows now, as a message for the conversation |
| `POST /agent/summary` | `{agent, stage, mock, dataset, loras, job, verification, deployment, transcript}`: the journey so far as a short illustrated story (`create_summary.py`) -- `{title, markdown, charts, journey, facts, by, note}` |

**Every block can be asked about.** Each card on the right, the charts and
tables inside them, the step's controls and the guide itself carry a small
question mark. Pressing one sends `POST /agent/help` the block's key and the
text it shows at that moment -- the hover titles of a chart included, which
is where its numbers are -- and the answer is written into the conversation
on the left like any other message. What a block *is* comes from
`prompts.UI_BLOCKS`, never from the page; what it *shows* is handed to the
model as a fact to point at ("2 answers of 7-8 words"), cut to
`ui_help.SHOWN_CHARS`. With no model, the block's own description is the
answer. A new block is a key in `UI_BLOCKS` and a `helped()`/`sub()` call in
the page; a test checks the two name the same blocks.

**The journey so far, as a story.** *Summary* in the top bar (or "a recap",
"summarise what we learned" in the chat -- the `show_summary` tool) opens a
centred overlay: the six chapters (data, training, search, testing,
verification, live) drawn as stops on a road, then a short story of what was
done and what it showed, with charts and small tables, ending in what was
learned and what is next. `POST /agent/summary` (`create_summary.py`) is sent
the ids the page holds and the dataset's numbers, and reads the rest as the
user -- a LoRA, search, verification or deployment of someone else's is left
out. The numbers are ours: the charts come back as *data* (`charts()`: answer
lengths, each LoRA's final loss, the search round by round, the top blends
before and after testing, the blend beside its LoRAs) and the page draws them
with the board's own chart code; the model only writes the story, placing a
chart with `[[chart:ID]]`, and one it leaves out is drawn at the end. Its
prompt is `prompts.SUMMARY` -- sent without `PERSONA`, whose "no headings, no
tables" is the chat bubbles' rule -- which is where to tweak what it writes.
With no model, `create_summary.fallback()` tells the same story in the
`SUMMARY_*` wording. *Download PDF* is the browser's own "Save as PDF" of the
overlay alone (the page's print styles hide the rest and force light
colours); *Markdown* saves the text.

**The conversation can be kept.** *Export* in the guide's header saves the
whole conversation on the left, as this browser holds it: *Markdown* to read
or share (every message with its time and the model that wrote it, the notes
under them, the tool calls and events between, grouped by day, under a line
naming the dataset, LoRAs, search, verification and deployment it touched),
or *JSON* with the same and nothing lost. It is made in the page -- no
endpoint -- and carries only what the conversation already shows: never the
API key, a live blend's key or the dataset itself.

**The second half: blending them.** A search needs adapters, and the first
half has just made some, so the guide carries on: *Blend them* (or "blend
them" in the chat, or *Blend my LoRAs* on the welcome for someone who already
has some) lists the user's own ready LoRAs, ticks the ones just trained, and
offers the search's size -- rounds after the first (`BLEND_GENERATIONS`, 3),
blends in each (`BLEND_POPULATION`, 8) and questions every blend is judged on
(`BLEND_QUESTIONS`, 10). `/agent/blend/plan` turns that into a `POST /jobs`
body the page submits as it is, then watches through `GET /jobs/{id}/status`,
its log and `GET /jobs/{id}` (the best and mean score of each round, drawn),
and `/agent/blend/debrief` says what was found: the best blend **in words**
(`stack(poem-r16 ×0.18, merge(poem-r8 ×0.65, ...))` -- `CAT`, `SVD`, `LIN` and
the individual's drawn weights, with the user's LoRA names in place of the
slots), its score, its score on questions the search never saw, and whether
the rounds improved. The third part, below, tests, verifies and puts live.

Four rules shape that body. **Only the user's LoRAs**: every id is looked up
among their catalogue rows (`blending.own()`), the body names the slots by id,
and `submit.own_slots()` checks them again on the way in -- the `Lora00N`
folders are never used. **One place per LoRA, at least five, one model**: the grammar blends up
to L1-L10, a LoRA each, and fewer than five go round again to fill five places
(two LoRAs are L1=A, L2=B, L3=A, ...), and all
of them must share a base model and chat template, which the job then runs
under. **A practice LoRA makes a practice search**: a mocked LoRA has no
weights, so any chosen one (or *Practice run*) sets `TEMPLATE` to the mocked
template -- scores are random, and the guide says so. **The questions are
somebody's answers**: the source the chat chose (a shared dataset, or a LoRA's
training data), else the dataset of this conversation, else the copy of its
data the first chosen LoRA kept; the first `questions` records are the
training split, and of the rest up to `BLEND_TEST_QUESTIONS` go to the
testing split and then up to `BLEND_VALIDATION_QUESTIONS` to the validation
split -- shared out when too few are left for both (`blending.split()`). The
search itself runs **no testing pass** (`no_test`): testing is a step of its
own, below. The estimate is a stated guess (`SECONDS_PER_INDIVIDUAL`), since
nothing measures a search's pace yet.

**The third part: testing, verifying, going live.** The search keeps the
blends whose answers scored best on its own questions, so its scores flatter
them; the guide then asks the questions it never saw, lets the person choose,
checks the choice, and puts one live -- each step one of the API's own
requests, sent by the page:

1. **Test all blends** -- `POST /jobs/{id}/test`: every blend of the finished
   search answers the testing split, graded the way the search was. The page
   watches the job; `/agent/test/debrief` then lists every blend with its
   search score and its tested score, ranked by the tested one, with a
   scatter of the two on the right (above the diagonal: better on new
   questions). *Skip testing* goes straight to choosing, by search score.
2. **Verify it** -- the person picks the blend they think is best (the best
   tested one is selected) and what to check it on: the validation split by
   default, or another split, a demo dataset, or one of the blend's LoRAs'
   own training data, and how many questions. `/agent/verify/plan` builds the
   `POST /jobs/{id}/verify` body: the blend against **every LoRA it uses, once
   each** -- with fewer than five LoRAs one fills several places, and asking it
   once per place would only repeat its answers (`release.lora_slots()`). The
   report comes back as each LoRA's average beside the blend's, and won / tied
   / lost with the sign test's *p*.
3. **Go live** -- the person picks the model to put live (the verified blend is
   selected) and the page calls `POST /jobs/{id}/live`. The token comes back
   once and stays in the page -- on the right, with a `curl` line -- and the
   guide is told the deployment, never the key. A box on the left asks the
   blend through `POST /infer`, the answer streaming in; *Take it down* is
   `DELETE /live/{id}`.

**Every run has an address, and a page of them.** The guide keeps the search
on screen in its address -- `/guide.html?job=12` -- so a reload, a bookmark or a
link brings that search back into the conversation: finished, it is read back
and offers its next step; still running, it is watched. **`/runs.html`**
(*Runs* in the top bar) lists every search of the user's from
`GET /runs`: status, when, rounds and blends, the LoRAs, the best score, and
tags for tested, verified and live, with a filter and a search box, and on
each *Open in the guide* and *Open in Console* -- the console takes
`/console.html?job=12` too, opening that job's detail when it is the caller's own. Both are **private to the key**: the pages are
static and every row comes from the API, which answers only with the caller's
own jobs -- someone else's `?job=` is "no search of yours" -- and the guide's
saved conversation is kept per key (under a hash of it), so two people
sharing a browser never see each other's.

**Every default the guide starts from can be a person's own.**
**`/settings.html`** (*Settings* in the top bar) is, below its *Appearance*
card, one form for the whole process: the chat's provider and model; for
training, the base model, chat template, the ranks, the number of LoRAs
(`LORA_COUNT`: empty is one per rank; a number takes the ranks in turn until
there are that many, at most `MAX_LORAS`, and a rank's second LoRA is trained
under create_lora's seed plus one, its third plus two, so repeats are
different adapters rather than copies), the epochs
behind the three "how long can you wait?" answers, the create_lora recipe
(learning rate, alpha, dropout, batch, scheduler, ...) and whether a practice
run is ticked; for blending, the generations, the blends per generation, the
questions each is judged on, how many are kept back for testing and for
verification, the evaluator and the judge model, and (under *More blending
options*) the search's five seeds -- a number repeats that part of a search,
empty draws one when the search is made; and the size of a
verification asked of a file. Each model box has a drop-down of what its
endpoint lists beside it, and once a model is chosen, a **Thinking** checkbox:
on, off, or (its third, indeterminate state) the model's own habit.

**Thinking is a switch on both models.** `THINKING` (gep_lora/assistant/settings.py,
or the page's per-conversation `agent.thinking`) for the guide's chat model and
`JUDGE_THINKING` (gep_lora/core/config/settings.py, frozen into a sweep like every JUDGE_*
knob) for the judge; `None` sends nothing. On an OpenAI-compatible endpoint it
is `reasoning_effort` -- `"none"` off, `"medium"` on -- because that is the one
control LM Studio was found to honour: on a loaded reasoning model it took a
short answer from 251 completion tokens to 4, while `chat_template_kwargs`,
`reasoning` and a `/no_think` in the prompt changed nothing. Anthropic's wire
gets its `thinking` block (`disabled` / `adaptive`), the Responses API its
`reasoning.effort`, and the unsloth judge the chat template's
`enable_thinking`. An endpoint that refuses the field (400/422) is asked again
without it -- and dropping a refused field no longer spends one of the judge's
`JUDGE_RETRIES`, which it used to for `response_format` too. Each field shows the server's value beside it,
and a card at the top shows the journey a new conversation would take under
what is on screen.

They are saved per user in `api.sqlite` (`guide_defaults`, one JSON document
each) through `GET`/`PUT`/`DELETE /agent/defaults`, and
`gep_lora/assistant/guide_defaults.py` owns the fields and checks them -- by
train.py's rules for a training option, the chat's for ranks, settings.py's
limits for a search. **Only what differs from the server is kept**, so a field
left alone follows the server's value when it changes. Every `/agent/*`
request then runs under the caller's saved values (`facade.theirs`): the
page's config, the session's ranks and blend sizes, the wait choices, the
estimate, the POST /loras bodies (which carry the saved recipe, under the
chat's own changes) and the POST /jobs body (`EVALUATOR`, `JUDGE_MODEL`), so
train.py and submit.py check them as they check anything else. A default is
a start, never a limit: the chat can still change any of it, within the
server's maxima, which are not offered. It applies to a **new** conversation;
one already under way keeps the plan it was shown. Saving a new provider,
model or practice-run default also forgets the guide's remembered choice of
it in that browser, which would otherwise win.

All three need a search that **finished** (the API refuses the rest). A search
that was stopped, cancelled or failed gets **Resume the search** instead
(`POST /jobs/{id}/resume`), and **Use a finished search** -- on the blend step
too -- lists the user's `done` jobs (`GET /jobs?status=done`) so any of them
can be tested, verified and put live, not only the latest.

What the person chose -- which search, which blend, which questions, how many
-- is the session's `release` part (`release.release_of()`), so the chat can do
all of it too: "test the blends", "#7 is the best", "verify it on the poem
dataset", "put #3 live". A blend the search does not hold, or cannot build, is
refused by the tool; another user's search is "no search of yours".

---

### The visual guide — a blend drawn by hand, and tested

`/visual_guide.html` is the other way to a blend: instead of a search finding one,
you **draw** it. The page is laid out like the guide -- the same top bar, themes,
model chip and chat on the left -- and on the right:

- **Your blend**, the tree, drawn the way the report page draws a chromosome
  (`generate_html_db_stats.tree_svg`): each LoRA at its weight, folded in pairs up
  to one fold at the top -- CAT, SVD or LIN -- or a single LoRA on its own, used at
  full strength since only a fold applies weights. A new drawing starts as an empty
  CAT; select the top to change its fold or **Remove** it. Every node shows the rank PEFT gives it; a node PEFT cannot
  build -- a LIN over two ranks -- is red, with the reason under the tree. Click a
  node to change its fold, weight or sides, **Delete** empties a place, **Ctrl+Z**
  undoes. Under it, the chromosome a search would hold the drawing as, and the
  **weights**: what w1..w10 are worth under the drawing's seed, with *New weights*
  to draw again. A weight tile is a piece too: drag it onto a LoRA in the tree (one
  a fold weights -- not a LoRA alone at the top) to give it that weight, or click
  it while a LoRA is selected. On the comparison a weight is a name, so one dragged
  from the other blend's tiles lands worth what this blend's seed makes it. Above it, an **Open** box: pick one of your jobs (a search or a
  drawn blend) and then one of its blends, best first, and it is drawn at once --
  a searched one with the weights it was scored with, marked *edited* once it
  changes. **🎲 Random** replaces the drawing with a random blend of your LoRAs,
  a new one each click (`POST /blends/random`, `drawn.random_drawing()`): grown by
  the search's own `random_tree()` under `MAX_DEPTH`, `BRANCH_PROB` and
  `ROOT_LEAF_PROB`, over your LoRAs on the blend's base model, drawn again until
  it can be built, with a new seed. The chat does it too ("surprise me").
- **Pieces**, under the tree, a row of what a blend is made of: your ready LoRAs
  (each with its rank) and the three folds, **CAT** (stack: ranks add up), **SVD**
  (merge: the larger rank) and **LIN** (average: equal ranks only). Drag one onto
  the tree, or click a place in the tree and then a piece.
- **Test**: the questions (a demo dataset, the data one of your LoRAs was trained
  on, or your own file or paste), how many, and **Test this blend**.
- **Result**: the blend and each of its LoRAs alone, the same questions, the
  same judge -- mean scores, wins/ties/losses with a sign test, and every answer.

The chat can do all of it too ("stack poem-r8 and story-r16", "make the right one
an SVD", "new weights", "test it on 30 questions"), through tools that edit the
drawing the page sends and hand it back (`gep_lora/assistant/visual.py`), exactly as
the guide's tools edit its plan; starting a test is an action the page carries out
with its own button's call.

**Nothing about it is a second pipeline.** `gep_lora/service/drawn.py` turns the drawing
into the chromosome and `LORA_SLOTS` a search would have held it as (slots in
reading order, one per LoRA however often it is used), and checks it with
`generate_population.check()` and `generate_runs.plan()` -- the grammar and the
rank rule every individual meets (`POST /blends/check`). The weights are the draw
`start_run.step_runs` makes for individual 1 (or the searched blend's own number,
when one was opened from a search) under the drawing's
`WEIGHT_MASTER_SEED`, so the numbers on screen are the ones the scripts use. And a
test (`POST /blends/test`) is:

1. a prepared sweep written by `submit.prepare()` -- the chosen questions as its
   training split, `COUNT` 1, the drawing's seed, your LoRAs as its slots;
2. the drawing put in as its only individual through start_run's own `trees` and
   `runs` steps, and the job settled `done` at once with task `blend`: it is never
   queued, and never resumed, evaluated or tested as a search would be (those
   endpoints refuse it with a 409);
3. a **verification** of that individual on those questions, queued for the worker
   -- `gep_lora/core/testing/evaluate_chromosome_against_loras.py`, the same comparison as
   *Verification* above.

So a drawn blend is one of your runs like any other: it is listed on the Runs page
(tagged *drawn*, opened in the visual guide with `?job=N`, which reads it back
through `GET /blends/{job id}`), its database downloads from the console, and it
can be put live with `POST /jobs/{id}/live`.

**Saving an edited blend.** Once a blend opened from a job is *edited*, a **Save into
job N** button appears beside where it came from (on both pages). It saves the
drawing into that job's run as a **brand new blend** (`POST /blends/{job}/save`,
`drawn.save()`): the next number, the run's own slots for its LoRAs (so every LoRA in
it must be one the run blends), its tree and script made by start_run's `trees` and
`runs` steps for it alone, and its weights **pinned**. A run draws each blend's
weights from its seed and the blend's number, so a new number would mean new values;
instead the new individual stores the weight seed it was shown under in
`individuals.weight_pin`, and `step_runs` keeps a pin rather than deriving one --
what was saved is exactly what runs, and copies selection makes of it inherit the
pin. It then joins the search like any individual: a resumed search runs and scores
it, and may select, mutate or cull it. Only a job at rest takes one (done, stopped,
failed or cancelled). The page then holds that new blend -- its number, its pin --
and opening it again (`GET /blends/{job}?individual=N` returns `pin`) draws the same
weights. Older sweep databases gain the column when opened (`store._migrate()`).

**Editing a weight's value.** The **Edit values** switch beside *Weights* (on both
pages, per blend on the comparison) turns the ten tiles into boxes: type what a weight
is worth, above 0 and at most 1 -- the draw's own range -- and the tree, the formula
and everything after follow. A value set by hand is marked, with ↺ to put back what
the seed drew; *New weights* draws afresh and clears them. The chat does it too
("make w3 0.4"). Such values are **set over the seed's draw**, the rest of it
unchanged, and they travel with the blend: each blend template carries a
`WEIGHT_VALUES` block applied over its `WEIGHTS` (`{}` for any other blend), a drawn
test's sweep and a blend saved into a run store them in `individuals.weight_values`,
and the runs step, the verification's re-render and going live all honour them. A
value that differs from where a job's blend came from marks it *edited*.

**Show code.** Every tree has a **Show code** button (both pages). It opens an
overlay with the Python the blend runs as when it is processed -- coloured, with
line numbers, **Copy** and **Save as…** (the browser's own save dialog where it has
one, a download otherwise). It is never a second generator: `POST /blends/code`
(`drawn.code()`) hands back

- the script the run **stored** for that blend, byte for byte, while the blend is
  still a job's own (same chromosome in that run's slots, same weight seed) -- the
  overlay says *Stored script*;
- once it is edited, what **saving** it into that run would make: that run's
  template and settings through `generate_runs.render()`, as its next number, with
  the weights shown;
- for a blend drawn here, what **testing** it would run: this server's `TEMPLATE`
  (the mocked one for a practice run), its LoRAs as slots, and the questions the
  test writes to `training.jsonl` beside its sweep when it starts.

The overlay is `gep_lora/apps/web/static/code_view.js` (`gepCode.show()`), served beside `nav.js`.

### Compare blends — two blends side by side

`/blend_comparison.html` (**Compare** in the top bar) is the visual guide twice
over: the same chat on the left, and on the right **blend A** and **blend B** side
by side, each with the visual guide's whole tree editor. Either side can be:

- **opened from a job** with the box above its tree -- first the job (any search
  of yours, or a blend drawn on the visual guide), then one of its blends, best
  first. It is drawn the moment it is picked. A searched blend keeps its search's
  seed *and its own individual number*, so its weights are the ones it was scored
  with; the side says which job and blend it came from, and *edited* once its
  drawing or its weights have changed since;
- **drawn** from the Pieces under the trees (they go to the blend a place is
  selected in), drawn at **🎲 Random**, **copied** onto the other side, **swapped**
  with it or cleared.

Then both are tested on **the same questions** (**Test both blends**): two drawn
tests, one per side, each the blend beside its own LoRAs alone. The results are
shown side by side, and a card puts **A against B**: the two blends' own answers
paired by question, how many each won, the mean difference and the same exact
sign test a verification uses, one line per question from A's score to B's, and
every question with both answers. The address carries what is open
(`?a=JOB.N&b=JOB.N`), so a link brings the same two blends back.

The chat can do all of it ("open the best of job 3 in A", "copy A to B and make its
top an SVD", "test both on 30 questions"), through `gep_lora/assistant/compare.py`: a
`visual.Toolbox` whose tools take `blend` ("A" or "B") and run the visual guide's
own tool on that side. It adds no pipeline either:

- `GET /blends` lists every blend of yours a page can open, job by job
  (`drawn.sources()`), and `GET /blends/{job}?individual=N` opens any one of them
  (`drawn.opened()`; a search's best without `N`) -- the visual guide opens
  searched blends the same way now;
- a drawing carries an individual **number** beside its seed (`POST /blends/check`
  and `/blends/test` take `number`, 1 by default), and a drawn test stores its
  individual under that number (`store.add_individuals(first=)`), so the sweep of
  one draws exactly the searched blend's weights;
- `POST /agent/compare/plan` reads back the two `POST /blends/test` bodies, the
  page sends them, and `POST /agent/compare/outcome` / `.../debrief` are each
  side's verification and `compare.head_to_head()` -- computed, never the model's.

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
`python start_run.py runs` (or keep them with `--keep-scripts`) before running it by
hand.

A script generated from `template_remote_code.py` needs a server to talk to, and
finds it in `$GEP_LORA_SERVER` — falling back to `http://127.0.0.1:8770`, which
is the default port `gep_lora/core/blends/lora_server.py` listens on:

```bash
D:\sage-is\loras\.venv\Scripts\python.exe -m gep_lora.core.blends.lora_server --base-model unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit
D:\sage-is\loras\.venv\Scripts\python.exe run_db\run_004.py
```

The client itself imports nothing but the standard library, so it runs under any
Python 3 — but the server does not, and the pipeline starts servers with
`sys.executable`, so the venv rule above is unchanged for a sweep.

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

## Where the time goes

A sweep records what it cost as carefully as it records what it found. Every step
of every pass writes a `step_timings` row -- wall seconds, how many items it
worked on, how many it skipped -- and the phases inside it write `phase_timings`
rows with `calls` beside the seconds. `python -m gep_lora.core.metrics.report` reads them back:

```
where the time went -- 41.2m over 9 step(s)
    step          seconds   share  passes did                       per item worst pass
    process         38.1m   92.5%       3 30 individuals (+9 skipped)     76.2s      13.4m
    evaluate         2.6m    6.3%       3 150 answers                      1.0s      55.1s
    runs             8.1s    0.3%       3 39 scripts (+9 skipped)         208ms       3.1s
    ...

inside process -- 38.1m of wall time over 3 pass(es)
  what the step itself did (share of its 38.1m):
    materialise              3     380ms    0.0%     127ms     140ms  per pass
    waiting on scripts      30     38.0m   99.7%                      2.1h of script time, overlapping
  inside those 30 script(s) -- 2.1h of script time, 4.2m each:
    phase                calls   seconds   share      mean     worst  shape
    model_load              30     31.2m   24.7%      62.4s     71.0s  per script
    generate                30     19.0m   15.0%      38.0s     51.1s  per script
    combine.svd             18     12.1m    9.6%      40.3s     56.1s  0.6 per script
    ...
```

(Layout only -- those numbers are invented. Run it on a sweep for real ones.)

The two levels are the two questions. The step table says **which step to
optimise first** -- and it is `process`, every time, because every individual is
another base-model load. The phase table says **what to do about it**, which the
step total cannot: `calls` separates a fixed cost from a per-item one, and they
want opposite fixes. A `model_load` called once per script gets cheaper only by
running fewer scripts -- a smaller `COUNT`, fewer generations, the `has_changed`
skip doing its job, or, since `template_remote_code.py`, not running it per
script at all: a sweep against a
[pool of servers](#paying-for-the-model-load-once--blendslora_serverpy) has no
`model_load` row, which is the clearest way to see what it was costing.
`generate` is one call per `ANSWER_BATCH`
prompts, so `TRAINING_COUNT` moves the tokens in a call rather than the number
of them, and halving it no longer halves that row. `combine.svd` at 40s a node
is an argument about the alphabet the search draws from.

Where the phases come from: the step measures its own work (`materialise`,
`prepare`, one `score` per judge call), and each generated script measures
itself, printing `TIMING: <phase> <seconds>` lines that `process_run.timings()`
reads back with the transcript. Script phases are shares of **script** time, not
of the step's wall time, because `PROCESS_RUN_BATCH_SIZE` of them run at once and
their seconds overlap; the report says so rather than quietly adding them up.
All three templates print the lines, so a mocked sweep on a machine with no GPU
exercises the whole path. A remote script prints the server's phases as its own,
so the rows read the same; the `build` round trip it adds is nested inside them
the way `compact` is nested inside `combine.svd`.

`--step process` prints one step in full, `--csv` prints the rows behind the
tables, and `--run N` picks a sweep. Sweeps run before the tables existed have no
rows, and none can be worked out after the fact.

---

## Things to tune

Both live as tables at the top of every generated script, so they are easy to
change per individual or globally in `template_code.py`.

**`WEIGHTS`** — nothing in the repo defines what `w1`–`w10` are worth, so each
run draws them fresh, strictly between 0 and 1 -- all ten, whichever a tree
uses; the draw is sequential, so `w1`–`w5` are the values they were when there
were only five:

```python
WEIGHTS = {"w%d" % n: _weight() for n in range(1, 11)}
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
repointed at a different adapter without touching the others. It lives in
`settings.py`, not in the templates:

```python
LORA_SLOTS = {
    "L1": "loras/Lora001/my_planning_coach-lora_adapter",
    "L2": "loras/Lora002/my_planning_coach-lora_adapter",
    "L3": "loras/Lora003/my_planning_coach-lora_adapter",
    "L4": "loras/Lora004/my_planning_coach-lora_adapter",
    "L5": "loras/Lora005/my_planning_coach-lora_adapter",
}
```

The slots *are* the search space, so they belong with the rest of the knobs: one
edit reaches both templates, and the sweep records which five adapters its
fitness numbers were earned on. `generate_runs.lora_slots()` resolves them once —
relative to the repo folder — and writes the result into every generated script
as a literal dict, so a script carries real paths rather than deriving them.

Five genuinely distinct adapters, one per slot, all trained on the same base
model (`unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit` — they must share a base
for PEFT to combine them). Their ranks differ:

| Slot | Folder | `r` |
|---|---|---|
| `L1` | `loras/Lora001` | 16 |
| `L2` | `loras/Lora002` | 16 |
| `L3` | `loras/Lora003` | 8 |
| `L4` | `loras/Lora004` | 4 |
| `L5` | `loras/Lora005` | 32 |

A relative entry is taken from the folder holding this README, so scripts in
`run_db/` and `test.py`'s output in `run/` both find the same adapters. An
absolute path is used as it stands, and so is anything that is neither — a Hub
repo id, say — though that is as far as it has ever gone: every rank check reads
the adapter's own `adapter_config.json` off disk.

Ranks are **not** assumed equal. Each slot's rank is read from its own
`adapter_config.json` — at generation time for the docstring and the `state` and
`rank` recorded on the individual, and again at runtime by the script, which
tracks the rank of
every intermediate adapter in `RANKS` and refuses a `linear` step whose two
inputs disagree. Point a slot at an `r=8` LoRA and `CAT(r16, r8)` reports rank
24, with more trees turning up `BAD` because their `LIN` nodes no longer match.

**`EVAL_PROMPTS`** — the questions each individual answers, read from a file at
startup rather than baked into the scripts, so editing that file changes the eval
set without regenerating anything:

```python
TRAINING_SET = 'D:\...\training_set.txt'   # stamped in from settings.py
EVAL_PROMPTS = _prompts(TRAINING_SET)
```

*Which* file is `TRAINING_SET` in `settings.py`, not a line in either template —
one knob, resolved once by `generate_runs.training_set_path()` (relative to the
repo folder unless it is already absolute) and written into every script as a
literal. Repointing the eval set is therefore a settings change followed by
`python start_run.py runs`, and the sweep records which prompts it was scored against,
which matters because the prompts are half of what a fitness number means. A
resumed sweep keeps reading its own stored value: `gep_lora/core/pipeline/start_run.py` passes
`conf["TRAINING_SET"]`, so editing `settings.py` cannot silently move the eval
set out from under a sweep already under way.

One prompt per line. Surrounding double or single quotes are optional (the file
currently uses them), blank lines are skipped, and a missing or empty file fails
with a clear message — at generation time as well as at runtime, since otherwise
every script would die at startup.

A JSON record per line works too, and is what `datasets/*.json` hold: the prompt
is the record's first `user` turn. The `assistant` turn beside it never reaches
the model being scored — that would be showing it the answer — but it *is* the
reference the `llm_judge_reference` and `similarity` evaluators grade against,
read back from the same file by `generate_runs.eval_records()`.

**`EVALUATOR`** — how an answer becomes a number: a judge model, a judge shown
the dataset's own answer, a judge shown what the base model said and asked how
much the blend improved on it, overlap with that answer, local checks, or a
panel.
The registry and every knob behind it are described under
[`gep_lora/core/evaluators/`](#5-evaluators--exchangesquality) above; `python start_run.py
--evaluators` lists what is registered. It is the setting the search's direction
hangs from, so it is worth choosing before a long run rather than after one.

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

One package, `gep_lora/`, in four layers that each import only the ones below
them -- apps -> assistant -> service -> core -- so any number of user
interfaces can sit on the same features (`unittests/test_layers.py` holds the
rule). Three launchers stay at the top, so `python main.py` is still the
command.

```
main.py  start_run.py  continue_run.py     launchers for the three drivers

gep_lora/
  paths.py        where the repo is; every relative setting is against it
  core/           the engine: a sweep, from population to fitness
    pipeline/     the three drivers themselves
    config/       every knob the pipeline reads
    search/       the GEP search itself
    blends/       a chromosome, turned into a script and run
    templates/    what those scripts are made from
    storage/      the database, and the datasets it keeps
    metrics/      what each step cost, measured and read back
    evaluators/   how an answer is scored
    testing/      the held-out pass
    reporting/    a sweep, written out as something to look at
    adapters/     making and checking the LoRAs a sweep blends; the catalogue
  service/        the search as a web service's features: jobs, a worker,
                  trainings, verifications, drawn blends, live inference --
                  facade.py is every one of them as a method, for any UI
  assistant/      the guides behind /guide.html, /visual_guide.html and
                  /blend_comparison.html: prompts, providers, tools, a dataset's facts
  apps/web/       the HTTP API (server.py, agent_routes.py) and its pages (static/)
  tools/          dev aids that are not part of the pipeline
```

### The drivers

| Path | What it is |
|---|---|
| `gep_lora/core/pipeline/main.py` | gep_lora/core/pipeline/start_run.py, gep_lora/core/pipeline/continue_run.py, then the testing pass — a whole search in one command |
| `gep_lora/core/pipeline/start_run.py` | the pipeline driver; add future steps to its `STEPS` list |
| `gep_lora/core/pipeline/continue_run.py` | runs the generation loop on over a sweep already in the database |

### The search

| Path | What it is |
|---|---|
| `gep_lora/core/config/settings.py` | COUNT, SEED, TEMPLATE and the rest — every knob, in one place |
| `gep_lora/core/search/generate_population.py` | the alphabet, `encode`/`decode`, and the random draw |
| `gep_lora/core/search/draw_trees.py` | draws one chromosome as a tree → `individuals.tree` |
| `gep_lora/core/search/calculate_fitness.py` | folds each transcript into one number → `individuals.fitness` |
| `gep_lora/core/search/elitism.py` | marks the fittest individual as the one to keep → `individuals.is_best` |
| `gep_lora/core/search/selection.py` | roulette wheel sampling; appends each pick as a full copy of its parent, culls the weakest, draws one newcomer |
| `gep_lora/core/search/mutation.py` | point-mutates every chromosome but the elite's, within its grammar; clears the fitness it invalidates |
| `gep_lora/core/search/weight_mutation.py` | swaps a fixed share of all the non-elite weight symbols; never the elite's, never clears `has_changed` |

### Building and running a blend

| Path | What it is |
|---|---|
| `gep_lora/core/blends/generate_runs.py` | fills `template_code.py`, one runnable script per individual |
| `gep_lora/core/blends/process_run.py` | launches a generated script and reads its transcript back |
| `gep_lora/core/blends/baseline_run.py` | produces and caches what the base model itself answers, for `llm_judge_baseline` |
| `gep_lora/core/blends/lora_server.py` | one base model held open, blending LoRAs on request — the load, paid once instead of per individual |
| `gep_lora/core/blends/server_pool.py` | starts those servers, hands one to each script of a batch, recycles and stops them |
| `gep_lora/core/templates/template_code.py` | the generated script with `@@MARKERS@@` for the varying parts |
| `gep_lora/core/templates/template_code_mocked.py` | the same, mocked: no model load, random answers and scores |
| `gep_lora/core/templates/template_remote_code.py` | the same, remote: the blend is built on a `lora_server.py` process that already holds the model |
| `gep_lora/core/templates/template_baseline.py` | the base model alone on the eval prompts — the control a blend is measured against |
| `gep_lora/core/templates/template_baseline_mocked.py` | the same, mocked: no model load, invented answers, cached under `mock:` |

### Scoring

| Path | What it is |
|---|---|
| `gep_lora/core/evaluators/` | the evaluators, one module each: `llm_judge.py`, `llm_judge_reference.py`, `llm_judge_answers.py`, `llm_judge_baseline.py`, `jev_judge_reference.py`, `similarity.py`, `heuristic.py`, `panel.py` |
| `gep_lora/core/evaluators/common.py` | what they share: the registry, the judge transport, the reference answers, the tokeniser |
| `gep_lora/core/evaluators/local_model.py` | the other half of the judge transport: the judge loaded here with unsloth, for `JUDGE_BACKEND = "unsloth"` |

### The database, and reading it back

| Path | What it is |
|---|---|
| `gep_lora/core/storage/store.py` | the database: schema, helpers, `--list/--show/--export` |
| `gep_lora/core/storage/add_dataset.py` | the only way into the `datasets` table: every split as a sweep is created, or one afterwards from the command line |
| `gep_lora/core/storage/db_datasets.py` | the way back out: a sweep's stored splits written beside its database, for `--from-db` |
| `gep_lora/core/metrics/record.py` | the Meter a step is timed with, and the only writer of `step_timings` / `phase_timings` |
| `gep_lora/core/metrics/report.py` | what a sweep spent its time on, ranked: `python -m gep_lora.core.metrics.report` |
| `gep_lora/core/reporting/generate_html_db_stats.py` | writes one stored sweep out as a self-contained HTML page, beside its database -- including its cost section |
| `gep_lora/core/testing/test_run_with_dataset.py` | runs a sweep's best individuals against a dataset they were never scored on, and grades what they say |
| `gep_lora/core/testing/evaluate_chromosome_against_loras.py` | runs a sweep's best individual beside each of its LoRAs applied alone, grades all of them with one judge, and says which the blend beats |
| `gep_lora/service/verify.py` | the API's half of that comparison: what a verification may ask for, the command the worker runs, and the report it leaves behind |
| `gep_lora/apps/web/static/guide.html` | the LoRA guide page at `/guide.html`: a chat model walks a user from a dataset to trained LoRAs, with the progress drawn beside it |
| `gep_lora/apps/web/static/visual_guide.html` | the visual guide at `/visual_guide.html`: a blend of the user's LoRAs drawn as a tree and tested, with the guide's chat beside it |
| `gep_lora/apps/web/static/blend_comparison.html` | the comparison at `/blend_comparison.html`: two blends side by side, opened from any job or drawn, edited and tested on the same questions, with its own guide |
| `gep_lora/assistant/compare.py` | the comparison's guide: the visual guide's tools on either side, the two tests' plan, and blend A against blend B question by question |
| `gep_lora/service/drawn.py` | a drawn blend in the pipeline's terms: the tree as a chromosome, its rank check and weights, and the sweep of one plus verification that tests it |
| `gep_lora/apps/web/static/nav.js` | the top bar every page shares: the pages in order, the page's own buttons, the server's health and the API key |
| `gep_lora/assistant/prompts.py` | every system prompt the guide sends, and the wording it falls back to without a model |
| `gep_lora/assistant/settings.py` | which provider and model the guide talks through, and what it plans |
| `gep_lora/assistant/tools.py` | what the guide's chat can do: choose part of the dataset, change ranks, epochs and options, switch dataset, start and stop |
| `gep_lora/assistant/ui_help.py` | "what is this?" for each block of the page: the blocks that can be explained, and one explained from what it shows |
| `gep_lora/assistant/visual.py` | the visual guide's chat: tools that edit the drawing, its intro, the test read back, and the debrief |
| `gep_lora/assistant/create_summary.py` | the journey so far: the user's own facts, the charts as data, the story in the model's words or built-in ones |

### The adapters, and the dev aids

| Path | What it is |
|---|---|
| `gep_lora/core/adapters/create_lora.py` | trains one adapter into a folder, with its recipe, data and loss history beside it, and catalogues it |
| `gep_lora/core/adapters/catalog.py` | the LoRA catalogue, the `loras` table of `api_jobs/api.sqlite`: `scan`, `list`, `show`, `own`, `forget`, `models` |
| `gep_lora/core/adapters/create_all_loras.py` | trains the whole set, varying rank or learning rate |
| `gep_lora/core/adapters/test_lora.py` | asks one adapter a question, without a blend, and prints where the time went; `--no-unsloth` loads through plain transformers |
| `gep_lora/core/adapters/base_models_and_loras_comparison.py` | times loading and inference for every base model under `loras/`, bare and with each of its adapters, into a markdown report in `loras/` -- run it on an idle GPU |
| `gep_lora/tools/test.py` | try a single chromosome → `run/test_*` |
| `gep_lora/tools/combination.py` | the original two-adapter script the generated code is modelled on |

### What is on disk

| Path | What it is |
|---|---|
| `plan.txt` | the original spec |
| `run_db/gep.sqlite3` | every sweep ever run, with its settings, seeds, transcripts and scores |
| `run_db/run_001.py` … | the generated combination scripts, until `process` has run them |
| `datasets/training_set.txt` | the eval prompts, one per line, read by every generated script; the path is `TRAINING_SET` in `gep_lora/core/config/settings.py` |
| `run_testing/` | where the testing pass's scripts run, and are deleted from; `TESTING_RUN_DIR` in `gep_lora/core/config/settings.py` |
| `run/test_tree.txt`, `run/test_run.py` | output for the chromosome currently set in `gep_lora/tools/test.py` |

### Pipeline

`gep_lora/core/pipeline/start_run.py` runs all of this in order. Every module below is a library it calls;
the arrows end in tables, not files.

```
plan.txt                              the rules
   |
add_dataset.save_all                  -->  datasets (once, as the sweep is made)
generate_population.build_population  -->  individuals (chromosome)
draw_trees.draw                       -->  individuals.tree
generate_runs.plan/render             -->  individuals.script_source
   +                                       + run_db/run_NNN.py  (must be files)
gep_lora/core/templates/template_code.py                 (the shape of those scripts)

process_run.launch/exchanges          -->  executions, exchanges
baseline_run.ensure                   -->  baselines (once per model+question,
                                           only for llm_judge_baseline)
evaluators.get(EVALUATOR).score      -->  exchanges.quality
calculate_fitness.assign              -->  individuals.fitness
                                           + fitness_history (per generation)
elitism.elect                         -->  individuals.is_best
selection.select                      -->  individuals (n copies + 1 drawn
                                           fresh, the n+1 weakest deleted)
mutation.apply                        -->  individuals.chromosome + has_changed
                                           (and fitness back to NULL)
weight_mutation.apply                 -->  the same, for weight symbols only

store --show / --export                    reads any of it back out
generate_html_db_stats                -->  <db>_run<N>_stats.html (the same,
                                           as a page, beside the database)
add_dataset --split testing           -->  datasets (a split added afterwards)
test_run_with_dataset                 -->  test_results (the best individuals,
                                           re-pointed at unseen questions,
                                           then graded by the same evaluators)

gep_lora.tools.test  -->  run/test_tree.txt + run/test_run.py  (one chromosome, same builders)

gep_lora/core/pipeline/continue_run.py  -->  that whole column again, once per generation
gep_lora/core/pipeline/main.py          -->  gep_lora/core/pipeline/start_run.py, gep_lora/core/pipeline/continue_run.py, then the testing pass
gep_lora/core/pipeline/main.py --db <a prepared database>
                 -->  the same, into the sweep that database already holds:
                      its settings and its stored dataset, nothing local, and
                      everything on disk in one folder beside the database
```

Running the whole thing:

```bash
python start_run.py
```

Or one stage at a time — the same steps, named:

```bash
python start_run.py population trees runs
```

```bash
python start_run.py process --limit 3
```

```bash
python start_run.py evaluate
```
