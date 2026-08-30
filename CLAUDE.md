# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Gene Expression Programming search over **LoRA adapter blends**. A chromosome is a
K-expression (Karva notation, level-order, dot-separated, always rooted at `CAT`) that
describes how to fold five LoRA adapters into one model. Each chromosome is compiled into a
standalone Python script that builds that blend with PEFT and answers the eval prompts;
a judge model then scores the answers.

`plan.txt` is the original spec. `README.md` is long and current — read it before changing
pipeline behaviour, and update it when behaviour changes.

## Interpreter — read this first

The generated scripts need the venv **one level up**, outside this repo:

```bash
D:\sage-is\loras\.venv\Scripts\python.exe main.py
```

Python 3.13 is first on PATH on this machine and has no torch/unsloth. `process_run.py`
launches each generated script with `sys.executable`, so running the pipeline under the
wrong interpreter fails every individual; it guards this with `check_interpreter()`.
Only `process`/`evaluate` and the generated scripts need the venv — the tree-manipulation
steps (`population`, `trees`, `runs`) run under any Python 3.

## Commands

All commands run from the repo root.

```bash
python main.py
```

Whole pipeline:
`population -> trees -> runs -> process -> evaluate -> fitness -> elitism -> selection
-> mutation`, stopping at the first
failure. `python main.py --list` shows the steps; naming steps runs a subset, always in
pipeline order regardless of typing order.

Everything a sweep produces goes into one database, `run_db/gep.sqlite3` — the population,
the settings, the seeds, the transcripts and the scores. See `store.py`. A sweep is a row,
so sweeps accumulate rather than replacing each other, and `--run` resumes one.

The generated scripts are a cache with a life cycle: `store.materialise()` writes any that
are missing or stale before `process` runs, and `store.remove_scripts()` deletes each one it
processed afterwards, so a finished sweep leaves only the database. `--keep-scripts` opts
out; `python main.py runs` brings them back. Only scripts that actually ran are removed —
ones skipped as `BAD` or held back by `--limit` stay.

```bash
python continue_run.py --generations 3
```

`main.py` runs a sweep through **one** generation; `continue_run.py` carries an existing one
on, running `trees -> runs -> process -> evaluate -> fitness -> elitism -> selection ->
mutation` per generation (`GENERATIONS` in `settings.py`, default 10). It never draws a
population and never creates a sweep -- it resumes one from the database (`--db`, `--run`)
under the settings that sweep was created with, reusing `main.STEPS` and `main.run()` rather
than a second copy of the driver. The `_run` suffix is forced: `continue` is a keyword, so a
`continue.py` could never be imported. **Mind the growth** -- selection appends, so with
`SELECTION_COUNT = None` the population doubles every generation; the driver prints the
projection before starting, and carries straight on -- it never prompts. Because a sweep reads its own stored settings, editing
`settings.py` does nothing to one already under way -- `--set NAME=VALUE` (JSON values,
repeatable, name must already exist) writes the change into the sweep's settings table
instead, which is how SELECTION_COUNT gets fixed mid-run without the sweep losing track of
what it ran under.

```bash
python full_run.py
```

`main.py` then `continue_run.py` against the same sweep -- a whole search, `1 + GENERATIONS`
generations, in one command. It calls both drivers as **libraries in this interpreter** (a
subprocess would be another chance to run under the wrong Python, since `process` uses
`sys.executable`), and hands the sweep on **by id** rather than by "the latest", so a
database that gains a sweep in between cannot be picked up by mistake. `--label` goes to
`main.py`, `--generations`/`--set` to `continue_run.py`, the rest to both. A failing first
half stops the run.

```bash
python store.py --show 0
```

Reads a stored sweep back: `--list` the sweeps, `--show` one (`0` = latest), `--export`
one into a folder of text files (population, trees, index, scripts, outputs, transcripts,
results) — a view of the sweep, derived from the database, never the sweep itself.

```bash
python main.py population trees runs
```

The fast half — everything before a base-model load. Use this while iterating on tree code
or `template_code.py`.

```bash
python main.py process --limit 3
```

Smoke-test the expensive step against the latest sweep. `process` costs one base-model load
per individual, in a separate process each (the scripts cannot share an interpreter).
`PROCESS_RUN_BATCH_SIZE` in `settings.py` is how many of those processes run **at once** --
`process_run.batches()` cuts the selected individuals into consecutive groups of that size
and `launch_batch()` runs a group together, waiting it out before starting the next. A fixed
ceiling, not a refilling queue, and one paid in memory: a batch of N holds N copies of the
base model at the same time, and an individual that runs out of memory is recorded as a
failed execution like any other rather than stopping the sweep. `1` restores the old
one-at-a-time behaviour, output included. Results are stored in the order the batch was
asked for and only from the driver thread, so the database is written exactly as it was
sequentially; the per-individual commit means an interrupted batch keeps what came back.
An execution's `seconds` is that script's own wall clock, so within a batch they overlap.

`process` **streams** its children rather than collecting them at the end: `launch()` runs
the script under `Popen` with `-u`, drains stdout and stderr on a thread each, and calls
`on_line` per stdout line plus `on_tick` every `TICK` seconds so a silence can be reported
too. `process_run.Progress` turns that into the occasional line -- the model coming ready
(always, with the blend's rank), the last prompt starting (always), and the running
`prompt k/N` throttled to one per `PROCESS_RUN_PROGRESS_SECONDS` per script (`0` = milestones
only). It formats; `main.py` prints, under a lock, because the callbacks arrive on the
children's drain threads. **The transcript is never echoed** -- it belongs in the database.
A killed script now keeps the output it had already printed, so a timeout stores a partial
transcript rather than an empty one.

`context.generation` ("2/5", or None) is set by `continue_run.py` around each generation and
read only by `main.run()`'s step banner and the batch line. Display only: no step may behave
differently in one generation than another, and nothing stores it.

Dry-run the whole pipeline by setting `TEMPLATE = "template_code_mocked.py"` in
`settings.py`. The mocked template produces the same scripts minus the model: random
answers, random qualities, no GPU, and no judge. Use it whenever the thing under test is
the plumbing — but a mocked quality is noise, never a result.

```bash
python test.py CAT.SVD.LIN.L1.L2.L3.L1.w3.w3.w2.w1
```

The closest thing to a unit test: exercises one chromosome through the same builders the
pipeline uses (`draw_trees.draw`, `generate_runs.plan/render`), prints tree + build order +
verdict, and writes `run/test_tree.txt` / `run/test_run.py` — its own folder, beside
`run_db/`, so it never collides with a sweep. Output is byte-identical to what that
chromosome would get as an individual. There is no pytest suite.

The `evaluate` step needs an OpenAI-compatible judge endpoint (`evaluate_run.py` defaults to
LMStudio at `http://172.22.208.1:1234/v1`). It is resumable — already-scored exchanges are
skipped unless `--force`.

Population size for a full run is `COUNT` in `settings.py` (currently 10, kept small for
iteration; the README's worked numbers assume 100). `settings.py` holds every knob the
pipeline reads — add one there rather than at the top of `main.py`, or the sweep records a
value it did not use.

## Architecture

`main.py` is the entry point and the driver: it owns the `STEPS` list, the `Context` each
step gets, and the argument parser. `settings.py` is the one copy of the knobs; `store.py`
owns the sqlite schema (`runs -> settings, individuals -> executions -> exchanges`,
plus `fitness_history` hanging off `runs`), its
helpers, and `--list/--show/--export`. Nothing else imports sqlite3.

`generate_population.py` is the root module — it owns the alphabet (`BINARY_OPS`,
`UNARY_OPS`, `VARIABLES`, `ARITY`), the `Node` type, and `decode`/`encode`/`levels`.
Everything else imports from it; there is no second parser. Grammar invariants enforced
there: root is `CAT`; `CAT`/`SVD`/`LIN` take two *operators*; `L1`–`L5` take one *variable*.

`calculate_fitness.py` folds a judged transcript into one number: `assign(conn, run_id)`
averages `exchanges.quality` over each individual's most recent execution -- the
`individual_quality` view -- and writes it to `individuals.fitness`. An individual with
nothing to average (never run, `BAD`, crashed, still unjudged) gets `0.0`, not NULL, so a
selection step never has to decide what a missing score means. It **also** writes the whole
population's scores to `fitness_history` -- one row per individual per generation, stamped
with `recorded_at` -- because `individuals.fitness` only ever holds *now*: the next
generation overwrites it and mutation clears it outright, so without the history a sweep
cannot say whether the search went anywhere. Each history row keeps the chromosome and
state **as they were then**, so reading the past never goes through the present population.
Nothing in the schema counts generations (neither driver stores one), so
`store.fitness_generation()` derives it from population size -- which selection grows by
appending and never shrinks, the same thing `selection` and `mutation` seed their
generators from: grown since the last snapshot means a new generation, unchanged means the
same one restated, which is what keeps a re-run of the cheap `fitness` step from inventing
a generation. `assign()` returns a `Snapshot(generation, recorded_at, rows)`.
`store.fitness_by_generation()` / `best_of_generation()` are the read side, printed by the
step, by `store.py --show`, and exported as `fitness_history.txt`.

`elitism.py` names the survivor: `elect(conn, run_id)` marks one individual with the
highest `fitness` as `is_best` and clears every other, in one statement, so a sweep never
carries two elites or last generation's. Ties break on the lowest individual number -- a
fixed rule, so re-running elects the same one. An all-zero population elects nobody and
writes nothing: `fitness` defaults to 0.0, so that means either the fitness step never ran
or nothing scored, and neither has an elite worth keeping. It reads the stored `fitness`
column and never the transcripts -- one definition of "best", living in
[calculate_fitness.py](calculate_fitness.py).

`selection.py` is roulette wheel sampling: `select(conn, run_id, count, rng)` gives each
individual a slice of the wheel as wide as its `fitness`, spins it `count` times **with
replacement**, and **appends** the picks -- it deletes nothing and overwrites nothing, so an
existing individual keeps its number, script, executions and transcripts either way. A pick
is a new row that copies its parent **field for field** (`store.append_copies`, whose column
list comes from the table so a new field is copied too): only `id`, `number` and `is_best`
are its own -- `is_best` picks one individual out of the population rather than describing
one, so no copy inherits it and every copy starts at 0. It does inherit the parent's
`script_name`, `weight_seed` and `fitness` -- transient, and cleared up by the next run of
`runs` (names and seeds, re-derived from the number) and `fitness`. That inheritance is meant
to be spent by whatever varies the copy, not kept. The step is therefore **not idempotent**: running it twice is two
generations, not one done twice. Fitness 0.0 is a zero-width slice and is never picked; an
all-zero population has no wheel, selects nobody and writes nothing. Each round derives its
generator from `SELECTION_MASTER_SEED` and the size of the population it spins over, the way
weight seeds derive from `WEIGHT_MASTER_SEED` and an individual's number.

`mutation.py` is point mutation with the grammar built in: `mutate(chromosome, rate, rng)`
gives each symbol probability `rate` of becoming a *different symbol of its own class*
(`CAT`/`SVD`/`LIN`, `L1`-`L5`, `w1`-`w5`) and never touches the root. Class-local swaps are
the only ones that preserve both arity and the child alphabet -- `children_alphabet()`
depends on the class, not the symbol -- so the tree keeps its shape and every result still
decodes; `generate_population.check()` is run on each one before it is stored, so a broken
mutant is a bug rather than a result. `apply()` skips the `is_best` row entirely (mutating
the elite would discard what elitism protects) and writes `has_changed` for every
individual, 1 where the chromosome moved and 0 everywhere else -- this round's answer, not a
running total. **has_changed = 1 clears that row's `fitness` to NULL** (`store.set_chromosome`):
the score belonged to the chromosome just replaced, and keeping it would let a mutant be
elected or win a slice of the wheel on a blend it no longer describes. Every reader of the
column already coalesces NULL to 0.0, so a mutant is simply passed over until it has been
judged again. Its `tree`, `script_source` and `rank` are stale too; `trees`/`runs` re-derive
those. Re-run `process` before `fitness` -- `fitness` reads the latest *execution*, so
running it first would recompute the old chromosome's score and put it back. A `CAT`->`LIN` swap over mismatched
ranks is expected and is culled by the usual `BAD` path.

`generate_runs.py` turns a decoded tree into a script:

- `plan(root, ranks)` — post-order walk producing ordered `Step`s. Nodes are numbered in
  build order, so a step never references a name defined after it. Each *occurrence* of an
  `L*` gets its own adapter name (`n1_L2`, `n4_L1`), so one slot may appear several times at
  different weights.
- `training_set_path()` / `training_count()` / `lora_slots()` — the eval prompts file, the
  cap on how many of its records are used, and the five adapter
  paths, from `TRAINING_SET`, `TRAINING_COUNT` and `LORA_SLOTS` in `settings.py` rather than
  from either template. A relative value is resolved against the repo folder — for a slot, only when it
  really names a folder there, so an absolute path or a Hub repo id passes through as
  written. Both resolved values are stamped into each script as literals, so a script
  carries real paths instead of walking up from wherever it lands. `main.py` passes the
  *sweep's stored* values, so editing `settings.py` cannot move the eval set, change how
  much of it counts, or swap the adapters under a sweep already running. **There is no `resolve_from_template()` any
  more** — nothing reads constants back out of the templates, because nothing that varies
  lives there. A new knob goes in `settings.py` and reaches the script through a marker.
- `render()`/`fill()` — substitutes `@@MARKER@@`s. An unfilled marker raises rather than
  reaching a generated file.

`template_code.py` is the generated script with the varying parts marked, deliberately kept
as **valid Python** so editors, linters and `python -m compileall` still work on it. Markers:
`@@NAME@@` inline; a line that is only `@@NAME@@` or `# @@NAME@@` becomes a block; a line
starting with `#~` is a template-only note that never reaches the output. Blocks: `TREE`,
`BUILD_ORDER`, `NOTE`, `ATTACH_LEAVES`, `COMBINE_NODES`, `WEIGHT_SEED`, `TRAINING_SET`,
`TRAINING_COUNT`, `LORA_SLOTS`. Inline: `SCRIPT_NAME`, `PROVENANCE`, `LABEL`, `EXPRESSION`, `LEAF_COUNT`,
`FINAL_ADAPTER`, `FINAL_RANK`.

`WEIGHT_SEED`, `TRAINING_SET`, `TRAINING_COUNT` and `LORA_SLOTS` are blocks rather than
inline markers because each stands in for the assignment itself, so a generated script gets
`WEIGHT_SEED = 12345` (or `= None`), `TRAINING_SET = '...'`, `TRAINING_COUNT = 10` (or
`= None`) and the whole `LORA_SLOTS = {...}` dict as plain literals. They are why those four
are the names a linter calls undefined in both templates.

The eval file is read **a line at a time**, in either of two shapes, told apart by the line
itself rather than by its extension: a **JSON record per line** carrying a `messages` list
(what `datasets/*.json` hold, and what `create_lora.py` trains on) — the prompt is its first
`user` turn — or **plain one-prompt-per-line text**, quotes optional. For a JSON record only
the user turn is asked; the assistant turn beside it is somebody else's answer to the same
question, and sending it would be showing the model the answer and then scoring the reply.
A whole-file JSON array is rejected with a message saying so, by `eval_prompt_count()` at
generation time and by the templates at startup, because a line count cannot describe one.

`TRAINING_COUNT` caps the eval set at its **first N records** — the top of the file, not a
sample of it, and the same N for every individual, because fitness numbers are only
comparable when they were earned answering the same questions. A cap larger than the file
is not an error; it just means all of them, which is also what `None` means. It is the
cheapest knob in `settings.py`: one prompt is one `generate()` call per individual per
generation, so turning it down while iterating cuts the eval half of a sweep
proportionally — at the cost of a noisier fitness signal.

**Change what every generated script does by editing `template_code.py`, then re-running
`python main.py runs`.** Only touch the generator when the new part varies per individual.

`template_code_mocked.py` is the second template: same markers, same generated shape, same
rank arithmetic and same `linear` guard — so the `ok`/`BAD` split is identical — but nothing
is loaded and `ask()`/`grade()` invent the answer and its score. It prints `QUALITY:` and
`REASON:` lines that `process_run.py` folds into the transcript, which is why a mocked sweep
comes out pre-scored. **A change to one template usually belongs in both**; anything that
differs between them is by definition part of the mock.

## The rank rule

The operators map onto PEFT `add_weighted_adapter` combination types — `CAT`→`cat`,
`SVD`→`svd`, `LIN`→`linear` — and PEFT constrains the resulting rank: `cat` **sums** input
ranks, `svd` takes the **max**, `linear` **requires equal** input ranks or raises.

The five slots were trained at different ranks (`L1`=16, `L2`=16, `L3`=8, `L4`=4, `L5`=32),
so `LIN` above mismatched inputs — especially above a `CAT` — often cannot run. This is a
property of the search space, not a bug. Ranks are computed statically at generation time
(such individuals get `state = 'BAD'` and carry a `NOTE` in their script), and again at
runtime by the generated script, which refuses the bad `linear` step with the same message
instead of letting a bare `ValueError` out of PEFT. The `process` step skips `BAD`
individuals unless `--include-blocked`.

`process` also skips individuals with `has_changed = 0` **that already have an execution**
-- their stored result is of the chromosome they still hold, so re-running one costs a
base-model load to learn nothing. Never having run is not the same as being unchanged, so a
fresh population and every copy `selection` appends still run in full. `--include-unchanged`
overrides it; a step where nothing needs running is reported, not a failure.

**Never assume a shared rank.** Every rank is read from that slot's own
`adapter_config.json` (`slot_ranks()` at generation time, `_rank()` at runtime), taking
`max(r, *rank_pattern.values())` the way PEFT does.

## Conventions that matter

- **Individual failures are results, not pipeline failures.** A chromosome that crashes is
  recorded as an execution row with its exit code and the sweep carries on; only a sweep
  where *nothing* ran exits non-zero. Keep this when adding steps.
- **The pipeline adapts to which template it ran.** `process` drops its unsloth check when
  the generated scripts don't import it (`process_run.imports_unsloth`), and `evaluate`
  contacts the judge only when some answer still lacks a quality. Both keep a mocked sweep
  runnable on a plain Python 3 with nothing else up; don't reintroduce an unconditional
  check.
- **An execution is self-contained.** Its row carries the weight seed, the full `weights`
  draw (all five, not just the referenced ones), stdout, stderr and the `exchanges`, so a
  scorer never has to look anywhere else — `individual_quality` is the view that does the
  join. Exchanges are parsed from stdout only, so stderr progress bars cannot leak in; a
  missing reply keeps an empty `answer` rather than vanishing.
- **Every individual carries its own weight seed.** It is derived from
  `WEIGHT_MASTER_SEED` and the individual's number, stamped into that individual's script
  and stored, so a whole sweep repeats without every individual sharing one draw.
  `settings.py`'s `SEED` seeds the *chromosomes* only. Keep the two separate.
- **A step reads the settings its sweep was created with**, not `settings.py` as it stands
  now; that is what makes resuming a sweep still be the same sweep. A seed left `None` is
  drawn once at sweep creation and stored as the number drawn.
- `evaluate_run.py`'s `SYSTEM_PROMPT` **is** the fitness criterion — the whole search
  optimises toward whatever it rewards. Two parsing traps already fixed there: the score is
  requested *before* the reason (a long reason must not truncate it away), and `MAX_TOKENS`
  is generous because a reasoning judge returns an empty message if it runs out mid-thought.
- Steps are added to `main.py`'s `STEPS` list as a `Step(name, callable, description)`,
  where the callable takes the `Context` — the connection, the run id, that sweep's settings,
  the run dir and the parsed options.
- **`main.py` calls the other modules as libraries; none of them has a `main()`.** That is
  what keeps the pipeline from writing text files: `build_population`, `draw`,
  `plan`/`render`, `launch`, `exchanges`, `judge` are all pure enough to use directly. If a
  new step needs something out of one of them, extract a function there rather than teaching
  that module about the database.
- The five adapters live under `loras/Lora001`..`loras/Lora005`, each holding its own
  `my_planning_coach-lora_adapter/`. `LORA_SLOTS` in `settings.py` spells those paths out
  relative to the repo folder — one place, so repointing a slot needs no template edit and
  the sweep records which five adapters it was scored on. `create_all_loras.py` writes
  there via its `LORA_DIR` and prints a paste-ready block (`create_lora.slot_line()` emits
  the settings form, relative with forward slashes); `test_lora.py` accepts
  `--lora Lora003` with or without the `loras/` prefix.
- `run/`, `run_db/` and `run_real/` are gitignored, as is everything under
  `loras/Lora00*/` except each folder's `main.py` and `inference.py` -- the ignore matches
  the folders' *contents* (`loras/Lora00*/*`), because git cannot re-include a file whose
  parent directory is excluded. Adapter weights and the sweep database are not tracked.
  `run/` is only ever written by `test.py`.
- `combination.py` is the original hardcoded two-adapter script the generated code is
  modelled on. Reference, not part of the pipeline.
