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

Whole pipeline: `population -> trees -> runs -> process -> evaluate -> fitness -> elitism`,
stopping at the first
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
owns the sqlite schema (`runs -> settings, individuals -> executions -> exchanges`), its
helpers, and `--list/--show/--export`. Nothing else imports sqlite3.

`generate_population.py` is the root module — it owns the alphabet (`BINARY_OPS`,
`UNARY_OPS`, `VARIABLES`, `ARITY`), the `Node` type, and `decode`/`encode`/`levels`.
Everything else imports from it; there is no second parser. Grammar invariants enforced
there: root is `CAT`; `CAT`/`SVD`/`LIN` take two *operators*; `L1`–`L5` take one *variable*.

`calculate_fitness.py` folds a judged transcript into one number: `assign(conn, run_id)`
averages `exchanges.quality` over each individual's most recent execution -- the
`individual_quality` view -- and writes it to `individuals.fitness`. An individual with
nothing to average (never run, `BAD`, crashed, still unjudged) gets `0.0`, not NULL, so a
selection step never has to decide what a missing score means.

`elitism.py` names the survivor: `elect(conn, run_id)` marks one individual with the
highest `fitness` as `is_best` and clears every other, in one statement, so a sweep never
carries two elites or last generation's. Ties break on the lowest individual number -- a
fixed rule, so re-running elects the same one. An all-zero population elects nobody and
writes nothing: `fitness` defaults to 0.0, so that means either the fitness step never ran
or nothing scored, and neither has an elite worth keeping. It reads the stored `fitness`
column and never the transcripts -- one definition of "best", living in
[calculate_fitness.py](calculate_fitness.py).

`generate_runs.py` turns a decoded tree into a script:

- `plan(root, ranks)` — post-order walk producing ordered `Step`s. Nodes are numbered in
  build order, so a step never references a name defined after it. Each *occurrence* of an
  `L*` gets its own adapter name (`n1_L2`, `n4_L1`), so one slot may appear several times at
  different weights.
- `resolve_from_template()` — `exec`s selected module-level assignments (`LORA_SLOTS`,
  `TRAINING_SET`) straight out of `template_code.py` rather than keeping a second copy that
  could drift. **If the generator needs another template constant, add it here — do not
  duplicate the value.**
- `render()`/`fill()` — substitutes `@@MARKER@@`s. An unfilled marker raises rather than
  reaching a generated file.

`template_code.py` is the generated script with the varying parts marked, deliberately kept
as **valid Python** so editors, linters and `python -m compileall` still work on it. Markers:
`@@NAME@@` inline; a line that is only `@@NAME@@` or `# @@NAME@@` becomes a block; a line
starting with `#~` is a template-only note that never reaches the output. Blocks: `TREE`,
`BUILD_ORDER`, `NOTE`, `ATTACH_LEAVES`, `COMBINE_NODES`, `WEIGHT_SEED`. Inline:
`SCRIPT_NAME`, `PROVENANCE`, `LABEL`, `EXPRESSION`, `LEAF_COUNT`, `FINAL_ADAPTER`,
`FINAL_RANK`.

`WEIGHT_SEED` is a one-line block rather than an inline marker because it stands in for the
assignment itself, so a generated script gets `WEIGHT_SEED = 12345` (or `= None`) as a plain
literal. It is why `WEIGHT_SEED` is the one name a linter calls undefined in both templates.

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
- `run/`, `run_db/` and `run_real/` are gitignored, as is everything under `Lora00*/` except
  each folder's `main.py` and `inference.py`. Adapter weights and the sweep database are not
  tracked. `run/` is only ever written by `test.py`.
- `combination.py` is the original hardcoded two-adapter script the generated code is
  modelled on. Reference, not part of the pipeline.
