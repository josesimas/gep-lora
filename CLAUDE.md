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

Whole pipeline: `population -> trees -> runs -> process -> evaluate`, stopping at the first
failure. `python main.py --list` shows the steps; naming steps runs a subset, always in
pipeline order regardless of typing order.

```bash
python main.py population trees runs
```

The fast half — everything before a base-model load. Use this while iterating on tree code
or `template_code.py`.

```bash
python process_run.py --limit 3
```

Smoke-test the expensive step. `process` costs one base-model load per individual, in a
separate process each (the scripts cannot share an interpreter).

```bash
python generate_runs.py --template template_code_mocked.py
```

Dry-run the pipeline. The mocked template produces the same scripts minus the model:
random answers, random qualities, no GPU, and no judge. `TEMPLATE` at the top of `main.py`
does the same for a whole `python main.py`. Use it whenever the thing under test is the
plumbing — but a mocked quality is noise, never a result.

```bash
python test.py CAT.SVD.LIN.L1.L2.L3.L1.w3.w3.w2.w1
```

The closest thing to a unit test: exercises one chromosome through the same builders the
batch tools use (`draw_trees.draw`, `generate_runs.plan/render`), prints tree + build order +
verdict, and writes `run/test_tree.txt` / `run/test_run.py`. Output is byte-identical to what
that chromosome would get as a population line. There is no pytest suite.

`evaluate_run.py` needs an OpenAI-compatible judge endpoint (defaults to LMStudio at
`http://172.22.208.1:1234/v1`). It is resumable — already-scored exchanges are skipped
unless `--force`.

Population size for a full run is `COUNT` at the top of `main.py` (currently 3, kept small
for iteration; the README's worked numbers assume 100).

## Architecture

`generate_population.py` is the root module — it owns the alphabet (`BINARY_OPS`,
`UNARY_OPS`, `VARIABLES`, `ARITY`), the `Node` type, and `decode`/`encode`/`levels`.
Everything else imports from it; there is no second parser. Grammar invariants enforced
there: root is `CAT`; `CAT`/`SVD`/`LIN` take two *operators*; `L1`–`L5` take one *variable*.

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
`BUILD_ORDER`, `NOTE`, `ATTACH_LEAVES`, `COMBINE_NODES`. Inline: `SCRIPT_NAME`,
`PROVENANCE`, `LABEL`, `EXPRESSION`, `LEAF_COUNT`, `FINAL_ADAPTER`, `FINAL_RANK`.

**Change what every generated script does by editing `template_code.py`, then re-running
`generate_runs.py`.** Only touch the generator when the new part varies per individual.

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
(such individuals are marked `BAD` in `run/index.txt` and carry a `NOTE`), and again at
runtime by the generated script, which refuses the bad `linear` step with the same message
instead of letting a bare `ValueError` out of PEFT. `process_run.py` skips `BAD` individuals
unless `--include-blocked`.

**Never assume a shared rank.** Every rank is read from that slot's own
`adapter_config.json` (`slot_ranks()` at generation time, `_rank()` at runtime), taking
`max(r, *rank_pattern.values())` the way PEFT does.

## Conventions that matter

- **Individual failures are results, not pipeline failures.** A chromosome that crashes is
  recorded in `run/results.txt` and the sweep carries on; only a sweep where *nothing* ran
  exits non-zero. Keep this when adding steps.
- **The pipeline adapts to which template it ran.** `process_run.py` drops its unsloth check
  when the generated scripts don't import it, and `evaluate_run.py` contacts the judge only
  when some answer still lacks a quality. Both keep a mocked sweep runnable on a plain
  Python 3 with nothing else up; don't reintroduce an unconditional check.
- **Transcripts are self-contained.** `run/output_result_NNN.json` carries the `chromosome`,
  the full `weights` draw (all five, not just the referenced ones) and the `exchanges`, so a
  scorer never cross-references `index.txt`. Exchanges are parsed from stdout only, so
  stderr progress bars cannot leak in; a missing reply keeps an empty `answer` rather than
  vanishing.
- **Weights are redrawn per execution** (`WEIGHT_SEED = None`), so the same chromosome scores
  differently each run. `main.py`'s `SEED` seeds the *chromosomes* only. Set `WEIGHT_SEED` to
  an int when a comparison must repeat.
- `evaluate_run.py`'s `SYSTEM_PROMPT` **is** the fitness criterion — the whole search
  optimises toward whatever it rewards. Two parsing traps already fixed there: the score is
  requested *before* the reason (a long reason must not truncate it away), and `MAX_TOKENS`
  is generous because a reasoning judge returns an empty message if it runs out mid-thought.
- Steps are added to `main.py`'s `STEPS` list as a `Step(name, callable, args, description)`;
  the callable is another script's `main(argv)`, so a step behaves exactly like running that
  script by hand.
- `run/` is gitignored, as is everything under `Lora00*/` except each folder's `main.py` and
  `inference.py`. Adapter weights are not tracked.
- `combination.py` is the original hardcoded two-adapter script the generated code is
  modelled on. Reference, not part of the pipeline.
