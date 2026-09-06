# Performance work

Where the sweep's time goes, what has been done about it, what was tried and
did not work, and what is worth doing next. Written to be picked up cold.

Status as of **2026-09-06**. Machine: RTX A6000 48 GB, Windows, torch
2.11.0+cu128, peft 0.20.0, transformers 5.5.0, unsloth 2026.8.15. Every number
below was measured on this machine, not estimated; where a number is a single
sample it says so.

---

## 1. How to get the numbers

A sweep records what it cost as it runs. Nothing to switch on.

```bash
python -m metrics.report                        # the latest sweep
python -m metrics.report --run 3 --step process # one step in full
python -m metrics.report --csv                  # the rows behind the tables
python -m reporting.generate_html_db_stats run_db/gep.sqlite3   # the same, drawn
```

Two tables in `run_db/gep.sqlite3` hold it: `step_timings` (one row per step per
pass — wall seconds, items, skipped) and `phase_timings` (one row per phase —
calls, total, worst). `metrics/record.py` is the write side, `metrics/report.py`
the read side, and the HTML report's **Cost** section draws the same rows. See
the schema comments in `storage/store.py` and the "What each step cost" section
of `CLAUDE.md`.

**Read these two rules before drawing conclusions from the tables:**

- **Script phases are shares of script time, not of wall time.**
  `PROCESS_RUN_BATCH_SIZE` scripts run at once, so their seconds overlap each
  other and the clock. The report says which denominator each table uses.
- **A pass's wall time is its slowest script.** Verified: `wall == longest` in
  all six passes of the baseline sweep. So the straggler is what to optimise,
  not the average.

---

## 2. Baseline: `run_db/gep.sqlite3`, run 1, 6 passes

Real sweep, `template_code.py`, `COUNT=4`, `TRAINING_COUNT=5`,
`PROCESS_RUN_BATCH_SIZE=4`, `EVALUATOR=llm_judge` over an endpoint. 19 scripts
ran, 5 individuals were skipped by the `has_changed` rule.

```
where the time went -- 8.3m over 9 step(s)
    process    7.6m  92.0%   19 individuals (+5 skipped)   24.1s each
    evaluate  39.6s   8.0%   95 answers                   417ms each
    runs        69ms   0.0%   24 scripts
    ...everything else under 70ms combined
```

Inside the 19 scripts — **18.6m of script time, 58.8s each**:

| phase | calls | total | each | share |
|---|---|---|---|---|
| `model_load` | 19 | 5m 53s | 18.6s | 31.6% |
| `generate` | 95 | 4m 41s | 2.96s | 25.2% |
| `import` | 19 | 4m 12s | 13.3s | 22.6% |
| `combine.svd` | 2 | 2m 44s | **82.3s** | 14.7% |
| `attach` | 40 | 22.0s | 549ms | 2.0% |
| `combine.cat` | 19 | 10.8s | 568ms | 1.0% |
| `inference_setup` | 19 | 805ms | 42ms | 0.1% |
| *(before it runs)* | 19 | 31.4s | 1.7s | 2.8% |
| `compact` | 2 | 109ms | 55ms | *(nested in svd)* |

`evaluate`: `score` 95 calls, 39.2s, 412ms each; `prepare` 6 calls, 49ms.

The whole GEP tail — `fitness` + `elitism` + `selection` + `mutation` — came to
**103ms across six generations**. Do not spend time there.

**The two expensive generations are the two containing an `SVD` node**: passes
1 and 5 cost 116.6s and 127.8s wall against ~45–53s for the rest, and in both
the slowest script *is* the svd script. Two nodes inflated the sweep by roughly
150s of its 497s.

---

## 3. Landed

### Batched generation (2026-09-06)

`templates/template_code.py` and `template_code_mocked.py` now answer
`ANSWER_BATCH` (8) prompts in one `generate()` call instead of one call per
prompt. Left padding with an eos fallback for the pad token; each batch's
exchanges are printed as it completes, so a timed-out script still leaves the
batches it finished.

Measured, same blend, same process, best of three: **7.35s → 2.06s** for five
prompts (**3.56×**), replies 5/5 identical, peak VRAM 1.75 GB.

Confirmed on a real generation afterwards:

| | before | after |
|---|---|---|
| `generate` calls | 95 (one per answer) | 3 (one per script) |
| per answer | 2.96s | **0.62s** |
| share of script time | 25.2% | **9.8%** |

Two consequences to keep in mind:

- **`generate`'s `calls` counts calls, not answers.** The phase label carries
  the answer count (`"5 answer(s)"`). `TRAINING_COUNT` now moves tokens per
  call rather than the number of calls, until it exceeds `ANSWER_BATCH`.
- **Batched and sequential decoding can word an answer differently** — the
  reduction order in the matmuls differs. Each sweep stays internally
  consistent; a sweep straddling the change is not strictly comparable with
  itself.

**After this change the composition flipped**: on the confirmation run,
`import` 44.4% + `model_load` 32.8% = **77% of script time**, `generate` 9.8%.
Everything below is ranked against that new picture.

---

## 4. Tried and rejected — do not repeat

### `svd_driver="gesvdj"` on `add_weighted_adapter`

No effect. Measured on the real adapters, base model loaded once:

```
svd driver=None      70.42s
svd driver=gesvdj    70.43s
svd driver=gesvda    torch._C._LinAlgError (error code 978, ill-conditioned)
```

`gesvdj` is what the default already picks for these shapes, and `gesvda`
cannot be used at all. The 70s is not a driver choice — see §5.2 for what it
actually is.

---

## 5. Open work, ranked

### 5.1 The per-script fixed cost — `import` + `model_load`, now ~77% of a script

31.9s per script in the baseline (13.3s import + 18.6s load), paid per
individual per generation, producing nothing. It partly hides today because
four scripts pay it concurrently, so one pass absorbs it once — but that stops
being true the moment the population exceeds `PROCESS_RUN_BATCH_SIZE`. At
`COUNT=100` with batch 4 it is 25 sequential batches and the majority of the
sweep.

**Do the cheap experiment first.** `model_load` swung 9.6s → 34.6s across
passes of one sweep, which is unexplained, and 13.3s for unsloth's import may
be partly hub round-trips. Measure `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` against a cold and a warm run before building
anything. If 32s becomes 12s for free, the case for the rest weakens a lot.

**Then the model server.** The design discussed (and the right one for this
codebase): a long-lived process that loads the base model once and exposes an
API — build this blend, answer these prompts — with a new
`templates/template_remote_code.py` whose generated scripts are thin clients,
and a settings flag for how many servers to start.

Why it fits: the pipeline's contract is *"a generated script prints a
transcript on stdout and its exit code is a result"*. Everything hangs off
that — `executions`, `exchanges`, the `TIMING:` lines, `verdict_of()`, partial
transcripts on timeout, the mocked path. A remote template keeps the contract
completely and moves the invasiveness into a new operational component, which
is a better place for it than the pipeline. In-process evaluation of several
individuals per interpreter — the other option — breaks one-script-per-
individual, which is what makes an execution self-contained.

Risks, in the order they will bite:

1. **State leakage across builds.** Adapters accumulate on a long-lived model;
   `delete_adapter` must actually return the VRAM (PEFT pins buffers behind svd
   nodes — that is what `_compact()` exists for), and `for_inference()` mutates
   the model, so calling it after every rebuild is unexplored.
   **Acceptance test:** the same chromosome built as the 1st and as the 300th
   blend on one server must produce a *byte-identical* transcript to a
   cold-process run, with resting VRAM flat across the sequence. If that fails,
   stop — no protocol design fixes it.
2. **Reproducibility.** Today every result comes from a fresh interpreter.
   After this, a result can depend on what the server built before it. Record
   the server's identity and how many blends it had built on the execution row
   so drift is detectable, and recycle a worker every N individuals to bound it.
3. **A pool may not buy what a pool suggests.** The current 4-way concurrency
   yields only 2.4–3.0× because what overlaps well is the CPU-bound import and
   load. Remove those and the rest is GPU-bound on one card, so N servers will
   time-share rather than multiply. **Expect one warm server to be close to a
   pool of four** — measure that before building port allocation and pool
   lifecycle.
4. **Blend-building lands in a third place.** The server needs
   `attach`/`combine`/`_compact`/the rank rule, and `template_code.py` keeps
   its inline copy because generated scripts must stand alone. `CLAUDE.md`'s
   "a change to one template belongs in both" becomes "belongs in three". Best
   mitigation: a shared module the server imports, with a note in both
   templates saying they must agree.

### 5.2 The `svd` fold — 70–83s a node, and it sets a generation's wall clock

**What it is.** PEFT runs one exact `torch.linalg.svd` per LoRA-carrying module
over that module's whole delta weight. This base model has **196** such
modules, shaped `(1536,1536)`×56, `(256,1536)`×56, `(8960,1536)`×56,
`(1536,8960)`×28. At 0.20–0.58s each that is the 70s, and it is inherent to
doing an exact SVD — not a driver choice.

**Why it is waste.** A LoRA delta is `B @ A`: its rank is the adapter's rank.
The summed delta of two adapters has rank ≤ r1 + r2. Measured on the real
adapters, the worst-case module's summed delta had **true rank 36 inside an
8960×1536 matrix**, and PEFT was finding 1536 singular values of it.

**The headroom, measured on that real delta:**

```
exact torch.linalg.svd      0.577s
torch.svd_lowrank(q=48,niter=2)  0.011s     52x
singular values, max relative difference   7.67e-04
rank-32 reconstruction, relative Frobenius 9.76e-04
(the rank-32 part holds 99.89% of the delta)
```

So ~70s → ~2s on the phase that decides the wall clock of any generation
containing an `SVD` node.

**Two ways to take it**, in increasing order of "no argument to have":

- `torch.svd_lowrank` with `q = (sum of input ranks) + 16`. Simple. Since q
  exceeds the true rank, it is not an approximation of the subspace, but it is
  not bit-exact either (~1e-3 above).
- **Exact and fast:** the delta is a product of thin factors, so its SVD can be
  had via QR of the factors in O((m+n)r²) instead of O(mn·min(m,n)). No
  approximation argument at all. More code.

Either way it means not calling PEFT's `add_weighted_adapter` for the svd case
and doing the fold in our own code (`_svd_generalized_task_arithmetic_weighted_adapter`
in `peft/tuners/lora/model.py` is what to mirror — mind `scaling`,
`fan_in_fan_out`, embeddings and `rank_pattern`). The mocked template needs no
change: it builds no adapters, only the rank arithmetic.

Sample size is thin — **2 svd nodes in the baseline sweep**. Re-measure on a
sweep with more of them before and after.

### 5.3 `evaluate` — 8% now, and it does not shrink

95 sequential judge calls at 412ms. It is the one cost that stays flat while
everything else falls, so its share grows with every win above. A few
concurrent requests against the OpenAI-compatible endpoint is easy; mind that
`JUDGE_ABANDON_FRACTION` reads answers in order, so concurrency must not break
the give-up rule.

### 5.4 `PROCESS_RUN_BATCH_SIZE`

Currently 4 with `COUNT=4`, so a generation is one batch and everything already
overlaps. Resting cost is ~3.6 GB per script (~4.7 GB peak during an svd node)
against 48 GB, so there is room — but raising it mostly buys more overlap of
import and load, which is what §5.1 would delete outright. Revisit after that.

### Not worth touching

`attach` (549ms), `combine.cat` (568ms), `inference_setup` (42ms),
`materialise`/`clear scripts` (ms), and the entire GEP search tail (103ms over
six generations).

---

## 6. Gotchas that cost real time

- **`tools.test` emits `WEIGHT_SEED = None`.** Two runs of one chromosome are
  therefore *different blends*, not one blend measured twice. This invalidated
  an A/B and produced a garbage transcript that looked like a batching bug for
  an hour. Worth adding a `--seed` flag. Until then, pin the seed by hand or
  compare inside one process.
- **A bad blend produces fluent-looking gibberish.** That is a result, not a
  bug — it is what the search exists to score 0. Do not diagnose it as a code
  fault without pinning the weights first.
- **sqlite reuses `executions` rowids after a cull** (`MAX(rowid)+1`), so two
  generations' phase rows can carry the same `execution_id`. Per-individual
  reads key on **(step_timings row, execution id)**, and the `script` phase
  carries the individual's number and chromosome in its `detail` for that
  reason. Sweeps recorded before that fix fall back to a join and show culled
  individuals as `culled` with no chromosome.
- **Two clocks measure each script**: the step's, from launch, and the script's
  own `total`. The gap (~1.6–1.7s) is interpreter startup, and the report shows
  it rather than hiding it.
- **`_compact()` is correctness-neutral.** Verified: 196 weights cloned,
  answers unchanged. It is purely the VRAM measure it is documented as.
- **Judge endpoint.** `evaluate` reads the sweep's *stored* `EVALUATOR`, not
  `settings.py`'s — re-scoring an old sweep will reach for whatever judge that
  sweep was created with.

---

## 7. How to verify a change

1. **Mocked sweep** — plumbing, no GPU: run a sweep with
   `TEMPLATE = "template_code_mocked.py"` and check transcripts still parse
   (answers intact, no `TIMING:` text inside one) and the phase rows appear.
2. **Real generation** — `python start_run.py --db run_db/<scratch>.sqlite3
   population trees runs process` with the venv's python
   (`D:\sage-is\loras\.venv\Scripts\python.exe`). `process` alone needs no
   judge.
3. **Compare** — `python -m metrics.report --db run_db/<scratch>.sqlite3
   --step process` against §2 above, phase by phase.
4. Delete the scratch database and any `run_db/run_0*.py` left by a `BAD`
   individual.

Benchmarks quoted here were one-off scripts in a session scratch directory and
are gone; the recipes are in §4 and §5.2 and are a few minutes to rewrite. If
they are wanted permanently, `tools/` is where dev aids live.
