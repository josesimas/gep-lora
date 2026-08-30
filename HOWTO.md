# HOWTO

Train the adapters, test them, run the search. See `README.md` for how any of it
works.

## The venv

Anything that loads a model needs a Python environment with the training and
inference stack installed. The repo does not ship one and does not care where it
lives — on the reference machine it sits one level above the repo, but any
location works.

What it must have:

| | |
| --- | --- |
| Python | 3.11 (3.13 is not supported by unsloth) |
| `torch` | with CUDA matching your GPU driver |
| `unsloth` | loads the base model |
| `peft` | 0.20 or later; provides `add_weighted_adapter` |
| `transformers`, `trl`, `datasets` | training and chat templating |

An NVIDIA GPU is assumed. The base model is `unsloth/Qwen2.5-1.5B-Instruct`,
downloaded on first use.

Create it once, from wherever you want it:

```bash
python -m venv .venv
```

Then install into it — follow unsloth's own instructions for the torch build
matching your CUDA version, since that is the part that varies by machine.

**Which commands need it.** Everything under [1. Train](#1-train-the-adapters),
`test_lora.py`, and the `process` and `evaluate` steps — so in practice all of
`full_run.py` and `continue_run.py`. The rest (`test.py`, `store.py`,
`main.py population trees runs`, anything with the mocked template) runs on any
Python 3.

Activate the venv before those commands. On Windows:

```bash
.venv\Scripts\activate.bat
```

Check it took:

```bash
python -c "import torch, unsloth, peft; print(torch.cuda.is_available())"
```

`True` means you are ready. `ModuleNotFoundError` means the activation did not
stick — a `(.venv)` prompt does not prove it. `where python` shows what is
actually resolving; the venv's full path to `python.exe` always works regardless
of PATH.

This matters more than it looks: `main.py process` launches every generated
script with `sys.executable`, so the wrong interpreter fails the whole population
at once rather than one script.

---

## 1. Train the adapters

Five adapters go in `loras/Lora001` .. `loras/Lora005`, each in a subfolder named
`my_planning_coach-lora_adapter`. That is what `LORA_SLOTS` in `settings.py`
already points at.

Datasets are in `datasets/`, one JSON object per line, each with a `messages`
list. Not a JSON array.

Print the plan first:

```bash
python create_all_loras.py --dataset poem --values 16 16 8 4 32 --dry-run
```

Then train (hours):

```bash
python create_all_loras.py --dataset poem --values 16 16 8 4 32
```

`--values` sets the ranks explicitly and sets the count from its length. Without
it, ranks double from `--rank-min` (4). Ranks matter: `CAT` sums them, `SVD` takes
the max, `LIN` requires them equal or fails.

The script refuses to overwrite existing adapters — use `--force` or `--start N`.
When it finishes it prints a `LORA_SLOTS` block; paste it into `settings.py` if
you did not use the default folders.

One extra adapter:

```bash
python create_lora.py loras/Lora006/shout_adapter --dataset uppercase --rank 8
```

---

## 2. Test the adapters

Answer one question with and without the adapter (needs the venv):

```bash
python test_lora.py --lora Lora003 "Describe autumn."
```

With no question it keeps asking. `--demo` runs three built-in prompts and exits.
Do this for all five; if an answer looks like the base model, that adapter did not
train.

Check a chromosome without a GPU:

```bash
python test.py CAT.SVD.LIN.L1.L2.L3.L1.w3.w3.w2.w1
```

Prints the tree, the build order, the final rank and `ok` or `BAD`. Writes
`run/test_tree.txt` and `run/test_run.py`.

---

## 3. Run the search

One generation is:

```
population -> trees -> runs -> process -> evaluate -> fitness -> elitism -> selection -> mutation
```

### 3a. Dry run

Set in `settings.py`:

```python
TEMPLATE = "template_code_mocked.py"
```

```bash
python full_run.py --label "mock"
```

No model, no judge, no GPU, finishes in seconds. Scores are random — the point is
that the pipeline runs. Set `TEMPLATE` back to `"template_code.py"` afterwards.

### 3b. Start the judge

`evaluate` needs an OpenAI-compatible endpoint. Default in `evaluate_run.py`:

```python
BASE_URL = "http://172.22.208.1:1234/v1"
MODEL = None
```

Load a model in LMStudio and start its server, then point `BASE_URL` at it. For a
cloud judge, use that provider's URL, name `MODEL`, and set `JUDGE_API_KEY` in the
environment.

`SYSTEM_PROMPT` in that file is what the search optimises for. Read it first.

### 3c. Set the size

In `settings.py`:

| Setting | Current | Effect |
| --- | --- | --- |
| `COUNT` | 4 | starting population; one model load each |
| `TRAINING_COUNT` | 20 | eval prompts per individual per generation |
| `SELECTION_COUNT` | 2 | individuals added per generation |
| `GENERATIONS` | 5 | generations after the first |
| `PROCESS_RUN_BATCH_SIZE` | 4 | scripts running at once; each holds a copy of the base model |

`SELECTION_COUNT = None` doubles the population every generation.

### 3d. Smoke test

```bash
python main.py population trees runs
```

```bash
python main.py process --limit 1 --run 0 --keep-scripts
```

```bash
python main.py evaluate fitness --run 0
```

`--run 0` is the latest sweep.

### 3e. Full run

```bash
python full_run.py --label "poem blends 1"
```

`1 + GENERATIONS` generations. Flags: `--generations N`, `--set NAME=VALUE`,
`--limit N`, `--timeout N`, `--keep-scripts`.

Editing `settings.py` does not affect a sweep already created. Use `--set`.

Resume:

```bash
python continue_run.py --run 0 --generations 3
```

Skipped individuals are normal: `BAD` ones cannot run (`--include-blocked`), and
unmutated ones with a stored result are not re-run (`--include-unchanged`). A
crashed individual is stored as a failed execution; the sweep continues.

---

## 4. Read the results

```bash
python store.py --list
```

```bash
python store.py --show 0
```

```bash
python store.py --export 0 --into export
```

`--export` writes population, trees, scripts, outputs, transcripts, results and
`fitness_history.txt`. The history file is the one that shows whether fitness
improved — `individuals.fitness` only holds the current generation.

Run the winning blend by hand (needs the venv):

```bash
python main.py runs --run 0
```

```bash
python run_db\run_003.py "Help me plan my week."
```

---

## Problems

| Symptom | Cause |
| --- | --- |
| `No module named 'unsloth'` | venv not active; check `where python` |
| `cannot import name 'load_dataset'` | same — the repo's `datasets/` folder shadowed the library |
| every individual failed | same — `process` runs the scripts under `sys.executable` |
| all fitness 0.0 | judge unreachable; fix it and re-run `evaluate fitness` |
| most individuals `BAD` | rank spread makes `LIN` illegal; use repeated ranks or `--vary learning-rate` |
| JSON array rejected | eval set must be one record per line |
| out of memory | lower `PROCESS_RUN_BATCH_SIZE` |
| population exploding | set `SELECTION_COUNT` to a number |
