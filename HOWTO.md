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

Create it once, from wherever you want it. The two walkthroughs below build the
same environment; the only part that really varies by machine is the torch build,
so if your GPU wants a different CUDA version take that line from unsloth's own
instructions instead.

### Windows

```bash
py -3.11 -m venv .venv
```

```bash
.venv\Scripts\activate.bat
```

```bash
python -m pip install --upgrade pip setuptools wheel
```

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

```bash
pip install unsloth
```

On the reference machine `create_env.bat` and `setup_torch.bat` (one level above
the repo) do exactly this, with a local wheelhouse so a second machine can
install offline.

### Linux (Ubuntu)

Ubuntu does not ship Python 3.11 — 22.04 has 3.10 and 24.04 has 3.12, and unsloth
supports neither — so take it from deadsnakes:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt update
```

```bash
sudo apt install -y python3.11 python3.11-venv python3.11-dev build-essential git
```

What you need next is the **NVIDIA driver**, not the CUDA toolkit: the torch
wheels carry their own CUDA runtime, and `cu128` wants a driver of 525 or later.
Check what is there:

```bash
nvidia-smi
```

If that prints nothing, install a driver before going further — torch installs
happily without one and simply reports `torch.cuda.is_available()` as `False`:

```bash
sudo ubuntu-drivers install
```

Reboot, then build the venv wherever you want it:

```bash
python3.11 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
python -m pip install --upgrade pip setuptools wheel
```

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

```bash
pip install unsloth
```

That last install is the whole rest of the stack — `unsloth_zoo`, `peft`,
`transformers`, `trl`, `datasets`, `accelerate`, `bitsandbytes`, `xformers` — so
there is nothing else to name. Two Linux-only notes. **Do not** install
`triton-windows`; it is in the Windows environment only because Windows has no
triton wheel, and on Linux triton comes with torch. And run `pip` from anywhere
except the repo folder — the repo's own `datasets/` directory shadows the
`datasets` library, which is the `cannot import name 'load_dataset'` in
[Problems](#problems).

The repo itself needs no porting: `LORA_SLOTS`, `TRAINING_SET` and `DB_PATH` in
`settings.py` are relative and already written with forward slashes, and every
path the pipeline generates goes through `pathlib`. Two things do not carry over
from a Windows machine, though. Adapter folder names are **case-sensitive** here,
so `loras/Lora001` has to be exactly that if you copy trained adapters across
rather than training them again. And `JUDGE_BASE_URL`'s default
(`http://172.22.208.1:1234/v1`) is a WSL address for an LMStudio running on the
Windows host — on native Linux, with the judge on the same box, that is
`http://localhost:1234/v1`.

### Either way

**Which commands need it.** Everything under [1. Train](#1-train-the-adapters),
`test_lora.py`, and the `process` and `evaluate` steps — so in practice all of
`full_run.py` and `continue_run.py`. The rest (`test.py`, `store.py`,
`main.py population trees runs`, anything with the mocked template) runs on any
Python 3.

Activate the venv before those commands — `.venv\Scripts\activate.bat` on
Windows, `source .venv/bin/activate` on Linux — and check it took:

```bash
python -c "import torch, unsloth, peft; print(torch.__version__, torch.cuda.is_available())"
```

A version and `True` means you are ready. `ModuleNotFoundError` means the
activation did not stick — a `(.venv)` prompt does not prove it. `where python`
(Windows) or `which python` (Linux) shows what is actually resolving; the venv's
full path to the interpreter — `.venv\Scripts\python.exe` or `.venv/bin/python` —
always works regardless of PATH.

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

The last generation of a run stops after `fitness`. The three steps behind it
build the *next* generation, so a finished run leaves the population that was
actually scored, each individual still described by the script that earned its
transcript. `python main.py` on its own is a one-generation run and stops there
too; `--next-generation` runs them anyway.

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

### 3b. Choose how answers are scored

`EVALUATOR` in `settings.py` picks one of five ways:

```bash
python main.py --evaluators
```

| `EVALUATOR` | What it does | Needs an endpoint |
| --- | --- | --- |
| `llm_judge` | a judge model grades the answer on its own merits | yes |
| `llm_judge_reference` | the same judge, also shown the dataset's own answer | yes |
| `similarity` | word overlap with the dataset's own answer | no |
| `heuristic` | length, repetition, required/forbidden patterns | no |
| `panel` | several judge models, aggregated | yes |

The last two need nothing running at all, which makes them the quick way to
exercise a real sweep. The reference ones need an eval set with `assistant`
turns — `datasets/*.json` have them, `datasets/training_set.txt` does not.

Whichever you pick is frozen into the sweep when it starts, so choose before the
run rather than during it. To change it in a sweep already going:

```bash
python continue_run.py --set EVALUATOR='"similarity"'
```

### 3c. Start the judge (for the three that need one)

Defaults in `settings.py`:

```python
JUDGE_BASE_URL = "http://172.22.208.1:1234/v1"
JUDGE_MODEL = None
```

Load a model in LMStudio and start its server, then point `JUDGE_BASE_URL` at it.
For a cloud judge, use that provider's URL, name `JUDGE_MODEL`, and set
`JUDGE_API_KEY` in the environment — the key is the one judge setting that is not
in `settings.py`, because settings are written into the sweep's database.

`JUDGE_SYSTEM_PROMPT` (or `JUDGE_REFERENCE_SYSTEM_PROMPT`) is what the search
optimises for. Read it first.

### 3d. Set the size

In `settings.py`:

| Setting | Current | Effect |
| --- | --- | --- |
| `COUNT` | 4 | starting population; one model load each |
| `TRAINING_COUNT` | 20 | eval prompts per individual per generation |
| `SELECTION_COUNT` | 2 | copies added per generation; the same number plus one is culled, so the population holds its size |
| `GENERATIONS` | 5 | generations after the first |
| `PROCESS_RUN_BATCH_SIZE` | 4 | scripts running at once; each holds a copy of the base model |

A round appends `SELECTION_COUNT` copies plus one brand-new individual and
culls that many of the weakest, so the population stays at `COUNT` and what
turns over is its membership. `SELECTION_COUNT = None` is the exception: it
asks for as many copies as the population holds, one more than the cull can
take, so it grows the population by two a generation.

### 3e. Smoke test

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

### 3f. Full run

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
| `No module named 'unsloth'` | venv not active; check `where python` / `which python` |
| `cannot import name 'load_dataset'` | same — the repo's `datasets/` folder shadowed the library |
| every individual failed | same — `process` runs the scripts under `sys.executable` |
| all fitness 0.0 | judge unreachable; fix it and re-run `evaluate fitness` |
| most individuals `BAD` | rank spread makes `LIN` illegal; use repeated ranks or `--vary learning-rate` |
| JSON array rejected | eval set must be one record per line |
| out of memory | lower `PROCESS_RUN_BATCH_SIZE` |
| `python3.11: command not found` | Ubuntu ships 3.10 or 3.12; install 3.11 from deadsnakes |
| `torch.cuda.is_available()` is `False` | no NVIDIA driver; `nvidia-smi`, then `sudo ubuntu-drivers install` and reboot |
| population growing when it should not | `SELECTION_COUNT = None`, or a population too small to spare `SELECTION_COUNT + 1` non-elite individuals; set it to a number well below `COUNT` |
