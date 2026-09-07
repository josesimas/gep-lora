# unittests/search

Unit tests for `search/` — the GEP search itself. 173 tests, about 15 seconds,
on **plain Python 3**: no GPU, no base model, no judge endpoint, no adapters on
disk. Nothing here loads a model, so the venv one level up is not needed.

```bash
python -m unittest discover -s unittests -t .
```

`-t .` is the top-level directory, and it matters: the tests import `search`
and `storage` the way the pipeline does, so discovery has to start from the
repo root. One module at a time:

```bash
python -m unittest unittests.search.test_selection -v
```

## What is here

| module | what it holds down |
| --- | --- |
| `test_generate_population.py` | the alphabet, the arity/class rule, encode/decode round-tripping, what `decode` refuses, the draw staying inside `MAX_DEPTH`, and `build_population` giving up rather than looping |
| `test_draw_trees.py` | the stored drawing, and that its rows read back as the chromosome |
| `test_mutation.py` | class-local swaps preserving tree shape, the root and the elite going untouched, `has_changed` written for everyone, and a mutant's fitness cleared to NULL |
| `test_calculate_fitness.py` | the mean over the *latest* execution, 0.0 rather than NULL for nothing to average, and the generation a snapshot is filed under |
| `test_elitism.py` | exactly one `is_best` or none, ties on the lowest number, an all-zero population electing nobody |
| `test_selection.py` | the wheel, zero-width slices never picked, the cull's `n+1` arithmetic, a copy being its parent field for field but never `is_best`, retired numbers |
| `test_generation_cycle.py` | the four steps run as a search |
| `support.py` | the sqlite fixtures the four database-backed steps need |

## Two things worth knowing before changing them

**The database is real.** `support.SweepTestCase` builds an actual sqlite
database with the real schema in a temp folder and throws it away afterwards.
Most of what these steps promise is a promise about what ends up in the tables
— that a cull takes an individual's executions with it, that a copy inherits a
column added later, that `mark_best` leaves no window with two elites — and a
mocked connection would be the suite agreeing with itself about SQL nobody ran.
`store.py` is the only module in the repo that speaks sqlite; that stays true
here.

**`test_generation_cycle.py` scores on something with a known answer.** The
stand-in judge gives an individual the fraction of its leaves that are `w1`.
That target is reachable by variable swaps alone, so a working search climbs to
1.0 and a broken one does not — which is what lets the file assert that the
search *works* rather than merely that it runs. The starting population is
restricted to trees of four leaves or more for the same reason: a two-leaf tree
scores 1.0 whenever both variables happen to be `w1`, one draw in twenty-five,
so a population of small trees usually arrives with the target already in it
and every claim about the climb would be a claim about the first draw's luck.

Those tests were checked against broken steps rather than only against working
ones. Replacing `selection.select` with one that picks nobody fails
`test_the_search_climbs`, `test_it_reaches_the_target_it_is_pointed_at` and
`test_the_numbers_keep_rising`; neutering `elitism.elect` fails
`test_the_best_never_goes_backwards`; neutering `mutation.apply` fails
`test_it_reaches_the_target_it_is_pointed_at`. Keep that property when adding
to the file — a cycle test that passes with the step it names removed is
measuring the harness.

## What is not covered

Only `search/`. The steps that need a model or a judge — `blends/`,
`evaluators/`, `templates/` — are not exercised here, and neither are
`storage/store.py`'s own readers beyond what these four steps call. The sibling
folders under `unittests/` are where those would go.
