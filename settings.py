"""
settings.py - The knobs for a complete run, in one place.

main.py reads this module. Keeping the values here rather than at the top of it
means there is no second copy to drift, and it is what lets a sweep record the
settings it ran under without listing them by hand -- snapshot() takes every
upper-case name below, so a knob added here is a knob stored there.

Change a value and re-run; nothing else needs editing.
"""

# --- the population --------------------------------------------------------

# How many individuals the population holds.
COUNT = 3

# Seed for the population draw. An int repeats the same population every run;
# None grows a fresh one each time -- and, since a sweep records what it drew,
# even that stays repeatable afterwards. Note this is separate from the LoRA
# blend weights each individual is evaluated under -- those come from
# WEIGHT_MASTER_SEED below.
SEED = 42

# Reject duplicate chromosomes when building the population.
UNIQUE = True

# Deepest level an operator may sit at, and the chance an operator is arity 2
# and keeps the branch growing -- the shape a population is drawn with, recorded
# alongside the rest so a stored sweep says what shape that was.
MAX_DEPTH = 4
BRANCH_PROB = 0.6


# --- the generated scripts -------------------------------------------------

# Which template generate_runs.py fills. None means its own default,
# template_code.py -- the real thing. Set it to "template_code_mocked.py" for a
# dry run: same trees, same ranks, same BAD verdicts, but no model load, random
# answers, and scores that arrive with the transcript, so the whole pipeline
# finishes in seconds on a machine with no GPU and no judge running. Mocked
# scores are noise; never read one as a result.
TEMPLATE = "template_code_mocked.py"


# --- the blend weights -----------------------------------------------------

# Where the per-individual weight seeds come from. Each individual's script is
# stamped with a seed derived from this one and its own number, so it draws the
# same w1..w5 every time it runs and re-running a stored sweep rebuilds the same
# blends. An int makes a whole sweep reproducible from the start; None draws a
# master seed at run time and stores it, which is just as repeatable after the
# fact -- the value used is written to the run's settings either way.
WEIGHT_MASTER_SEED = None


# --- selection -------------------------------------------------------------

# Where the roulette wheel's spins come from. Each application of the selection
# step derives its own generator from this seed and the size of the population
# it is spinning over, the way each individual derives its weight seed from
# WEIGHT_MASTER_SEED and its own number: one sweep, one recorded seed, and every
# draw in it repeatable -- but a second generation still draws its own parents
# rather than the first one's again. An int makes a sweep reproducible from the
# start; None draws a master seed at run time and stores it.
SELECTION_MASTER_SEED = None

# How many individuals each spin of the wheel adds. None means as many as the
# population already holds, which doubles it: N parents in, N offspring out.
SELECTION_COUNT = 3


# --- mutation --------------------------------------------------------------

# The chance each symbol of a chromosome is replaced by another of its own kind
# -- per symbol, not per chromosome, so an eleven-symbol individual at 0.1
# expects about one change and may well come through untouched. A symbol only
# ever becomes one of its own class (CAT/SVD/LIN, L1-L5, w1-w5) and the root is
# never touched, which is what keeps every mutated chromosome readable; 0.0
# turns mutation off without removing the step.
MUTATION_RATE = 0.1

# Where the mutation dice come from, on the same terms as the two seeds above:
# an int makes a sweep reproducible from the start, None draws one at run time
# and records it.
MUTATION_MASTER_SEED = None


# --- continuing a sweep ----------------------------------------------------

# How many generations continue_run.py runs when it is not told otherwise. One
# generation is trees -> runs -> process -> evaluate -> fitness -> elitism ->
# selection -> mutation over the population already in the database.
#
# Mind what this costs: process loads the base model once per individual, and
# selection *appends* its picks, so with SELECTION_COUNT left at None the
# population doubles every generation and the work of a run grows with it. Fix
# SELECTION_COUNT to a number to grow by that much per generation instead.
GENERATIONS = 10


# --- where things go -------------------------------------------------------

# Where the generated scripts go, and the database itself, relative to this file.
#
# This folder must stay exactly one level below the project folder: a generated
# script finds the LoRA folders and training_set.txt by going up one from its
# own directory, so a deeper path breaks every one of them. Siblings are fine;
# subfolders are not.
DB_RUN_DIR = "run_db"
DB_PATH = "run_db/gep.sqlite3"


def snapshot():
    """Every setting above as {name: value}, for recording what a run used."""
    return {name: value for name, value in sorted(globals().items())
            if name.isupper() and not name.startswith("_")}
