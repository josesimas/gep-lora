"""
settings.py - The knobs for a complete run, in one place.

Both entry points read this module: main_txt.py, which drives the pipeline over
text files, and main_sqlite.py, which drives it over a sqlite database. Keeping
the values here rather than at the top of one of them means there is no second
copy to drift, and it is what lets the sqlite mode record the settings a sweep
ran under without listing them by hand -- it snapshots every upper-case name
below, so a knob added here is a knob stored there.

Change a value and re-run; nothing else needs editing.
"""

# --- the population --------------------------------------------------------

# How many individuals the population holds.
COUNT = 10

# Seed for the population draw. An int repeats the same population every run;
# None grows a fresh one each time. Note this is separate from the LoRA blend
# weights each individual is evaluated under -- those are WEIGHT_SEED below.
SEED = 42

# Reject duplicate chromosomes when building the population.
UNIQUE = True

# Deepest level an operator may sit at, and the chance an operator is arity 2
# and keeps the branch growing. These are generate_population.py's defaults,
# repeated here so the sqlite mode can record the shape a population was drawn
# with; the text mode leaves them alone and lets that script default.
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

# The seed each generated script draws its w1..w5 from.
#
# None is the historical behaviour and the text mode's: every execution redraws
# the weights from the OS, so the same chromosome scores differently each time
# and the draw itself cannot be recovered -- only the weights it produced, which
# the transcript records.
#
# The sqlite mode wants a sweep to be repeatable, so it does not use this value
# directly. It derives one seed per individual from WEIGHT_MASTER_SEED below and
# stamps that into the individual's script, so re-running a stored sweep rebuilds
# the same blends. Set this to an int if you want the text mode pinned the same
# way.
WEIGHT_SEED = None

# Where the sqlite mode's per-individual weight seeds come from. An int makes a
# whole sweep reproducible; None draws a master seed at run time and stores it,
# which is just as repeatable after the fact -- the value used is written to the
# run's settings either way.
WEIGHT_MASTER_SEED = None


# --- where things go -------------------------------------------------------

# Where the text mode puts everything, relative to this file.
#
# Any of these folders must stay exactly one level below the project folder: a
# generated script finds the LoRA folders and training_set.txt by going up one
# from its own directory, so a deeper path breaks every one of them. Siblings
# are fine; subfolders are not.
RUN_DIR = "run"

# Where the sqlite mode puts the generated scripts, and the database itself.
#
# A folder of its own rather than run/, because the two modes would otherwise
# overwrite each other's run_NNN.py: the text mode's index.txt would still name
# a script that had since been replaced by a sqlite sweep's, and process_run.py
# would run the wrong blend under the right chromosome's name. Keeping them
# apart means a sweep in either mode can be left sitting there safely.
DB_RUN_DIR = "run_db"
DB_PATH = "run_db/gep.sqlite3"


def snapshot():
    """Every setting above as {name: value}, for recording what a run used."""
    return {name: value for name, value in sorted(globals().items())
            if name.isupper() and not name.startswith("_")}
