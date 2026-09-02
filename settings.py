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
COUNT = 4

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


# --- the adapters being blended --------------------------------------------

# The model every one of the five adapters was trained on, and the model the
# generated scripts load before attaching anything -- it reaches them through a
# marker, like the slots below, so there is one copy of the name rather than one
# per template. Change it only alongside adapters trained on the same base:
# PEFT loads a LoRA against the model its adapter_config.json names.
#
# It is also the identity of the "llm_judge_baseline" evaluator's cached
# base-model answers: those are stored under this name, so repointing this at
# another model asks for that model's own baseline rather than reusing the old
# one's.
BASE_MODEL = "unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit"

# Where each of the five LoRAs the trees refer to lives -- the search space
# itself, so it belongs with the rest of the knobs rather than in the templates:
# repoint a slot here and both templates follow, and the sweep records which
# five adapters its fitness numbers were earned on.
#
# One independent entry per slot. A relative path is taken from this file's
# folder; an absolute one is used as it stands. Anything that is neither -- a
# Hub repo id, say -- is passed through untouched, which is as far as that has
# ever worked: every rank check reads the adapter's own adapter_config.json off
# disk. generate_runs.py resolves these once and writes the result into each
# generated script, so a script carries real paths rather than working them out
# from where it happens to sit.
#
# These five were trained at different ranks (r=16, 16, 8, 4, 32), which the
# code handles -- nothing assumes they match, because PEFT's cat sums input
# ranks, svd takes the max, and linear refuses inputs whose ranks differ.
LORA_SLOTS = {
    "L1": "loras/Lora001/my_planning_coach-lora_adapter",
    "L2": "loras/Lora002/my_planning_coach-lora_adapter",
    "L3": "loras/Lora003/my_planning_coach-lora_adapter",
    "L4": "loras/Lora004/my_planning_coach-lora_adapter",
    "L5": "loras/Lora005/my_planning_coach-lora_adapter",
}


# --- the generated scripts -------------------------------------------------

# Which template generate_runs.py fills. None means its own default,
# template_code.py -- the real thing. Set it to "template_code_mocked.py" for a
# dry run: same trees, same ranks, same BAD verdicts, but no model load, random
# answers, and scores that arrive with the transcript, so the whole pipeline
# finishes in seconds on a machine with no GPU and no judge running. Mocked
# scores are noise; never read one as a result.
TEMPLATE = "template_code.py"


# --- running the generated scripts -----------------------------------------

# How many generated scripts the process step keeps in flight at once. The
# scripts are launched in batches of this size and a batch is waited out before
# the next one starts, so this is a fixed ceiling on concurrency rather than a
# queue that refills: a batch takes as long as its slowest member.
#
# 1 is the old behaviour, one script at a time. Anything higher trades memory
# for wall clock, and the trade is steep for a real run: every script loads the
# base model into its own process, so a batch of N is N copies of the model
# resident at the same time. Set this to what the GPU can actually hold -- an
# individual that runs out of memory is recorded as a failed execution like any
# other, so an over-large batch does not stop a sweep, it quietly fills it with
# failures. Mocked runs (TEMPLATE = "template_code_mocked.py") load nothing and
# can go much higher.
#
# The scripts in a batch share the run folder as their working directory, so
# they also share the caches unsloth drops there.
PROCESS_RUN_BATCH_SIZE = 4

# How often a running script says where it has got to, in seconds. A generated
# script prints nothing at all while it loads the base model and then one
# question-and-answer pair per eval prompt, so left to itself the console shows
# a long hang and then a wall of transcript. The process step reads that output
# as it arrives and reports a line for that script every so often instead: the
# prompt it is on, or that it is still loading.
#
# This is the cadence, not the detail. Milestones -- the model coming ready, the
# last prompt starting -- are one line each and are always said; the running
# count waits for this many seconds to have passed since that script last said
# anything, so a fifty-prompt run speaks a handful of times rather than fifty.
# Turn it down to watch a run closely, up for a quieter log, and to 0 for the
# milestones alone. The transcript itself is never echoed -- it goes to the
# database, and store.py --show reads it back.
PROCESS_RUN_PROGRESS_SECONDS = 30


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
SELECTION_COUNT = 2


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
GENERATIONS = 5


# --- where things go -------------------------------------------------------

# Where the generated scripts go, and the database itself, relative to this file.
#
# This folder must stay exactly one level below the project folder: a generated
# script finds the LoRA folders by going up one from its own directory, so a
# deeper path breaks every one of them. Siblings are fine; subfolders are not.
# (The eval prompts no longer come into it -- TRAINING_SET below is resolved at
# generation time and stamped into each script as an absolute path.)
DB_RUN_DIR = "run_db"
DB_PATH = "run_db/gep.sqlite3"

# The eval prompts every generated script is judged on, one per line. It lives
# here rather than in the templates so the eval set can be repointed without
# editing generated-script code, and so a sweep records which file it was scored
# against -- the prompts are half of what a fitness number means. A relative
# path is taken from this file's folder, like DB_RUN_DIR above; an absolute one
# is used as it stands, so the file need not sit in the project folder at all.
# generate_runs.py resolves it and stamps the result into each script, so a
# script no longer has to find this file by walking up from itself.
TRAINING_SET = "datasets/poem_lora_dataset.json"

# How many records of TRAINING_SET each run is judged on. The first
# TRAINING_COUNT of them, in file order, or the whole file when it holds fewer
# -- and None for no cap at all, which is every record.
#
# The top N rather than a sample of N, and the same N for everyone: fitness only
# compares across individuals because they all answered the same questions, so a
# per-individual draw would make two scores incomparable and a re-run of one
# individual incomparable with itself. File order is already arbitrary; taking a
# prefix of it keeps that arbitrariness fixed instead of adding a second source
# of it.
#
# This is the cheapest knob in the file. Every prompt is one generate() call per
# individual per generation, so halving it halves the eval half of a sweep --
# worth turning down while iterating and back up for a real search, remembering
# that a short eval set makes a noisier fitness signal.
TRAINING_COUNT = 20



# --- how an answer is scored -----------------------------------------------

# Which evaluator the evaluate step uses. One name, out of the registry in
# evaluate_run.py:
#
#   "llm_judge"            a judge model grades the answer on its own merits.
#                          Needs an endpoint. The original behaviour, and the
#                          default a sweep created before this setting existed
#                          is read back under.
#   "llm_judge_reference"  the same judge, shown the answer the dataset carries
#                          for that question as well. Needs an endpoint and a
#                          dataset with assistant turns. This is the one that
#                          can see *style*: a judge grading on merit alone
#                          happily rewards a helpful prose answer from a blend
#                          that was supposed to rhyme.
#   "llm_judge_baseline"   the same judge, shown what the *base model* replied
#                          to the same question as well, and asked how much the
#                          blend improved on it. Needs an endpoint, and one run
#                          of the base model over the eval set -- cached in the
#                          database the first time and read from there ever
#                          after. This is the one that scores what this search
#                          is actually for: 0.5 means the blend changed nothing
#                          worth having, above it means it helped, below it
#                          means it did harm.
#   "similarity"           token or character overlap with the dataset's answer.
#                          No endpoint, deterministic, free -- and a measure of
#                          agreement with one particular answer rather than of
#                          quality.
#   "heuristic"            local checks: length, repetition, a pattern the
#                          answer must or must not match. No endpoint. Measures
#                          whether an answer is malformed, not whether it is
#                          good.
#   "panel"                several judge models, aggregated. Less noise per
#                          score, N times the cost.
#
# Like every setting here, this is frozen into a sweep when it starts: changing
# it does nothing to a sweep already running, which is what keeps every fitness
# number in one sweep comparable with the others. `python main.py --evaluators`
# lists what is registered.
EVALUATOR = "llm_judge"


# --- the judge model, for the evaluators that ask one -----------------------
#
# Read by "llm_judge", "llm_judge_reference", "llm_judge_baseline", and by
# "panel" for everything except which models sit on it. The API key is deliberately *not* here: a sweep
# writes its settings into the database, so the key is read from the
# JUDGE_API_KEY environment variable by evaluate_run.py instead.

# Where the judge lives. The default is the local LMStudio instance; its API is
# OpenAI-compatible, so a cloud endpoint is a drop-in replacement:
#   OpenAI      https://api.openai.com/v1
#   OpenRouter  https://openrouter.ai/api/v1
#   vLLM        http://<host>:8000/v1
# Claude is not OpenAI-compatible; using a Claude model as the judge needs a
# separate backend through the anthropic SDK.
JUDGE_BASE_URL = "http://172.22.208.1:1234/v1"

# Which model does the grading. None asks the endpoint what it has loaded, which
# is what you want with LMStudio, and stores the answer on every exchange it
# grades; name it explicitly for a cloud model.
JUDGE_MODEL = None

# Grading should be repeatable, so keep the temperature at zero.
JUDGE_TEMPERATURE = 0.0

# The judge emits a short JSON object, but reasoning models spend tokens
# thinking first and return an empty message if they run out mid-thought, so
# this needs far more headroom than the answer itself requires.
JUDGE_MAX_TOKENS = 2000

# Seconds to wait for one grading call, how many times to retry a call that
# fails for a transient reason (connection dropped, 5xx, rate limit), and how
# long to wait between tries.
JUDGE_TIMEOUT = 300
JUDGE_RETRIES = 2
JUDGE_RETRY_WAIT = 3

# Ask for a JSON object back. None for an endpoint that rejects the parameter --
# the prompt asks for JSON anyway, and a 400 falls back to that by itself.
JUDGE_RESPONSE_FORMAT = {"type": "json_object"}

# How the judge is told to grade, with no reference to compare against. This is
# the rubric the whole search selects on, so it is worth tuning deliberately.
JUDGE_SYSTEM_PROMPT = """\
You are grading the quality of a single answer given by an AI planning coach.

You will be shown the QUESTION a user asked and the ANSWER the coach gave.
Judge the answer only, on how well it serves the person who asked.

Consider:
- Relevance: does it address what was actually asked?
- Usefulness: could the person act on it, or are they left stuck?
- Specificity: concrete and grounded rather than vague filler.
- Coherence: well formed and consistent, free of contradictions, repetition,
  broken grammar or nonsense.
- Appropriateness: sensible length and tone. Asking one focused clarifying
  question is fine when the request genuinely needs it; deflecting every
  request without helping is not.

Score from 0.0 to 1.0:
  1.0  excellent - directly useful, specific, clear
  0.7  good - helpful, minor weaknesses
  0.5  mixed - partly useful, vague or padded
  0.3  poor - barely addresses the question
  0.0  useless - incoherent, empty, or entirely off topic

Reply with JSON and nothing else, with the score FIRST:
{"quality": <number between 0 and 1>, "reason": "<at most 12 words>"}
"""

# How the judge is told to grade when it is shown the dataset's own answer --
# the rubric "llm_judge_reference" selects on, and "panel" too when
# PANEL_USE_REFERENCE is on.
#
# The reference is what the adapters were fine-tuned to produce, so this prompt
# asks about the thing merit-only grading cannot see: did the blend answer in
# the manner the training data answers in. It deliberately does not ask for a
# copy -- a blend that reproduced the reference word for word would score well
# here and have learned nothing but that one answer.
JUDGE_REFERENCE_SYSTEM_PROMPT = """\
You are grading a single answer produced by a fine-tuned AI model.

You will be shown:
  QUESTION          what the user asked
  REFERENCE ANSWER  how the model's training data answers that question
  ANSWER            what the model actually replied

The REFERENCE ANSWER is an example of the intended behaviour, not the only
correct answer. Do not reward copying it, and do not punish different wording,
different examples or different details.

Judge the ANSWER on:
- Manner: does it answer in the same style, voice, form and register as the
  reference? This matters most -- it is what the model was trained for.
- Substance: does it actually answer the QUESTION, as the reference does?
- Coherence: well formed and consistent, free of contradictions, repetition,
  broken grammar or nonsense.

Score from 0.0 to 1.0:
  1.0  excellent - same manner as the reference, and a sound answer
  0.7  good - recognisably the same manner, minor slips
  0.5  mixed - part of the manner, or the manner without the substance
  0.3  poor - answers, but in nothing like the intended manner
  0.0  useless - incoherent, empty, or entirely off topic

Reply with JSON and nothing else, with the score FIRST:
{"quality": <number between 0 and 1>, "reason": "<at most 12 words>"}
"""

# How the judge is told to grade when it is shown what the *base model* said to
# the same question -- the rubric "llm_judge_baseline" selects on, and the one
# that matches what this whole search is for: not "is this answer good" but "did
# folding these adapters in make it better than the model was without them".
#
# The scale is centred, and that is the point. 0.5 is "the blend changed nothing
# worth having", so a fitness of 0.5 across a transcript says an individual is
# the base model with extra steps, above it says the blend earned its keep, and
# below it says it did harm. A merit-only rubric cannot say any of that: a blend
# that ruins nothing scores well on merit because the base model was already
# competent, and the search then has nothing to climb.
JUDGE_BASELINE_SYSTEM_PROMPT = """\
You are measuring what a LoRA adapter blend did to a base model.

You will be shown:
  QUESTION      what the user asked
  BASE ANSWER   what the base model replied, with no adapter attached
  TUNED ANSWER  what the same model replied with the adapter blend attached

Both answers come from the same model and the same question. Judge only how
the TUNED ANSWER compares with the BASE ANSWER as a reply to that question.

Consider, in this order:
- Usefulness: does the tuned answer serve the person who asked better?
- Specificity: more concrete and grounded, rather than more words.
- Manner: a clearer, better suited style, voice or form for the request.
- Coherence: the tuned answer must not be more repetitive, contradictory,
  truncated or malformed than the base one.

Do not reward length, padding, restated questions or lists for their own sake,
and do not reward a change of subject. If the two answers are equally good in
different words, that is no improvement.

Score from 0.0 to 1.0, where 0.5 is "no real difference":
  1.0  transformed - the tuned answer is far better in every way that matters
  0.8  clearly better
  0.6  slightly better
  0.5  no meaningful difference, or a difference not worth having
  0.3  slightly worse
  0.2  clearly worse
  0.0  ruined - empty, incoherent, repetitive or off topic where the base
       answer was not

Reply with JSON and nothing else, with the score FIRST:
{"quality": <number between 0 and 1>, "reason": "<at most 12 words>"}
"""


# --- the base-model answers "llm_judge_baseline" grades against -------------
#
# Producing them costs one base-model load and one generate() per eval prompt,
# once. They are then cached in the database under BASE_MODEL and the question,
# outside any one sweep, so every later sweep on the same base model and the
# same prompts reads them and loads nothing.

# Which template the one baseline script is filled from. None picks it from the
# sweep's own TEMPLATE: template_baseline.py for a real sweep, and
# template_baseline_mocked.py for one generated from template_code_mocked.py,
# so a mocked sweep still needs no GPU. Mocked baselines are cached apart from
# real ones -- under "mock:<model>" -- so a dry run can never leave made-up base
# answers where a real sweep would find them.
BASELINE_TEMPLATE = None

# Seconds to allow the baseline script. It answers every eval prompt in one
# process, so give it roughly what one individual gets, times the prompts.
BASELINE_TIMEOUT = 1800


# --- the "similarity" evaluator --------------------------------------------

# How an answer is compared with the dataset's answer:
#   "token_f1"     bag-of-words F1, repeats counted. Balanced: an answer that
#                  is all reference words plus padding loses precision, one
#                  that covers half of them loses recall.
#   "containment"  how much of the reference's vocabulary turns up at all.
#                  Forgiving about length and about everything else added.
#   "sequence"     character-level overlap (difflib), so word order and
#                  phrasing count. The strictest of the three.
SIMILARITY_METRIC = "token_f1"

# Whether case counts. Off by default, since a lowercase answer to an uppercase
# reference is usually the same answer -- turn it on for an adapter whose whole
# job is a matter of case, or use HEURISTIC_REQUIRE for that instead.
SIMILARITY_CASE_SENSITIVE = False


# --- the "heuristic" evaluator ---------------------------------------------

# The length band an answer is expected to fall in, in words. Under the floor
# scores proportionally; over the ceiling falls away from it. None for no
# ceiling.
HEURISTIC_MIN_WORDS = 8
HEURISTIC_MAX_WORDS = 400

# A pattern the answer must contain, and one it must not, or None for neither.
# Each is a Python regular expression, searched with re.MULTILINE, and each
# counts as one of the equally weighted checks that make up the score. This is
# where a task whose rule is genuinely checkable gets checked -- r"^[^a-z]*$"
# for an all-uppercase adapter, say.
HEURISTIC_REQUIRE = None
HEURISTIC_FORBID = None


# --- the "panel" evaluator --------------------------------------------------

# The models on the panel, by id, all served by one endpoint. An empty list asks
# the endpoint what it has loaded and sits a panel of one on it -- which is
# "llm_judge" with extra steps, so name at least two to get the point of this.
PANEL_MODELS = []

# Where the panel is served, or None for JUDGE_BASE_URL. Everything else about
# a member -- temperature, token budget, timeouts, the rubric -- comes from the
# JUDGE_* settings above, so a panel is several models grading identically
# rather than several differently configured judges.
PANEL_BASE_URL = None

# How the members' scores become one: "mean", "median", "min" or "max". Median
# ignores a single outlying judge; min is the pessimistic reading, and selects
# for answers no member objected to.
PANEL_AGGREGATE = "mean"

# Whether the panel grades against the dataset's own answer -- the
# JUDGE_REFERENCE_SYSTEM_PROMPT rubric -- rather than on merit alone. Needs a
# dataset with assistant turns, exactly like "llm_judge_reference".
PANEL_USE_REFERENCE = False



def snapshot():
    """Every setting above as {name: value}, for recording what a run used."""
    return {name: value for name, value in sorted(globals().items())
            if name.isupper() and not name.startswith("_")}
