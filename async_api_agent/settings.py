"""
settings.py - The knobs the LoRA agent reads.

Kept out of config/settings.py for the reason async_api/settings.py is: that
module's snapshot() freezes every upper-case name into every sweep, and which
model chats with a user is not a fact about a search.

Every value can be overridden by an environment variable of the same name,
prefixed GEP_AGENT_ (GEP_AGENT_PROVIDER=anthropic, GEP_AGENT_MODEL=claude-opus-5,
GEP_AGENT_LORA_RANKS=[4,8,16]) -- JSON when it parses as JSON, a string when not.

API keys are never settings. Each provider names the environment variable its
key is read from (`key_env`), so no key is written to a file here, sent to the
browser, or stored in a database.
"""

import json
import os

# --- which model the agent talks through ------------------------------------

# The provider and model a conversation starts on. The page can switch both for
# its own conversation; these are what it is offered first. MODEL None means
# the provider's own `model` below, and for a provider whose `model` is None
# too (a local server) the first chat model it lists -- the way the judges
# discover what LM Studio has loaded.
PROVIDER = "lmstudio"
MODEL = None

# Every provider the agent can talk through. `kind` is the wire format:
# "openai" is POST {base_url}/chat/completions (and GET {base_url}/models),
# "anthropic" is POST {base_url}/messages with Anthropic's headers, and
# "scripted" asks nobody -- the fallback wording from prompts.py, always.
# A gateway serving models in more than one format adds `wires`: {wire:
# [model-id prefix]}, where a wire is "openai", "anthropic" or "responses"
# (POST {base_url}/responses, OpenAI's Responses API).
#
# `base_url` None means the judges' endpoint, config/settings.py's
# JUDGE_BASE_URL, so the agent talks to the same LM Studio the judges grade on.
# `local` providers may be pointed at another URL from the page; a hosted one
# may not, because its key would go wherever the URL said.
# `temperature` says whether the provider takes one: the newest Claude and
# OpenAI reasoning models refuse the parameter outright.
PROVIDERS = {
    "lmstudio": {"label": "LM Studio (the judges' endpoint)", "kind": "openai",
                 "base_url": None, "key_env": "JUDGE_API_KEY", "model": None,
                 "local": True, "temperature": True},
    "ollama": {"label": "Ollama", "kind": "openai",
               "base_url": "http://127.0.0.1:11434/v1", "key_env": None, "model": None,
               "local": True, "temperature": True},
    "anthropic": {"label": "Anthropic (Claude)", "kind": "anthropic",
                  "base_url": "https://api.anthropic.com/v1", "key_env": "ANTHROPIC_API_KEY",
                  "model": "claude-opus-5", "local": False, "temperature": False},
    "openai": {"label": "OpenAI", "kind": "openai",
               "base_url": "https://api.openai.com/v1", "key_env": "OPENAI_API_KEY",
               "model": "gpt-5-mini", "local": False, "temperature": False},
    "gemini": {"label": "Google Gemini", "kind": "openai",
               "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
               "key_env": "GEMINI_API_KEY", "model": "gemini-2.5-flash",
               "local": False, "temperature": True},
    "mistral": {"label": "Mistral", "kind": "openai",
                "base_url": "https://api.mistral.ai/v1", "key_env": "MISTRAL_API_KEY",
                "model": "mistral-small-latest", "local": False, "temperature": True},
    "openrouter": {"label": "OpenRouter", "kind": "openai",
                   "base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY",
                   "model": "openrouter/auto", "local": False, "temperature": True},
    # One key, one URL, three wire formats: OpenCode Go serves each model in
    # the shape its maker speaks, so `wires` picks the format by model-id
    # prefix and `kind` is what every model not named there speaks. The prefixes
    # follow https://opencode.ai/docs/go/ -- a new family on another endpoint
    # needs a line here. It also refuses (400) a request without a stable id
    # per conversation, so `session_header` names the header the page's
    # conversation id goes in.
    "opencode-go": {"label": "OpenCode Go", "kind": "openai",
                    "base_url": "https://opencode.ai/zen/go/v1", "key_env": "OPENCODE_API_KEY",
                    "model": "kimi-k3", "local": False, "temperature": True,
                    "wires": {"anthropic": ["minimax-", "qwen"],
                              "responses": ["grok-", "gpt-", "muse-"]},
                    "session_header": "x-opencode-session"},
    "scripted": {"label": "No model (built-in wording)", "kind": "scripted",
                 "base_url": None, "key_env": None, "model": "scripted",
                 "local": True, "temperature": False},
}

# How warm the agent's wording is, where the provider takes a temperature.
TEMPERATURE = 0.4

# The longest reply asked for. Generous for the reason JUDGE_MAX_TOKENS is: a
# reasoning model (a Qwen3 in LM Studio, a Claude) thinks first, the thinking
# counts, and one that runs out mid-thought answers with nothing at all.
MAX_TOKENS = 4096
ANTHROPIC_MAX_TOKENS = 6000

# How hard a Claude model thinks before phrasing a step: a guide's reply is
# short and latency is what the user feels, so low.
ANTHROPIC_EFFORT = "low"

# Seconds to wait for a reply, and for a provider to list its models. The list
# is asked whenever the page's provider changes, so a dead endpoint should say
# so in a few seconds rather than hold the page up.
TIMEOUT = 120
MODELS_TIMEOUT = 6

# How many earlier turns of the conversation go with a free-form question.
HISTORY_TURNS = 12

# --- what the agent reads of a dataset --------------------------------------

# Records shown to the model when it summarises a dataset, and how much of
# each turn. The statistics cover every record; the model sees a sample.
SAMPLE_RECORDS = 8
SAMPLE_CHARS = 500

# Below this many records the agent says the dataset is small.
FEW_RECORDS = 30

# --- what the agent plans ----------------------------------------------------

# One LoRA per rank, all trained on the user's dataset. More than one because
# a search blends adapters, and adapters of different ranks are what the rank
# rule (CAT sums, SVD takes the max, LIN needs equal) has to work with -- the
# five default slots are 16, 16, 8, 4 and 32 for that reason.
LORA_RANKS = [8, 16]

# The most LoRAs one plan may train, when the chat is asked for more ranks.
# Five, because that is how many slots a search blends (LORA_SLOTS L1-L5):
# one chat can train a whole set. Each is a full training of its own, one
# after another on the worker, so this also caps what one sentence can queue.
MAX_LORAS = 5

# The name each LoRA gets: the dataset's stem and the rank, made free in the
# user's catalogue by adding -2, -3 ... when taken.
LORA_NAME = "{stem}-r{rank}"

# How long the user can wait, offered as choices. Each is a number of epochs
# -- passes over the dataset -- and the page shows the time each would take
# on this dataset beside it. The last is create_lora.py's own default.
WAIT_CHOICES = [
    {"id": "quick", "label": "A quick look", "epochs": 3,
     "blurb": "Enough to see whether the LoRA picks up the style."},
    {"id": "balanced", "label": "A solid result", "epochs": 8,
     "blurb": "A good middle: most of what the data can teach."},
    {"id": "thorough", "label": "The full recipe", "epochs": 20,
     "blurb": "What the LoRAs in this project were trained with."},
]

# The range a typed answer ("an hour", "5 epochs") is held to.
MIN_EPOCHS = 0.5
MAX_EPOCHS = 100

# The estimate: seconds per optimiser step when no LoRA on this base model has
# been trained here yet to measure it from, and seconds per training for what
# is not steps -- loading the base model, saving, the smoke test.
SECONDS_PER_STEP = 1.2
OVERHEAD_SECONDS = 60

# Whether the page starts with "practice run" ticked: a mocked training
# (create_lora.py --mock) that trains nothing, needs no GPU and finishes in
# seconds -- the plumbing check the demo page's Mock box is.
MOCK = False

# --- what the agent blends ---------------------------------------------------
#
# The second half of the guide: the user's LoRAs combined by a search (a job
# on POST /jobs). Its defaults are small on purpose -- a first search is for
# seeing what a blend does, and the chat can make it bigger.

# Generations after the first (a search is 1 + this), individuals in each, and
# how many questions every blend is judged on.
BLEND_GENERATIONS = 3
BLEND_POPULATION = 8
BLEND_QUESTIONS = 10

# The limits the chat is held to. A population under four leaves selection
# nothing to cull beside the copies it adds (SELECTION_COUNT is 2).
MAX_BLEND_GENERATIONS = 20
MIN_BLEND_POPULATION = 4
MAX_BLEND_POPULATION = 40
MAX_BLEND_QUESTIONS = 100

# How many of the questions left over after the judged ones go to the testing
# step (every blend is asked them) and then to the verification (the chosen
# blend beside each of its LoRAs): questions the search never saw, kept apart
# from each other. See blending.split().
BLEND_TEST_QUESTIONS = 10
BLEND_VALIDATION_QUESTIONS = 10

# The estimate: seconds one blend takes to build and answer, and to start a
# search (the lora servers loading the base model). A guess until measured;
# the page says so.
SECONDS_PER_INDIVIDUAL = 40
MOCK_SECONDS_PER_INDIVIDUAL = 0.5
BLEND_OVERHEAD_SECONDS = 60

# The job's label when the chat names none: the first LoRA's stem.
BLEND_LABEL = "{stem}-blend"

# --- after the search: testing, verification, going live ----------------------
#
# How many questions a verification asks when they come from a file -- a demo
# dataset, a LoRA's training data -- rather than one of the search's own
# splits, which carry their own size; and the most the chat may ask for.
VERIFY_QUESTIONS = 10
MAX_VERIFY_QUESTIONS = 100


def _override():
    for name, value in list(globals().items()):
        if not name.isupper():
            continue
        raw = os.environ.get("GEP_AGENT_" + name)
        if raw is None:
            continue
        try:
            globals()[name] = json.loads(raw)
        except ValueError:
            globals()[name] = raw


_override()
