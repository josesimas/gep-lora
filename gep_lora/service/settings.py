"""
settings.py - The knobs the async API reads.

Deliberately *not* in gep_lora/core/config/settings.py: that module's snapshot() freezes every
upper-case name into every sweep, and where a web server listens is not a fact
about a search. What a submitted sweep runs under still comes from
gep_lora/core/config/settings.py (plus the submission's overrides), exactly as a sweep
started from the command line does.

Every value can be overridden by an environment variable of the same name,
prefixed GEP_API_ (GEP_API_PORT=9000), so a deployment needs no edit here.
"""

import json
import os

from gep_lora import paths

# Where the registry and every job's folder live. Relative to the repo folder.
JOBS_DIR = paths.JOBS_DIR

# The HTTP server.
HOST = "127.0.0.1"
PORT = 8780

# How long a loaded model stays in memory after its last request, in seconds.
MODEL_TTL = 300

# How often the worker looks for a new job, and for a cancel on the one it runs.
WORKER_POLL_SECONDS = 2.0

# The largest request body the server reads (datasets arrive inline).
MAX_BODY_BYTES = 64 * 1024 * 1024

# Default and ceiling for max_new_tokens on an inference request.
MAX_NEW_TOKENS = 512
MAX_NEW_TOKENS_LIMIT = 4096

# Settings a submission may not override. The paths are the service's to
# choose (a job keeps to its own folder), and the dataset settings are filled
# from the datasets the submission carries rather than named by it.
LOCKED_SETTINGS = (
    "DB_PATH", "DB_RUN_DIR", "TESTING_RUN_DIR",
    "TRAINING_SET", "VALIDATION_SET", "TESTING_SET",
    "LORA_SERVER_HOST",
)

# The options a submission may pass to gep_lora/core/pipeline/main.py, and what they become.
MAIN_OPTIONS = ("no_test", "limit", "timeout", "test_min_quality")

# How long GET /judge/models waits for a judge endpoint to list its models. The
# demo asks it whenever an endpoint URL changes, so it is short: a dead
# endpoint should say so in a few seconds, not hold a form up.
JUDGE_MODELS_TIMEOUT = 5

# Datasets a submission may name by file rather than send inline: only files
# under this folder (relative to the repo folder).
SHARED_DATASETS_DIR = "datasets"

# Where a LoRA trained through the API is written, as user<N>/<name> (relative
# to the repo folder). Inside loras/, so the catalogue's scan finds
# them and a search's LORA_SLOTS can name them like any other.
TRAINED_LORAS_DIR = "loras/trained"

# Seconds each step of a mocked training takes, so one can be watched.
MOCK_TRAINING_DELAY = 0.05


def _override():
    for name, value in list(globals().items()):
        if not name.isupper():
            continue
        raw = os.environ.get("GEP_API_" + name)
        if raw is None:
            continue
        try:
            globals()[name] = json.loads(raw)
        except ValueError:
            globals()[name] = raw           # a bare string, such as a host


_override()
