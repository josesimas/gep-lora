"""
paths.py - Where the repo is, for every layer.

Every relative path a setting holds -- TRAINING_SET, LORA_SLOTS, DB_RUN_DIR,
JOBS_DIR -- is resolved against ROOT, never against the module reading it or
the cwd, so moving a module never moves what it reads.

API_DATABASE is here rather than in the service because the LoRA catalogue
(core/adapters/catalog.py) is a table of that file, and core may not import
the service to learn where it is. service/settings.py's JOBS_DIR defaults to
JOBS_DIR below, and both honour $GEP_API_JOBS_DIR.
"""

import os

# The repo folder: gep_lora/'s parent.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

JOBS_DIR = "api_jobs"
API_DATABASE = "api.sqlite"


def jobs_dir(value=None):
    """The API's folder: `value`, $GEP_API_JOBS_DIR or JOBS_DIR, against ROOT."""
    value = value or os.environ.get("GEP_API_JOBS_DIR") or JOBS_DIR
    return os.path.abspath(value if os.path.isabs(value) else os.path.join(ROOT, value))


def api_database(jobs=None):
    """The API's database: users, jobs, trainings, deployments and LoRAs."""
    return os.path.join(jobs_dir(jobs), API_DATABASE)
