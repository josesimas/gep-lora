"""
support.py - Fixtures for the async API tests.

A throwaway JOBS_DIR, five fake adapter folders (an adapter_config.json each,
which is all the mocked template and the rank rule read), and a small dataset
-- so a job can be submitted, run through main.py and put live on a machine
with nothing else installed.
"""

import json
import os
import shutil
import tempfile
import unittest

from async_api import registry as reg

# The ranks the real slots were trained at, so the rank rule behaves as it does.
RANKS = {"L1": 16, "L2": 16, "L3": 8, "L4": 4, "L5": 32}

RECORDS = [
    {"messages": [{"role": "user", "content": "What is question %d?" % number},
                  {"role": "assistant", "content": "It is answer %d." % number}]}
    for number in range(1, 7)
]


def make_slots(folder):
    """{slot: folder} of fake adapters under `folder`."""
    slots = {}
    for slot, rank in RANKS.items():
        where = os.path.join(folder, "adapters", slot)
        os.makedirs(where, exist_ok=True)
        with open(os.path.join(where, "adapter_config.json"), "w", encoding="utf-8") as handle:
            json.dump({"r": rank}, handle)
        slots[slot] = where
    return slots


class JobsTestCase(unittest.TestCase):
    """A registry in a temp folder, a user, and fake adapters."""

    def setUp(self):
        self.folder = tempfile.mkdtemp(prefix="gep-api-tests-")
        self.registry = reg.Registry(os.path.join(self.folder, "jobs"))
        self.key = self.registry.add_user("alice")
        self.user = self.registry.user_for_key(self.key)
        self.slots = make_slots(self.folder)

    def tearDown(self):
        shutil.rmtree(self.folder, ignore_errors=True)

    def submission(self, **settings):
        """A small mocked job: one generation beyond the first, four individuals."""
        conf = {"TEMPLATE": "template_code_mocked.py", "GENERATIONS": 1, "COUNT": 4,
                "TRAINING_COUNT": 3, "LORA_SLOTS": self.slots, "SEED": 7}
        conf.update(settings)
        return {"label": "test", "settings": conf,
                "datasets": {"training": RECORDS[:4],
                             "testing": "\n".join(json.dumps(one) for one in RECORDS[4:])},
                "options": {"test_min_quality": 0.0}}
