"""The whole search in one command: gep_lora/core/pipeline/main.py, run from here.

    python main.py [--db ...] [--run N] [--resume | --evaluate] ...

Kept at the top so the command every document gives still works; the driver
itself is gep_lora.core.pipeline.main (python -m gep_lora.core.pipeline.main).
"""

import sys

from gep_lora.core.pipeline.main import cli

if __name__ == "__main__":
    sys.exit(cli())
