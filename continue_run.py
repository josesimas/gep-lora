"""More generations of a stored sweep: gep_lora/core/pipeline/continue_run.py,
run from here.

    python continue_run.py --generations 3 ...

Kept at the top so the command every document gives still works; the driver
itself is gep_lora.core.pipeline.continue_run.
"""

import sys

from gep_lora.core.pipeline.continue_run import cli

if __name__ == "__main__":
    sys.exit(cli())
