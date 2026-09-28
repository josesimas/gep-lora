"""One generation of a sweep: gep_lora/core/pipeline/start_run.py, run from here.

    python start_run.py [steps ...] [--list] ...

Kept at the top so the command every document gives still works; the driver
itself is gep_lora.core.pipeline.start_run.
"""

import sys

from gep_lora.core.pipeline.start_run import main

if __name__ == "__main__":
    sys.exit(main())
