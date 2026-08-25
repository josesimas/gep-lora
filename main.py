"""
main.py - The entry point. Runs the pipeline in one of two modes.

    population -> trees -> runs -> process -> evaluate

The five steps are the same either way; what differs is where a sweep is kept.

    txt      main_txt.py     everything lands in run/ as text files:
                             population.txt, trees.txt, index.txt, run_NNN.py,
                             output_NNN.txt, output_result_NNN.json, results.txt.
                             Easy to open and diff, which is what you want while
                             changing the tree code or a template. The next
                             sweep overwrites the last one.

    sqlite   main_sqlite.py  everything lands in one database (see store.py):
                             the population, every setting, every seed, every
                             transcript and every score, with the sweep itself
                             as a row so sweeps accumulate instead of replacing
                             each other, and can be queried across. Because the
                             seeds are stored -- the population's, and one per
                             individual for its blend weights -- a stored sweep
                             can be repeated. That is what you want when the
                             numbers are meant to be kept.

Pick with --mode, or set MODE below and just run `python main.py`. Everything
else on the command line is passed through to the mode, so the step names and
its own options work exactly as they would if you ran that file directly:

    python main.py                              # every step, in the default mode
    python main.py --list                       # the steps of that mode
    python main.py population trees runs        # the fast half, before a model load
    python main.py --mode sqlite                # a new sweep, into the database
    python main.py --mode sqlite process evaluate   # resume the latest sweep
    python main.py --mode sqlite --limit 3      # sqlite-only option, passed through
    python main.py --mode txt --help            # the mode's own help

Both modes read their settings from settings.py -- COUNT, SEED, TEMPLATE and the
rest -- so the two cannot drift apart.

Run it with the **venv's python**: `process` launches each generated script with
sys.executable, so the wrong interpreter fails every individual. That does not
apply to a sweep generated from template_code_mocked.py, which loads nothing.
"""

import argparse
import sys

import main_sqlite
import main_txt

# --- which mode ------------------------------------------------------------

# The mode used when --mode is not given. "txt" for the file-per-thing layout
# while you are iterating; "sqlite" for a sweep whose results are meant to be
# kept, compared and repeated.
MODE = "txt"

MODES = {
    "txt": (main_txt.main,
            "text files in run/ -- easy to read, overwritten by the next sweep"),
    "sqlite": (main_sqlite.main,
               "one sqlite database -- queryable, keeps every sweep and its seeds"),
}


def main(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)

    # Only --mode is claimed here; everything else belongs to the mode, which
    # has its own parser, its own steps and its own options. parse_known_args
    # keeps this file from having to know any of them.
    parser = argparse.ArgumentParser(
        description=__doc__.strip().splitlines()[0],
        epilog="every other argument is passed straight to the chosen mode; "
               "try `python main.py --mode sqlite --help`",
        add_help=False,
    )
    parser.add_argument("--mode", choices=sorted(MODES), default=MODE,
                        help="where the sweep is kept (default %s)" % MODE)
    parser.add_argument("--modes", action="store_true",
                        help="describe the modes and exit")
    parser.add_argument("-h", "--help", action="store_true",
                        help="show this message, or the mode's own help with --mode")
    args, rest = parser.parse_known_args(raw)

    if args.modes:
        for name, (_, description) in sorted(MODES.items()):
            print("%-8s %s%s" % (name, description,
                                 "   (default)" if name == MODE else ""))
        return 0

    # `--help` on its own explains the dispatcher. Asked together with a mode,
    # or with anything else, it is that mode's help that is wanted.
    if args.help:
        named_mode = any(one == "--mode" or one.startswith("--mode=") for one in raw)
        if not named_mode and not rest:
            parser.print_help()
            return 0
        rest.append("--help")

    return MODES[args.mode][0](rest)


if __name__ == "__main__":
    sys.exit(main())
