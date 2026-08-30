"""
create_all_loras.py - Fill loras/Lora001..Lora00N with one adapter each, from one
dataset and one base model, varying a single training parameter across them.

create_lora.py makes one adapter. This makes the set the search actually needs,
and it makes them *comparable*: same data, same base model, same seed, same
everything except the rank. That is what a slot difference then means -- if two
blends score differently, the rank is the only thing that could have caused it.
A folder per adapter, numbered the way the existing five are, so the paths
LORA_SLOTS already holds keep working:

    python create_all_loras.py --dataset poem --count 5
    python create_all_loras.py --dataset poem --values 16 16 8 4 32
    python create_all_loras.py --dataset poem --dry-run

Rank is the parameter with teeth downstream: PEFT's cat sums input ranks, svd
takes the max, and linear refuses inputs whose ranks differ, so the spread of
ranks across the slots is what decides which trees can run at all. Sweeping it
is therefore not just a way to get five different adapters -- it is choosing the
shape of the search space, which is why --values is worth reaching for when a
particular spread is wanted (--values 16 16 8 4 32 rebuilds the existing one).

What varies
    --vary rank            (default) powers of two from --rank-min, doubling.
    --vary learning-rate   log-spaced between --lr-min and --lr-max, every rank
                           left equal at --rank -- which changes what the
                           adapters know while leaving every LIN legal.
    --values ...           either axis, given explicitly; sets --count from its
                           own length.

Each adapter is trained in its own process. That is not the interpreter dodge
full_run.py makes -- sys.executable is this same python either way -- it is
memory: a training run holds the base model, the optimiser and the gradients,
and letting the process exit is the one reliable way to give all of it back
before the next load.

Interpreter: this trains, so it needs the venv one level up --

    D:\\sage-is\\loras\\.venv\\Scripts\\python.exe create_all_loras.py --dataset poem
"""

import argparse
import math
import os
import subprocess
import sys

import create_lora

_HERE = os.path.dirname(os.path.abspath(__file__))

# The subfolder each loras/Lora00N folder keeps its adapter in. This is the name
# the existing five use and the one settings.py's LORA_SLOTS spells out, so
# leaving it alone means the generated scripts need no edit at all.
ADAPTER_NAME = "my_planning_coach-lora_adapter"

# The script this one drives, and the folders it fills: loras/Lora001,
# loras/Lora002, ... One place for the parent, so moving the set again is one
# line here rather than a hunt through the joins below.
CREATE_LORA = os.path.join(_HERE, "create_lora.py")
LORA_DIR = os.path.join(_HERE, "loras")
FOLDER_FORMAT = "Lora%03d"


def learning_rates(count, low, high):
    """`count` learning rates, log-spaced from `low` to `high`.

    Log rather than linear because learning rate is felt multiplicatively: the
    step from 1e-4 to 2e-4 is the same size of change as 2e-4 to 4e-4, and a
    linear spread would put almost every sample at the top of the range.
    """
    if low <= 0 or high <= 0:
        raise SystemExit("learning rates must be positive, got %g and %g" % (low, high))
    if count == 1:
        return [low]
    ratio = math.log(high / low) / (count - 1)
    return [low * math.exp(ratio * i) for i in range(count)]


def ranks(count, smallest):
    """`count` ranks, doubling from `smallest`.

    Doubling rather than stepping, for the same reason the learning rates are
    log-spaced, and because it is what makes the rank rule visible: consecutive
    slots never match, so LIN across them is illegal and the search has to route
    around it, while a slot always matches the cat of the two below it.
    """
    if smallest < 1:
        raise SystemExit("rank must be at least 1, got %d" % smallest)
    return [smallest * 2 ** i for i in range(count)]


def parse_values(raw, vary):
    """The explicit --values list, typed for the axis it is describing."""
    if vary == "rank":
        try:
            values = [int(v) for v in raw]
        except ValueError:
            raise SystemExit("--vary rank needs whole-number --values, got: "
                             + " ".join(raw))
        if any(v < 1 for v in values):
            raise SystemExit("ranks must be at least 1, got: " + " ".join(raw))
        return values
    try:
        values = [float(v) for v in raw]
    except ValueError:
        raise SystemExit("--vary learning-rate needs numeric --values, got: "
                         + " ".join(raw))
    if any(v <= 0 for v in values):
        raise SystemExit("learning rates must be positive, got: " + " ".join(raw))
    return values


def plan(options):
    """The whole batch as a list of (folder, value), before anything is trained."""
    if options.values:
        values = parse_values(options.values, options.vary)
    elif options.vary == "rank":
        values = ranks(options.count, options.rank_min)
    else:
        values = learning_rates(options.count, options.lr_min, options.lr_max)

    return [(os.path.join(LORA_DIR, FOLDER_FORMAT % (options.start + i),
                          options.adapter_name), value)
            for i, value in enumerate(values)]


def check_targets(batch, force):
    """Refuse the whole batch before training any of it.

    create_lora.py guards its own output folder, but on its own that guard fires
    one adapter at a time -- long enough into the batch to have paid for the runs
    before it. Checking every target up front means the answer to "will this
    clobber something" arrives while it is still free.
    """
    occupied = [folder for folder, _ in batch
                if os.path.isdir(folder) and os.listdir(folder)]
    if occupied and not force:
        raise SystemExit(
            "these already hold an adapter:\n  "
            + "\n  ".join(occupied)
            + "\n\nTraining would replace trained weights that nothing else has a "
              "copy of. Pass --force to overwrite them, or --start %d to write "
              "alongside instead." % next_free(len(batch))
        )
    return occupied


def next_free(count):
    """The lowest start number whose whole run of `count` folders is free."""
    start = 1
    while True:
        wanted = [os.path.join(LORA_DIR, FOLDER_FORMAT % (start + i))
                  for i in range(count)]
        if not any(os.path.isdir(w) and os.listdir(w) for w in wanted):
            return start
        start += 1


def command(folder, value, options):
    """The create_lora.py argv for one adapter of the batch."""
    argv = [sys.executable, CREATE_LORA, folder,
            "--dataset", options.dataset,
            "--epochs", str(options.epochs),
            "--alpha", str(options.alpha),
            "--batch-size", str(options.batch_size),
            "--grad-accum", str(options.grad_accum),
            "--max-seq", str(options.max_seq),
            "--base-model", options.base_model,
            # One seed for the whole batch: the swept parameter is meant to be
            # the only difference between two adapters, and a per-adapter seed
            # would quietly be a second one.
            "--seed", str(options.seed)]

    if options.vary == "rank":
        argv += ["--rank", str(value), "--learning-rate", str(options.learning_rate)]
    else:
        argv += ["--rank", str(options.rank), "--learning-rate", repr(value)]

    if options.no_sample:
        argv.append("--no-sample")
    if options.prompt:
        argv += ["--prompt", options.prompt]
    # Always force downstream: check_targets() has already settled the question
    # for the whole batch, and without it a folder cleared by an earlier failure
    # could still trip create_lora.py's own guard.
    argv.append("--force")
    return argv


def describe(value, vary):
    return ("rank %d" % value) if vary == "rank" else ("lr %.3g" % value)


def show_plan(batch, options):
    print("%s, %s, seed %d, %g epochs"
          % (options.dataset, options.base_model, options.seed, options.epochs))
    print("varying %s across %d adapter(s); everything else is held fixed."
          % (options.vary, len(batch)))
    for folder, value in batch:
        print("  %-14s %s" % (describe(value, options.vary),
                              os.path.relpath(folder, _HERE)))
    print("\nEach one loads the base model once, in its own process.")


def train_all(batch, options):
    """Run the batch, returning a result row per adapter."""
    results = []
    for index, (folder, value) in enumerate(batch, start=1):
        label = describe(value, options.vary)
        print("\n" + "=" * 72)
        print("[%d/%d] %s -> %s"
              % (index, len(batch), label, os.path.relpath(folder, _HERE)))
        print("=" * 72, flush=True)

        code = subprocess.run(command(folder, value, options)).returncode
        # A rank is only trustworthy once it has been read back off disk, so a
        # run that exited non-zero reports no rank rather than the one it meant.
        rank = None
        if code == 0:
            try:
                rank = create_lora.rank_of(folder)
            except (OSError, KeyError, ValueError) as error:
                code = code or 1
                print("warning: %s finished but its adapter_config.json is "
                      "unreadable (%s)" % (folder, error))
        results.append((folder, value, code, rank))
    return results


def known_slot(folder):
    """Whether settings.py's LORA_SLOTS already points at `folder`.

    Asked of the setting itself rather than of a copy kept here, so a
    hand-edited LORA_SLOTS is answered by the file that holds it.
    generate_runs.lora_slots() is what resolves the relative entries, which is
    the same resolution a generated script will be built with.
    """
    try:
        import generate_runs
        slots = generate_runs.lora_slots()
    except Exception:
        # Never worth failing a finished batch over: the block is printed either
        # way, and "paste this in" is the safe half of the advice.
        return False
    return any(os.path.normcase(os.path.abspath(path))
               == os.path.normcase(os.path.abspath(folder))
               for path in slots.values())


def report(results, options):
    """The summary, and the LORA_SLOTS block that makes the set usable."""
    print("\n" + "=" * 72)
    print("Done: %d of %d trained."
          % (sum(1 for r in results if r[2] == 0), len(results)))
    print("=" * 72)
    for folder, value, code, rank in results:
        # The rank is read back off disk rather than assumed, so this column is
        # a check: on a rank sweep it either confirms what was asked for or says
        # it did not get it, and on any other sweep it reports what came out.
        if code:
            status = "FAILED, exit %d" % code
        elif options.vary == "rank" and rank != value:
            status = "MISMATCH: asked rank %d, got %d" % (value, rank)
        elif options.vary == "rank":
            status = "ok"
        else:
            status = "ok, rank %d" % rank
        print("  %-14s %-42s %s" % (describe(value, options.vary),
                                    os.path.relpath(folder, _HERE), status))

    good = [(folder, rank) for folder, _, code, rank in results if code == 0]
    if not good:
        return

    if len(good) != len(results):
        # The block below numbers slots over the adapters that exist, so a gap
        # in the middle of the batch pulls the ones after it up a slot. Worth
        # saying out loud rather than letting it read as loras/Lora00N -> LN.
        print("\nNote: %d adapter(s) failed, so the slots below are numbered over "
              "the ones that trained -- a gap shifts everything after it. Re-run "
              "the failures before leaning on this block."
              % (len(results) - len(good)))

    print("\nLORA_SLOTS for settings.py:")
    for index, (folder, rank) in enumerate(good, start=1):
        # Past L5 there is no slot to fill: L1..L5 is the grammar's alphabet
        # (UNARY_OPS in generate_population.py), so a sixth adapter needs the
        # alphabet widened before any tree can name it.
        slot = "L%d" % index if index <= 5 else "L? (no slot: the grammar stops at L5)"
        print("    " + create_lora.slot_line(folder, slot))

    # Whether the block above is what settings.py already says, rather than
    # something to paste over it. Read, not assumed: --start, --adapter-name and
    # a hand-edited LORA_SLOTS can each put the two out of step.
    if all(known_slot(folder) for folder, _ in good):
        print("\nThose are the paths settings.py already holds, so nothing "
              "needs editing -- the generated scripts pick the new adapters up "
              "as they are.")
    else:
        print("\nPaste that into settings.py's LORA_SLOTS, then re-run "
              "`python main.py runs` so the generated scripts pick it up.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a whole set of LoRA adapters into loras/Lora001..Lora00N, "
                    "varying one parameter across them.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
               "  python create_all_loras.py --dataset poem --count 5\n"
               "  python create_all_loras.py --dataset poem --values 16 16 8 4 32\n"
               "  python create_all_loras.py --dataset poem --dry-run",
    )
    parser.add_argument(
        "--dataset", "-d", required=True,
        help="the one dataset every adapter is trained on: a path, or a bare "
             "name looked up under datasets/ (so 'poem' finds "
             "datasets/poem_lora_dataset.json).")
    parser.add_argument(
        "--count", "-n", type=int, default=5,
        help="how many adapters to train (default 5). Ignored when --values is "
             "given, which sets its own length.")
    parser.add_argument(
        "--start", type=int, default=1,
        help="the number the first folder takes (default 1, so loras/Lora001 up). Use "
             "it to write a new set alongside an existing one.")
    parser.add_argument(
        "--vary", choices=("rank", "learning-rate"), default="rank",
        help="which parameter differs between adapters (default rank). Rank is "
             "the one the search feels directly -- it decides which CAT/SVD/LIN "
             "combinations PEFT will allow.")
    parser.add_argument(
        "--values", nargs="+", metavar="V",
        help="the exact values to sweep, instead of a computed spread. Sets the "
             "count from its own length.")
    parser.add_argument(
        "--lr-min", type=float, default=5e-5,
        help="low end of the learning-rate sweep (default 5e-5).")
    parser.add_argument(
        "--lr-max", type=float, default=5e-4,
        help="high end of the learning-rate sweep (default 5e-4). The two "
             "bracket create_lora.py's own 2e-4 default.")
    parser.add_argument(
        "--rank-min", type=int, default=4,
        help="smallest rank of a rank sweep, doubling from there (default 4, "
             "giving 4, 8, 16, 32, 64 for five adapters).")
    parser.add_argument(
        "--rank", "-r", type=int, default=16,
        help="the rank every adapter gets when the sweep is over learning rate "
             "(default 16). Holding it equal makes every LIN legal.")
    parser.add_argument(
        "--learning-rate", type=float, default=2e-4,
        help="the learning rate every adapter gets when the sweep is over rank "
             "(default 2e-4).")
    parser.add_argument(
        "--epochs", type=float, default=20, help="training epochs (default 20).")
    parser.add_argument(
        "--alpha", type=int, default=16, help="lora_alpha (default 16).")
    parser.add_argument(
        "--batch-size", type=int, default=2,
        help="per-device training batch size (default 2).")
    parser.add_argument(
        "--grad-accum", type=int, default=4,
        help="gradient accumulation steps (default 4).")
    parser.add_argument(
        "--max-seq", type=int, default=create_lora.MAX_SEQ,
        help="max sequence length (default %d)." % create_lora.MAX_SEQ)
    parser.add_argument(
        "--seed", type=int, default=3407,
        help="training seed, shared by every adapter in the batch (default 3407) "
             "so the swept parameter is the only difference between them.")
    parser.add_argument(
        "--base-model", default=create_lora.BASE_MODEL,
        help="the one base model every adapter is trained on (default %s)."
             % create_lora.BASE_MODEL)
    parser.add_argument(
        "--adapter-name", default=ADAPTER_NAME,
        help="subfolder each adapter is written to inside its loras/Lora00N folder "
             "(default %s, which is the path template_code.py already expects)."
             % ADAPTER_NAME)
    parser.add_argument(
        "--prompt", default=None,
        help="prompt each adapter answers once after training. Handy here: the "
             "same question across the set is the quickest read on whether the "
             "sweep changed anything.")
    parser.add_argument(
        "--no-sample", action="store_true",
        help="skip those samples.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the plan and stop. Worth doing first -- a full batch is "
             "several hours of GPU time.")
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite target folders that already hold an adapter.")
    return parser.parse_args(argv)


def main(argv=None):
    options = parse_args(argv)
    if options.count < 1:
        raise SystemExit("--count must be at least 1, got %d" % options.count)
    if options.start < 1:
        raise SystemExit("--start must be at least 1, got %d" % options.start)

    batch = plan(options)
    # Resolved once, here, so all N runs are provably reading the same file and
    # a typo costs nothing instead of failing on the first training run.
    options.dataset = create_lora.resolve_dataset(options.dataset)

    show_plan(batch, options)

    if options.dry_run:
        print("\n--dry-run: nothing trained.")
        return 0

    create_lora.check_interpreter()
    occupied = check_targets(batch, options.force)
    if occupied:
        print("\n--force: overwriting %d existing adapter(s)." % len(occupied))

    results = train_all(batch, options)
    report(results, options)

    # Unlike the pipeline, where a failed individual is a result, a failed
    # adapter here is a missing *input* -- nothing downstream can score its
    # absence -- so any failure at all is worth a non-zero exit.
    return 1 if any(code for _, _, code, _ in results) else 0


if __name__ == "__main__":
    sys.exit(main())
