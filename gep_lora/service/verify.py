"""
verify.py - One blend of a finished job, run beside the LoRAs it is made of.

The API's half of `gep_lora/core/testing/evaluate_chromosome_against_loras.py`: what a form
needs to offer (`choices`), what a request may ask for (`options_for`), the
command the worker runs (`command`), and the report that comes back
(`report`). The comparison itself lives in that script and is not duplicated
here -- a verification is that command line, queued.

A verification is **work, like a job**: it loads a base model once per
contestant, so it cannot happen inside a request. It goes into the registry's
own `verifications` table and the same worker takes it, after the jobs. What it
writes goes in `verify<id>/` inside the job's folder, so deleting the run takes
its verifications with it -- a reading of a sweep is worth nothing without the
sweep.

**The default contestants are the LoRAs the chromosome actually names**
(`blend_slots`), not every slot the sweep holds: the question a verification
answers is whether folding *these* adapters together beat using one of them, and
a slot the blend never mentions is not part of that. `slots: "all"` asks for
every slot anyway, which is the other question worth asking.

**The questions are the sweep's own by default** -- one of the splits it
holds, testing first, since the blend was selected on training -- or another
dataset (`dataset`): a shared file, or the data one of the user's own LoRAs was
trained on, or a file the page uploaded (`upload`). The server resolves which
file that is (it knows whose LoRA is whose); this module only takes the path it
is handed and passes it to the script's --dataset, which asks those questions
instead of a split.
"""

# Where uploaded question files go, inside the job's folder: a reading of the
# sweep made on them is worth nothing without the sweep, so they go with it.
UPLOADS = "verify_datasets"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

import glob
import hashlib
import json
import os
import re

from gep_lora.core import evaluators
from gep_lora.core.blends import generate_runs
from gep_lora.core.search.generate_population import UNARY_OPS, VARIABLES, decode, slot_key
from gep_lora.core.storage import store
from gep_lora.core.testing import evaluate_chromosome_against_loras as compare


class VerifyError(ValueError):
    """What is wrong with a verification request, in words for whoever sent it."""


def blend_slots(chromosome):
    """The slots one chromosome names, in slot order. -> ["L2", "L5"].

    Read off the decoded tree rather than by splitting the string, so it is the
    same parse the build plan is made from -- and `L1` cannot be found inside
    some other symbol.
    """
    try:
        root, _ = decode(chromosome)
    except ValueError as error:
        raise VerifyError("cannot read the chromosome %r: %s" % (chromosome, error))
    found, frontier = set(), [root]
    while frontier:
        node = frontier.pop()
        if node.symbol not in VARIABLES and node.children:
            frontier.extend(node.children)
        if node.symbol in UNARY_OPS:              # L1..L10, never LIN
            found.add(node.symbol)
    return sorted(found, key=slot_key)


def choices(db_path, run_id):
    """What a verification form needs, out of the job's own sweep.

    Every individual that could be verified (with the slots its chromosome
    names, so the page can say what it would be compared against), the splits
    the sweep holds, the evaluators registered here, and the sweep's own
    evaluator and judge as the defaults -- a verification graded by the sweep's
    rubric is the one comparable with the fitness the search produced.
    """
    conn = store.connect(db_path)
    try:
        conf = store.get_settings(conn, run_id)
        individuals = []
        quality = {row["number"]: row for row in store.quality_rows(conn, run_id)}
        for row in store.individuals(conn, run_id):
            view = quality.get(row["number"])
            individuals.append({
                "number": row["number"], "chromosome": row["chromosome"],
                "state": row["state"], "fitness": row["fitness"],
                "is_best": bool(row["is_best"]),
                "quality": view["quality"] if view else None,
                "answers": view["answers"] if view else 0,
                "slots": blend_slots(row["chromosome"]) if row["chromosome"] else []})
        splits = sorted({entry["split"] for entry in store.dataset_summary(conn, run_id)}
                        & set(compare.SPLIT_COUNTS))
        return {
            "individuals": individuals,
            "slots": sorted(generate_runs.lora_slots(conf.get("LORA_SLOTS")), key=slot_key),
            "splits": splits,
            "evaluators": [{"value": name, "label": name, "description": description,
                            "needs_judge": evaluators.get(name).needs_judge}
                           for name, description in evaluators.available()],
            "backends": list(evaluators.BACKENDS),
            "defaults": {"evaluator": conf.get("EVALUATOR"),
                         "judge_model": conf.get("JUDGE_MODEL"),
                         "judge_backend": conf.get("JUDGE_BACKEND"),
                         "judge_base_url": conf.get("JUDGE_BASE_URL"),
                         "split": "testing" if "testing" in splits else (
                             splits[0] if splits else None),
                         "count": conf.get("TESTING_COUNT")},
        }
    finally:
        conn.close()


def _upload_lines(text):
    """An uploaded file's text as dataset lines: JSON Lines, plain prompts one
    per line, or a whole-file JSON array, which is turned into JSON Lines here
    because the scripts read a line at a time. -> [str]."""
    stripped = text.strip()
    if stripped.startswith("["):
        try:
            records = json.loads(stripped)
        except ValueError:
            records = None
        if isinstance(records, list):
            lines = []
            for number, record in enumerate(records, 1):
                if isinstance(record, dict):
                    lines.append(json.dumps(record, ensure_ascii=False))
                elif isinstance(record, str) and record.strip() and "\n" not in record:
                    lines.append(record.strip())
                else:
                    raise VerifyError("record %d: a record is an object (with a "
                                      "'messages' list) or a one-line prompt" % number)
            return lines
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    for number, line in enumerate(lines, 1):
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except ValueError:
            raise VerifyError("line %d starts like JSON but is not valid JSON" % number)
        messages = record.get("messages") if isinstance(record, dict) else None
        if not isinstance(messages, list) or not any(
                isinstance(turn, dict) and turn.get("role") == "user" for turn in messages):
            raise VerifyError("line %d: a JSON record needs a 'messages' list with a "
                              "user turn -- that turn is the question" % number)
    return lines


def upload(job_folder, text, name=None):
    """Store a question file the page sent, for a verification. -> (path, label).

    Written into the job's own folder under a name made from its content, so
    the same file uploaded twice is one file, and deleting the run takes it.
    """
    if not isinstance(text, str) or not text.strip():
        raise VerifyError("the uploaded dataset is empty")
    if len(text.encode("utf-8")) > MAX_UPLOAD_BYTES:
        raise VerifyError("the uploaded dataset is over %d MB"
                          % (MAX_UPLOAD_BYTES // (1024 * 1024)))
    if name is not None and not isinstance(name, str):
        raise VerifyError("the uploaded dataset's name must be a string")
    lines = _upload_lines(text)
    if not lines:
        raise VerifyError("the uploaded dataset has no questions in it")
    body = "\n".join(lines) + "\n"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.splitext(
        os.path.basename(name or ""))[0]).strip("._")[:60] or "uploaded"
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()[:12]
    ext = ".jsonl" if lines[0].startswith("{") else ".txt"
    folder = os.path.join(job_folder, UPLOADS)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "%s_%s%s" % (stem, digest, ext))
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(body)
    label = "%s (uploaded, %d questions)" % (os.path.basename(name) if name else "a file",
                                             len(lines))
    return path, label


def _positive(body, name):
    value = body.get(name)
    if value in (None, ""):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or value != int(value) or value < 1:
        raise VerifyError("%s must be a whole number of at least 1, or left out "
                          "for all of them" % name)
    return int(value)


def _text(body, name):
    value = body.get(name)
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise VerifyError("%s must be a string" % name)
    return value.strip() or None


def options_for(body, offered, dataset=None):
    """A request, checked against what this sweep offers. -> (number, options).

    Checked here rather than by the script, for the reason a submission is
    checked in submit.py: a bad request should be a 400 now, not a task that
    fails on the worker in ten minutes' time.

    `dataset` turns the request's `dataset` field into (path, label), raising
    VerifyError for one it cannot -- the server's, since whose file is whose
    is the server's to know. Without it, a request naming a dataset is refused.
    """
    body = body or {}
    if not isinstance(body, dict):
        raise VerifyError("the body must be an object")

    runnable = {one["number"]: one for one in offered["individuals"]
                if one["state"] != "BAD"}
    number = body.get("individual")
    if number is None:
        best = [one for one in offered["individuals"] if one["is_best"]] or sorted(
            runnable.values(), key=lambda one: (-(one["fitness"] or 0), one["number"]))
        if not best:
            raise VerifyError("this job has no individual that can be built")
        number = best[0]["number"]
    if not isinstance(number, int) or isinstance(number, bool):
        raise VerifyError("individual must be an individual's number")
    if number not in runnable:
        raise VerifyError("no individual %d worth verifying in this job (a BAD one "
                          "cannot be built)" % number)
    blend = runnable[number]

    # Named only when asked for. Left out, the script grades by the sweep's own
    # EVALUATOR -- the same default -- except for a mocked sweep, whose answers
    # it scores with the numbers its scripts printed: a judge asked about
    # invented answers would only cost time (or wait on an endpoint that is
    # not there).
    name = _text(body, "evaluator")
    if name and name not in {one["value"] for one in offered["evaluators"]}:
        raise VerifyError("unknown evaluator %r; there are: %s"
                          % (name, ", ".join(one["value"] for one in offered["evaluators"])))
    backend = _text(body, "judge_backend")
    if backend and backend not in offered["backends"]:
        raise VerifyError("unknown judge backend %r; there are: %s"
                          % (backend, ", ".join(offered["backends"])))
    split = _text(body, "split")
    path = label = None
    if body.get("dataset") not in (None, ""):
        if split:
            raise VerifyError("ask either a split of the sweep or another dataset, not both")
        if dataset is None:
            raise VerifyError("this server takes no dataset for a verification")
        path, label = dataset(body["dataset"])
    elif split and split not in offered["splits"]:
        raise VerifyError("this job holds no %s split; it has: %s"
                          % (split, ", ".join(offered["splits"]) or "none"))

    wanted = body.get("slots")
    if wanted in (None, "", "blend"):
        slots = blend["slots"]
    elif wanted == "all":
        slots = offered["slots"]
    elif isinstance(wanted, list) and all(isinstance(one, str) for one in wanted):
        slots = [one.strip() for one in wanted if one.strip()]
    else:
        raise VerifyError('slots must be "blend", "all", or a list of slot names')
    unknown = [one for one in slots if one not in offered["slots"]]
    if unknown:
        raise VerifyError("no slot %s in this job's LORA_SLOTS (%s)"
                          % (", ".join(unknown), ", ".join(offered["slots"])))
    if not slots:
        raise VerifyError("there is nothing to compare individual %d against: its "
                          "chromosome names no adapter" % number)

    return number, {"evaluator": name, "judge_model": _text(body, "judge_model"),
                    "judge_backend": backend,
                    "judge_base_url": _text(body, "judge_base_url"),
                    "split": None if path else split or offered["defaults"]["split"],
                    "dataset": path, "dataset_label": label,
                    "count": _positive(body, "count"),
                    "slots": slots,
                    "timeout": _positive(body, "timeout") or 900}


def command(python, db_path, run_id, number, options, folder, script=None):
    """The evaluate_chromosome_against_loras.py command line for one verification.

    Run as a module from the repo root, the way every CLI below the top level
    is, and with --into pointing inside the job's folder so the answers and the
    report land where the API can find them again.
    """
    argv = [python, "-u", "-m", script or compare.__name__,
            "--db", db_path, "--run", str(run_id), "--individual", str(number),
            "--into", folder, "--timeout", str(options.get("timeout") or 900)]
    for flag, name in (("--evaluator", "evaluator"), ("--judge-model", "judge_model"),
                       ("--judge-backend", "judge_backend"),
                       ("--judge-base-url", "judge_base_url"), ("--split", "split")):
        if options.get(name):
            argv += [flag, str(options[name])]
    if options.get("dataset"):
        argv += ["--dataset", options["dataset"]]
    if options.get("count"):
        argv += ["--count", str(options["count"])]
    if options.get("slots"):
        argv += ["--slots", ",".join(options["slots"])]
    return argv


def report(folder):
    """The newest scores file a verification wrote, as JSON. -> dict or None.

    The script names it for the evaluator and the judge, and writes one per
    grading; a verification runs one grading, so the newest is its own. None
    means it has not got that far -- queued, running, or failed before scoring.
    """
    found = sorted(glob.glob(os.path.join(folder, "scores_*.json")),
                   key=os.path.getmtime, reverse=True)
    if not found:
        return None
    try:
        with open(found[0], encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None
