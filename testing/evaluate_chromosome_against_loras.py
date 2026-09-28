"""
evaluate_chromosome_against_loras.py - Is the blend a search found better than
the adapters it was blended from?

    python -m testing.evaluate_chromosome_against_loras --db run_db/gep.sqlite3
    python -m testing.evaluate_chromosome_against_loras --db api_jobs/user1/job6/job.sqlite3 \\
        --judge-model qwen3-32b --count 20

A sweep scores every individual, but only ever against the *bare* base model
(llm_judge_baseline) or against the dataset's own answers -- never against the
obvious control, each LoRA applied on its own. So a blend at 0.72 says nothing
about whether L3 alone would have scored 0.72 too, and the search could have
been skipped. This puts them side by side:

  1. takes the sweep's best individual (`is_best`, else the highest fitness;
     --individual picks another);
  2. builds one script for it and one per adapter in the sweep's LORA_SLOTS,
     each adapter attached **alone, at full strength** -- the way it was
     trained, with no weight drawn for it, since a weight only ever enters a
     tree through the combine() that folds it into something else;
  3. runs them on the same questions -- the sweep's stored testing split by
     default, since the blend was selected on the training one -- through the
     same launcher, batches and lora_server pool the process step uses;
  4. grades every answer with one evaluator -- the sweep's own EVALUATOR unless
     --evaluator says otherwise, and its judge unless --judge-model,
     --judge-backend or --judge-base-url say otherwise;
  5. prints each contestant's mean and, per adapter, how many questions the
     blend won, tied and lost against it, with a two-sided sign test on the
     ones that were not ties.

**Every contestant is rendered from the same template, the same settings and
the same questions**, the blend included: the comparison is between adapters,
and re-pointing the blend's stored script while rendering the adapters fresh
would compare template versions as well. The blend keeps its own weight seed,
so it is the blend the search scored. --stored-script runs the individual's
stored script instead (re-pointed, as the testing pass does), for when the
exact script that earned the score matters more than the controls matching it.

It writes nothing to the sweep. Everything goes into one folder beside the
database (`<db>_run<N>/lora_comparison/`, or --into): the questions it asked,
`answers.json` -- every contestant's transcript, keyed on a hash of the script
that produced it -- and one `scores_<evaluator>_<judge>.json` plus a `.md`
report per grading. The answers are the expensive half, so a second run with
another judge reuses them and only grades; a changed script, split or count
changes the hash and runs again, and --rerun forces it.

Two things this deliberately does not do. It does not abandon a contestant
whose first answers score 0, as the evaluate step does -- a comparison wants
every question graded on every side, or the means stop sharing a denominator.
And a mocked sweep's answers arrive pre-scored, as they do everywhere else; they
are used as they are unless an evaluator or judge is named, and a mocked score
is noise, never a result.
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import threading
import time

from blends import generate_runs
from blends import process_run
from blends import server_pool
from config import settings as config
import evaluators
from evaluators import common
from search.generate_population import Node, decode
from storage import add_dataset
from storage import db_datasets
from storage import store
from testing import test_run_with_dataset

# The repo folder, one above this one: paths given on the command line fall
# back to it, the way every other module resolves a setting.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FOLDER = "lora_comparison"
ANSWERS = "answers.json"

# Which setting caps each split, read from the sweep. Validation has none.
SPLIT_COUNTS = {"training": "TRAINING_COUNT", "testing": "TESTING_COUNT",
                "validation": None}

# The weight symbol a lone adapter's leaf carries. plan() needs a child to name
# the weight, and nothing ever reads it: a leaf that is the final adapter is
# never combined, so no weight is applied to it.
_UNUSED_WEIGHT = "w1"


class Contestant:
    """One script in the comparison: the blend, or one adapter on its own."""

    __slots__ = ("key", "label", "expression", "script_name", "source")

    def __init__(self, key, label, expression, script_name, source):
        self.key = key                  # "blend" or the slot, e.g. "L3"
        self.label = label
        self.expression = expression
        self.script_name = script_name
        self.source = source

    @property
    def digest(self):
        return hashlib.sha256(self.source.encode("utf-8")).hexdigest()


# --- what is being compared --------------------------------------------------


def best_individual(conn, run_id, number=None):
    """The individual to put against the adapters. -> a row with `quality`.

    `is_best` first, since that is what elitism named; then the stored fitness,
    then the mean quality over its latest execution, for a sweep whose fitness
    or elitism step never ran. Ties go to the lowest number, the rule elitism
    breaks them by.
    """
    sql = ("SELECT i.*, q.quality AS quality"
           "  FROM individuals i"
           "  LEFT JOIN individual_quality q"
           "    ON q.run_id = i.run_id AND q.number = i.number"
           " WHERE i.run_id = ?")
    args = [run_id]
    if number is not None:
        sql += " AND i.number = ?"
        args.append(number)
    sql += (" ORDER BY i.is_best DESC, COALESCE(i.fitness, 0) DESC,"
            " COALESCE(q.quality, 0) DESC, i.number LIMIT 1")
    row = conn.execute(sql, args).fetchone()
    if row is None:
        raise SystemExit("run %d holds %s" % (
            run_id, "no individual %d" % number if number is not None
            else "no individuals yet"))
    if row["state"] == "BAD":
        raise SystemExit("individual %d (%s) is BAD -- PEFT cannot build it, so "
                         "there is nothing to compare. Pick another with "
                         "--individual." % (row["number"], row["chromosome"]))
    return row


def question_setup(conn, run_id, conf, args, folder, say=print):
    """(dataset path, split name, count) for the questions every contestant asks.

    --dataset names a file; otherwise the split comes out of the sweep's own
    rows, written into the comparison folder. The default split is testing when
    the sweep holds one -- the blend was selected on training, so those are the
    questions it has an unfair advantage on -- and training when it does not.
    """
    held = {entry["split"] for entry in store.dataset_summary(conn, run_id)}
    if args.dataset:
        path, split = add_dataset.resolve_dataset(args.dataset), None
    else:
        split = args.split or ("testing" if "testing" in held else "training")
        if split not in held:
            raise SystemExit(
                "run %d holds no %s split. Name a file with --dataset, or pick "
                "one it does hold: %s" % (run_id, split,
                                          ", ".join(sorted(held)) or "none"))
        path, _ = db_datasets.export(conn, run_id, split, folder)

    if args.count is not None:
        count = args.count or None
    elif split and SPLIT_COUNTS[split] and SPLIT_COUNTS[split] in conf:
        count = conf[SPLIT_COUNTS[split]]
    else:
        count = None
    if count is not None:
        count = test_run_with_dataset.testing_count(count)
    say("questions: %s%s, %s" % (
        path, "" if split is None else " (the sweep's %s split)" % split,
        "all of them" if count is None else "the first %d" % count))
    return path, split, count


def contestants(row, conf, slots, dataset, count, stored_script=False):
    """The blend and one lone adapter per slot, as rendered scripts."""
    template = generate_runs.template_path(conf.get("TEMPLATE"))
    lines = generate_runs.load_template(template)
    all_slots = conf.get("LORA_SLOTS")
    ranks = generate_runs.slot_ranks(all_slots)
    shared = dict(template_lines=lines, weight_seed=row["weight_seed"],
                  weight_values=store.weight_values(row),
                  training_set=dataset, slots=all_slots, count=count,
                  base_model=conf.get("BASE_MODEL"),
                  chat_template=generate_runs.chat_template_name(conf))

    if stored_script:
        if not row["script_source"]:
            raise SystemExit("individual %d has no stored script to run; drop "
                             "--stored-script" % row["number"])
        blend_source = test_run_with_dataset.repoint(row["script_source"],
                                                     dataset, count)
        expression = (test_run_with_dataset.script_chromosome(row["script_source"])
                      or row["chromosome"])
    else:
        expression = row["chromosome"]
        steps, final = generate_runs.plan(decode(expression)[0], ranks)
        blend_source = generate_runs.render(
            expression, steps, final, "blend_%03d.py" % row["number"],
            "Generated by evaluate_chromosome_against_loras.py: the blend.",
            "Individual %d" % row["number"], **shared)
    out = [Contestant("blend", "individual %d" % row["number"], expression,
                      "blend_%03d.py" % row["number"], blend_source)]

    for slot in slots:
        leaf = Node(slot)
        leaf.children.append(Node(_UNUSED_WEIGHT))
        steps, final = generate_runs.plan(leaf, ranks)
        source = generate_runs.render(
            slot, steps, final, "lora_%s.py" % slot,
            "Generated by evaluate_chromosome_against_loras.py: %s on its own, "
            "at full strength." % slot,
            "%s alone" % slot, root=leaf, **shared)
        out.append(Contestant(slot, "%s alone" % slot, slot,
                              "lora_%s.py" % slot, source))
    return out


# --- running them --------------------------------------------------------------


def load_answers(folder):
    try:
        with open(os.path.join(folder, ANSWERS), encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def save_answers(folder, answers):
    path = os.path.join(folder, ANSWERS)
    with open(path + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(answers, handle, indent=1, ensure_ascii=False)
    os.replace(path + ".tmp", path)


def run_scripts(todo, folder, conf, answers, prompts, timeout, keep_scripts,
                say=print):
    """Run the contestants whose answers are missing or stale, into `answers`.

    The testing pass's loop, keyed on contestants rather than individuals: the
    sweep's batch size, its lora_server pool when the scripts are clients, the
    same streaming progress, and a crash recorded as a result. Saved after each
    batch, so an interrupted run keeps what came back.
    """
    if process_run.imports_unsloth(todo[0].source) or server_pool.wanted(todo[0].source):
        process_run.check_interpreter()
    for contestant in todo:
        with open(os.path.join(folder, contestant.script_name), "w",
                  encoding="utf-8") as handle:
            handle.write(contestant.source)

    size = process_run.batch_size(
        conf.get("PROCESS_RUN_BATCH_SIZE", config.PROCESS_RUN_BATCH_SIZE))
    pool = server_pool.pool_for(
        conf, todo[0].source, generate_runs.base_model_name(conf.get("BASE_MODEL")),
        folder, config, say=say)
    if pool is not None and size > pool.size:
        size = pool.size
    every = conf.get("PROCESS_RUN_PROGRESS_SECONDS",
                     config.PROCESS_RUN_PROGRESS_SECONDS)
    console = threading.Lock()

    def progress(script, message):
        with console:
            say("        %s%s" % ("" if size == 1 else script + "  ", message))

    def watch(script):
        return process_run.Progress(script, prompts, every, progress)

    try:
        for group in process_run.batches(todo, size):
            for contestant in group:
                say("running %-14s %s" % (contestant.label, contestant.expression))
            envs = None if pool is None else pool.envs_for(len(group))
            results = process_run.launch_batch(
                folder, [c.script_name for c in group], timeout, watch, envs)
            for contestant, (code, seconds, out, err) in zip(group, results):
                transcript = process_run.exchanges(out)
                answers[contestant.key] = {
                    "label": contestant.label, "expression": contestant.expression,
                    "digest": contestant.digest, "exit_code": code,
                    "verdict": process_run.verdict_of(code),
                    "seconds": round(seconds, 1), "transcript": transcript,
                    "stderr_tail": [line for line in err.splitlines()
                                    if line.strip()][-5:] if code else []}
                say("        %-14s %-9s %6.1fs  %d answer(s)"
                    % (contestant.label, process_run.verdict_of(code), seconds,
                       len(transcript)))
                if code != 0:
                    tail = [line for line in (out + err).splitlines() if line.strip()][-1:]
                    if tail:
                        say("        %s" % tail[0][:100])
            save_answers(folder, answers)
            if pool is not None:
                pool.maintain()
    finally:
        if pool is not None:
            pool.stop()
        if not keep_scripts:
            for contestant in todo:
                try:
                    os.remove(os.path.join(folder, contestant.script_name))
                except OSError:
                    pass


# --- grading them --------------------------------------------------------------


class _Options:
    """What llm_judge_baseline reads off a step's options."""

    def __init__(self, keep_scripts):
        self.keep_scripts = keep_scripts


def judged_conf(conf, dataset, count, args):
    """The sweep's settings as the evaluator should read them.

    The eval set pointed at the questions actually asked -- the substitution the
    testing pass makes, for the reason it makes it: the reference-reading
    evaluators would otherwise grade these answers against another file's --
    and the judge swapped for whichever one the command line named.
    """
    graded = test_run_with_dataset.testing_conf(conf, dataset, count)
    for name, value in (("JUDGE_MODEL", args.judge_model),
                        ("JUDGE_BACKEND", args.judge_backend),
                        ("JUDGE_BASE_URL", args.judge_base_url)):
        if value is not None:
            graded[name] = value
    return graded


def grade(entries, graded, name, context, say=print):
    """Score every answer. -> ({key: [quality or None per answer]}, evaluator, label, reasons).

    One evaluator over every contestant's answers, prepared once so all of them
    are judged by the same judge in the same state. A blank answer is 0.0 without
    asking; a failed grading call is None -- neither a zero nor a score -- and
    drops that question from the pairwise counts of whoever it belongs to.
    """
    evaluator = evaluators.get(name)
    if evaluator.check is not None:
        evaluator.check(graded)
    items = [{"question": item.get("question", ""), "answer": item.get("answer", ""),
              "position": position, "number": 0}
             for entry in entries.values()
             for position, item in enumerate(entry["transcript"], 1)]
    say("\nevaluator: %s -- %s" % (evaluator.name, evaluator.description))
    try:
        prepared = evaluator.prepare(graded, items, context)
        for note in prepared.notes:
            say(note)
        scores, reasons = {}, {}
        for key, entry in entries.items():
            say("grading %s" % entry["label"])
            scores[key], reasons[key] = [], []
            for position, item in enumerate(entry["transcript"], 1):
                answer = item.get("answer") or ""
                if not answer.strip():
                    quality, reason = 0.0, "no answer given"
                else:
                    try:
                        quality, reason = evaluator.score(
                            {"question": item.get("question", ""), "answer": answer,
                             "position": position, "number": 0}, prepared)
                        quality = round(quality, 3)
                    except (RuntimeError, ValueError) as error:
                        quality, reason = None, "FAILED: %s" % error
                say("    [%d] %s  %s" % (position, "  -- " if quality is None
                                         else "%.2f" % quality, reason))
                scores[key].append(quality)
                reasons[key].append(reason)
        return scores, evaluator.name, prepared.label, reasons
    finally:
        evaluators.release_models()


def generated_scores(entries):
    """The scores a mocked script printed beside its answers, or None if any lack one."""
    scores, reasons = {}, {}
    for key, entry in entries.items():
        transcript = entry["transcript"]
        if not transcript or any(item.get("quality") is None for item in transcript):
            return None
        scores[key] = [item["quality"] for item in transcript]
        reasons[key] = [item.get("reason", "") for item in transcript]
    return scores, reasons


# --- saying what it found ------------------------------------------------------


def sign_test(wins, losses):
    """Two-sided exact sign test over the questions that were not ties."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def mean(values):
    kept = [value for value in values if value is not None]
    return sum(kept) / len(kept) if kept else None


def compare(scores, questions):
    """The blend against each adapter, question by question. -> rows."""
    blend = scores["blend"]
    rows = []
    for key, theirs in scores.items():
        row = {"key": key, "mean": mean(theirs),
               "graded": sum(value is not None for value in theirs),
               "failed": sum(value is None for value in theirs),
               "missing": max(0, questions - len(theirs))}
        if key != "blend":
            wins = ties = losses = 0
            deltas = []
            for position in range(questions):
                ours = blend[position] if position < len(blend) else None
                them = theirs[position] if position < len(theirs) else None
                if ours is None or them is None:
                    continue
                deltas.append(ours - them)
                if ours > them:
                    wins += 1
                elif ours < them:
                    losses += 1
                else:
                    ties += 1
            row.update(wins=wins, ties=ties, losses=losses, delta=mean(deltas),
                       p=sign_test(wins, losses))
        rows.append(row)
    return rows


def report_lines(meta, entries, rows):
    """The comparison as Markdown -- printed as it is, and written beside the scores."""
    lines = [
        "# %s against each LoRA alone" % entries["blend"]["label"],
        "",
        "- sweep: run %d in `%s`" % (meta["run"], meta["db"]),
        "- blend: `%s` (training quality %s)" % (
            entries["blend"]["expression"],
            "n/a" if meta["training_quality"] is None else "%.3f" % meta["training_quality"]),
        "- questions: %d from `%s`" % (meta["questions"], meta["dataset"]),
        "- graded by: %s (%s)" % (meta["evaluator"], meta["judge"]),
        "",
        "| contestant | mean | graded | blend wins | ties | blend loses | blend - it | sign test p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in sorted(rows, key=lambda r: (r["key"] != "blend", -(r["mean"] or 0))):
        entry = entries[row["key"]]
        name = "**%s** `%s`" % (entry["label"], entry["expression"]) \
            if row["key"] == "blend" else entry["label"]
        if entry["verdict"] != "ok":
            name += " (%s)" % entry["verdict"]
        cells = [name, "-" if row["mean"] is None else "%.3f" % row["mean"],
                 "%d/%d" % (row["graded"], meta["questions"])]
        if row["key"] == "blend":
            cells += ["", "", "", "", ""]
        else:
            cells += [str(row["wins"]), str(row["ties"]), str(row["losses"]),
                      "-" if row["delta"] is None else "%+.3f" % row["delta"],
                      "%.3f" % row["p"]]
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    better = [row for row in rows if row["key"] != "blend"
              and row["wins"] > row["losses"] and row["p"] < 0.05]
    worse = [row for row in rows if row["key"] != "blend"
             and row["losses"] > row["wins"] and row["p"] < 0.05]
    others = [row for row in rows if row["key"] != "blend"
              and row not in better and row not in worse]
    if better:
        lines.append("The blend is significantly better (p < 0.05) than: %s."
                     % ", ".join(entries[r["key"]]["label"] for r in better))
    if worse:
        lines.append("The blend is significantly worse (p < 0.05) than: %s."
                     % ", ".join(entries[r["key"]]["label"] for r in worse))
    if others:
        lines.append("No significant difference from: %s."
                     % ", ".join(entries[r["key"]]["label"] for r in others))
    return lines


def _safe(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-")[:60] or "judge"


# --- the command line ----------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare a sweep's best individual with each of its LoRAs "
                    "applied on its own.")
    parser.add_argument("--db", default=None,
                        help="database file (default: settings.DB_PATH)")
    parser.add_argument("--run", type=int, default=0, metavar="RUN",
                        help="the sweep (0 = the most recent, default)")
    parser.add_argument("--individual", type=int, default=None, metavar="N",
                        help="compare this individual instead of the best")
    parser.add_argument("--slots", default=None, metavar="L1,L3",
                        help="only these adapters (default: every slot in the "
                             "sweep's LORA_SLOTS)")
    parser.add_argument("--split", choices=sorted(SPLIT_COUNTS), default=None,
                        help="which of the sweep's stored splits to ask "
                             "(default: testing if it holds one, else training)")
    parser.add_argument("--dataset", default=None, metavar="FILE",
                        help="ask this file's questions instead of a stored split")
    parser.add_argument("--count", type=int, default=None, metavar="N",
                        help="ask only the first N questions, 0 for all of them "
                             "(default: the split's own TRAINING_COUNT or "
                             "TESTING_COUNT; all of a --dataset file)")
    parser.add_argument("--evaluator", default=None, metavar="NAME",
                        help="grade with this evaluator instead of the sweep's "
                             "EVALUATOR (python start_run.py --evaluators lists them)")
    parser.add_argument("--judge-model", default=None, metavar="MODEL",
                        help="the judge model to ask, instead of the sweep's JUDGE_MODEL")
    parser.add_argument("--judge-backend", choices=common.BACKENDS, default=None,
                        help="where that judge runs, instead of the sweep's JUDGE_BACKEND")
    parser.add_argument("--judge-base-url", default=None, metavar="URL",
                        help="the endpoint to ask, instead of the sweep's JUDGE_BASE_URL")
    parser.add_argument("--stored-script", action="store_true",
                        help="run the individual's stored script, re-pointed, "
                             "rather than rendering it from the template the "
                             "adapters are rendered from")
    parser.add_argument("--rerun", action="store_true",
                        help="run every script again, even when answers from "
                             "the same script are already saved")
    parser.add_argument("--no-score", action="store_true",
                        help="run the scripts and save the answers, grading nothing")
    parser.add_argument("--into", default=None, metavar="DIR",
                        help="the folder to work in (default: "
                             "<db>_run<N>/%s beside the database)" % FOLDER)
    parser.add_argument("--timeout", type=int, default=900,
                        help="seconds to allow each script (default 900)")
    parser.add_argument("--keep-scripts", action="store_true",
                        help="leave the generated scripts on disk afterwards")
    args = parser.parse_args(argv)

    db = args.db or config.DB_PATH
    where = db if os.path.isabs(db) else os.path.join(_ROOT, db)
    if not os.path.exists(db) and not os.path.exists(where):
        raise SystemExit("no database at %s" % os.path.abspath(where))
    conn = store.connect(os.path.abspath(db) if os.path.exists(db) else where)
    run_id = store.latest_run(conn) if args.run == 0 else args.run
    if run_id is None or store.get_run(conn, run_id) is None:
        raise SystemExit("%s holds no run %s" % (conn.path, args.run or "yet"))
    conf = store.get_settings(conn, run_id)

    folder = os.path.abspath(args.into or os.path.join(
        db_datasets.run_folder(conn, run_id), FOLDER))
    os.makedirs(folder, exist_ok=True)

    row = best_individual(conn, run_id, args.individual)
    slots = sorted(generate_runs.lora_slots(conf.get("LORA_SLOTS")))
    if args.slots:
        wanted = [slot.strip() for slot in args.slots.split(",") if slot.strip()]
        unknown = [slot for slot in wanted if slot not in slots]
        if unknown:
            raise SystemExit("no slot %s in this sweep's LORA_SLOTS (%s)"
                             % (", ".join(unknown), ", ".join(slots)))
        slots = wanted

    print("sweep %d in %s" % (run_id, conn.path))
    print("individual %d  %s  (fitness %s)" % (
        row["number"], row["chromosome"],
        "n/a" if row["fitness"] is None else "%.3f" % row["fitness"]))
    dataset, split, count = question_setup(conn, run_id, conf, args, folder)
    _, prompts, _ = generate_runs.eval_prompt_count(dataset, count)

    field = contestants(row, conf, slots, dataset, count, args.stored_script)
    answers = load_answers(folder)
    todo = [c for c in field if args.rerun
            or answers.get(c.key, {}).get("digest") != c.digest]
    reused = [c.label for c in field if c not in todo]
    if reused:
        print("reusing saved answers for: %s" % ", ".join(reused))
    print("working in %s\n" % folder)
    if todo:
        run_scripts(todo, folder, conf, answers, prompts, args.timeout,
                    args.keep_scripts)
    # Only this field's entries: answers.json may hold slots left out this time.
    entries = {c.key: answers[c.key] for c in field}
    if all(not entry["transcript"] for entry in entries.values()):
        raise SystemExit("no contestant produced a single answer -- see the "
                         "failures above")
    if args.no_score:
        print("\nanswers saved to %s; --no-score, so nothing was graded"
              % os.path.join(folder, ANSWERS))
        return 0

    named = args.evaluator or args.judge_model or args.judge_backend or args.judge_base_url
    generated = None if named else generated_scores(entries)
    if generated is not None:
        scores, reasons = generated
        evaluator_name, label = "generated", "scores the mocked scripts printed"
        print("\nthese answers came pre-scored by mocked scripts -- noise, not a "
              "result. Name an --evaluator to grade them.")
    else:
        graded = judged_conf(conf, dataset, count, args)
        context = test_run_with_dataset._Context(
            conn, run_id, graded, folder, _Options(args.keep_scripts))
        scores, evaluator_name, label, reasons = grade(
            entries, graded, args.evaluator or conf.get("EVALUATOR"), context)

    rows = compare(scores, prompts)
    meta = {"run": run_id, "db": conn.path, "dataset": dataset, "split": split,
            "questions": prompts, "evaluator": evaluator_name, "judge": label,
            "individual": row["number"], "training_quality": row["quality"],
            "chromosome": entries["blend"]["expression"],
            "graded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            # Who each key in `summary` and `answers` is, so a reader of the
            # file alone can label it: the report is the product here, and a
            # page drawing it should not have to guess that "L3" is an adapter
            # and "blend" is an individual.
            "contestants": {key: {"label": entry["label"],
                                  "expression": entry["expression"],
                                  "verdict": entry["verdict"],
                                  "seconds": entry["seconds"]}
                            for key, entry in entries.items()}}
    lines = report_lines(meta, entries, rows)
    print("\n" + "\n".join(lines))

    stem = os.path.join(folder, "scores_%s_%s" % (_safe(evaluator_name), _safe(label)))
    with open(stem + ".json", "w", encoding="utf-8") as handle:
        json.dump({"meta": meta, "summary": rows,
                   "answers": {key: [dict(item, quality=scores[key][i], reason=reasons[key][i])
                                     for i, item in enumerate(entries[key]["transcript"])]
                               for key in entries}},
                  handle, indent=1, ensure_ascii=False)
    with open(stem + ".md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\nwrote %s.json and .md" % stem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
