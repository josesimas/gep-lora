"""
test_run_with_dataset.py - Put a sweep's best individuals in front of a dataset
they were never scored on.

    python test_run_with_dataset.py datasets/medical_testing_lora_dataset.json
    python test_run_with_dataset.py testing.json --db run_db1/gep.sqlite3 --run 3

The search judges every individual on the training split, generation after
generation, and then selects, elects and mutates on the strength of that number.
An individual that scores well there has been selected *for* those questions.
This asks the other question -- does the blend hold up on questions it was never
picked for? -- and it is the only honest way to ask it, because the training
split stopped being unseen the moment the first generation was scored on it.

What it does, in order:

  1. stores the dataset as this sweep's `testing` split, the way a sweep stores
     the ones its settings named (add_dataset.py), so what the pass asked is
     recorded beside what the search asked;
  2. takes the individuals whose mean training quality is above
     TESTING_MIN_QUALITY -- 0.5 by default, --min-quality for one pass;
  3. writes each one's **own script** into run_testing/, re-pointed at the new
     dataset and nothing else: same chromosome, same weight seed, same blend,
     same base model, different questions;
  4. runs them the way the process step runs a generation -- in batches of
     PROCESS_RUN_BATCH_SIZE, streaming progress, a crash being a result rather
     than a failure;
  5. stores every answer in `test_results`, one row per individual, transcript
     and all;
  6. grades them, with the sweep's own evaluator unless told otherwise, and
     writes each score beside the answer it belongs to and the mean onto the
     row.

The two halves are separable because only the first costs a base-model load:
`--no-score` stores the answers and stops, `--score-only` grades what an earlier
pass stored, `--force` re-grades, and an interrupted scoring run resumes where
it left off. That is what makes it safe to grade with a judge that may not be
up yet, or to change your mind about which evaluator a testing set deserves.

The scoring half reads the sweep's settings with one substitution -- the eval
set is the testing dataset (`testing_conf()`), the same swap `repoint()` makes
to the scripts. Three of the six evaluators grade against the eval set's own
answers, and left pointing at TRAINING_SET every one of them would silently
compare a testing answer with a training question's reference.

`test_answers` is the view that reads the graded transcripts back one answer at
a time.

Nothing here is a pipeline step. main.py runs a sweep; this runs *after* one,
against a sweep already in the database, and touches neither the population nor
any fitness number -- which is why the results live in their own table rather
than in `executions`, where fitness, elitism and selection would find them and
decide the next generation on questions the search is not judged on.
"""

import argparse
import json
import os
import re
import sys
import threading
import time

import add_dataset
import evaluators
import generate_runs
import process_run
import settings as config
import store

_HERE = os.path.dirname(os.path.abspath(__file__))

# The two lines a generated script carries that say which questions it asks.
# They are whole-line blocks in template_code.py -- "TRAINING_SET = '...'" and
# "TRAINING_COUNT = 20" -- filled with plain literals, which is what makes
# re-pointing a stored script a two-line rewrite rather than a re-render.
_ASSIGNMENT = "^%s = .*$"


def repoint(source, dataset, count):
    """The same script, asking a different file's questions.

    The stored script is what actually earned this individual its training
    quality: its weight seed, its adapters, its base model and the template code
    of the day it ran. Re-rendering from the chromosome would rebuild all of
    that out of whatever the template says *now*, which would make a testing
    pass a comparison of two template versions as much as of a blend. So the
    source is taken as it is and exactly two assignments are rewritten.

    Either one missing, or appearing twice, is a script this cannot honestly
    re-point -- it is said rather than guessed at, because a silent no-op would
    run the whole pass against the training set and store the answers as if
    they were the test's.
    """
    for name, literal in (("TRAINING_SET", repr(dataset)),
                          ("TRAINING_COUNT", str(count))):
        line = "%s = %s" % (name, literal)
        # A lambda rather than the string itself: a Windows path is full of
        # backslashes, and re reads those as escapes in a replacement template.
        source, hits = re.subn(_ASSIGNMENT % name,
                               lambda _match, line=line: line,
                               source, flags=re.MULTILINE)
        if hits != 1:
            raise SystemExit(
                "cannot re-point this script: it has %d line(s) assigning %s, "
                "and there should be exactly one. It was generated from a "
                "template this script does not recognise." % (hits, name))
    return source


# What a generated script says it builds: template_code.py stamps the
# chromosome in as EXPRESSION = "...", one line, in both templates.
_EXPRESSION = re.compile(r'^EXPRESSION = "(.*)"$', re.MULTILINE)


def script_chromosome(source):
    """The chromosome the script builds, or None if it does not say.

    Not the same question as individuals.chromosome, and the difference is the
    whole reason this exists: mutation rewrites that column and leaves the
    script, the rank and the stored quality describing the chromosome the
    individual *used to be*. A testing pass runs the script, so what it ran is
    what the script builds, and that is what its row records.
    """
    found = _EXPRESSION.search(source or "")
    return found.group(1) if found else None


def candidates(conn, run_id, minimum):
    """The individuals worth testing: mean training quality above `minimum`.

    Read from the individual_quality view rather than individuals.fitness --
    the same average over the same latest execution, but computed from the
    transcripts themselves, so a sweep whose fitness step never ran still has
    an answer here. An individual with no execution, or none yet scored, has a
    NULL quality and is not above anything.

    Ordered best first, so --limit takes the best few rather than the lowest
    numbered few.
    """
    return conn.execute(
        "SELECT i.*, q.quality AS quality, q.answers AS answers"
        "  FROM individual_quality q"
        "  JOIN individuals i ON i.run_id = q.run_id AND i.number = q.number"
        " WHERE q.run_id = ? AND q.quality > ? AND i.script_source IS NOT NULL"
        " ORDER BY q.quality DESC, i.number", (run_id, minimum)).fetchall()


def record_dataset(conn, run_id, path, replace=False, say=print):
    """Store the dataset as this sweep's `testing` split. -> its records.

    Through add_dataset.add(), so a testing set recorded here is stored exactly
    as one a sweep was created with. The one thing added on top: re-running a
    pass over the same file is not a conflict. add() refuses a split the run
    already holds, which is right when the file has changed and wrong when it
    is the same file with the same lines, so that case is checked for and
    passed over quietly.
    """
    records = generate_runs.dataset_records(path)
    held = [entry for entry in store.dataset_summary(conn, run_id)
            if entry["split"] == "testing"]
    if held and not replace:
        stored = store.dataset(conn, run_id, "testing")
        unchanged = (held[0]["source"] == path and len(stored) == len(records)
                     and all(was["content"] == now["content"]
                             for was, now in zip(stored, records)))
        if unchanged:
            say("testing    %3d record(s) already recorded, unchanged, from %s"
                % (len(records), path))
            return records
    add_dataset.add(conn, run_id, "testing", path, replace=replace, say=say)
    return records


def write_scripts(rows, run_dir, dataset, count):
    """Put each individual's re-pointed script in the testing folder.

    The folder has to sit exactly one level below the project folder, like
    run_db/: a generated script finds the LoRA folders by going up one from
    itself. The scripts keep their own names -- run_007.py is individual 7's
    script here as much as there -- which is also why they are somewhere else:
    two files of the same name asking different questions have no business in
    one folder.
    """
    os.makedirs(run_dir, exist_ok=True)
    for row in rows:
        source = repoint(row["script_source"], dataset, count)
        with open(os.path.join(run_dir, row["script_name"]), "w",
                  encoding="utf-8") as handle:
            handle.write(source)
    return len(rows)


def run_all(conn, run_id, rows, run_dir, dataset, prompts, conf, options,
            say=print):
    """Run every selected script and store what it said. -> (tested, failed).

    The process step's loop, over a different folder and into a different
    table: batches of PROCESS_RUN_BATCH_SIZE, the same streaming progress, the
    same rule that an individual which crashes is a row rather than a stopped
    pass, and the same per-individual commit, so an interrupted pass keeps what
    came back.
    """
    size = process_run.batch_size(
        conf.get("PROCESS_RUN_BATCH_SIZE", config.PROCESS_RUN_BATCH_SIZE))
    groups = process_run.batches(rows, size)
    every = conf.get("PROCESS_RUN_PROGRESS_SECONDS",
                     config.PROCESS_RUN_PROGRESS_SECONDS)

    # Progress arrives on the threads draining the children, several at once in
    # a batch, so one lock owns the console for a whole line.
    console = threading.Lock()

    def progress(script, message):
        with console:
            say("        %s%s" % ("" if size == 1 else script + "  ", message))

    def watch(script):
        return process_run.Progress(script, prompts, every, progress)

    failures = 0
    started = time.time()
    number = 0
    for position, group in enumerate(groups, 1):
        first = number + 1
        for offset, row in enumerate(group):
            say("[%d/%d] %s  %s  (training quality %.3f)"
                % (first + offset, len(rows), row["script_name"],
                   script_chromosome(row["script_source"]) or row["chromosome"],
                   row["quality"]))
        if size > 1:
            say("        batch %d/%d: %d running at once"
                % (position, len(groups), len(group)))

        results = process_run.launch_batch(
            run_dir, [row["script_name"] for row in group], options.timeout,
            watch)

        # Stored in the order they were asked for rather than the order they
        # finished, and from this thread only, so the table is written exactly
        # as it would have been one at a time.
        for row, (code, seconds, out, err) in zip(group, results):
            number += 1
            verdict = process_run.verdict_of(code)
            if code != 0:
                failures += 1
            transcript = process_run.exchanges(out)
            result_id = store.add_test_result(
                conn, run_id, row,
                script_chromosome(row["script_source"]) or row["chromosome"],
                dataset, prompts, row["quality"], seconds, code, verdict,
                process_run.drawn_weights(out), out, err, transcript)
            conn.commit()
            say("        %s%-9s %6.1fs  %d answer(s) -> test result %d"
                % ("" if size == 1 else row["script_name"] + "  ",
                   verdict, seconds, len(transcript), result_id))
            if code != 0:
                # Show why, so a systemic problem is obvious without a query.
                tail = [line for line in (out + err).splitlines()
                        if line.strip()][-1:]
                if tail:
                    say("        %s" % tail[0][:100])

    say("\ntested %d in %.1fs, %d failed"
        % (len(rows), time.time() - started, failures))
    return len(rows), failures


class _Context:
    """What an evaluator's prepare() is handed, for a pass that has no Context.

    main.py builds a real one per step; this is the same five attributes for a
    script that is not a step. Only llm_judge_baseline looks at it -- it reads
    and fills the cache of base-model answers, so it wants the database, the
    folder to run a baseline script in, and whether to keep it.
    """

    def __init__(self, conn, run_id, conf, run_dir, options):
        self.conn = conn
        self.run_id = run_id
        self.conf = conf
        self.run_dir = run_dir
        self.options = options
        self.generation = None


def testing_conf(conf, dataset, count):
    """The sweep's settings, with the eval set pointed at the testing dataset.

    The same substitution repoint() makes to the scripts, made to the settings
    the evaluators read -- and it has to be made, because three of the six
    grade against the eval set's own answers: llm_judge_reference and
    similarity read the reference beside each question, and llm_judge_baseline
    asks the base model the questions it is comparing against. Left pointing at
    TRAINING_SET, every one of them would grade a testing answer against a
    training question's reference -- silently, since both files parse.

    Everything else is the sweep's own, so the rubric, the endpoint and the
    judge are the ones the search was scored under.
    """
    out = dict(conf)
    out["TRAINING_SET"] = dataset
    out["TRAINING_COUNT"] = count
    return out


def settle(conn, run_id, dataset, say=print):
    """Give a row its mean when its answers were scored by something else.

    A mocked script grades its own answers and prints the scores, which
    process_run.exchanges() folds into the transcript -- so a mocked pass
    arrives scored and never reaches an evaluator, exactly as a mocked sweep
    does. Those rows still deserve the mean the graded ones get, or a mocked
    pass would report nothing and look like a failure of the plumbing rather
    than the dry run it is. Labelled "generated", the way add_exchanges() marks
    a score the script brought with it.
    """
    settled = 0
    for row in store.test_results(conn, run_id, dataset):
        if row["quality"] is not None:
            continue
        transcript = json.loads(row["exchanges"] or "[]")
        if not transcript or any(item.get("quality") is None
                                 for item in transcript):
            continue
        store.score_test_result(conn, row["id"], transcript, "generated", None)
        settled += 1
    if settled:
        conn.commit()
        say("%d row(s) took their mean from the scores their script printed"
            % settled)
    return settled


def score_pass(conn, run_id, dataset, conf, options, run_dir, say=print):
    """Grade the answers of a testing pass. -> (scored, failed).

    The evaluate step, over test_results rather than exchanges. Same registry,
    same prepare()/score() contract, same rule that an evaluator which cannot
    score one answer fails that answer and nothing else -- so a half-graded
    pass is resumable and re-running grades only what is still missing.

    The scores go back into the row's own transcript, beside the answers, and
    the row keeps the mean, which evaluator gave it and what that evaluator's
    label was. Storing the mean is what makes a testing number comparable with
    a training one: individual_quality takes the same average over the same
    kind of answers.

    Which evaluator is the sweep's own EVALUATOR unless --evaluator says
    otherwise, for the reason a sweep's steps read the sweep's settings: a
    testing quality graded by a different rubric than the training quality it
    is put beside would make the comparison meaningless.
    """
    pending = store.test_results_to_score(conn, run_id, dataset, options.force)
    if not pending:
        held = store.test_results(conn, run_id, dataset)
        if not held:
            raise SystemExit(
                "run %d has no testing results on %s to score. Run the pass "
                "first, without --score-only." % (run_id, dataset))
        say("nothing to score -- every answer of this pass already has a "
            "quality (pass --force to re-score them)")
        settle(conn, run_id, dataset, say)
        return 0, 0

    # Flattened out of the transcripts: an evaluator scores answers, and the
    # rows they arrived in are this module's business rather than its.
    items = []
    for row in pending:
        transcript = json.loads(row["exchanges"])
        for position, item in enumerate(transcript, 1):
            if options.force or item.get("quality") is None:
                items.append({"question": item.get("question", ""),
                              "answer": item.get("answer", ""),
                              "position": position,
                              "number": row["number"],
                              "chromosome": row["chromosome"]})

    name = options.evaluator or conf.get("EVALUATOR")
    evaluator = evaluators.get(name)
    graded = testing_conf(conf, dataset, options.count or None)
    context = _Context(conn, run_id, graded, run_dir, options)

    # Said before prepare(), which can take minutes -- the baseline evaluator
    # may have a base model to run -- so a silent console says what it is for.
    say("\nevaluator: %s -- %s" % (evaluator.name, evaluator.description))
    say("grading against %s" % dataset)
    prepared = evaluator.prepare(graded, items, context)
    for note in prepared.notes:
        say(note)
    say("scoring %d answer(s)%s\n"
        % (len(items), " (--force: re-scoring)" if options.force else ""))

    # The same rule the evaluate step applies, for the same reason: a judge call
    # is what a pass costs, and an individual whose first answers all score 0
    # spends the rest of them confirming it. A row here is one individual, so
    # "the first 10%" is of the answers this pass has left to grade for it.
    limit_fraction = (graded.get("JUDGE_ABANDON_FRACTION")
                      if evaluator.needs_endpoint else None)
    if limit_fraction:
        say("giving up on an individual once its first %g%% of graded answers "
            "have all scored 0" % (100 * limit_fraction))

    scored = failed = abandoned = 0
    started = time.time()
    for row in pending:
        transcript = json.loads(row["exchanges"])
        say("individual %d  %s" % (row["number"], row["chromosome"]))
        todo = [position for position, item in enumerate(transcript, 1)
                if options.force or item.get("quality") is None]
        limit = (evaluators.abandon_after(graded, len(todo))
                 if limit_fraction else None)
        judged = zeros = 0
        giving_up = ""

        for position, item in enumerate(transcript, 1):
            if not options.force and item.get("quality") is not None:
                continue
            if giving_up:
                # Scored rather than left ungraded, so the row's mean is the
                # mean of the whole individual and a re-run does not come back
                # for these -- the same bargain the evaluate step makes.
                item["quality"], item["reason"] = 0.0, giving_up
                scored += 1
                abandoned += 1
                continue
            answer = item.get("answer") or ""
            if not answer.strip():
                # Nothing to grade: an unanswered question is worth nothing,
                # and asking a judge about an empty string wastes a call. It is
                # not an evaluation either, so it neither counts toward the
                # first 10% nor condemns the individual on its own.
                item["quality"], item["reason"] = 0.0, "no answer given"
                scored += 1
                say("    [%d] 0.00  (no answer)" % position)
                continue
            try:
                quality, reason = evaluator.score(
                    {"question": item.get("question", ""), "answer": answer,
                     "position": position, "number": row["number"]}, prepared)
            except (RuntimeError, ValueError) as error:
                # One answer's failure, not the pass's: it keeps its NULL and a
                # re-run picks it up again. A failure is not a zero, so it never
                # counts toward giving up either.
                failed += 1
                say("    [%d] FAILED  %s" % (position, error))
                continue
            item["quality"], item["reason"] = round(quality, 3), reason
            scored += 1
            judged += 1
            zeros += 1 if item["quality"] == 0.0 else 0
            say("    [%d] %.2f  %s" % (position, item["quality"], reason))

            if limit and judged >= limit and zeros == judged:
                giving_up = ("abandoned: the first %d graded answer(s) all "
                             "scored 0" % judged)
                say("    giving up on individual %d -- %s, so its remaining "
                    "%d answer(s) are scored 0 unasked"
                    % (row["number"], giving_up,
                       len([left for left in todo if left > position])))
        mean = store.score_test_result(conn, row["id"], transcript,
                                       evaluator.name, prepared.label)
        # Committed per row, so an interrupted scoring run keeps what it did.
        conn.commit()
        say("    -> %s over %d answer(s)"
            % ("no score" if mean is None else "%.3f" % mean, len(transcript)))

    say("\nscored %d, %d failed, in %.1fs"
        % (scored, failed, time.time() - started))
    if abandoned:
        say("%d answer(s) scored 0 unasked, on individuals given up on early"
            % abandoned)
    if failed and not scored:
        raise SystemExit("nothing could be scored by the %s evaluator -- check "
                         "its settings, and the judge endpoint if it uses one"
                         % evaluator.name)
    return scored, failed


def report(conn, run_id, dataset, say=print):
    """Test quality beside the training quality that picked each individual.

    The point of the whole exercise, in one table: an individual that holds its
    score here answered questions it was never selected on, and one that drops
    was selected for the training file rather than for the job.
    """
    rows = [row for row in store.test_quality(conn, run_id, dataset)
            if row["quality"] is not None]
    if not rows:
        return
    say("\n%-4s %-8s %-8s %-7s %s"
        % ("#", "training", "testing", "delta", "chromosome"))
    for row in rows:
        delta = row["quality"] - (row["selected_on"] or 0.0)
        say("%-4d %-8.3f %-8.3f %+-7.3f %s"
            % (row["number"], row["selected_on"] or 0.0, row["quality"], delta,
               row["chromosome"]))
    testing = [row["quality"] for row in rows]
    training = [row["selected_on"] or 0.0 for row in rows]
    say("mean over %d individual(s): training %.3f, testing %.3f (%+.3f)"
        % (len(rows), sum(training) / len(training), sum(testing) / len(testing),
           sum(testing) / len(testing) - sum(training) / len(training)))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a sweep's best individuals against another dataset.")
    parser.add_argument("dataset",
                        help="the dataset to test on, JSON Lines or one prompt "
                             "per line")
    parser.add_argument("--db", default=None,
                        help="database file (default: settings.DB_PATH)")
    parser.add_argument("--run", type=int, default=0, metavar="RUN",
                        help="the sweep to test (0 = the most recent, default)")
    parser.add_argument("--min-quality", type=float, default=None, metavar="Q",
                        help="test the individuals scoring above this on the "
                             "training split (default: settings.TESTING_MIN_QUALITY)")
    parser.add_argument("--count", type=int, default=0, metavar="N",
                        help="ask only the first N questions of the dataset "
                             "(0 = all of them, the default)")
    parser.add_argument("--limit", type=int, default=0, metavar="N",
                        help="test only the best N of the individuals selected")
    parser.add_argument("--into", default=None, metavar="DIR",
                        help="where the scripts run (default: "
                             "settings.TESTING_RUN_DIR)")
    parser.add_argument("--timeout", type=int, default=900,
                        help="seconds to allow each script (default 900)")
    parser.add_argument("--keep-scripts", action="store_true",
                        help="leave the scripts on disk afterwards (default: "
                             "delete them; they are re-pointed copies of what "
                             "the database already holds)")
    parser.add_argument("--replace", action="store_true",
                        help="overwrite a different testing split already "
                             "stored for this sweep")
    parser.add_argument("--evaluator", default=None, metavar="NAME",
                        help="score with this evaluator instead of the sweep's "
                             "own EVALUATOR (python main.py --evaluators lists them)")
    parser.add_argument("--no-score", action="store_true",
                        help="run the scripts and store the answers, leaving "
                             "them ungraded for a later pass")
    parser.add_argument("--score-only", action="store_true",
                        help="grade the answers of an earlier pass over this "
                             "dataset; runs nothing")
    parser.add_argument("--force", action="store_true",
                        help="re-score answers that already have a quality")
    args = parser.parse_args(argv)

    if args.no_score and args.score_only:
        raise SystemExit("--no-score and --score-only ask for opposite halves "
                         "of the same pass; pick one.")

    if args.db is None:
        args.db = config.DB_PATH
    # connect() would create an empty database with the schema in it, which for
    # a typo'd --db means being told a fresh file holds no runs rather than that
    # the file does not exist.
    where = args.db if os.path.isabs(args.db) else os.path.join(_HERE, args.db)
    if not os.path.exists(where):
        raise SystemExit("no database at %s" % os.path.abspath(where))
    conn = store.connect(args.db)

    run_id = store.latest_run(conn) if args.run == 0 else args.run
    if run_id is None:
        raise SystemExit("%s holds no runs yet" % conn.path)
    if store.get_run(conn, run_id) is None:
        raise SystemExit("%s holds no run %d" % (conn.path, run_id))

    # The sweep's own settings, not settings.py's: the individuals being tested
    # were built under them, and a batch size or a progress interval belongs to
    # the sweep the way every other knob does. The two knobs this script adds
    # fall back to settings.py, since a sweep created before they existed
    # recorded neither.
    conf = store.get_settings(conn, run_id)
    minimum = (config.TESTING_MIN_QUALITY if args.min_quality is None
               else args.min_quality)

    dataset = add_dataset.resolve_dataset(args.dataset)
    print("sweep %d in %s" % (run_id, conn.path))
    # Asked before the dataset is recorded, because --score-only with --replace
    # would otherwise overwrite a stored split on the way to finding out it has
    # nothing to grade -- a write for a pass that never happened.
    if args.score_only and not store.test_results(conn, run_id, dataset):
        raise SystemExit(
            "run %d has no testing results on %s to score. Run the pass first, "
            "without --score-only." % (run_id, dataset))
    records = record_dataset(conn, run_id, dataset, replace=args.replace)
    prompts = len(records) if not args.count else min(args.count, len(records))

    run_dir = args.into or conf.get("TESTING_RUN_DIR") or config.TESTING_RUN_DIR
    if not os.path.isabs(run_dir):
        run_dir = os.path.join(_HERE, run_dir)
    run_dir = os.path.abspath(run_dir)

    # The second half on its own: grading answers an earlier pass stored. It
    # runs nothing, so it needs neither the venv nor a candidate -- which is
    # the point of separating them, since scoring can be re-run, re-pointed at
    # another evaluator, or picked up after an interrupted judge, and none of
    # that should cost a base-model load per individual again.
    if args.score_only:
        score_pass(conn, run_id, dataset, conf, args, run_dir)
        report(conn, run_id, dataset)
        return 0

    rows = candidates(conn, run_id, minimum)
    if not rows:
        raise SystemExit(
            "no individual in run %d scored above %.3f on the training split, "
            "so there is nothing worth testing. Lower it with --min-quality, or "
            "check that the sweep has been processed and evaluated (python "
            "store.py --show %d)." % (run_id, minimum, run_id))
    if args.limit:
        rows = rows[:args.limit]

    # A run that finishes normally now stops after `fitness`, so its population
    # is the one that was scored and this list is usually empty. It stops being
    # empty when the sweep was stopped mid-generation, or its steps were run by
    # hand, or an older sweep ran to the end of mutation: the chromosome column
    # has then moved on while the script, the rank and the quality that picked
    # it all still describe the individual as it was when it last ran. The pass
    # tests what it can actually run, and says so rather than reporting a
    # mutant's number against an ancestor's answers without comment.
    moved = [row for row in rows
             if (script_chromosome(row["script_source"]) or row["chromosome"])
             != row["chromosome"]]
    if moved:
        print("note: %d of them have been mutated since they last ran -- their "
              "scripts, and the quality that picked them, describe the "
              "chromosome they used to be, and that is what is tested and "
              "stored. Re-run trees/runs/process/evaluate first to test the "
              "current population." % len(moved))

    # Whether the venv is needed is a property of the scripts themselves: the
    # mocked ones load nothing, and demanding it for them would put a GPU-less
    # machine out of reach.
    real = process_run.imports_unsloth(rows[0]["script_source"])
    if real:
        process_run.check_interpreter()

    written = write_scripts(rows, run_dir, dataset,
                            args.count if args.count else None)
    print("\ntesting %d individual(s) scoring above %.3f, on %d question(s)"
          % (len(rows), minimum, prompts))
    print("wrote %d script(s) to %s, each re-pointed at the testing set"
          % (written, run_dir))
    if real:
        print("each one loads the base model, so this takes a while\n")
    else:
        print("these are mocked scripts: nothing is loaded, so this is quick\n")

    tested, failures = run_all(conn, run_id, rows, run_dir, dataset, prompts,
                               conf, args)

    # The scripts have done their job: everything they printed is in the
    # database, and they are re-pointed copies of a source that is in there too,
    # so leaving them about only invites someone to run one by hand and take it
    # for the sweep's own.
    if args.keep_scripts:
        print("kept the scripts in %s (--keep-scripts)" % run_dir)
    else:
        gone = store.remove_scripts(conn, run_id, run_dir,
                                    [row["script_name"] for row in rows])
        print("removed %d spent script(s) from %s -- they are re-pointed copies "
              "of what the database already holds" % (gone, run_dir))

    # A blend that cannot answer is a result. Only a pass where nothing at all
    # ran points at something systemic -- the wrong interpreter, a moved
    # adapter, a dataset no script can read -- and there is nothing to grade
    # either way, so this is said before the scoring half rather than after it.
    if failures == tested:
        raise SystemExit("every individual failed -- try: python store.py --show %d"
                         % run_id)

    # The answers are stored either way; grading them is the second half of the
    # pass, and separable because it costs no model load. --no-score leaves
    # them for a later --score-only, which is what to do when the judge is not
    # up, or when which evaluator should grade this dataset is still a question.
    if args.no_score:
        print("\nnot scored (--no-score). Grade them later with:")
        print("    python test_run_with_dataset.py %s --score-only"
              % os.path.basename(dataset))
    else:
        score_pass(conn, run_id, dataset, conf, args, run_dir)
        report(conn, run_id, dataset)

    print("\nresults are in test_results, and the test_answers view reads them "
          "one answer at a time:")
    print("    SELECT number, chromosome, position, question, answer, quality"
          "\n      FROM test_answers WHERE run_id = %d;" % run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
