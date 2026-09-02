"""
baseline_run.py - What the base model says on its own, produced once and kept.

The "llm_judge_baseline" evaluator does not ask whether an answer is good; it
asks whether the blend made it better than the model was without any adapter.
That needs a second answer to every eval prompt -- the control -- and this
module is where it comes from.

    generate_runs.render_baseline()   fill template_baseline.py
    process_run.launch()              run it once, in its own process
    process_run.exchanges()           read the YOU:/COACH: pairs back out
    store.add_baselines()             file them under the model and question

The cost is one base-model load and one generate() per prompt, paid once. After
that the answers live in the `baselines` table and every later sweep reads them
from there: the table hangs off no run, because a base-model answer belongs to
the model and the question and to nothing else. Repointing BASE_MODEL asks for
that model's own control rather than reusing the old one's, and adding prompts
to the eval set produces only the ones that are missing.

Two things it is careful about:

  * **A mocked sweep gets a mocked baseline**, filled from
    template_baseline_mocked.py, and cached under "mock:<model>" rather than
    under the model's own name. An invented control can therefore never end up
    in front of a real judge, and a dry run still exercises the whole path.
  * **The control is generated the way the individuals were.** Same base model,
    same eval file, same cap, same chat template, same max_new_tokens, no
    sampling. A control produced under different settings would make every
    improvement score a comparison of the settings as much as of the blend.

Nothing here decides *how* a base answer is used -- that is the evaluator's
part, in evaluate_run.py. This module only makes sure the answers exist.
"""

import os

import generate_runs
import process_run
import settings as config
import store

# The one script this module ever writes, in the sweep's run folder, deleted
# again once it has been read -- the same life cycle the individuals' scripts
# have, and for the same reason: its answers are in the database, so the file
# is spent the moment it has run.
SCRIPT_NAME = "baseline.py"

# The mocked baseline, chosen when the sweep itself was mocked.
MOCKED_TEMPLATE = "template_baseline_mocked.py"

_HERE = os.path.dirname(os.path.abspath(__file__))


def _resolve(name):
    """Absolute path of a template, from a name or a path."""
    if not os.path.isabs(name) and not os.path.exists(name):
        return os.path.join(_HERE, name)
    return os.path.abspath(name)


def template_for(conf):
    """(path, mocked) -- which baseline template this sweep's control comes from.

    BASELINE_TEMPLATE names one outright. Left as None it follows the sweep's
    own TEMPLATE, which is the answer that is right by default: a mocked sweep
    loads nothing anywhere else, and a baseline that reached for a GPU would be
    the one part of a dry run that could not run dry.
    """
    named = conf.get("BASELINE_TEMPLATE", config.BASELINE_TEMPLATE)
    if named:
        path = _resolve(named)
        return path, "mocked" in os.path.basename(path)
    mocked = "mocked" in os.path.basename(conf.get("TEMPLATE") or "")
    return _resolve(MOCKED_TEMPLATE if mocked else "template_baseline.py"), mocked


def model_key(conf):
    """The name this sweep's base answers are cached under.

    The model itself, or "mock:<model>" when the control is invented. Two
    namespaces in one table rather than two tables, because they are the same
    kind of row answering the same question -- one of them just cannot be
    believed.
    """
    model = generate_runs.base_model_name(conf.get("BASE_MODEL"))
    return ("mock:" + model) if template_for(conf)[1] else model


def cached(conn, conf):
    """{question_key: answer} already stored for this sweep's base model."""
    return store.baselines(conn, model_key(conf))


def answer_for(question, answers):
    """What the base model said to this question, out of ensure()'s map.

    Matched on the normalised question rather than on the position it held in
    the eval file, for the reason the reference answers are: an exchange keeps
    the question it actually asked, so matching on it cannot pair an answer
    with the wrong control after the file has been edited.
    """
    return answers.get(store.question_key(question))


def missing(conn, conf, questions):
    """The questions of `questions` that have no cached base answer yet."""
    have = cached(conn, conf)
    seen, absent = set(), []
    for question in questions:
        key = store.question_key(question)
        if key in have or key in seen:
            continue
        seen.add(key)
        absent.append(question)
    return absent


def generate(conn, conf, run_dir, timeout=None, keep_script=False, say=print):
    """Run the baseline script once and cache what it said. -> (added, total).

    `added` is how many answers were new, `total` how many the script produced.
    Failing here is fatal to the caller by design: an evaluator that grades
    against the base model has nothing to grade against without this, and
    scoring a whole sweep's transcript against a missing control would be
    worse than stopping.
    """
    template, mocked = template_for(conf)
    model = model_key(conf)
    source = generate_runs.render_baseline(
        SCRIPT_NAME,
        "Generated by baseline_run.py: the control the llm_judge_baseline "
        "evaluator measures every blend against.",
        template_path=template,
        training_set=conf.get("TRAINING_SET"),
        count=conf.get("TRAINING_COUNT"),
        base_model=conf.get("BASE_MODEL"),
    )

    # The same interpreter check the process step makes, for the same reason:
    # this script is launched with sys.executable, so the wrong python fails it
    # after paying for a process launch. A mocked baseline imports nothing.
    if process_run.imports_unsloth(source):
        process_run.check_interpreter()

    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, SCRIPT_NAME)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(source)

    try:
        prompts = generate_runs.eval_prompt_count(
            conf.get("TRAINING_SET"), conf.get("TRAINING_COUNT"))[1]
    except SystemExit:
        prompts = 0

    wait = conf.get("BASELINE_TIMEOUT", config.BASELINE_TIMEOUT) if timeout is None \
        else timeout
    say("baseline: running %s (%s) over %d prompt(s) -- once, then cached"
        % (SCRIPT_NAME, os.path.basename(template), prompts))
    if not mocked:
        say("baseline: this loads the base model, so it takes about as long as "
            "one individual")

    every = conf.get("PROCESS_RUN_PROGRESS_SECONDS",
                     config.PROCESS_RUN_PROGRESS_SECONDS)
    reporter = process_run.Progress(SCRIPT_NAME, prompts, every,
                                    lambda script, message: say("        " + message))
    code, seconds, out, err = process_run.launch(run_dir, SCRIPT_NAME, wait,
                                                 reporter.line, reporter.tick)

    transcript = process_run.exchanges(out)
    if code != 0 or not transcript:
        # Leave the script where it is: without it there is nothing to re-run
        # by hand, and this is the one path where that is worth doing.
        tail = [line for line in (out + err).splitlines() if line.strip()][-1:]
        raise SystemExit(
            "the baseline script %s in %s %s, so there is nothing for "
            "EVALUATOR = 'llm_judge_baseline' to compare against.%s"
            % (SCRIPT_NAME, run_dir, process_run.verdict_of(code),
               ("\n    " + tail[0][:200]) if tail else "")
        )

    added = store.add_baselines(
        conn, model,
        [(item["question"], item["answer"]) for item in transcript],
        source=os.path.basename(template))

    if keep_script:
        say("baseline: kept %s (--keep-scripts)" % path)
    else:
        try:
            os.remove(path)
        except OSError:
            pass                       # a cache file, never worth failing over
    say("baseline: %d answer(s) in %.1fs, %d new, cached under %r"
        % (len(transcript), seconds, added, model))
    return added, len(transcript)


def ensure(conn, conf, questions, run_dir, timeout=None, keep_script=False,
           say=print):
    """Every base answer these questions need. -> {question_key: answer}.

    Reads the cache first and only runs the base model for what is not in it,
    which is the whole point of storing them: the first sweep on a base model
    pays for the control and every sweep after it reads the same rows.

    A question still missing afterwards is left missing rather than guessed at
    -- the evaluator fails that one exchange and says why, which is a better
    account of it than a score measured against nothing.
    """
    absent = missing(conn, conf, questions)
    if not absent:
        return cached(conn, conf)
    say("baseline: %d of %d question(s) have no cached base-model answer"
        % (len(absent), len(questions)))
    generate(conn, conf, run_dir, timeout=timeout, keep_script=keep_script, say=say)
    return cached(conn, conf)
