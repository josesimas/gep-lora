"""
submit.py - A submission turned into a *prepared* sweep database.

A prepared database is what `python main.py --db <it>` already knows how to
run: a run row, its settings, its dataset, and no individuals (see main.py's
`prepared()`). So submitting a job is writing one of those and nothing else --
the worker then hands the file to main.py, which adopts it, reads the questions
out of its rows and keeps everything it writes in a folder beside it. There is
no second definition of what a job needs.

A submission is JSON:

    {
      "label":    "overnight",                       optional
      "settings": {"GENERATIONS": 3, "COUNT": 10},   overrides of config/settings.py
      "datasets": {
        "training": [{"messages": [...]}, ...],      required
        "testing":  "one JSON record per line\\n...", or a string of lines
        "validation": {"file": "medical_validation_lora_dataset.json"}
      },
      "options":  {"no_test": false, "limit": 0, "timeout": 900,
                   "test_min_quality": 0.5}           main.py flags, optional
    }

A dataset is a list (a dict per record, or a one-line prompt string per
record), the text of a JSON Lines / plain file, or `{"file": name}` naming a
file under SHARED_DATASETS_DIR on the server. Splits not given are not part of
the job: their settings are cleared rather than left pointing at whatever the
server's settings.py names.

Settings are config/settings.py as it stands at submission, with the overrides
applied, then checked and seeded by `start_run.freeze()` -- the same function a
sweep created at the command line goes through. The frozen values are what the
job runs under, however settings.py is edited while it waits in the queue.
"""

import json
import os
import shutil

from async_api import settings
from blends import generate_runs
from config import settings as config
import start_run
from storage import add_dataset
from storage import store
from testing import test_run_with_dataset

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class SubmissionError(ValueError):
    """What is wrong with a submission, in words for whoever sent it."""


def _checked(call, *args):
    """The pipeline says no by raising SystemExit; here that is a 400."""
    try:
        return call(*args)
    except SystemExit as error:
        raise SubmissionError(str(error))


def settings_for(overrides):
    """config/settings.py's values with `overrides` applied, checked. -> conf."""
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise SubmissionError("settings must be an object of NAME: value")
    conf = config.snapshot()
    unknown = sorted(name for name in overrides if name not in conf)
    if unknown:
        raise SubmissionError("unknown setting(s): %s" % ", ".join(unknown))
    locked = sorted(name for name in overrides if name in settings.LOCKED_SETTINGS)
    if locked:
        raise SubmissionError("setting(s) a submission may not change: %s"
                              % ", ".join(locked))
    conf.update(overrides)

    _checked(start_run.freeze, conf)
    _checked(generate_runs.base_model_name, conf.get("BASE_MODEL"))
    _checked(generate_runs.training_count, conf.get("TRAINING_COUNT"))
    _checked(test_run_with_dataset.testing_count, conf.get("TESTING_COUNT"))
    # The adapters have to be on this machine: the worker blends them here.
    _checked(generate_runs.slot_ranks, conf.get("LORA_SLOTS"))
    template = generate_runs.template_path(conf.get("TEMPLATE"))
    if not os.path.exists(template):
        raise SubmissionError("no template %r; there are: %s"
                              % (conf.get("TEMPLATE"),
                                 ", ".join(generate_runs.templates_available())))
    generations = conf.get("GENERATIONS")
    if not isinstance(generations, int) or generations < 1:
        raise SubmissionError("GENERATIONS must be a whole number of at least 1")
    return conf


def form():
    """What a submission form needs: settings.py's values, and the choices.

    -> {defaults: {NAME: value}, choices: {NAME: [{value, label, ...}]},
        locked: [NAME]}. Every setting a submission may override is in
    `defaults`, so a form offers a subset and a client never has to guess a
    value the server would have used.
    """
    import evaluators                   # the registry: importing it registers them
    from evaluators import composite

    conf = config.snapshot()
    templates = [name for name in generate_runs.templates_available()
                 if not name.startswith("template_baseline")]
    return {
        "defaults": {name: value for name, value in conf.items()
                     if name not in settings.LOCKED_SETTINGS},
        "choices": {
            "TEMPLATE": [{"value": name, "label": name} for name in templates],
            "EVALUATOR": [{"value": name, "label": name, "description": description,
                           "needs_judge": evaluators.get(name).needs_judge}
                          for name, description in evaluators.available()],
            "JUDGE_BACKEND": [{"value": name, "label": name} for name in evaluators.BACKENDS],
            "COMPOSITE_AGGREGATE": [{"value": name, "label": name,
                                     "description": composite.AGGREGATE_DESCRIPTIONS[name]}
                                    for name in composite.AGGREGATES],
            "COMPOSITE_ON_FAILURE": [{"value": name, "label": name,
                                      "description": composite.ON_FAILURE_DESCRIPTIONS[name]}
                                     for name in composite.ON_FAILURE],
        },
        "locked": list(settings.LOCKED_SETTINGS),
    }


def options_for(options):
    """The main.py options a submission asked for, checked. -> dict."""
    options = options or {}
    if not isinstance(options, dict):
        raise SubmissionError("options must be an object")
    unknown = sorted(set(options) - set(settings.MAIN_OPTIONS))
    if unknown:
        raise SubmissionError("unknown option(s): %s; there are: %s"
                              % (", ".join(unknown), ", ".join(settings.MAIN_OPTIONS)))
    out = {}
    if "no_test" in options:
        out["no_test"] = bool(options["no_test"])
    for name in ("limit", "timeout"):
        if name in options:
            value = options[name]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise SubmissionError("%s must be a whole number >= 0" % name)
            out[name] = value
    if options.get("test_min_quality") is not None:
        try:
            out["test_min_quality"] = float(options["test_min_quality"])
        except (TypeError, ValueError):
            raise SubmissionError("test_min_quality must be a number")
    return out


def dataset_lines(split, value):
    """One split of a submission as the lines of a dataset file. -> [str]."""
    if isinstance(value, dict):
        name = value.get("file")
        if not isinstance(name, str) or set(value) != {"file"}:
            raise SubmissionError("%s: a dataset object must be {\"file\": name}" % split)
        shared = os.path.abspath(os.path.join(_ROOT, settings.SHARED_DATASETS_DIR))
        path = os.path.abspath(os.path.join(shared, name))
        if os.path.commonpath([shared, path]) != shared or not os.path.isfile(path):
            raise SubmissionError("%s: no shared dataset %r" % (split, name))
        with open(path, encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    if isinstance(value, list):
        lines = []
        for number, record in enumerate(value, 1):
            if isinstance(record, dict):
                lines.append(json.dumps(record, ensure_ascii=False))
            elif isinstance(record, str) and record.strip() and "\n" not in record:
                lines.append(record.strip())
            else:
                raise SubmissionError(
                    "%s record %d: a record is an object (with a 'messages' list) "
                    "or a one-line prompt" % (split, number))
        return lines
    raise SubmissionError("%s: a dataset is a list of records, the text of a file, "
                          "or {\"file\": name}" % split)


def write_datasets(folder, datasets, conf):
    """Write each split into the job folder and point its setting at it."""
    if not isinstance(datasets, dict):
        raise SubmissionError("datasets must be an object keyed by split")
    unknown = sorted(set(datasets) - set(store.SPLITS))
    if unknown:
        raise SubmissionError("unknown dataset split(s): %s; there are: %s"
                              % (", ".join(unknown), ", ".join(store.SPLITS)))
    if not datasets.get("training"):
        raise SubmissionError("a job needs a training dataset -- the questions "
                              "every individual is judged on")
    where = os.path.join(folder, "datasets")
    os.makedirs(where, exist_ok=True)
    for split in store.SPLITS:
        setting = add_dataset.SPLIT_SETTINGS[split]
        if datasets.get(split) is None:
            conf[setting] = None
            continue
        lines = dataset_lines(split, datasets[split])
        if not lines:
            raise SubmissionError("%s: the dataset is empty" % split)
        path = os.path.join(where, "%s.jsonl" % split)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.writelines(line + "\n" for line in lines)
        # Read back through the pipeline's own reader, so a record it cannot
        # parse is refused now and not an hour into the job.
        _checked(generate_runs.dataset_records, path)
        conf[setting] = path


def submit(registry, user, payload):
    """Create a job from a submission. -> the queued job row.

    The row is reserved first (the folder is named for its id) and only queued
    once the database is whole; anything that goes wrong in between removes
    both, so a failed submission leaves nothing for the worker to find.
    """
    if not isinstance(payload, dict):
        raise SubmissionError("a submission is a JSON object")
    unknown = sorted(set(payload) - {"label", "settings", "datasets", "options"})
    if unknown:
        raise SubmissionError("unknown field(s): %s" % ", ".join(unknown))
    label = payload.get("label")
    if label is not None and not isinstance(label, str):
        raise SubmissionError("label must be a string")
    conf = settings_for(payload.get("settings"))
    options = options_for(payload.get("options"))

    job = registry.reserve_job(user["id"], label, options)
    folder = registry.folder(job)
    try:
        os.makedirs(folder, exist_ok=True)
        write_datasets(folder, payload.get("datasets") or {}, conf)
        with open(os.path.join(folder, "submission.json"), "w", encoding="utf-8") as handle:
            json.dump({"label": label, "settings": payload.get("settings") or {},
                       "options": options, "user": user["name"]}, handle, indent=2)
        conn = store.connect(registry.database(job))
        try:
            run_id = store.create_run(conn, template=conf.get("TEMPLATE") or "template_code.py",
                                      label=label)
            store.save_settings(conn, run_id, conf)
            _checked(add_dataset.save_all, conn, run_id, conf, lambda *_: None)
        finally:
            conn.close()
    except BaseException:
        registry.discard(job["id"])
        shutil.rmtree(folder, ignore_errors=True)
        try:
            os.rmdir(os.path.dirname(folder))       # the user's folder, if now empty
        except OSError:
            pass
        raise
    return registry.enqueue(job["id"], run_id)

