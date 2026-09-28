"""
train.py - A LoRA trained through the API: the form, the checks, the command.

The API's half of `python -m gep_lora.core.adapters.create_lora`: what a form needs to offer
(`form`, `base_models`), what a request may ask for (`options_for`), the queue
entry it becomes (`submit`), the command line the worker runs (`command`) and
what the folder says about how far it got (`progress`). The training itself --
the recipe, the loss curve, the record written beside the weights and the
catalogue row it keeps up -- is create_lora.py's, and is not repeated here.

A training is **work, like a job**: it loads a base model and holds a card for
minutes to hours, so it cannot happen inside a request. It goes into the
registry's `trainings` table and the same worker takes it, and its console goes
to `train.log` in `user<N>/lora<id>/` beside the user's jobs, with the dataset
it was sent.

**The adapter is its trainer's.** It is written to
`TRAINED_LORAS_DIR/user<N>/<name>` -- inside loras/, where every other adapter
is, so a search's LORA_SLOTS can name it and `python -m gep_lora.core.adapters.catalog scan`
finds it again -- and its catalogue row is reserved here as `queued`, owned by
the user who asked, so their list shows it from the moment it is asked for.
The API shows a catalogue row to its owner and to nobody else (server.py), and
names are unique per owner, so no request here can learn what another user
has: a name another user holds is simply free.

The defaults are create_lora.py's own (`create_lora.defaults()`), except the
base model and chat template, which are the search's -- a LoRA trained here with
nothing changed is one the server's default search can blend.
"""

import json
import os
import re
import shutil

from gep_lora.core.adapters import catalog as lora_catalog
from gep_lora.core.adapters import create_lora
from gep_lora.service import settings
from gep_lora.service import submit as job_submit
from gep_lora.core.config import settings as config
from gep_lora import paths

_ROOT = paths.ROOT

# A name is also a folder name, and a word in a URL and a log line.
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

# (type, lowest, highest, may be left out) for every number a request may set.
NUMBERS = {
    "rank": (int, 1, 1024, False),
    "alpha": (int, 1, 4096, False),
    "dropout": (float, 0.0, 0.9, False),
    "epochs": (float, 0.01, 10000, False),
    "max_steps": (int, 1, 10 ** 7, True),
    "learning_rate": (float, 1e-9, 1.0, False),
    "warmup_steps": (int, 0, 10 ** 6, False),
    "weight_decay": (float, 0.0, 1.0, False),
    "batch_size": (int, 1, 4096, False),
    "grad_accum": (int, 1, 4096, False),
    "max_seq": (int, 16, 1 << 20, False),
    "seed": (int, 0, 2 ** 31 - 1, False),
}
CHOICES = {"scheduler": create_lora.SCHEDULERS, "optim": create_lora.OPTIMIZERS}
TEXTS = ("base_model", "chat_template", "prompt")

# Names unsloth's get_chat_template() takes that the adapters here are likely
# to want -- offered, not enforced: the box takes any name unsloth knows.
CHAT_TEMPLATES = ("qwen-2.5", "qwen3", "chatml", "llama-3.1", "gemma-3", "mistral", "phi-4")


class TrainError(ValueError):
    """What is wrong with a training request, in words for whoever sent it."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def trained_dir():
    where = settings.TRAINED_LORAS_DIR
    return os.path.abspath(where if os.path.isabs(where) else os.path.join(_ROOT, where))


def user_dir(user):
    """Where one user's trained LoRAs go: a folder each, as in JOBS_DIR."""
    return os.path.join(trained_dir(), "user%d" % user["id"])


def defaults():
    out = create_lora.defaults()
    out["base_model"] = config.BASE_MODEL
    out["chat_template"] = config.CHAT_TEMPLATE
    return out


def base_models(catalog, owner):
    """Every base model worth offering, and why. -> [{id, loras, ready,
    local, bytes, search}], the search's own first, then by how many LoRAs.

    Two places say a model is usable here: `owner`'s LoRAs in the catalogue
    (trained on it, so it loads and the pipeline can blend on it) and the
    Hugging Face cache (it is downloaded, so a training starts without a
    fetch). A judge endpoint's models are a third list the page asks for itself
    through /judge/models; those are served names, not Hub ids, so they are
    offered last and for what they are.
    """
    found = {}

    def entry(model_id):
        return found.setdefault(model_id, {"id": model_id, "loras": 0, "ready": 0,
                                           "local": False, "bytes": None,
                                           "search": model_id == config.BASE_MODEL})
    entry(config.BASE_MODEL)
    for row in catalog.all(owner=owner):
        if row["base_model"]:
            one = entry(row["base_model"])
            one["loras"] += 1
            one["ready"] += row["status"] == lora_catalog.READY
    for model in lora_catalog.local_models():
        one = entry(model["id"])
        one["local"] = True
        one["bytes"] = model["bytes"]
    return sorted(found.values(), key=lambda one: (not one["search"], -one["loras"],
                                                   not one["local"], one["id"].lower()))


def form(catalog, user):
    """What a training form needs. -> {defaults, choices, base_models, ...}."""
    return {
        "defaults": defaults(),
        "choices": {"scheduler": list(create_lora.SCHEDULERS),
                    "optim": list(create_lora.OPTIMIZERS),
                    "target_modules": list(create_lora.TARGET_MODULES),
                    "chat_template": list(CHAT_TEMPLATES)},
        "base_models": base_models(catalog, user["name"]),
        "folder": lora_catalog.stored_folder(user_dir(user)),
        "search": {"base_model": config.BASE_MODEL, "chat_template": config.CHAT_TEMPLATE},
    }


def _number(name, value):
    kind, low, high, optional = NUMBERS[name]
    if value is None or value == "":
        if optional:
            return None
        raise TrainError("%s must be given" % name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainError("%s must be a number" % name)
    if kind is int:
        if value != int(value):
            raise TrainError("%s must be a whole number" % name)
        value = int(value)
    else:
        value = float(value)
    if not low <= value <= high:
        raise TrainError("%s must be between %s and %s" % (name, low, high))
    return value


def options_for(body, catalog, user):
    """A request from `user`, checked. -> (name, options, mock).

    Checked here rather than by create_lora.py, for the reason a submission is
    checked in submit.py: a bad request should be a 400 now, not a training
    that fails on the worker in an hour.
    """
    if not isinstance(body, dict):
        raise TrainError("a training request is a JSON object")
    unknown = sorted(set(body) - {"name", "dataset", "settings", "mock"})
    if unknown:
        raise TrainError("unknown field(s): %s" % ", ".join(unknown))

    name = body.get("name")
    if not isinstance(name, str) or not NAME.match(name.strip()):
        raise TrainError("name must be 1-64 letters, digits, '.', '_' or '-', "
                         "starting with a letter or digit")
    name = name.strip()
    if catalog.by_name(name, user["name"]) is not None:
        raise TrainError("you already have a LoRA called %r" % name, 409)
    if os.path.exists(os.path.join(user_dir(user), name)):
        raise TrainError("%s already exists on disk"
                         % lora_catalog.stored_folder(os.path.join(user_dir(user), name)), 409)

    wanted = body.get("settings") or {}
    if not isinstance(wanted, dict):
        raise TrainError("settings must be an object")
    options = defaults()
    unknown = sorted(set(wanted) - set(options))
    if unknown:
        raise TrainError("unknown setting(s): %s; there are: %s"
                         % (", ".join(unknown), ", ".join(sorted(options))))
    options.update(wanted)

    for key in NUMBERS:
        options[key] = _number(key, options.get(key))
    for key, allowed in CHOICES.items():
        if options[key] not in allowed:
            raise TrainError("%s must be one of %s" % (key, ", ".join(allowed)))
    for key in TEXTS:
        value = options.get(key)
        if value is not None and not isinstance(value, str):
            raise TrainError("%s must be a string" % key)
        options[key] = value.strip() or None if isinstance(value, str) else None
    if not options["base_model"]:
        raise TrainError("base_model must name the model to train on")
    if not options["prompt"]:
        options["prompt"] = create_lora.defaults()["prompt"]
    options["no_sample"] = bool(options.get("no_sample"))

    modules = options.get("target_modules")
    if isinstance(modules, str):
        modules = [part.strip() for part in modules.split(",") if part.strip()]
    if not isinstance(modules, list) or not modules \
            or not all(isinstance(one, str) for one in modules):
        raise TrainError("target_modules must be a list of module names")
    strange = sorted(set(modules) - set(create_lora.TARGET_MODULES))
    if strange:
        raise TrainError("unknown target module(s) %s; there are: %s"
                         % (", ".join(strange), ", ".join(create_lora.TARGET_MODULES)))
    # In the order create_lora names them, so two recipes over one set read the same.
    options["target_modules"] = [one for one in create_lora.TARGET_MODULES if one in modules]
    return name, options, bool(body.get("mock"))


def dataset_lines(value):
    """The training data a request sent, as JSON Lines. -> [str].

    Any shape a job's dataset may take (submit.dataset_lines), but every record
    has to be a conversation: a LoRA learns the assistant turns, and a bare
    prompt has none.
    """
    try:
        lines = job_submit.dataset_lines("dataset", value)
    except job_submit.SubmissionError as error:
        raise TrainError(str(error))
    if not lines:
        raise TrainError("the dataset is empty")
    for number, line in enumerate(lines, 1):
        try:
            record = json.loads(line)
        except ValueError:
            raise TrainError("dataset record %d is not a JSON object with a 'messages' "
                             "list -- a LoRA is trained on conversations, not bare prompts"
                             % number)
        messages = record.get("messages") if isinstance(record, dict) else None
        if not isinstance(messages, list) or not any(
                isinstance(turn, dict) and turn.get("role") == "assistant" for turn in messages):
            raise TrainError("dataset record %d has no 'messages' list with an assistant "
                             "turn in it; that turn is what the LoRA learns" % number)
    return lines


def submit(registry, catalog, user, body):
    """Queue one training. -> (training row, catalogue row).

    The training row is reserved first (its folder is named for its id), the
    dataset written into it, the catalogue row reserved as `queued`, and only
    then is the training queued -- anything failing in between takes all of it
    back, so the worker never finds half a request.
    """
    name, options, mock = options_for(body or {}, catalog, user)
    if not isinstance(body, dict) or body.get("dataset") is None:
        raise TrainError("a training needs a dataset: the conversations to learn")
    lines = dataset_lines(body["dataset"])
    folder = os.path.join(user_dir(user), name)
    # What the record calls the data: the shared file it named, or an upload.
    # Never the work folder's copy, which goes when the training row does.
    shared = body["dataset"].get("file") if isinstance(body["dataset"], dict) else None
    source = ("%s/%s" % (settings.SHARED_DATASETS_DIR, shared) if shared
              else "uploaded by %s" % user["name"])
    options["mock"] = mock
    options["dataset_source"] = source
    options["owner"] = user["name"]

    row = registry.reserve_training(user["id"], name, lora_catalog.stored_folder(folder), options)
    work = registry.training_folder(row)
    lora = None
    try:
        os.makedirs(work, exist_ok=True)
        with open(os.path.join(work, "dataset.jsonl"), "w", encoding="utf-8", newline="\n") as handle:
            handle.writelines(line + "\n" for line in lines)
        try:
            lora = catalog.add(name, folder, lora_catalog.QUEUED, "trained",
                               owner=user["name"], base_model=options["base_model"],
                               rank=options["rank"], alpha=options["alpha"],
                               target_modules=options["target_modules"],
                               chat_template=options["chat_template"], mock=int(mock),
                               recipe={key: value for key, value in options.items()
                                       if key not in ("mock", "dataset_source", "owner")},
                               dataset=source, records=len(lines))
        except ValueError as error:
            raise TrainError(str(error), 409)
        queued = registry.enqueue_training(row["id"], lora["id"])
    except BaseException:
        registry.discard_training(row["id"])
        shutil.rmtree(work, ignore_errors=True)
        if lora is not None:
            catalog.remove(lora["id"])
        raise
    return queued, lora


def command(python, row, work, catalog_path):
    """The create_lora.py command line for one queued training."""
    options = json.loads(row["options"] or "{}")
    argv = [python, "-u", "-m", "gep_lora.core.adapters.create_lora",
            lora_catalog.absolute(row["folder"]),
            "--dataset", os.path.join(work, "dataset.jsonl"),
            "--dataset-source", options.get("dataset_source") or "uploaded",
            "--name", row["name"], "--catalog", catalog_path,
            "--base-model", options["base_model"],
            "--target-modules", ",".join(options["target_modules"]),
            "--scheduler", options["scheduler"], "--optim", options["optim"]]
    for key in NUMBERS:
        if options.get(key) is not None:
            argv += ["--" + key.replace("_", "-"), repr(options[key])
                     if isinstance(options[key], float) else str(options[key])]
    if options.get("owner"):
        argv += ["--owner", options["owner"]]
    if options.get("chat_template"):
        argv += ["--chat-template", options["chat_template"]]
    if options.get("prompt"):
        argv += ["--prompt", options["prompt"]]
    if options.get("no_sample"):
        argv.append("--no-sample")
    if options.get("mock"):
        argv += ["--mock", "--mock-delay", str(settings.MOCK_TRAINING_DELAY)]
    return argv


def progress(folder):
    """How far the training writing `folder` has got, from its training.json.
    -> {status, step, of, epoch, loss, seconds} or None before it has started."""
    try:
        with open(os.path.join(lora_catalog.absolute(folder), lora_catalog.RECIPE),
                  encoding="utf-8") as handle:
            recipe = json.load(handle)
    except (OSError, ValueError):
        return None
    out = dict(recipe.get("progress") or {})
    out["status"] = recipe.get("status")
    return out


def dataset_path(row, training=None, registry=None):
    """The data a LoRA was (or will be) trained on: the copy beside its weights,
    else the one its queued training was sent. -> a path, or None."""
    copy = os.path.join(lora_catalog.absolute(row["folder"]), create_lora.DATASET_COPY)
    if os.path.isfile(copy):
        return copy
    if training is not None and registry is not None:
        sent = os.path.join(registry.training_folder(training), "dataset.jsonl")
        if os.path.isfile(sent):
            return sent
    return None


def preview(path, limit=20):
    """The first `limit` records of a dataset as {question, answer}, and how
    many it holds. -> {records, total}."""
    records, total = [], 0
    with open(path, encoding="utf-8-sig", errors="replace") as handle:
        text = handle.read()
    stripped = text.strip()
    if stripped.startswith("["):
        try:
            items = json.loads(stripped)
        except ValueError:
            items = []
    else:
        items = []
        for line in stripped.splitlines():
            if line.strip():
                try:
                    items.append(json.loads(line))
                except ValueError:
                    items.append(line.strip())
    total = len(items)
    for item in items[:limit]:
        messages = item.get("messages") if isinstance(item, dict) else None
        if isinstance(messages, list):
            turns = [turn for turn in messages if isinstance(turn, dict)]
            ask = next((t.get("content") for t in turns if t.get("role") == "user"), None)
            answer = next((t.get("content") for t in turns if t.get("role") == "assistant"), None)
            records.append({"question": ask, "answer": answer})
        else:
            records.append({"question": str(item), "answer": None})
    return {"records": records, "total": total}
