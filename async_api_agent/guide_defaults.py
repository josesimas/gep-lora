"""
guide_defaults.py - One user's defaults for the whole guide: training, blending, after.

The guide starts every conversation from the knobs in settings.py (and
create_lora.py's recipe, through train.defaults()). Those are the server's, the
same for everyone; this module is the layer above them that is a person's own:
the values they want a new conversation to start from, set once on the
/settings.html page and kept in the API's database (registry.py's
`guide_defaults` table, one JSON document per user).

    FIELDS         every default a person may set: its group, its words, its
                   type and limits, and the server's own value beside it
    check(values)  what a page sent, checked -> the values to keep
    form(saved)    what the page needs: the fields, the server's values, the
                   saved ones and what is in force
    applied(...)   a context manager: the user's saved values are the ones
                   `value()`, `wait_choices()`, `recipe()` and `training()`
                   read for the length of one request

**A saved default is only a default.** It is what a conversation starts from,
never a limit on it: the chat may still change the ranks, the epochs, a
learning rate or the size of a search, within the limits settings.py sets
(MAX_LORAS, MAX_BLEND_POPULATION, ...), which are the server's and are not
offered here. And it reaches the pipeline only through the requests the guide
already makes -- the POST /loras bodies carry the training values, the POST
/jobs body the search's -- so train.py and submit.py check a saved default
exactly as they check anything else.

**Nothing is copied.** A value left unset is not stored as the server's value
of the day: it goes on meaning "whatever the server says", so a person who
changed one learning rate still gets the next server default for everything
else.

Why a context variable rather than a `defaults` argument on every planner and
blending function: settings.py is read in a dozen places across the guide, and
each of them is already right about *which* knob it wants. `value(NAME)` is
settings.NAME with the user's saved value in front of it, set by routes.py
around each /agent request, so a function called outside one -- a test, the
command line -- reads settings.py as it always did.
"""

import contextlib
import contextvars

from async_api import train
from async_api_agent import settings
from config import settings as config

GROUPS = (
    ("guide", "The guide", "Which model phrases what the guide says."),
    ("training", "Creating the LoRAs", "What each LoRA is trained on and how."),
    ("blend", "Blending them", "The search that combines your LoRAs, and how it is judged."),
    ("release", "Testing and going live", "The questions a chosen blend is checked on."),
)


def _evaluators():
    import evaluators                              # the registry; loads every evaluator
    return [{"value": name, "label": name, "description": description}
            for name, description in evaluators.available()]


def _providers():
    return [{"value": name, "label": entry["label"]} for name, entry in settings.PROVIDERS.items()]


# One entry per default. `setting` is the settings.py name value() stands in
# for; `option` the create_lora option (a POST /loras settings key) a training
# default is; `job` the sweep setting a blend default becomes. `server` is a
# callable, so the server's value is read when asked, after any GEP_AGENT_
# override. Kinds: int, float, bool, text, choice, ranks, seed -- a whole
# number, or "" for random (None) -- and tristate -- on,
# off, or "" for the model's own habit (None). `attach` names the field a
# default is drawn beside rather than on a row of its own: a thinking switch
# belongs to the model it switches.
FIELDS = [
    # --- the guide
    {"key": "provider", "group": "guide", "label": "Chat provider", "kind": "choice",
     "choices": _providers, "setting": "PROVIDER", "server": lambda: settings.PROVIDER,
     "help": "Who the guide talks through. A hosted one needs its key on the server."},
    {"key": "model", "group": "guide", "label": "Chat model", "kind": "text", "optional": True,
     "setting": "MODEL", "server": lambda: settings.MODEL,
     "help": "Empty: the provider's own default, or the first model a local server lists."},
    {"key": "thinking", "group": "guide", "label": "Thinking", "kind": "tristate", "attach": "model",
     "setting": "THINKING", "server": lambda: settings.THINKING,
     "help": "Whether the chat model thinks before it replies. Off is much faster on a "
             "reasoning model; unset leaves it to the model."},

    # --- training
    {"key": "base_model", "group": "training", "label": "Base model", "kind": "text",
     "option": "base_model", "server": lambda: train.defaults()["base_model"],
     "help": "What the LoRAs are trained on. LoRAs are blended only with others on the same one."},
    {"key": "chat_template", "group": "training", "label": "Chat template", "kind": "text",
     "optional": True, "option": "chat_template", "choices": lambda: list(train.CHAT_TEMPLATES),
     "server": lambda: train.defaults()["chat_template"],
     "help": "The words every prompt is written in; empty is the base model's own."},
    {"key": "ranks", "group": "training", "label": "Ranks", "kind": "ranks",
     "setting": "LORA_RANKS", "server": lambda: list(settings.LORA_RANKS),
     "help": "The ranks the LoRAs are trained at, taken in turn."},
    {"key": "lora_count", "group": "training", "label": "Number of LoRAs", "kind": "int",
     "optional": True, "low": 1, "high": settings.MAX_LORAS, "setting": "LORA_COUNT",
     "server": lambda: settings.LORA_COUNT,
     "help": "How many LoRAs a plan trains. Empty: one per rank. More than the ranks "
             "repeats them, each repeat under its own seed so the LoRAs differ."},
    {"key": "epochs_quick", "group": "training", "label": "Epochs: “A quick look”", "kind": "float",
     "low": settings.MIN_EPOCHS, "high": settings.MAX_EPOCHS, "wait": 0,
     "server": lambda: settings.WAIT_CHOICES[0]["epochs"],
     "help": "The three answers to “how long can you wait?”, as passes over the data."},
    {"key": "epochs_balanced", "group": "training", "label": "Epochs: “A solid result”",
     "kind": "float", "low": settings.MIN_EPOCHS, "high": settings.MAX_EPOCHS, "wait": 1,
     "server": lambda: settings.WAIT_CHOICES[1]["epochs"]},
    {"key": "epochs_thorough", "group": "training", "label": "Epochs: “The full recipe”",
     "kind": "float", "low": settings.MIN_EPOCHS, "high": settings.MAX_EPOCHS, "wait": 2,
     "server": lambda: settings.WAIT_CHOICES[2]["epochs"]},
    {"key": "learning_rate", "group": "training", "label": "Learning rate", "kind": "number",
     "option": "learning_rate", "server": lambda: train.defaults()["learning_rate"]},
    {"key": "alpha", "group": "training", "label": "LoRA alpha", "kind": "number",
     "option": "alpha", "server": lambda: train.defaults()["alpha"]},
    {"key": "dropout", "group": "training", "label": "LoRA dropout", "kind": "number",
     "option": "dropout", "server": lambda: train.defaults()["dropout"]},
    {"key": "max_seq", "group": "training", "label": "Max sequence length", "kind": "number",
     "option": "max_seq", "server": lambda: train.defaults()["max_seq"]},
    {"key": "batch_size", "group": "training", "label": "Batch size", "kind": "number",
     "option": "batch_size", "server": lambda: train.defaults()["batch_size"]},
    {"key": "grad_accum", "group": "training", "label": "Gradient accumulation", "kind": "number",
     "option": "grad_accum", "server": lambda: train.defaults()["grad_accum"]},
    {"key": "warmup_steps", "group": "training", "label": "Warm-up steps", "kind": "number",
     "option": "warmup_steps", "server": lambda: train.defaults()["warmup_steps"]},
    {"key": "weight_decay", "group": "training", "label": "Weight decay", "kind": "number",
     "option": "weight_decay", "server": lambda: train.defaults()["weight_decay"]},
    {"key": "max_steps", "group": "training", "label": "Max steps", "kind": "number",
     "optional": True, "option": "max_steps", "server": lambda: train.defaults()["max_steps"],
     "help": "Empty: as many as the epochs make."},
    {"key": "scheduler", "group": "training", "label": "Scheduler", "kind": "choice",
     "choices": lambda: list(train.CHOICES["scheduler"]), "option": "scheduler",
     "server": lambda: train.defaults()["scheduler"]},
    {"key": "optim", "group": "training", "label": "Optimiser", "kind": "choice",
     "choices": lambda: list(train.CHOICES["optim"]), "option": "optim",
     "server": lambda: train.defaults()["optim"]},
    {"key": "mock", "group": "training", "label": "Practice run", "kind": "bool",
     "setting": "MOCK", "server": lambda: bool(settings.MOCK),
     "help": "Start with “practice run” ticked: nothing is trained and no GPU is needed."},

    # --- blending
    {"key": "blend_generations", "group": "blend", "label": "Generations", "kind": "int",
     "low": 1, "high": settings.MAX_BLEND_GENERATIONS, "setting": "BLEND_GENERATIONS",
     "server": lambda: settings.BLEND_GENERATIONS,
     "help": "After the first: a search is 1 + this many rounds."},
    {"key": "blend_population", "group": "blend", "label": "Blends per generation", "kind": "int",
     "low": settings.MIN_BLEND_POPULATION, "high": settings.MAX_BLEND_POPULATION,
     "setting": "BLEND_POPULATION", "server": lambda: settings.BLEND_POPULATION},
    {"key": "blend_questions", "group": "blend", "label": "Questions each blend is judged on",
     "kind": "int", "low": 1, "high": settings.MAX_BLEND_QUESTIONS,
     "setting": "BLEND_QUESTIONS", "server": lambda: settings.BLEND_QUESTIONS},
    {"key": "blend_test_questions", "group": "blend", "label": "Questions kept for testing",
     "kind": "int", "low": 0, "high": settings.MAX_BLEND_QUESTIONS,
     "setting": "BLEND_TEST_QUESTIONS", "server": lambda: settings.BLEND_TEST_QUESTIONS,
     "help": "From the questions left over: every blend is asked them after the search."},
    {"key": "blend_validation_questions", "group": "blend",
     "label": "Questions kept for verification", "kind": "int", "low": 0,
     "high": settings.MAX_BLEND_QUESTIONS, "setting": "BLEND_VALIDATION_QUESTIONS",
     "server": lambda: settings.BLEND_VALIDATION_QUESTIONS,
     "help": "Kept apart from the testing ones: the chosen blend beside its LoRAs."},
    {"key": "evaluator", "group": "blend", "label": "Evaluator", "kind": "choice",
     "choices": _evaluators, "job": "EVALUATOR", "server": lambda: config.EVALUATOR,
     "help": "How every answer is scored -- what the search optimises for."},
    {"key": "judge_model", "group": "blend", "label": "Judge model", "kind": "text",
     "optional": True, "job": "JUDGE_MODEL", "server": lambda: config.JUDGE_MODEL,
     "help": "For the evaluators that ask a model. Empty: the first the endpoint lists."},
    {"key": "judge_thinking", "group": "blend", "label": "Thinking", "kind": "tristate",
     "attach": "judge_model", "job": "JUDGE_THINKING", "server": lambda: config.JUDGE_THINKING,
     "help": "Whether the judge thinks before it grades. Off is most of a reasoning "
             "judge's time saved; unset leaves it to the model."},

    # The search's own seeds: an int repeats that part of a search exactly, ""
    # (None) draws one when the search is made -- and records it, so even a
    # random search can be repeated afterwards from its stored settings.
    {"key": "seed", "group": "blend", "label": "Seed: the first population", "kind": "seed",
     "job": "SEED", "server": lambda: config.SEED, "more": True,
     "help": "Which chromosomes the first generation starts from."},
    {"key": "weight_seed", "group": "blend", "label": "Seed: the blend weights", "kind": "seed",
     "job": "WEIGHT_MASTER_SEED", "server": lambda: config.WEIGHT_MASTER_SEED, "more": True,
     "help": "The weights w1-w5 each blend is drawn under."},
    {"key": "selection_seed", "group": "blend", "label": "Seed: selection", "kind": "seed",
     "job": "SELECTION_MASTER_SEED", "server": lambda: config.SELECTION_MASTER_SEED,
     "more": True, "help": "The roulette wheel that picks which blends are copied."},
    {"key": "mutation_seed", "group": "blend", "label": "Seed: mutation", "kind": "seed",
     "job": "MUTATION_MASTER_SEED", "server": lambda: config.MUTATION_MASTER_SEED,
     "more": True, "help": "Which symbols of a chromosome change."},
    {"key": "weight_mutation_seed", "group": "blend", "label": "Seed: weight mutation",
     "kind": "seed", "job": "WEIGHT_MUTATION_MASTER_SEED",
     "server": lambda: config.WEIGHT_MUTATION_MASTER_SEED, "more": True,
     "help": "Which blend weights are swapped for others."},

    # --- after the search
    {"key": "verify_questions", "group": "release", "label": "Verification questions from a file",
     "kind": "int", "low": 1, "high": settings.MAX_VERIFY_QUESTIONS,
     "setting": "VERIFY_QUESTIONS", "server": lambda: settings.VERIFY_QUESTIONS,
     "help": "When a verification asks a demo dataset or a LoRA's data rather than the "
             "search's own split, which carries its own size."},
]

BY_KEY = {field["key"]: field for field in FIELDS}

# The largest seed a search takes: start_run's own limit on the seeds it draws.
SEED_LIMIT = 2 ** 31 - 1
_SETTING = {field["setting"]: field["key"] for field in FIELDS if field.get("setting")}

_CURRENT = contextvars.ContextVar("guide_defaults", default=None)


class DefaultsError(ValueError):
    """A default that cannot be kept, in words for the person."""


# --- checking --------------------------------------------------------------------


def _choices(field):
    found = field["choices"]() if callable(field.get("choices")) else field.get("choices") or []
    return [one["value"] if isinstance(one, dict) else one for one in found]


def _one(field, value):
    kind, label = field["kind"], field["label"]
    if kind in ("bool", "tristate"):
        if not isinstance(value, bool):
            raise DefaultsError("%s must be true or false" % label)
        return value
    if kind == "ranks":
        from async_api_agent import planner       # the chat's own rule for ranks
        try:
            return planner.check_ranks(value)
        except planner.SessionError as error:
            raise DefaultsError("%s: %s" % (label, error))
    if kind == "number":                          # a create_lora option: train.py's rules
        try:
            return train._number(field["option"], value)
        except train.TrainError as error:
            raise DefaultsError(str(error))
    if kind == "seed":
        if isinstance(value, bool) or not isinstance(value, (int, float))                 or value != int(value) or not 0 <= value <= SEED_LIMIT:
            raise DefaultsError("%s must be a whole number from 0 to %d, or empty for random"
                                % (label, SEED_LIMIT))
        return int(value)
    if kind in ("int", "float"):
        whole = kind == "int"
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or (whole and value != int(value)) \
                or not field["low"] <= value <= field["high"]:
            raise DefaultsError("%s must be a %s from %s to %s"
                                % (label, "whole number" if whole else "number",
                                   field["low"], field["high"]))
        return int(value) if whole else float(value)
    if kind == "choice":
        allowed = _choices(field)
        if value not in allowed:
            raise DefaultsError("%s must be one of %s" % (label, ", ".join(map(str, allowed))))
        return value
    if not isinstance(value, str) or len(value) > 200:
        raise DefaultsError("%s must be some text" % label)
    return " ".join(value.split())


def check(values):
    """What a page sent, checked. -> {key: value}, only the keys given a value.

    A key sent as None (or an empty text) is not kept -- the server's value
    goes on applying -- except for an `optional` text, where "" is itself a
    choice: no chat model named, the base model's own template."""
    if not isinstance(values, dict):
        raise DefaultsError("defaults must be an object of {name: value}")
    unknown = sorted(set(values) - set(BY_KEY))
    if unknown:
        raise DefaultsError("unknown default(s): %s" % ", ".join(unknown))
    out = {}
    for key, value in values.items():
        field = BY_KEY[key]
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            if (field.get("optional") and field["kind"] == "text")                     or field["kind"] in ("tristate", "seed"):
                out[key] = ""
            continue
        out[key] = _one(field, value)
    if "provider" not in out and out.get("model"):
        raise DefaultsError("a chat model belongs to a provider: choose the provider too")
    return out


def stored(raw):
    """A saved document as it is read back: anything that no longer checks --
    an evaluator since removed, a limit since lowered -- is dropped rather than
    stopping the guide, and goes back to the server's value."""
    raw = raw if isinstance(raw, dict) else {}
    try:
        return check(raw)
    except DefaultsError:
        pass
    out = {}
    for key, value in raw.items():
        # The model is checked with its provider, which it cannot be saved without.
        one = dict({key: value}, **({"provider": raw["provider"]}
                                    if key == "model" and "provider" in raw else {}))
        try:
            kept = check(one)
        except DefaultsError:
            continue
        if key in kept:
            out[key] = kept[key]
    if "model" in out and "provider" not in out:
        out.pop("model")
    return out


# --- what is in force ------------------------------------------------------------


def server(field):
    return field["server"]()


def effective(saved=None):
    """{key: value}: the saved value where there is one, else the server's.
    An optional text saved as "" means None -- nothing named."""
    saved = _CURRENT.get() if saved is None else saved
    saved = saved or {}
    out = {}
    for field in FIELDS:
        if field["key"] in saved:
            value = saved[field["key"]]
            out[field["key"]] = None if value == "" else value
        else:
            out[field["key"]] = server(field)
    return out


def form(saved=None, updated_at=None):
    """What the defaults page draws. -> {groups, fields, saved, effective}."""
    saved = saved or {}
    fields = []
    for field in FIELDS:
        entry = {name: field[name] for name in ("key", "group", "label", "kind", "low", "high",
                                                "optional", "help", "attach", "more")
                 if name in field}
        if field["kind"] == "seed":
            entry.update(low=0, high=SEED_LIMIT)
        if field.get("choices"):
            found = field["choices"]() if callable(field["choices"]) else field["choices"]
            entry["choices"] = [one if isinstance(one, dict) else {"value": one, "label": one}
                                for one in found]
        entry["server"] = server(field)
        fields.append(entry)
    return {"groups": [{"id": key, "label": label, "about": about} for key, label, about in GROUPS],
            "fields": fields, "saved": saved, "effective": effective(saved),
            "updated_at": updated_at}


@contextlib.contextmanager
def applied(saved):
    """Make `saved` the defaults value() and the rest read, for a block."""
    token = _CURRENT.set(stored(saved))
    try:
        yield
    finally:
        _CURRENT.reset(token)


def value(name):
    """settings.<name>, or the user's saved default for it."""
    key = _SETTING.get(name)
    saved = _CURRENT.get() or {}
    if key is not None and key in saved:
        return None if saved[key] == "" else saved[key]
    return getattr(settings, name)


def wait_choices():
    """settings.WAIT_CHOICES with the user's epochs in them."""
    saved = _CURRENT.get() or {}
    out = [dict(choice) for choice in settings.WAIT_CHOICES]
    for field in FIELDS:
        if "wait" in field and field["key"] in saved and field["wait"] < len(out):
            out[field["wait"]]["epochs"] = saved[field["key"]]
    return out


def training():
    """The create_lora options the user saved: what a POST /loras body has to
    carry for their defaults to be the ones it trains under. -> {option: value}"""
    saved = _CURRENT.get() or {}
    out = {}
    for field in FIELDS:
        if field.get("option") and field["key"] in saved:
            value = saved[field["key"]]
            out[field["option"]] = None if value == "" else value
    return out


def recipe():
    """train.defaults() with the user's training defaults over it."""
    return dict(train.defaults(), **training())


def job_settings():
    """The sweep settings the user saved for a blend search. -> {NAME: value}"""
    saved = _CURRENT.get() or {}
    out = {}
    for field in FIELDS:
        if field.get("job") and field["key"] in saved:
            value = saved[field["key"]]
            out[field["job"]] = None if value == "" else value
    return out
