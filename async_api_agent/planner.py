"""
planner.py - "How long can you wait?" turned into epochs, a time, and a plan.

Three things the agent needs numbers for, and none of them is the model's:

  * `estimate()`  how long N epochs of this dataset take, for all the LoRAs
                  the agent trains: steps from the dataset's size and
                  create_lora.py's batch settings, seconds per step measured
                  from LoRAs already trained here on the same base model (a
                  straight line through their steps and seconds, so the fixed
                  cost of loading a model is not read as a slow step), else
                  settings.SECONDS_PER_STEP;
  * `read_wait()` a typed answer ("about an hour", "5 epochs", "the quick
                  one") as epochs, without a model -- the fallback for when
                  none answers, and the check on one that does;
  * `plan()`      the POST /loras bodies that train them: one per rank in
                  settings.LORA_RANKS, named free in the user's catalogue, and
                  with the dataset's own first question as the smoke test, so
                  the sample a finished LoRA gives is on its own subject.

What each of them works from is the **session**: what the person has asked
for so far in the chat -- which part of the dataset (selection.py), which
ranks, how many epochs, any training option they changed, a name, a
smoke-test prompt, a practice run. The page keeps it and sends it back with
every step; `session_of()` checks it on the way in, with the limits train.py
checks a POST /loras against, so a session the agent built is one the API
will take.

The plan is only proposed: the page sends it to the async API's POST /loras
itself, which checks every body exactly as it checks the demo page's.
"""

import math
import os
import re
import statistics

from adapters import catalog as lora_catalog
from async_api import settings as api_settings
from async_api import train
from async_api_agent import selection as picking
from async_api_agent import settings

# What a mocked training costs besides its steps: a python start, no model.
MOCK_OVERHEAD_SECONDS = 3


# The training options the chat may change, beside rank and epochs (which
# have their own keys). Each is checked by train.py's own rules.
OPTIONS = ("learning_rate", "alpha", "dropout", "max_seq", "batch_size", "grad_accum",
           "warmup_steps", "weight_decay", "max_steps", "scheduler", "optim")

# The ranks a plan may give its LoRAs. How many it may train is
# settings.MAX_LORAS.
RANKS = (1, 256)


class SessionError(ValueError):
    """A session value the plan cannot use, in words for the person."""


def recipe():
    """create_lora.py's defaults as the API would train them."""
    return train.defaults()


def check_ranks(ranks):
    if isinstance(ranks, int) and not isinstance(ranks, bool):
        ranks = [ranks]
    if not isinstance(ranks, list) or not ranks:
        raise SessionError("ranks must be a list of 1 to %d whole numbers" % settings.MAX_LORAS)
    if len(ranks) > settings.MAX_LORAS:
        raise SessionError("at most %d LoRAs can be trained at once" % settings.MAX_LORAS)
    out = []
    for rank in ranks:
        whole = (not isinstance(rank, bool) and isinstance(rank, (int, float))
                 and rank == int(rank))
        if not whole or not RANKS[0] <= rank <= RANKS[1]:
            raise SessionError("a rank must be a whole number from %d to %d" % RANKS)
        out.append(int(rank))
    return out


def check_options(options):
    """Training options, checked by train.py's own rules. -> dict."""
    if not isinstance(options, dict):
        raise SessionError("options must be an object")
    unknown = sorted(set(options) - set(OPTIONS))
    if unknown:
        raise SessionError("unknown option(s): %s; there are: %s"
                           % (", ".join(unknown), ", ".join(OPTIONS)))
    out = {}
    for key, value in options.items():
        if value is None:
            continue
        try:
            if key in train.CHOICES:
                if value not in train.CHOICES[key]:
                    raise train.TrainError("%s must be one of %s"
                                           % (key, ", ".join(train.CHOICES[key])))
                out[key] = value
            else:
                out[key] = train._number(key, value)
        except train.TrainError as error:
            raise SessionError(str(error))
    return out


def check_name(name):
    if name is None:
        return None
    stem = stem_of(str(name))
    if not train.NAME.match(stem):
        raise SessionError("a name must be letters, digits, '.', '_' or '-'")
    return stem


def session_of(raw):
    """What the page sent as its session, checked. -> the full session."""
    raw = raw if isinstance(raw, dict) else {}
    try:
        chosen = picking.clean(raw.get("selection"))
    except picking.SelectionError as error:
        raise SessionError(str(error))
    epochs = raw.get("epochs")
    if epochs is not None:
        number = not isinstance(epochs, bool) and isinstance(epochs, (int, float))
        if not number or not settings.MIN_EPOCHS <= epochs <= settings.MAX_EPOCHS:
            raise SessionError("epochs must be between %s and %s"
                               % (settings.MIN_EPOCHS, settings.MAX_EPOCHS))
        epochs = float(epochs)
    prompt = raw.get("prompt")
    if prompt is not None and (not isinstance(prompt, str) or not prompt.strip()):
        raise SessionError("the test prompt must be some text")
    from async_api_agent import blending          # it plans from this module's facts
    try:
        blend = blending.blend_of(raw.get("blend"))
    except blending.BlendError as error:
        raise SessionError(str(error))
    return {"selection": chosen,
            "blend": blend,
            "ranks": check_ranks(raw["ranks"]) if raw.get("ranks") else list(settings.LORA_RANKS),
            "epochs": epochs,
            "options": check_options(raw.get("options") or {}),
            "name": check_name(raw.get("name")),
            "prompt": " ".join(prompt.split())[:300] if prompt else None,
            "mock": bool(raw.get("mock"))}


def steps_per_epoch(records, options=None):
    options = options or recipe()
    return max(1, math.ceil(max(1, records) / float(options["batch_size"] * options["grad_accum"])))


def speed(catalog, base_model, mock=False):
    """-> (seconds per step, overhead seconds, where that came from)."""
    if mock:
        return (float(api_settings.MOCK_TRAINING_DELAY), MOCK_OVERHEAD_SECONDS,
                "a practice run's fixed pace")
    points = []
    for row in catalog.all(base_model=base_model, status=lora_catalog.READY):
        if row["mock"] or not row["seconds"] or not row["steps"]:
            continue
        points.append((float(row["steps"]), float(row["seconds"])))
    if len(points) >= 2 and len({steps for steps, _ in points}) >= 2:
        mean_x = statistics.fmean(x for x, _ in points)
        mean_y = statistics.fmean(y for _, y in points)
        spread = sum((x - mean_x) ** 2 for x, _ in points)
        slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / spread
        overhead = mean_y - slope * mean_x
        if slope > 0 and overhead >= 0:
            return (slope, overhead, "measured from %d LoRA(s) trained here on this base model"
                    % len(points))
    if points:
        rates = [max(0.05, (seconds - settings.OVERHEAD_SECONDS) / steps)
                 for steps, seconds in points]
        return (statistics.median(rates), settings.OVERHEAD_SECONDS,
                "measured from %d LoRA(s) trained here on this base model" % len(points))
    return (settings.SECONDS_PER_STEP, settings.OVERHEAD_SECONDS,
            "a rough guess: no LoRA has been trained on this base model here yet")


def human(seconds):
    """A duration as a person says it: 'about 40 seconds', 'about 1 h 20 min'."""
    seconds = max(1, int(round(seconds)))
    if seconds < 90:
        return "about %d seconds" % seconds
    minutes = int(round(seconds / 60.0))
    if minutes < 90:
        return "about %d minute%s" % (minutes, "" if minutes == 1 else "s")
    hours, minutes = divmod(minutes, 60)
    return "about %d h %02d min" % (hours, minutes) if minutes else "about %d hours" % hours


def estimate(catalog, records, epochs, mock=False, base_model=None, session=None):
    """How long `epochs` epochs of `records` records take, all LoRAs together
    -- the session's LoRAs, under its options."""
    session = session or session_of(None)
    options = dict(recipe(), **session["options"])
    base_model = base_model or options["base_model"]
    per_epoch = steps_per_epoch(records, options)
    steps = max(1, int(math.ceil(per_epoch * epochs)))
    if options.get("max_steps"):
        steps = min(steps, options["max_steps"])
    per_step, overhead, source = speed(catalog, base_model, mock)
    loras = len(session["ranks"])
    per_lora = overhead + steps * per_step
    return {"epochs": epochs, "loras": loras, "steps_per_epoch": per_epoch,
            "steps": steps, "seconds_per_step": round(per_step, 3),
            "overhead_seconds": round(overhead, 1),
            "seconds_per_epoch": round(per_epoch * per_step * loras, 1),
            "seconds_per_lora": round(per_lora, 1), "seconds": round(per_lora * loras, 1),
            "time": human(per_lora * loras), "source": source,
            "batch": options["batch_size"] * options["grad_accum"]}


def choices(catalog, records, mock=False, session=None):
    """settings.WAIT_CHOICES, each with its estimate on this dataset."""
    out = []
    for choice in settings.WAIT_CHOICES:
        found = estimate(catalog, records, choice["epochs"], mock, session=session)
        out.append(dict(choice, estimate=found, time=found["time"]))
    return out


def epochs_for(catalog, records, minutes, mock=False, session=None):
    """The most epochs that fit in `minutes`, all LoRAs together."""
    one = estimate(catalog, records, 1, mock, session=session)
    budget = minutes * 60.0 - one["overhead_seconds"] * one["loras"]
    return clamp(max(settings.MIN_EPOCHS,
                     math.floor(max(0.0, budget) / max(one["seconds_per_epoch"], 1e-6))))


def clamp(epochs):
    return max(settings.MIN_EPOCHS, min(settings.MAX_EPOCHS, float(epochs)))


_NUMBER = r"(\d+(?:[.,]\d+)?)"
_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
          "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twenty": 20, "thirty": 30,
          "forty": 40, "fifty": 50, "sixty": 60, "a couple of": 2, "a few": 3}


def read_wait(text, per_epoch_seconds, overhead_seconds):
    """A typed answer as epochs, or None. -> (epochs or None, how it was read)."""
    said = " " + (text or "").lower().replace(",", ".") + " "
    for word, value in sorted(_WORDS.items(), key=lambda pair: -len(pair[0])):
        said = re.sub(r"(?<![a-z])%s(?![a-z])" % re.escape(word), str(value), said)
    said = said.replace("half 1 hour", "30 minutes").replace("half an hour", "30 minutes")

    found = re.search(_NUMBER + r"\s*(?:epochs?|passes|pass|rounds?)\b", said)
    if found:
        epochs = clamp(float(found.group(1)))
        return epochs, "%g epoch(s), as you said" % epochs

    budget = 0.0
    for pattern, scale in ((r"\s*(?:h|hr|hrs|hours?)\b", 3600),
                           (r"\s*(?:m|min|mins|minutes?)\b", 60),
                           (r"\s*(?:s|sec|secs|seconds?)\b", 1)):
        for match in re.finditer(_NUMBER + pattern, said):
            budget += float(match.group(1)) * scale
    if "overnight" in said:
        budget = budget or 8 * 3600
    if budget:
        epochs = math.floor(max(0.0, budget - overhead_seconds) / max(per_epoch_seconds, 1e-6))
        epochs = clamp(max(epochs, settings.MIN_EPOCHS))
        return epochs, "a budget of %s" % human(budget).replace("about ", "")

    for choice in settings.WAIT_CHOICES:
        names = [choice["id"], choice["label"].lower()] + choice["label"].lower().split()[-1:]
        if any(" %s " % name in said or name in said.strip() for name in names):
            return float(choice["epochs"]), "the option “%s”" % choice["label"]
    for hint, index in (("quick", 0), ("fast", 0), ("short", 0), ("middle", 1), ("medium", 1),
                        ("balanced", 1), ("thorough", -1), ("full", -1), ("longest", -1),
                        ("best", -1)):
        if hint in said:
            choice = settings.WAIT_CHOICES[index]
            return float(choice["epochs"]), "the option “%s”" % choice["label"]

    bare = re.fullmatch(r"\s*" + _NUMBER + r"\s*", said)
    if bare:
        epochs = clamp(float(bare.group(1)))
        return epochs, "%g epoch(s)" % epochs
    return None, "no time or number of epochs in it"


def stem_of(name):
    """A dataset's name as the start of a LoRA's: 'poem_lora_dataset.json' -> 'poem'."""
    stem = os.path.splitext(os.path.basename(name or ""))[0].lower()
    stem = re.sub(r"(_lora)?_dataset$|^finetuning_", "", stem)
    stem = re.sub(r"[^a-z0-9._-]+", "-", stem).strip("-._")
    return (stem or "my-data")[:40]


def free_name(catalog, user, wanted, taken):
    """`wanted`, or wanted-2, -3 ... -- free in the user's catalogue and on disk."""
    candidate, number = wanted, 2
    while (candidate in taken or catalog.by_name(candidate, user["name"]) is not None
           or os.path.exists(os.path.join(train.user_dir(user), candidate))):
        candidate = "%s-%d" % (wanted, number)
        number += 1
    taken.add(candidate)
    return candidate


def plan(catalog, user, analysis, epochs, mock=False, shared_file=None, session=None):
    """The POST /loras bodies that train the agent's LoRAs. -> [body].

    One per rank in the session, with its options, name and test prompt. The
    dataset goes as {"file": name} when it is a shared file sent whole and as
    it is -- so the catalogue records where it came from -- and as the
    normalised lines of the part chosen otherwise.
    """
    session = session or session_of(None)
    first = analysis["samples"][0]["user"] if analysis.get("samples") else None
    prompt = session["prompt"] or (" ".join(first.split())[:300] if first else None)
    as_file = (shared_file and analysis.get("format") == "jsonl"
               and not analysis["stats"]["skipped"] and not analysis.get("selection"))
    dataset = {"file": shared_file} if as_file else "\n".join(analysis["lines"])
    stem = session["name"] or stem_of(analysis.get("name"))
    taken, bodies = set(), []
    for rank in session["ranks"]:
        name = free_name(catalog, user, settings.LORA_NAME.format(stem=stem, rank=rank), taken)
        wanted = dict(session["options"], epochs=float(epochs), rank=int(rank))
        if prompt:
            wanted["prompt"] = prompt
        bodies.append({"name": name, "dataset": dataset, "settings": wanted, "mock": bool(mock)})
    return bodies
