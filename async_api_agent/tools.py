"""
tools.py - What the chat can do, not just say.

The model behind the chat box is given these tools, and a request such as
"only train on the first 20 records at rank 32" becomes calls to them. Each
tool acts on the **session** (planner.session_of) -- the plan so far, which
the page keeps and sends with every message -- or reads the dataset, and
answers with a small JSON result the model reports from. A `Toolbox` holds
one message's worth of that: the session as it changes, the dataset, and
what the page must do afterwards.

Three rules keep it honest:

  * **A tool changes the plan, never the world.** Starting and stopping a
    training, and switching datasets, are *actions* handed back to the page,
    which does them through the API's own endpoints -- POST /loras, its
    cancel, /agent/analyse -- exactly as its buttons do. Nothing here queues
    or trains.
  * **The stage decides what may be asked.** The plan cannot change while a
    training runs; a dataset has to be there before part of it is chosen;
    a training can be started only once its length is known. A tool asked at
    the wrong moment refuses with a reason the model can pass on.
  * **Every value is checked by the rule that will check it later** --
    selection.clean(), planner.check_ranks()/check_options(), train.py's
    limits -- so a plan the chat built is one POST /loras accepts.

The descriptions the model reads are prompts.TOOLS; this module holds only
their parameters. With no model, commands.py turns plain requests into the
same calls.
"""

import copy

from async_api_agent import analysis
from async_api_agent import planner
from async_api_agent import prompts
from async_api_agent import selection as picking
from async_api_agent import settings

# The page's stages, in order.
STAGES = ("intro", "dataset", "analysis", "wait", "confirm", "training", "done")

# Stages in which the plan may change (not while a training runs).
PLANNING = ("intro", "dataset", "analysis", "wait", "confirm", "done")

# Stages in which a training may be started from the chat.
STARTABLE = ("analysis", "wait", "confirm", "done")

_INT = {"type": "integer"}
_NUM = {"type": "number"}
_WORDS = {"type": "array", "items": {"type": "string"}}

# name -> (JSON schema properties, required, needs a dataset, stages, changes the plan)
SPECS = {
    "show_records": ({
        "start": dict(_INT, description="first record number to show, from 1"),
        "count": dict(_INT, description="how many to show, at most 20 (default 5)"),
        "order": {"type": "string", "enum": ["file", "longest", "shortest"]},
        "contains": {"type": "string", "description": "only records mentioning this"},
    }, [], True, STAGES, False),
    "dataset_facts": ({}, [], True, STAGES, False),
    "select_records": ({
        "first": _INT, "last": _INT,
        "start": dict(_INT, description="first record number, from 1"),
        "end": dict(_INT, description="last record number, inclusive"),
        "percent": dict(_NUM, description="this percent of the records, from the top"),
        "sample": dict(_INT, description="this many records picked at random"),
        "seed": dict(_INT, description="the random sample's seed"),
        "contains": dict(_WORDS, description="keep only records mentioning any of these"),
        "excludes": dict(_WORDS, description="drop records mentioning any of these"),
        "min_answer_words": _INT, "max_answer_words": _INT,
        "drop_duplicates": {"type": "boolean"},
        "replace": {"type": "boolean",
                    "description": "start from the whole dataset instead of the part chosen"},
    }, [], True, PLANNING, True),
    "use_all_records": ({}, [], True, PLANNING, True),
    "set_loras": ({
        "ranks": dict(_WORDS, items=_INT, description="one rank per LoRA, e.g. [8, 16]"),
    }, ["ranks"], False, PLANNING, True),
    "set_epochs": ({
        "epochs": dict(_NUM, description="passes over the data"),
        "minutes": dict(_NUM, description="or: a time budget for all the LoRAs together"),
    }, [], True, PLANNING, True),
    "estimate_time": ({"epochs": _NUM}, ["epochs"], True, STAGES, False),
    "set_training_options": (dict({key: _NUM for key in planner.OPTIONS
                                   if key not in ("scheduler", "optim")},
                                  scheduler={"type": "string"}, optim={"type": "string"},
                                  name={"type": "string"}, prompt={"type": "string"},
                                  reset={"type": "boolean"}),
                             [], False, PLANNING, True),
    "show_plan": ({}, [], False, STAGES, False),
    "set_practice_run": ({"on": {"type": "boolean"}}, ["on"], False, PLANNING, True),
    "list_demo_datasets": ({}, [], False, STAGES, False),
    "use_demo_dataset": ({"file": {"type": "string"}}, ["file"], False, PLANNING, False),
    "choose_another_dataset": ({}, [], False, PLANNING, False),
    "list_my_loras": ({}, [], False, STAGES, False),
    "start_training": ({}, [], True, STARTABLE, False),
    "stop_training": ({}, [], False, ("training",), False),
}


class ToolError(ValueError):
    """Why a tool refused, in words the model passes on."""


def schemas():
    """Every tool as (name, description, JSON schema) -- providers.py puts
    them into each provider's own shape."""
    return [(name, prompts.TOOLS[name],
             {"type": "object", "properties": spec[0], "required": spec[1]})
            for name, spec in SPECS.items()]


class Toolbox:
    """One chat message's tools: the session as they change it, and what the
    page must do once the reply is back."""

    def __init__(self, catalog, user, stage, session, dataset=None, shared_datasets=None):
        """`dataset` is (text, name, shared file or None), or None before one
        is given; `shared_datasets` a callable listing the demo datasets."""
        self.catalog = catalog
        self.user = user
        self.stage = stage if stage in STAGES else "intro"
        self.before = planner.session_of(session)
        self.session = copy.deepcopy(self.before)
        self.dataset = dataset
        self.shared_datasets = shared_datasets or (lambda: [])
        self.actions = []
        self.steps = []
        self.preview = None
        self.plan_changed = False
        self._records = None

    # --- what the tools read ---

    def records(self):
        """The whole dataset, parsed. -> [record]."""
        if self._records is None:
            self._records = analysis.parse(self.dataset[0])[0]
        return self._records

    def chosen(self):
        return picking.indexed(self.records(), self.session["selection"])

    def estimate(self, epochs):
        return planner.estimate(self.catalog, max(1, len(self.chosen())), epochs,
                                self.session["mock"], session=self.session)

    # --- running one ---

    def run(self, name, arguments):
        """Run one tool call. -> its result (a JSON-able dict); a refusal is
        {"error": ...}, never an exception, so the model can report it."""
        arguments = arguments if isinstance(arguments, dict) else {}
        try:
            if name not in SPECS:
                raise ToolError("there is no tool called %s" % name)
            _, _, needs_data, stages, changes = SPECS[name]
            if self.stage not in stages:
                raise ToolError(
                    "a training is running; stop it or wait for it first"
                    if self.stage == "training" else
                    "that cannot be done at this step (%s)" % self.stage)
            if needs_data and not self.dataset:
                raise ToolError("there is no dataset yet; give one first")
            result = getattr(self, "_" + name)(**arguments)
        except (ToolError, picking.SelectionError, planner.SessionError,
                analysis.DatasetError) as error:
            result = {"error": str(error)}
        except TypeError as error:            # an argument the tool does not take
            result = {"error": "bad arguments: %s" % error}
        if "error" in result:
            summary = prompts.TOOL_FAILED.format(tool=name, error=result["error"])
        else:
            if SPECS[name][4]:
                self.plan_changed = True
            summary = prompts.TOOL_DONE[name].format(**result)
        self.steps.append({"tool": name, "arguments": arguments,
                           "ok": "error" not in result, "summary": summary})
        return result

    def outcome(self):
        """What the page needs after the reply: the session, what to do, and
        the dataset's facts again when the part in use changed."""
        actions = list(self.actions)
        kinds = {one["type"] for one in actions}
        # The plan is read back again whenever it changed, and always before a
        # start -- a training starts from bodies the person has just been shown.
        replan = "start" in kinds or (self.plan_changed and
                                      not kinds & {"dataset", "choose_dataset", "stop"})
        if replan and self.session["epochs"] and self.stage in STARTABLE:
            actions.insert(0, {"type": "plan", "epochs": self.session["epochs"]})
        elif self.plan_changed and self.stage == "wait" and "dataset" not in kinds:
            actions.insert(0, {"type": "choices"})
        out = {"session": self.session, "actions": actions, "steps": self.steps,
               "preview": self.preview, "analysis": None}
        if self.dataset and self.session["selection"] != self.before["selection"]:
            found = analysis.analyse(self.dataset[0], self.dataset[1], self.session["selection"])
            found.pop("lines")
            out["analysis"] = found
        return out

    # --- the tools ---

    def _show_records(self, start=None, count=5, order="file", contains=None):
        count = max(1, min(20, int(count or 5)))
        kept = self.chosen()
        if contains:
            word = str(contains).lower()
            kept = [(n, r) for n, r in kept if word in picking._text(r)]
        if order in ("longest", "shortest"):
            kept = sorted(kept, key=lambda pair: len(picking._turn(pair[1], "assistant").split()),
                          reverse=order == "longest")
        elif start:
            kept = [(n, r) for n, r in kept if n >= int(start)]
        shown = [{"number": n,
                  "user": analysis._clip(picking._turn(r, "user"), settings.SAMPLE_CHARS),
                  "assistant": analysis._clip(picking._turn(r, "assistant"), settings.SAMPLE_CHARS),
                  "answer_words": len(picking._turn(r, "assistant").split())}
                 for n, r in kept[:count]]
        self.preview = shown
        return {"shown": len(shown), "records": shown}

    def _dataset_facts(self):
        found = analysis.analyse(self.dataset[0], self.dataset[1], self.session["selection"])
        return {"records": found["records"], "total_records": found["total_records"],
                "selection_text": found["selection_text"], "stats": found["stats"],
                "problems": found["problems"],
                "keywords": [one["word"] for one in found["keywords"][:8]]}

    def _select(self, selection):
        total = len(self.records())
        kept = len(picking.apply(self.records(), selection))
        if not kept:
            raise ToolError("that would leave no records (%s); nothing was changed"
                            % picking.describe(selection, 0, total))
        self.session["selection"] = selection
        # The whole selection now in force, so the model reports what *is*
        # set rather than what it meant to set.
        return {"kept": kept, "total": total, "selection_now": selection,
                "duplicates_dropped": bool(selection.get("drop_duplicates")),
                "selection_text": picking.describe(selection, kept, total)}

    def _select_records(self, replace=False, **change):
        base = {} if replace else self.session["selection"]
        return self._select(picking.merge(base, change))

    def _use_all_records(self):
        return self._select({})

    def _set_loras(self, ranks):
        self.session["ranks"] = planner.check_ranks(ranks)
        return {"count": len(self.session["ranks"]), "ranks": self.session["ranks"],
                "ranks_text": ", ".join(str(rank) for rank in self.session["ranks"])}

    def _set_epochs(self, epochs=None, minutes=None):
        if epochs is None and minutes is None:
            raise ToolError("say a number of epochs or of minutes")
        if epochs is None:
            if minutes <= 0:
                raise ToolError("minutes must be above 0")
            epochs = planner.epochs_for(self.catalog, max(1, len(self.chosen())), minutes,
                                        self.session["mock"], self.session)
        self.session["epochs"] = planner.session_of(dict(self.session, epochs=epochs))["epochs"]
        found = self.estimate(self.session["epochs"])
        return {"epochs": self.session["epochs"], "time": found["time"],
                "steps_per_lora": found["steps"]}

    def _estimate_time(self, epochs):
        found = self.estimate(planner.clamp(epochs))
        return {"epochs": found["epochs"], "time": found["time"], "steps_per_lora": found["steps"],
                "source": found["source"]}

    def _set_training_options(self, reset=False, name=None, prompt=None, **options):
        merged = {} if reset else dict(self.session["options"])
        merged.update({key: value for key, value in options.items() if value is not None})
        checked = planner.session_of(dict(self.session, options=merged,
                                          name=name if name is not None else self.session["name"],
                                          prompt=prompt if prompt is not None
                                          else self.session["prompt"]))
        for key in ("options", "name", "prompt"):
            self.session[key] = checked[key]
        said = ["%s=%s" % pair for pair in sorted(self.session["options"].items())]
        if self.session["name"]:
            said.append("name %s" % self.session["name"])
        if prompt is not None:
            said.append("test prompt set")
        return {"options": self.session["options"], "name": self.session["name"],
                "prompt": self.session["prompt"],
                "options_text": ", ".join(said) or "the defaults"}

    def _show_plan(self):
        out = {"ranks": self.session["ranks"], "epochs": self.session["epochs"],
               "options": self.session["options"], "name": self.session["name"],
               "prompt": self.session["prompt"], "practice_run": self.session["mock"],
               "defaults": {key: value for key, value in planner.recipe().items()
                            if key in planner.OPTIONS}}
        if self.dataset:
            out["selection_text"] = picking.describe(self.session["selection"],
                                                     len(self.chosen()), len(self.records()))
            if self.session["epochs"]:
                out["time"] = self.estimate(self.session["epochs"])["time"]
        return out

    def _set_practice_run(self, on):
        self.session["mock"] = bool(on)
        self.actions.append({"type": "mock", "on": bool(on)})
        return {"on": bool(on), "state": "on" if on else "off"}

    def _list_demo_datasets(self):
        sets = [{"file": one["file"], "records": one["records"], "usable": one["usable"],
                 "about": one["keywords"]} for one in self.shared_datasets()]
        return {"count": len(sets), "datasets": sets}

    def _use_demo_dataset(self, file):
        found = next((one for one in self.shared_datasets() if one["file"] == file), None)
        if found is None:
            names = [one["file"] for one in self.shared_datasets()]
            close = [one for one in names if str(file).lower() in one.lower()]
            if len(close) != 1:
                raise ToolError("no demo dataset %r; there are: %s" % (file, ", ".join(names)))
            found = next(one for one in self.shared_datasets() if one["file"] == close[0])
        if not found["usable"]:
            raise ToolError("%s has no answers for a LoRA to learn" % found["file"])
        self.session["selection"] = {}
        self.actions.append({"type": "dataset", "file": found["file"]})
        return {"file": found["file"], "records": found["records"],
                "next": "the page reads it and summarises it for the person right after "
                        "your reply; they have nothing to press"}

    def _choose_another_dataset(self):
        self.actions.append({"type": "choose_dataset"})
        return {}

    def _list_my_loras(self):
        rows = self.catalog.all(owner=self.user["name"])
        loras = [{"name": row["name"], "status": row["status"], "rank": row["rank"],
                  "final_loss": row["final_loss"], "base_model": row["base_model"],
                  "practice_run": bool(row["mock"])} for row in rows[:50]]
        return {"count": len(rows), "loras": loras}

    def _start_training(self):
        if not self.session["epochs"]:
            raise ToolError("how long to train is not decided yet; set the epochs first")
        if not self.chosen():
            raise ToolError("the part of the dataset chosen holds no records")
        self.actions.append({"type": "start"})
        return {"epochs": self.session["epochs"],
                "next": "the page reads the plan back and queues it right after your reply, "
                        "so say it is about to start, not that it has"}

    def _stop_training(self):
        self.actions.append({"type": "stop"})
        return {}
