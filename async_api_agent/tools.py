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
    training, a search or a testing pass, verifying a blend, putting one live
    and switching datasets are *actions* handed back to the page, which does
    them through the API's own endpoints -- POST /loras, POST /jobs, their
    cancels, POST /jobs/{id}/test, /verify and /live, /agent/analyse --
    exactly as its buttons do. Nothing here queues, trains, blends, tests or
    deploys.
  * **The stage decides what may be asked.** The plan cannot change while a
    training runs; a dataset has to be there before part of it is chosen;
    a training can be started only once its length is known. A tool asked at
    the wrong moment refuses with a reason the model can pass on.
  * **Every value is checked by the rule that will check it later** --
    selection.clean(), planner.check_ranks()/check_options(), train.py's
    limits, blending.py's -- so a plan the chat built is one POST /loras (or
    POST /jobs) accepts. A LoRA to blend is looked up among the person's own
    and nobody else's.

The descriptions the model reads are prompts.TOOLS; this module holds only
their parameters. With no model, commands.py turns plain requests into the
same calls.
"""

import copy

from async_api_agent import analysis
from async_api_agent import blending
from async_api_agent import planner
from async_api_agent import prompts
from async_api_agent import release
from async_api_agent import selection as picking
from async_api_agent import settings

# The page's stages, in order: training LoRAs, blending them, then testing the
# blends, verifying one and putting one live.
STAGES = ("intro", "dataset", "analysis", "wait", "confirm", "training", "done",
          "blend", "blend_confirm", "blending", "blended",
          "testing", "tested", "verifying", "verified", "live")

# Stages in which the training plan may change (not while a training runs,
# nor once the person has moved on to blending).
PLANNING = ("intro", "dataset", "analysis", "wait", "confirm", "done")

# Stages in which a training may be started from the chat.
STARTABLE = ("analysis", "wait", "confirm", "done")

# Stages in which what a search blends may change, and it may be started.
BLEND_PLANNING = ("intro", "done", "blend", "blend_confirm", "blended")
BLEND_STAGES = ("blend", "blend_confirm", "blending", "blended")

# After a search: when a blend may be picked (and put live), and when a
# verification may be asked for -- once there is something to pick from.
PICKING = ("blended", "tested", "verified", "live")
VERIFYING = ("tested", "verified")

# Nothing may change while one of these runs.
RUNNING = {"training": "a training is running; stop it or wait for it first",
           "blending": "a search is running; stop it or wait for it first",
           "testing": "the blends are being tested; stop it or wait for it first",
           "verifying": "a verification is running; wait for it first"}

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
    "choose_another_dataset": ({}, [], False, PLANNING + ("blend", "blend_confirm", "blended"),
                               False),
    "list_my_loras": ({}, [], False, STAGES, False),
    "start_training": ({}, [], True, STARTABLE, False),
    "stop_training": ({}, [], False, ("training",), False),
    "open_blending": ({}, [], False, PLANNING + PICKING, False),
    "choose_blend_loras": ({
        "loras": dict(_WORDS, description="LoRA ids or names, e.g. [3, \"poem-r16\"]"),
        "add": {"type": "boolean", "description": "add to the LoRAs already chosen"},
    }, ["loras"], False, BLEND_PLANNING, True),
    "set_blend_search": ({
        "generations": dict(_INT, description="rounds after the first"),
        "population": dict(_INT, description="blends in each round"),
        "questions": dict(_INT, description="questions every blend is judged on"),
        "label": {"type": "string", "description": "the search's name"},
    }, [], False, BLEND_PLANNING, True),
    "set_blend_questions": ({
        "file": {"type": "string", "description": "a demo dataset's file name"},
        "lora": {"type": "string", "description": "a LoRA of theirs, by id or name"},
        "conversation": {"type": "boolean",
                         "description": "the dataset given in this conversation"},
    }, [], False, BLEND_PLANNING, True),
    "show_blend_plan": ({}, [], False, STAGES, False),
    "start_blend": ({}, [], False, BLEND_PLANNING, False),
    "stop_blend": ({}, [], False, ("blending",), False),
    "start_testing": ({}, [], False, PICKING, False),
    "stop_testing": ({}, [], False, ("testing",), False),
    "choose_best_blend": ({
        "individual": dict(_INT, description="the blend's number, e.g. 7 for #7"),
    }, ["individual"], False, PICKING, True),
    "set_verify_questions": ({
        "split": {"type": "string", "enum": list(release.SPLITS),
                  "description": "one of the search's own sets of questions"},
        "file": {"type": "string", "description": "a demo dataset's file name"},
        "lora": {"type": "string", "description": "a LoRA of theirs, by id or name"},
        "count": dict(_INT, description="how many questions"),
    }, [], False, PICKING, True),
    "start_verification": ({}, [], False, VERIFYING, False),
    "go_live": ({
        "individual": dict(_INT, description="the blend's number; default the one picked"),
    }, [], False, PICKING, False),
}


# The tools that change what a search blends rather than what is trained.
BLEND_TOOLS = ("choose_blend_loras", "set_blend_search", "set_blend_questions")
# The tools that change what happens after a search: which blend, which questions.
RELEASE_TOOLS = ("choose_best_blend", "set_verify_questions")


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

    def __init__(self, catalog, user, stage, session, dataset=None, shared_datasets=None,
                 registry=None):
        """`dataset` is (text, name, shared file or None), or None before one
        is given; `shared_datasets` a callable listing the demo datasets;
        `registry` the API's, to read the blends of the session's search."""
        self.catalog = catalog
        self.registry = registry
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
        self.blend_changed = False
        self.release_changed = False
        self._records = None
        self._found = False

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
                raise ToolError(RUNNING.get(self.stage) or
                                "that cannot be done at this step (%s)" % self.stage)
            if needs_data and not self.dataset:
                raise ToolError("there is no dataset yet; give one first")
            result = getattr(self, "_" + name)(**arguments)
        except (ToolError, picking.SelectionError, planner.SessionError,
                analysis.DatasetError, blending.BlendError, release.ReleaseError) as error:
            result = {"error": str(error)}
        except TypeError as error:            # an argument the tool does not take
            result = {"error": "bad arguments: %s" % error}
        if "error" in result:
            summary = prompts.TOOL_FAILED.format(tool=name, error=result["error"])
        else:
            if SPECS[name][4]:
                if name in BLEND_TOOLS:
                    self.blend_changed = True
                elif name in RELEASE_TOOLS:
                    self.release_changed = True
                else:
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
        if self.blend_changed or kinds & {"start_blend", "blend"}:
            return self._blend_outcome(actions, kinds)
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

    def _blend_outcome(self, actions, kinds):
        """The page's orders when the chat was about blending: to the blend
        step first if it is not there, and the search read back whenever what
        it blends changed, and always before it starts."""
        if self.stage not in BLEND_STAGES and "blend" not in kinds:
            actions.insert(0, {"type": "blend"})
        if "start_blend" in kinds or (self.blend_changed and self.stage in BLEND_PLANNING):
            at = next((index for index, one in enumerate(actions)
                       if one["type"] == "start_blend"), len(actions))
            actions.insert(at, {"type": "blend_plan"})
        return {"session": self.session, "actions": actions, "steps": self.steps,
                "preview": self.preview, "analysis": None}

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
        loras = [{"id": row["id"], "name": row["name"], "status": row["status"],
                  "rank": row["rank"], "final_loss": row["final_loss"],
                  "base_model": row["base_model"], "practice_run": bool(row["mock"]),
                  "can_blend": row["status"] == blending.lora_catalog.READY}
                 for row in rows[:50]]
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

    # --- blending ---

    def _blend(self, **change):
        """The session's blend part with `change` applied, checked."""
        wanted = dict(self.session["blend"], **change)
        self.session["blend"] = blending.blend_of(wanted)
        return self.session["blend"]

    def _chosen(self):
        return blending.chosen(self.catalog, self.user, self.session["blend"])

    def _open_blending(self):
        self.actions.append({"type": "blend"})
        return {"next": "the page moves on to combining LoRAs right after your reply"}

    def _choose_blend_loras(self, loras, add=False):
        wanted = loras if isinstance(loras, list) else [loras]
        rows = [blending.own(self.catalog, self.user, key) for key in wanted]
        if add:
            rows = [blending.own(self.catalog, self.user, one)
                    for one in self.session["blend"]["loras"]] + rows
        rows = blending.check_together(list({row["id"]: row for row in rows}.values()))
        self._blend(loras=[row["id"] for row in rows])
        return {"loras": [blending.describe(row) for row in rows],
                "names_text": ", ".join(row["name"] for row in rows),
                "slots": blending.slots(rows)}

    def _set_blend_search(self, generations=None, population=None, questions=None, label=None):
        change = {key: value for key, value in (("generations", generations),
                                                ("population", population),
                                                ("questions", questions),
                                                ("label", label)) if value is not None}
        if not change:
            raise ToolError("say how many rounds, blends or questions")
        blend = self._blend(**change)
        found = blending.estimate(blend["population"], blend["generations"],
                                  self.session["mock"])
        return {"generations": blend["generations"], "population": blend["population"],
                "questions": blend["questions"], "label": blend["label"], "time": found["time"],
                "search_text": "%d round(s) after the first, %d blends each, %d question(s)"
                               % (blend["generations"], blend["population"], blend["questions"])}

    def _set_blend_questions(self, file=None, lora=None, conversation=False):
        if sum(bool(one) for one in (file, lora is not None, conversation)) != 1:
            raise ToolError("say one source: a demo dataset, one of your LoRAs, or this "
                            "conversation's dataset")
        if conversation:
            if not self.dataset:
                raise ToolError("no dataset was given in this conversation")
            self._blend(source=None)
            return {"source": None, "source_text": "the dataset in this conversation"}
        if lora is not None:
            row = blending.own(self.catalog, self.user, lora)
            self._blend(source={"lora": row["id"]})
            return {"source": {"lora": row["id"]},
                    "source_text": "%s's training data" % row["name"]}
        names = [one["file"] for one in self.shared_datasets() if one["usable"]]
        match = [name for name in names if name == file] or \
            [name for name in names if str(file).lower() in name.lower()]
        if len(match) != 1:
            raise ToolError("no demo dataset %r; there are: %s" % (file, ", ".join(names)))
        self._blend(source={"file": match[0]})
        return {"source": {"file": match[0]}, "source_text": match[0]}

    def _show_blend_plan(self):
        blend = self.session["blend"]
        rows = self._chosen()
        mock = self.session["mock"] or any(row["mock"] for row in rows)
        source = blend["source"]
        return {"loras": [blending.describe(row) for row in rows],
                "places": ({slot: next(row["name"] for row in rows if row["id"] == lora_id)
                            for slot, lora_id in blending.slots(rows).items()} if rows else {}),
                "generations": blend["generations"], "population": blend["population"],
                "questions": blend["questions"], "label": blend["label"],
                "questions_from": (source["file"] if source and source.get("file") else
                                   "a LoRA's training data" if source else
                                   "this conversation's dataset" if self.dataset else
                                   "the LoRAs' own training data"),
                "practice_run": mock,
                "estimate": blending.estimate(blend["population"], blend["generations"],
                                              mock)["time"]}

    def _start_blend(self):
        rows = self._chosen()
        if not rows:
            raise ToolError("there are no ready LoRAs of yours to blend yet")
        self.actions.append({"type": "start_blend"})
        return {"loras": [row["name"] for row in rows],
                "next": "the page reads the search back and queues it right after your reply, "
                        "so say it is about to start, not that it has"}

    def _stop_blend(self):
        self.actions.append({"type": "stop_blend"})
        return {}

    # --- after the search ---

    def _blends(self):
        """release.blends() for the session's search, or None when there is
        no registry to read it from. Read once per message."""
        if self._found is False:
            self._found = None
            if self.registry is not None:
                job = release.own_job(self.registry, self.user, self.session["release"]["job"])
                self._found = release.blends(self.registry, self.catalog, self.user, job)
        return self._found

    def _search(self):
        if self.session["release"]["job"] is None:
            raise ToolError("there is no finished search to work with yet")

    def _picked(self, individual=None):
        """The blend asked for, else the one picked, else the recommended one.
        -> (number, the blend's entry, or None when the search cannot be read)."""
        wanted = individual if individual is not None else self.session["release"]["individual"]
        if isinstance(wanted, float) and wanted == int(wanted):
            wanted = int(wanted)
        if isinstance(wanted, bool) or (wanted is not None and not isinstance(wanted, int)):
            raise ToolError("a blend is picked by its number")
        found = self._blends()
        if found is None:
            if wanted is None:
                raise ToolError("pick a blend first, by its number")
            return wanted, None
        one = release.pick(found, dict(self.session["release"], individual=wanted))
        return one["number"], one

    def _choose_best_blend(self, individual):
        self._search()
        number, one = self._picked(individual)
        self.session["release"]["individual"] = number
        out = {"individual": number}
        if one is not None:
            out.update(formula=one["formula"], search_score=one["quality"],
                       tested_score=one["tested"], loras=one["loras"])
        return out

    def _set_verify_questions(self, split=None, file=None, lora=None, count=None):
        self._search()
        given = [one for one in (split, file, lora) if one not in (None, "")]
        if len(given) > 1:
            raise ToolError("say one source: the search's own questions, a demo dataset, or "
                            "one of your LoRAs")
        if not given and count is None:
            raise ToolError("say where the questions come from, or how many")
        part = dict(self.session["release"])
        if split:
            found = self._blends()
            if found is not None and split not in found["splits"]:
                raise ToolError("that search holds no %s questions; it has: %s"
                                % (split, ", ".join(found["splits"]) or "none"))
            part["questions"] = {"split": split}
        elif file:
            names = [one["file"] for one in self.shared_datasets() if one["usable"]]
            match = [name for name in names if name == file] or \
                [name for name in names if str(file).lower() in name.lower()]
            if len(match) != 1:
                raise ToolError("no demo dataset %r; there are: %s" % (file, ", ".join(names)))
            part["questions"] = {"file": match[0]}
        elif lora not in (None, ""):
            row = blending.own(self.catalog, self.user, lora)
            part["questions"] = {"lora": row["id"]}
        if count is not None:
            part["count"] = count
        self.session["release"] = release.release_of(part)
        questions = self.session["release"]["questions"]
        if questions is None:
            text = "the validation questions"
        elif questions.get("split"):
            text = "its %s questions" % questions["split"]
        elif questions.get("file"):
            text = questions["file"]
        else:
            text = "%s's training data" % self.catalog.get(questions["lora"])["name"]
        if self.session["release"]["count"]:
            text += ", %d of them" % self.session["release"]["count"]
        return {"questions": questions, "count": self.session["release"]["count"],
                "questions_text": text}

    def _start_testing(self):
        self._search()
        self.actions.append({"type": "start_testing"})
        return {"next": "the page queues the testing right after your reply, so say it is about "
                        "to start, not that it has"}

    def _stop_testing(self):
        self.actions.append({"type": "stop_testing"})
        return {}

    def _start_verification(self):
        self._search()
        number, _ = self._picked()
        self.session["release"]["individual"] = number
        self.actions.append({"type": "start_verification"})
        return {"individual": number,
                "next": "the page reads the verification back and queues it right after your "
                        "reply, so say it is about to start, not that it has"}

    def _go_live(self, individual=None):
        self._search()
        number, _ = self._picked(individual)
        self.session["release"]["individual"] = number
        self.actions.append({"type": "go_live"})
        return {"individual": number,
                "next": "the page puts it live right after your reply and shows its key; never "
                        "say or invent a key yourself"}
