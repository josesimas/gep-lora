"""
routes.py - The /agent endpoints, served by the async API's own server.

async_api/server.py adds ROUTES to its own, so these sit behind the same API
key, the same user and the same catalogue as every other endpoint -- a user's
agent sees that user's LoRAs and nobody else's. Each handler takes the server's
App and the user, like an App method, and returns (status, payload).

    GET  /agent/config              providers, the defaults, the plan's ranks,
                                    the wait choices and the demo datasets
    GET  /agent/models?provider=    the chat models a provider lists
         [&base_url=]
    POST /agent/intro               {agent, mock}             -> the welcome
    POST /agent/analyse             {agent, dataset, session?} -> facts + summary
    POST /agent/wait                {agent, records, mock, session?, quiet?}
                                                              -> the question, choices
    POST /agent/plan                {dataset, epochs, mock, session?}
                                                              -> POST /loras bodies
    POST /agent/started             {agent, trainings, epochs, estimate, mock}
    POST /agent/chat                {agent, message, stage, session, dataset?,
                                     context, history}        -> the reply, and what the
                                                                 tools changed
    POST /agent/debrief             {agent, loras: [id], mock, history}

`agent` is the page's choice of model -- {"provider", "model", "base_url"} --
`dataset` is {"text", "name"} for pasted or uploaded data, or {"file"} for
one of the shared datasets, and `session` is what the chat has set up so far
(planner.session_of): the part of the dataset in use, the ranks, the epochs,
the options. None of these trains anything: the page sends the plan to
POST /loras itself, and carries out the actions a chat reply hands back.
"""

import os

from async_api import settings as api_settings
from async_api_agent import agent
from async_api_agent import analysis
from async_api_agent import planner
from async_api_agent import prompts
from async_api_agent import providers
from async_api_agent import settings

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The biggest dataset file the demo list reads to describe.
_DESCRIBE_BYTES = 4 * 1024 * 1024


class AgentError(Exception):
    """A request the agent cannot serve; server.py sends it as {"error"}."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def _shared_dir():
    return os.path.abspath(os.path.join(_ROOT, api_settings.SHARED_DATASETS_DIR))


def _shared_path(name):
    shared = _shared_dir()
    path = os.path.abspath(os.path.join(shared, str(name)))
    if os.path.commonpath([shared, path]) != shared or not os.path.isfile(path):
        raise AgentError(400, "no shared dataset %r" % name)
    return path


def _read(path):
    with open(path, encoding="utf-8-sig", errors="replace") as handle:
        return handle.read()


def dataset_of(body):
    """-> (text, name, shared file name or None)."""
    value = (body or {}).get("dataset")
    if not isinstance(value, dict):
        raise AgentError(400, "dataset must be {\"text\", \"name\"?} or {\"file\"}")
    if value.get("file"):
        return _read(_shared_path(value["file"])), value["file"], value["file"]
    text = value.get("text")
    if not isinstance(text, str) or not text.strip():
        raise AgentError(400, "the dataset is empty -- paste it, upload a file or pick one")
    name = value.get("name") if isinstance(value.get("name"), str) else None
    return text, name, None


def _analysed(body, selection=None):
    text, name, shared = dataset_of(body)
    try:
        return analysis.analyse(text, name, selection), shared
    except analysis.DatasetError as error:
        raise AgentError(400, "I could not read that dataset: %s" % error)


def _session(body):
    try:
        return planner.session_of((body or {}).get("session"))
    except planner.SessionError as error:
        raise AgentError(400, "session: %s" % error)


def _choice(body):
    choice = (body or {}).get("agent")
    return choice if isinstance(choice, dict) else None


def _history(body):
    history = (body or {}).get("history")
    return [one for one in history if isinstance(one, dict)] if isinstance(history, list) else []


def _records(body):
    records = (body or {}).get("records")
    if not isinstance(records, int) or isinstance(records, bool) or records < 1:
        raise AgentError(400, "records must be the number of conversations, at least 1")
    return records


def _epochs(body):
    epochs = (body or {}).get("epochs")
    if isinstance(epochs, bool) or not isinstance(epochs, (int, float)) \
            or not settings.MIN_EPOCHS <= epochs <= settings.MAX_EPOCHS:
        raise AgentError(400, "epochs must be between %s and %s"
                         % (settings.MIN_EPOCHS, settings.MAX_EPOCHS))
    return float(epochs)


def demo_datasets():
    """The shared datasets, each described well enough to choose from."""
    try:
        names = sorted(name for name in os.listdir(_shared_dir())
                       if os.path.isfile(os.path.join(_shared_dir(), name)))
    except OSError:
        return []
    out = []
    for name in names:
        path = os.path.join(_shared_dir(), name)
        entry = {"file": name, "bytes": os.path.getsize(path), "usable": False,
                 "records": 0, "format": None, "first": None, "keywords": []}
        if entry["bytes"] <= _DESCRIBE_BYTES:
            try:
                found = analysis.analyse(_read(path), name)
                entry.update(usable=found["usable"], records=found["records"],
                             format=found["format"],
                             first=found["samples"][0]["user"] if found["samples"] else None,
                             keywords=[one["word"] for one in found["keywords"][:5]])
            except (analysis.DatasetError, OSError):
                pass
        out.append(entry)
    return out


# --- the handlers ------------------------------------------------------------


def config(app, user):
    options = planner.recipe()
    return 200, {"providers": providers.describe(),
                 "default": {"provider": settings.PROVIDER, "model": settings.MODEL},
                 "ranks": settings.LORA_RANKS, "base_model": options["base_model"],
                 "chat_template": options["chat_template"],
                 "recipe": {key: options[key] for key in ("batch_size", "grad_accum",
                                                          "learning_rate", "alpha", "scheduler")},
                 "wait_choices": settings.WAIT_CHOICES,
                 "min_epochs": settings.MIN_EPOCHS, "max_epochs": settings.MAX_EPOCHS,
                 "mock": bool(settings.MOCK), "few_records": settings.FEW_RECORDS,
                 "datasets": demo_datasets(),
                 # The page's own lines, so every word the agent says is prompts.py's.
                 "words": {"instructions": prompts.STEP_INSTRUCTIONS,
                           "not_yet": prompts.NOT_YET, "ask_dataset": prompts.ASK_DATASET,
                           "another_dataset": prompts.ANOTHER_DATASET,
                           "stopped": prompts.STOPPED, "chat_hint": prompts.CHAT_HINT}}


def models(app, user, query):
    choice = {"provider": (query.get("provider") or [None])[0],
              "base_url": (query.get("base_url") or [""])[0]}
    try:
        resolved = providers.resolve(choice)
        found = providers.list_models(resolved)
    except providers.ProviderError as error:
        raise AgentError(502, str(error))
    return 200, {"provider": resolved["name"], "base_url": resolved["base_url"],
                 "models": found, "default": resolved["model"]}


def intro(app, user, body):
    return 200, agent.intro(_choice(body), bool((body or {}).get("mock")))


def analyse(app, user, body):
    found, shared = _analysed(body, _session(body)["selection"])
    message = agent.summarise(found, _choice(body))
    found.pop("lines")
    return 200, {"analysis": found, "shared_file": shared, "message": message}


def wait(app, user, body):
    return 200, agent.wait(app.catalog, _records(body), _choice(body),
                           bool((body or {}).get("mock")), _history(body), _session(body),
                           bool((body or {}).get("quiet")))


def plan(app, user, body):
    session = _session(body)
    found, shared = _analysed(body, session["selection"])
    if not found["usable"]:
        raise AgentError(400, "that dataset has no conversations to train on")
    epochs, mock = _epochs(body), bool((body or {}).get("mock"))
    trainings = planner.plan(app.catalog, user, found, epochs, mock, shared, session)
    estimate = planner.estimate(app.catalog, found["records"], epochs, mock, session=session)
    return 200, {"trainings": trainings, "estimate": estimate, "epochs": epochs,
                 "message": agent.confirm(trainings, estimate, mock)}


def started(app, user, body):
    trainings = (body or {}).get("trainings")
    if not isinstance(trainings, list) or not all(isinstance(one, dict) for one in trainings):
        raise AgentError(400, "trainings must be a list of {name, rank, queue_position}")
    estimate = (body or {}).get("estimate")
    return 200, {"message": agent.started(trainings, _epochs(body),
                                          estimate if isinstance(estimate, dict) else None,
                                          bool((body or {}).get("mock")), _choice(body),
                                          _history(body))}


def chat(app, user, body):
    message = (body or {}).get("message")
    if not isinstance(message, str) or not message.strip():
        raise AgentError(400, "message must be what you want to ask")
    context = (body or {}).get("context")
    session = _session(body)
    dataset = dataset_of(body) if (body or {}).get("dataset") else None
    return 200, agent.chat(app.catalog, user, message.strip()[:4000],
                           str((body or {}).get("stage") or ""), session, dataset,
                           context if isinstance(context, dict) else None,
                           _choice(body), _history(body), demo_datasets)


def debrief(app, user, body):
    ids = (body or {}).get("loras")
    if not isinstance(ids, list) or not ids:
        raise AgentError(400, "loras must list the LoRA ids to report on")
    return 200, agent.debrief(app.catalog, user, ids, _choice(body), _history(body),
                              bool((body or {}).get("mock")))


# (method, path pattern, handler, extras) -- the shape of server.ROUTES, with
# a function where the server's own routes name an App method.
ROUTES = [
    ("GET", r"/agent/config", config, ()),
    ("GET", r"/agent/models", models, ("query",)),
    ("POST", r"/agent/intro", intro, ("body",)),
    ("POST", r"/agent/analyse", analyse, ("body",)),
    ("POST", r"/agent/wait", wait, ("body",)),
    ("POST", r"/agent/plan", plan, ("body",)),
    ("POST", r"/agent/started", started, ("body",)),
    ("POST", r"/agent/chat", chat, ("body",)),
    ("POST", r"/agent/debrief", debrief, ("body",)),
]
