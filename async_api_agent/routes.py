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

  combining them (blending.py)
    POST /agent/blend/intro         {agent, session, prefer?: [id], history, quiet?}
                                                              -> the user's ready LoRAs,
                                                                 the pick, the search
    POST /agent/blend/plan          {session, dataset?, mock, prefer?}
                                                              -> the POST /jobs body, read back
    POST /agent/blend/started       {agent, job: {id, queue_position}, plan, history}
    POST /agent/blend/debrief       {agent, job: id, history} -> what the search found

  after the search (release.py)
    POST /agent/test/started        {agent, job: {id, queue_position}, history}
    POST /agent/test/debrief        {agent, job: id, session, history}
                                                              -> every blend, tested or not,
                                                                 and the one recommended
    POST /agent/verify/plan         {session}                 -> the POST /jobs/{id}/verify
                                                                 body, read back
    POST /agent/verify/debrief      {agent, verification: id, history}
                                                              -> the blend beside its LoRAs
    POST /agent/live/started        {agent, deployment: id, history}

`agent` is the page's choice of model -- {"provider", "model", "base_url",
"conversation"}, the last an id per conversation for the providers that route
by one --
`dataset` is {"text", "name"} for pasted or uploaded data, or {"file"} for
one of the shared datasets, and `session` is what the chat has set up so far
(planner.session_of): the part of the dataset in use, the ranks, the epochs,
the options. None of these trains anything: the page sends the plan to
POST /loras itself, and carries out the actions a chat reply hands back.

The blend half is the same bargain: /agent/blend/plan returns a POST /jobs
body naming the user's own LoRAs by catalogue id, the page submits it, and
watches GET /jobs/{id}/status; /agent/blend/debrief reads only a job of the
user's own (another user's is a 404, as on /jobs/{id}).

And so is the last part. The page queues the testing pass itself (POST
/jobs/{id}/test) and watches the job; /agent/verify/plan returns the body of a
POST /jobs/{id}/verify, which the page sends and watches through GET
/verifications/{id}; the page puts a blend live with POST /jobs/{id}/live and
tries it through POST /infer. The token that call returns stays in the page:
/agent/live/started is sent the deployment's id, never its token, so no model
is ever shown one.
"""

import os

from async_api import settings as api_settings
from async_api_agent import agent
from async_api_agent import analysis
from async_api_agent import blending
from async_api_agent import planner
from async_api_agent import prompts
from async_api_agent import providers
from async_api_agent import release
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
                 "blend": {"generations": settings.BLEND_GENERATIONS,
                           "population": settings.BLEND_POPULATION,
                           "questions": settings.BLEND_QUESTIONS,
                           "max_generations": settings.MAX_BLEND_GENERATIONS,
                           "min_population": settings.MIN_BLEND_POPULATION,
                           "max_population": settings.MAX_BLEND_POPULATION,
                           "max_questions": settings.MAX_BLEND_QUESTIONS,
                           "slots": len(blending.SLOTS)},
                 "words": {"instructions": prompts.STEP_INSTRUCTIONS,
                           "not_yet": prompts.NOT_YET, "ask_dataset": prompts.ASK_DATASET,
                           "another_dataset": prompts.ANOTHER_DATASET,
                           "stopped": prompts.STOPPED, "blend_stopped": prompts.BLEND_STOPPED,
                           "chat_hint": prompts.CHAT_HINT,
                           "blend_hint": prompts.BLEND_HINT,
                           "test_stopped": prompts.TEST_STOPPED,
                           "release_hint": prompts.RELEASE_HINT},
                 "release": {"splits": list(release.SPLITS),
                             "questions": settings.VERIFY_QUESTIONS,
                             "max_questions": settings.MAX_VERIFY_QUESTIONS}}


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
    ready = len(blending.mine(app.catalog, user))
    out = agent.intro(_choice(body), bool((body or {}).get("mock")), ready)
    out["ready_loras"] = ready
    return 200, out


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
                           _choice(body), _history(body), demo_datasets, app.registry)


def debrief(app, user, body):
    ids = (body or {}).get("loras")
    if not isinstance(ids, list) or not ids:
        raise AgentError(400, "loras must list the LoRA ids to report on")
    return 200, agent.debrief(app.catalog, user, ids, _choice(body), _history(body),
                              bool((body or {}).get("mock")))


# --- combining the LoRAs ------------------------------------------------------


def _prefer(body):
    """The LoRA ids the page would like picked -- the ones it just trained."""
    ids = (body or {}).get("prefer")
    return [one for one in ids if isinstance(one, int) and not isinstance(one, bool)] \
        if isinstance(ids, list) else []


def blend_intro(app, user, body):
    return 200, agent.blend_intro(app.catalog, user, _session(body), _choice(body),
                                  _history(body), _prefer(body),
                                  bool((body or {}).get("quiet")))


def blend_plan(app, user, body):
    session = _session(body)
    session["mock"] = session["mock"] or bool((body or {}).get("mock"))
    dataset = dataset_of(body) if (body or {}).get("dataset") else None
    try:
        found = blending.plan(app.catalog, app.registry, user, session, session["mock"],
                              dataset, _prefer(body))
    except blending.BlendError as error:
        raise AgentError(400, str(error))
    found["message"] = agent.blend_confirm(found)
    found["blend"] = dict(session["blend"], loras=[one["id"] for one in found["loras"]])
    return 200, found


def blend_started(app, user, body):
    job = (body or {}).get("job")
    plan = (body or {}).get("plan")
    if not isinstance(job, dict) or not isinstance(plan, dict):
        raise AgentError(400, "job must be {id, queue_position} and plan what /agent/blend/plan "
                              "returned")
    return 200, {"message": agent.blend_started(job, plan, _choice(body), _history(body))}


def blend_debrief(app, user, body):
    job_id = (body or {}).get("job")
    if not isinstance(job_id, int) or isinstance(job_id, bool):
        raise AgentError(400, "job must be the search's job id")
    job = app.registry.job(job_id, user["id"])
    if job is None:
        raise AgentError(404, "no job %d" % job_id)
    return 200, agent.blend_debrief(app.registry, app.catalog, user, job, _choice(body),
                                    _history(body))


# --- after the search ------------------------------------------------------------


def _id(body, name):
    value = (body or {}).get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise AgentError(400, "%s must be an id" % name)
    return value


def _blends(app, user, job_id):
    """release.blends() for one of the user's own jobs; another's is a 404."""
    job = app.registry.job(job_id, user["id"])
    if job is None:
        raise AgentError(404, "no job %d" % job_id)
    try:
        return release.blends(app.registry, app.catalog, user, job)
    except release.ReleaseError as error:
        raise AgentError(409, str(error))


def test_started(app, user, body):
    job = (body or {}).get("job")
    if not isinstance(job, dict) or not isinstance(job.get("id"), int):
        raise AgentError(400, "job must be {id, queue_position}")
    found = _blends(app, user, job["id"])
    return 200, {"message": agent.test_started(found, job, _choice(body), _history(body))}


def test_debrief(app, user, body):
    """Every blend of the search, tested or not; the session's release part
    comes back pointing at this job, with a blend picked if none was."""
    job_id = _id(body, "job")
    found = _blends(app, user, job_id)
    part = dict(_session(body)["release"], job=job_id)
    numbers = {one["number"] for one in found["blends"] if one["state"] != "BAD"}
    if part["individual"] not in numbers:
        part["individual"] = found["recommended"]
    out = agent.test_debrief(found, _choice(body), _history(body))
    return 200, {"message": out, "result": found, "release": part}


def verify_plan(app, user, body):
    """The POST /jobs/{id}/verify body for the session's picked blend."""
    part = _session(body)["release"]
    if part["job"] is None:
        raise AgentError(400, "the session names no search to verify a blend of")
    found = _blends(app, user, part["job"])
    questions = part["questions"] or {}
    name = None
    if questions.get("lora") is not None:
        row = app.catalog.get(questions["lora"])
        name = row["name"] if row is not None and row["owner"] == user["name"] else None
    try:
        planned = release.verify_request(found, part, name)
    except release.ReleaseError as error:
        raise AgentError(400, str(error))
    planned["message"] = agent.verify_start(planned, found["mock"])
    planned["release"] = dict(part, individual=planned["blend"]["number"])
    return 200, planned


def verify_debrief(app, user, body):
    verification_id = _id(body, "verification")
    row = app.registry.verification(verification_id, user["id"])
    if row is None:
        raise AgentError(404, "no verification %d" % verification_id)
    facts = release.verification_outcome(app.registry, app.catalog, user, row)
    return 200, {"message": agent.verify_debrief(facts, _choice(body), _history(body)),
                 "result": facts}


def live_started(app, user, body):
    deployment_id = _id(body, "deployment")
    row = app.registry.deployment(deployment_id, user["id"])
    if row is None or row["revoked_at"]:
        raise AgentError(404, "no live deployment %d" % deployment_id)
    from async_api import server                  # its view of a deployment row
    shown = server.deployment_json(row)
    found = _blends(app, user, row["job_id"])
    one = next((blend for blend in found["blends"] if blend["number"] == row["number"]), None)
    formula = one["formula"] if one else row["chromosome"]
    return 200, {"message": agent.live(shown, formula, _choice(body), _history(body)),
                 "deployment": shown, "formula": formula}


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
    ("POST", r"/agent/blend/intro", blend_intro, ("body",)),
    ("POST", r"/agent/blend/plan", blend_plan, ("body",)),
    ("POST", r"/agent/blend/started", blend_started, ("body",)),
    ("POST", r"/agent/blend/debrief", blend_debrief, ("body",)),
    ("POST", r"/agent/test/started", test_started, ("body",)),
    ("POST", r"/agent/test/debrief", test_debrief, ("body",)),
    ("POST", r"/agent/verify/plan", verify_plan, ("body",)),
    ("POST", r"/agent/verify/debrief", verify_debrief, ("body",)),
    ("POST", r"/agent/live/started", live_started, ("body",)),
]
