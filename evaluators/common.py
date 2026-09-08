"""
evaluators/common.py - What an evaluator is, and what they all share.

One file rather than six copies. Everything here is used by at least two of the
evaluators beside it, and nothing here is an evaluator itself:

    the registry        Prepared, Evaluator, register(), get(), available()
    the judge transport ask_judge(), judge_settings(), resolve_model(),
                        discover_model(), parse_reply(), judge_note() --
                        llm_judge, llm_judge_reference, llm_judge_answers,
                        llm_judge_baseline and panel all speak to a model
    the references      load_references(), reference_for(), prepare_references()
                        -- llm_judge_reference, llm_judge_answers, similarity
                        and panel all grade against the dataset's own answer
    the tokeniser       WORD, tokens() -- similarity and heuristic both count
                        words
    the steps' own      needs_grading(), abandon_after() -- what the evaluate
                        step and the testing pass both have to decide about a
                        set of pending answers before an evaluator sees them

The names are public because they cross module boundaries now: a helper an
evaluator file imports cannot be an underscore. The one that stays private,
_request(), is the only thing here that nothing outside this file calls.

**A judge is one idea with two transports.** JUDGE_BACKEND picks between them:
"endpoint" POSTs to an OpenAI-compatible /v1/chat/completions, and "unsloth"
loads a model in this process the way the generated scripts do -- see
local_model.py, the only other module here that is not an evaluator. ask_judge()
is where they meet, and everything above it is shared, so which one a sweep used
changes where the tokens came from and nothing else about the score.

**No knob lives in this package.** They are all in settings.py, prefixed by the
evaluator that reads them (JUDGE_*, BASELINE_*, SIMILARITY_*, HEURISTIC_*,
PANEL_*), and they reach a step as the sweep's *stored* settings -- the same
contract every other step works under. The single exception is the API key,
which is read from the environment on purpose: a sweep records its settings into
the database, and a bearer token has no business in there.

The judge endpoints are reached over the OpenAI-compatible /v1/chat/completions
API, which LMStudio, OpenAI, OpenRouter, vLLM and most gateways all speak.
"""

import json
import math
import os
import re
import time
import urllib.error
import urllib.request

from blends import generate_runs

from evaluators import local_model

# The two ways of reaching a judge, and what JUDGE_BACKEND names. ENDPOINT is
# the default and what a sweep created before the setting existed reads back as
# -- the behaviour it ran under.
ENDPOINT = "endpoint"
UNSLOTH = "unsloth"
BACKENDS = (ENDPOINT, UNSLOTH)

# Sent as "Authorization: Bearer <key>". LMStudio ignores it; cloud endpoints
# require it. Deliberately not a setting: settings are written into the sweep's
# database, and a real key must not end up there.
API_KEY = os.environ.get("JUDGE_API_KEY", "")

# Which evaluator a sweep uses when it never said -- every sweep created before
# EVALUATOR existed, so it has to be the behaviour those sweeps ran under.
DEFAULT = "llm_judge"


# ===========================================================================
# What an evaluator is
# ===========================================================================


class Prepared:
    """What one evaluate step's worth of scoring works from.

    Built once by `prepare()`, handed to every `score()` call. `label` is what
    lands in exchanges.judge_model -- the judge's model id for the judging
    evaluators, the method's own name for the local ones, so the column keeps
    answering the question it was added for: what gave this score.
    """

    __slots__ = ("conf", "label", "settings", "references", "baselines", "notes")

    def __init__(self, conf, label, settings=None, references=None, notes=(),
                 baselines=None):
        self.conf = conf
        self.label = label
        self.settings = settings or {}
        self.references = references or {}
        # {question key: what the base model said} -- the control
        # llm_judge_baseline grades against, read out of the database once.
        self.baselines = baselines or {}
        self.notes = list(notes)      # lines start_run.py prints before scoring


class Evaluator:
    """One way of turning an answer into a quality.

    A plain holder rather than a base class: an evaluator is two functions and
    two strings, and subclassing would only invite one of them to grow state
    that outlives a step.
    """

    __slots__ = ("name", "description", "prepare", "score", "wants_reference",
                 "needs_judge", "wants_baseline")

    def __init__(self, name, description, prepare, score,
                 wants_reference=False, needs_judge=False,
                 wants_baseline=False):
        self.name = name
        self.description = description
        self.prepare = prepare
        self.score = score
        self.wants_reference = wants_reference
        # Asks a model for every score, whichever backend produces it. What the
        # abandon rule is decided on: a judge call costs a request or a
        # generate(), and a local scorer costs neither.
        self.needs_judge = needs_judge
        # Needs the base model's own answers, which cost a model load the first
        # time they are wanted and come out of the database ever after.
        self.wants_baseline = wants_baseline


# Filled by the register() call at the foot of each evaluator module, as
# evaluators/__init__.py imports them. Nothing else writes to it.
_REGISTRY = {}


def register(evaluator):
    _REGISTRY[evaluator.name] = evaluator
    return evaluator


def get(name):
    """The evaluator called `name`, or a failure naming the ones there are."""
    name = name or DEFAULT
    try:
        return _REGISTRY[name]
    except KeyError:
        raise SystemExit(
            "unknown EVALUATOR %r. settings.py must name one of: %s"
            % (name, ", ".join(sorted(_REGISTRY)))
        )


def available():
    """[(name, description)] for every registered evaluator, for --list."""
    return [(name, _REGISTRY[name].description) for name in sorted(_REGISTRY)]


# ===========================================================================
# The reference answers, for the evaluators that compare against them
# ===========================================================================


def normalise(text):
    """A question reduced to what makes two of them the same question."""
    return " ".join((text or "").split()).strip().lower()


def load_references(conf):
    """{normalised question: reference answer} from the sweep's eval set.

    Keyed by the question rather than by position because that is what survives
    a re-run: an exchange stores the question it actually asked, so matching on
    it cannot quietly pair an answer with the wrong reference the way an index
    into a file that has since been edited could. Position is the fallback, for
    two prompts that really are the same string.
    """
    records = generate_runs.eval_records(conf.get("TRAINING_SET"),
                                         conf.get("TRAINING_COUNT"))
    by_question, by_position = {}, {}
    for record in records:
        if record["reference"]:
            by_question.setdefault(normalise(record["question"]), record["reference"])
            by_position[record["position"]] = record["reference"]
    return by_question, by_position


def reference_for(item, prepared):
    """The dataset's own answer to this exchange, or None."""
    by_question, by_position = prepared.references.get("by_question", {}), \
        prepared.references.get("by_position", {})
    return by_question.get(normalise(item["question"])) or by_position.get(item["position"])


def prepare_references(conf, evaluator_name):
    """The reference maps, or a clear failure when the eval set holds none.

    A missing reference is fatal *here* rather than per exchange: an evaluator
    that compares against the dataset's answers cannot score a plain
    prompt-per-line file at all, and finding that out one exchange at a time
    would spend a whole evaluate step to say so.
    """
    by_question, by_position = load_references(conf)
    if not by_position:
        raise SystemExit(
            "EVALUATOR = %r needs the answers that come with the eval set, and "
            "%s carries none. That file is either plain one-prompt-per-line "
            "text or JSON records with no assistant turn; point TRAINING_SET at "
            "a dataset that has both turns (datasets/*.json do), or choose an "
            "evaluator that does not compare against a reference."
            % (evaluator_name, generate_runs.training_set_path(conf.get("TRAINING_SET")))
        )
    return {"by_question": by_question, "by_position": by_position}


# ===========================================================================
# The judge transport, shared by every evaluator that asks a model
# ===========================================================================


def _request(url, payload, api_key, timeout):
    """POST JSON, return the decoded JSON reply. Raises urllib errors."""
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def discover_model(base_url, api_key, timeout):
    """Ask the endpoint which model it has loaded (LMStudio serves one)."""
    url = base_url.rstrip("/") + "/models"
    headers = {}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            listed = json.loads(response.read().decode("utf-8")).get("data") or []
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise SystemExit(
            "cannot reach the judge at %s (%s). Is LMStudio running with a model "
            "loaded and its server started? Point JUDGE_BASE_URL at a different "
            "endpoint to use another one."
            % (base_url, error)
        )
    # LMStudio lists embedding models alongside chat ones; those cannot grade.
    chat_models = [entry.get("id") for entry in listed
                   if "embed" not in (entry.get("id") or "").lower()]
    if not chat_models:
        raise SystemExit("%s lists no chat models. Load one in LMStudio first." % url)
    return chat_models[0]


def parse_reply(text):
    """Pull the quality and the judge's reason out of its reply.

    Prefers well-formed JSON, and falls back to the first number in 0..1 that
    the text contains, so a model that wraps its JSON in prose or code fences
    still scores rather than failing the whole run. The reason is best-effort:
    the score is what the search needs, so a missing reason is never fatal, and
    a reply truncated after the score still yields one.
    """
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()

    reason = ""
    try:
        parsed = json.loads(cleaned)
        value = parsed.get("quality")
        reason = (parsed.get("reason") or "").strip()
    except (ValueError, AttributeError):
        match = re.search(r'"quality"\s*:\s*([0-9]*\.?[0-9]+)', cleaned)
        if not match:
            match = re.search(r"\b(0?\.[0-9]+|0|1(?:\.0+)?)\b", cleaned)
        value = match.group(1) if match else None
        # The JSON did not parse -- usually truncated -- so recover the reason
        # textually if enough of it made it through.
        said = re.search(r'"reason"\s*:\s*"([^"]*)', cleaned)
        reason = said.group(1).strip() if said else ""

    if value is None:
        raise ValueError("no quality score in judge reply: %r" % text[:200])
    score = float(value)
    if not 0.0 <= score <= 1.0:
        raise ValueError("quality %r is outside 0..1" % score)
    return score, reason


def ask_judge(system_prompt, user_content, settings):
    """One grading call. Returns (quality, reason).

    `settings` is the resolved judge block -- backend, base_url, api_key, model,
    temperature, max_tokens, timeout, retries, retry_wait, response_format, and
    the JUDGE_LOCAL_* values the unsloth backend loads under -- which the
    evaluators build from the sweep's stored JUDGE_*/PANEL_* values through
    judge_settings().

    The two backends meet here rather than in the evaluators, so which one is in
    use changes only where the tokens come from: the rubric, the retries and
    reading a quality out of the reply are one code path either way, and that is
    what makes a sweep graded locally comparable with one graded over an API.
    """
    local = settings.get("backend") == UNSLOTH
    payload, url = None, None
    if not local:
        payload = {
            "model": settings["model"],
            "temperature": settings["temperature"],
            "max_tokens": settings["max_tokens"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        if settings.get("response_format"):
            payload["response_format"] = settings["response_format"]
        url = settings["base_url"].rstrip("/") + "/chat/completions"

    retries = settings["retries"]
    last_error = None
    for attempt in range(retries + 1):
        try:
            if local:
                text = local_model.generate(system_prompt, user_content, settings)
            else:
                reply = _request(url, payload, settings["api_key"], settings["timeout"])
                message = reply["choices"][0]["message"]
                text = message.get("content") or ""
                if not text.strip():
                    # A reasoning model that spent its whole budget thinking
                    # returns an empty content; the score may still be in the
                    # reasoning.
                    text = message.get("reasoning_content") or message.get("reasoning") or ""
            # A blank or unparseable reply is usually a truncation, so it is
            # worth another attempt rather than losing the answer's score.
            return parse_reply(text)
        except ValueError as error:
            last_error = error
            if local and not settings["temperature"]:
                # Greedy decoding: the same prompt returns the same reply, so a
                # retry would spend another generate() to fail in the same way.
                break
        except urllib.error.HTTPError as error:
            # Some endpoints reject response_format; the prompt asks for JSON
            # anyway, so drop it and try once more rather than failing.
            if error.code == 400 and "response_format" in payload:
                payload.pop("response_format")
                last_error = error
                continue
            if error.code not in (408, 409, 429) and error.code < 500:
                raise RuntimeError("judge returned HTTP %d: %s"
                                   % (error.code, error.read().decode("utf-8", "replace")[:200]))
            last_error = error
        except (urllib.error.URLError, OSError, KeyError, IndexError) as error:
            last_error = error

        if attempt < retries:
            time.sleep(settings["retry_wait"])
    if local:
        raise RuntimeError("the local judge produced nothing gradeable: %s" % last_error)
    raise RuntimeError("judge unreachable after %d attempts: %s" % (retries + 1, last_error))


def backend_of(conf):
    """Which transport a sweep grades through, checked. -> ENDPOINT or UNSLOTH.

    Its own function because start_run.py asks it when a sweep is *created*, an
    hour before any answer needs grading: a misspelt backend should cost a line
    at the top of the run, not a finished transcript nobody can score.
    """
    backend = conf.get("JUDGE_BACKEND") or ENDPOINT
    if backend not in BACKENDS:
        raise SystemExit("JUDGE_BACKEND must be %s, not %r"
                         % (" or ".join(repr(name) for name in BACKENDS), backend))
    return backend


def judge_settings(conf, model=None, base_url=None):
    """The JUDGE_* block of a sweep's settings, resolved for ask_judge().

    Defaults are spelled out here for one reason only: a sweep created before a
    knob existed has no value for it stored, and resuming one must not crash on
    a KeyError. settings.py is still where a knob is *set*. It is also why a
    sweep that predates JUDGE_BACKEND reads back as ENDPOINT -- that is the
    behaviour it ran under.

    The last three are the unsloth backend's own, and the endpoint backend
    ignores them exactly as the local one ignores base_url, api_key, timeout and
    response_format. One block rather than two, because a judge is one idea with
    two transports and a caller building the block should not have to know which
    one a sweep chose.
    """
    return {
        "backend": backend_of(conf),
        "base_url": base_url or conf.get("JUDGE_BASE_URL"),
        "api_key": API_KEY,
        "model": model or conf.get("JUDGE_MODEL"),
        "temperature": conf.get("JUDGE_TEMPERATURE", 0.0),
        "max_tokens": conf.get("JUDGE_MAX_TOKENS", 2000),
        "timeout": conf.get("JUDGE_TIMEOUT", 300),
        "retries": conf.get("JUDGE_RETRIES", 2),
        "retry_wait": conf.get("JUDGE_RETRY_WAIT", 3),
        "response_format": conf.get("JUDGE_RESPONSE_FORMAT", {"type": "json_object"}),
        "max_seq_length": conf.get("JUDGE_LOCAL_MAX_SEQ_LENGTH", 4096),
        "load_in_4bit": conf.get("JUDGE_LOCAL_LOAD_IN_4BIT", True),
        "chat_template": conf.get("JUDGE_LOCAL_CHAT_TEMPLATE"),
    }


def resolve_model(settings, grading):
    """Fill in the model a sweep did not name, for the backend it chose.

    An endpoint can be asked what it has loaded -- that is what JUDGE_MODEL =
    None means, and what you want with LMStudio. A machine cannot be asked, so
    the local backend needs the name in writing; and it must not quietly fall
    back to BASE_MODEL, which would leave the model under test grading its own
    descendants.
    """
    if settings["model"] or not grading:
        return settings["model"]
    if settings["backend"] == UNSLOTH:
        raise SystemExit(
            "JUDGE_BACKEND = 'unsloth' grades with a model loaded here, so "
            "JUDGE_MODEL has to name one -- a Hub repo id, or a folder. There "
            "is nothing to ask what it has loaded the way an endpoint can be "
            "asked, and falling back to BASE_MODEL would leave the model under "
            "test marking its own homework."
        )
    settings["model"] = discover_model(
        settings["base_url"], settings["api_key"], settings["timeout"])
    return settings["model"]


def judge_note(settings, grading, what="judge"):
    """The line a prepare() prints to say where the grading will come from."""
    if not grading:
        return "%s: not contacted -- no answer needs grading" % what
    if settings["backend"] == UNSLOTH:
        return "%s: %s, loaded here with unsloth" % (what, local_model.describe(settings))
    return "%s: %s at %s" % (what, settings["model"], settings["base_url"])


def release_models():
    """Hand back whatever grading loaded. A no-op unless a judge was loaded.

    Called by the evaluate step and by the testing pass when they are done,
    whichever evaluator ran: a step that has finished scoring has no further use
    for a model, and `main.py` goes straight on to a generation whose scripts
    each want the card.
    """
    local_model.release()


def needs_grading(pending):
    """Whether any pending exchange has an answer worth spending a call on.

    An all-blank set -- a sweep where every script failed -- is scored 0.0 by
    start_run.py without anyone being asked, and a mocked sweep arrives scored, so
    neither should make the step demand an endpoint that need not be up, or load
    a judge model that has nothing to grade.
    """
    return any((row["answer"] or "").strip() for row in pending)


def abandon_after(conf, answers):
    """How many graded zeros in a row condemn one individual, or None.

    JUDGE_ABANDON_FRACTION of the answers that individual has pending, rounded
    up and never fewer than one -- 0.1 is "the first 10%". None when the rule is
    off, which is what 0, None and an individual with nothing pending all mean.

    Here rather than in either caller because both of them apply it: the
    evaluate step over `exchanges` and the testing pass over `test_results`
    grade the same answers under the same evaluator, so an individual worth
    giving up on in one is worth giving up on in the other. Whether to apply it
    at all is the caller's -- only an evaluator that asks a model (needs_judge)
    has anything to save by stopping early, and it saves it either way: a
    request not sent, or a generate() not run.
    """
    fraction = conf.get("JUDGE_ABANDON_FRACTION", 0.0) or 0.0
    if fraction <= 0 or not answers:
        return None
    return max(1, math.ceil(fraction * answers))


# ===========================================================================
# Words, for the evaluators that count them
# ===========================================================================


WORD = re.compile(r"[\w']+", re.UNICODE)


def tokens(text, case_sensitive):
    text = text if case_sensitive else text.lower()
    return WORD.findall(text)
