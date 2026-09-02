"""
evaluators/common.py - What an evaluator is, and what they all share.

One file rather than six copies. Everything here is used by at least two of the
evaluators beside it, and nothing here is an evaluator itself:

    the registry        Prepared, Evaluator, register(), get(), available()
    the judge transport ask_judge(), endpoint_settings(), discover_model(),
                        parse_reply() -- llm_judge, llm_judge_reference,
                        llm_judge_baseline and panel all speak to a model
    the references      load_references(), reference_for(), prepare_references()
                        -- llm_judge_reference, similarity and panel all grade
                        against the dataset's own answer
    the tokeniser       WORD, tokens() -- similarity and heuristic both count
                        words

The names are public because they cross module boundaries now: a helper an
evaluator file imports cannot be an underscore. The one that stays private,
_request(), is the only thing here that nothing outside this file calls.

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
import os
import re
import time
import urllib.error
import urllib.request

import generate_runs

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
        self.notes = list(notes)      # lines main.py prints before scoring


class Evaluator:
    """One way of turning an answer into a quality.

    A plain holder rather than a base class: an evaluator is two functions and
    two strings, and subclassing would only invite one of them to grow state
    that outlives a step.
    """

    __slots__ = ("name", "description", "prepare", "score", "wants_reference",
                 "needs_endpoint", "wants_baseline")

    def __init__(self, name, description, prepare, score,
                 wants_reference=False, needs_endpoint=False,
                 wants_baseline=False):
        self.name = name
        self.description = description
        self.prepare = prepare
        self.score = score
        self.wants_reference = wants_reference
        self.needs_endpoint = needs_endpoint
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

    `settings` is the resolved endpoint block -- base_url, api_key, model,
    temperature, max_tokens, timeout, retries, retry_wait, response_format --
    which the evaluators build from the sweep's stored JUDGE_*/PANEL_* values.
    """
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
            reply = _request(url, payload, settings["api_key"], settings["timeout"])
            message = reply["choices"][0]["message"]
            text = message.get("content") or ""
            if not text.strip():
                # A reasoning model that spent its whole budget thinking returns
                # an empty content; the score may still be in the reasoning.
                text = message.get("reasoning_content") or message.get("reasoning") or ""
            # A blank or unparseable reply is usually a truncation, so it is
            # worth another attempt rather than losing the answer's score.
            return parse_reply(text)
        except ValueError as error:
            last_error = error
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
    raise RuntimeError("judge unreachable after %d attempts: %s" % (retries + 1, last_error))


def endpoint_settings(conf, model=None, base_url=None):
    """The JUDGE_* block of a sweep's settings, resolved for ask_judge().

    Defaults are spelled out here for one reason only: a sweep created before
    a knob existed has no value for it stored, and resuming one must not crash
    on a KeyError. settings.py is still where a knob is *set*.
    """
    return {
        "base_url": base_url or conf.get("JUDGE_BASE_URL"),
        "api_key": API_KEY,
        "model": model or conf.get("JUDGE_MODEL"),
        "temperature": conf.get("JUDGE_TEMPERATURE", 0.0),
        "max_tokens": conf.get("JUDGE_MAX_TOKENS", 2000),
        "timeout": conf.get("JUDGE_TIMEOUT", 300),
        "retries": conf.get("JUDGE_RETRIES", 2),
        "retry_wait": conf.get("JUDGE_RETRY_WAIT", 3),
        "response_format": conf.get("JUDGE_RESPONSE_FORMAT", {"type": "json_object"}),
    }


def needs_grading(pending):
    """Whether any pending exchange has an answer worth spending a call on.

    An all-blank set -- a sweep where every script failed -- is scored 0.0 by
    main.py without anyone being asked, and a mocked sweep arrives scored, so
    neither should make the step demand an endpoint that need not be up.
    """
    return any((row["answer"] or "").strip() for row in pending)


# ===========================================================================
# Words, for the evaluators that count them
# ===========================================================================


WORD = re.compile(r"[\w']+", re.UNICODE)


def tokens(text, case_sensitive):
    text = text if case_sensitive else text.lower()
    return WORD.findall(text)
