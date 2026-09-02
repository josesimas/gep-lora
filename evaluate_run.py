"""
evaluate_run.py - Score an answer. Which way is a setting.

The process step stores the question/answer pairs a blended model produced.
This module grades them, and returns the number main.py writes back onto that
exchange:

    question   "Help me organize my desktop."
    answer     "Before we lay anything out, ..."
    quality    0.65
    reason     "asks a useful clarifying question but gives no concrete step"

Quality runs 0.0 to 1.0, where 1.0 is the best answer and 0.0 the worst. That is
the number a fitness function selects on, whichever way it was arrived at.

There is more than one sensible way to arrive at it, so this module is a
**registry of evaluators** rather than one hardcoded judge. `EVALUATOR` in
settings.py names the one a sweep uses, and -- like every other setting -- the
name is frozen into the sweep when it starts, so a sweep is always scored the
way it was created to be scored, and a stored sweep can say how:

    llm_judge             a judge model grades the answer on its own merits
    llm_judge_reference   the same, but shown the dataset's own answer too
    llm_judge_baseline    the same, but shown what the base model itself said,
                          and asked how much the blend improved on it
    similarity            token overlap with the dataset's answer, no model
    heuristic             local checkable properties, no model
    panel                 several judge models, aggregated

Each is an Evaluator: a name, a description, `prepare()` and `score()`.
`prepare(conf, pending, context=None)` is called once per evaluate step and
returns the Prepared bundle every call then works from -- the endpoint it
discovered, the references it loaded, the label to record. `context` is the
step's own Context, for the one evaluator that needs more than the settings and
the pending rows: llm_judge_baseline reads and fills the base-answer cache, so
it needs the database and the run folder. Everything else ignores it.
`score(item, prepared)` grades one exchange and returns `(quality, reason)`.
Raising ValueError or RuntimeError fails that one exchange and no more: main.py
counts it and moves on, which is what makes a half-scored sweep resumable.

**No knob lives in this file.** They are all in settings.py, prefixed by the
evaluator that reads them (JUDGE_*, BASELINE_*, SIMILARITY_*, HEURISTIC_*,
PANEL_*), and
they reach a step as the sweep's *stored* settings -- the same contract every
other step works under. The single exception is the API key, which is read from
the environment on purpose: a sweep records its settings into the database, and
a bearer token has no business in there.

The judge endpoints are reached over the OpenAI-compatible /v1/chat/completions
API, which LMStudio, OpenAI, OpenRouter, vLLM and most gateways all speak.

Scoring is resumable: an exchange that already has a quality is left alone
unless the evaluate step is run with --force, so an interrupted sweep can simply
be re-run. A sweep generated from template_code_mocked.py arrives already scored
and never reaches an evaluator at all.
"""

import difflib
import json
import os
import re
import statistics
import time
import urllib.error
import urllib.request

import baseline_run
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


def _normalise(text):
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
            by_question.setdefault(_normalise(record["question"]), record["reference"])
            by_position[record["position"]] = record["reference"]
    return by_question, by_position


def reference_for(item, prepared):
    """The dataset's own answer to this exchange, or None."""
    by_question, by_position = prepared.references.get("by_question", {}), \
        prepared.references.get("by_position", {})
    return by_question.get(_normalise(item["question"])) or by_position.get(item["position"])


def _prepare_references(conf, evaluator_name):
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


def _needs_grading(pending):
    """Whether any pending exchange has an answer worth spending a call on.

    An all-blank set -- a sweep where every script failed -- is scored 0.0 by
    main.py without anyone being asked, and a mocked sweep arrives scored, so
    neither should make the step demand an endpoint that need not be up.
    """
    return any((row["answer"] or "").strip() for row in pending)


# ===========================================================================
# 1. llm_judge -- a judge model, grading the answer on its own merits
# ===========================================================================


def _prepare_judge(conf, pending, context=None):
    settings = endpoint_settings(conf)
    if _needs_grading(pending) and not settings["model"]:
        settings["model"] = discover_model(settings["base_url"], settings["api_key"],
                                           settings["timeout"])
    label = settings["model"] or "llm_judge"
    note = ("judge: %s at %s" % (settings["model"], settings["base_url"])
            if _needs_grading(pending) else
            "judge: not contacted -- no answer needs grading")
    return Prepared(conf, label, settings=settings, notes=[note])


def _score_judge(item, prepared):
    prompt = prepared.conf.get("JUDGE_SYSTEM_PROMPT")
    return ask_judge(prompt,
                     "QUESTION:\n%s\n\nANSWER:\n%s" % (item["question"], item["answer"]),
                     prepared.settings)


register(Evaluator(
    "llm_judge",
    "a judge model grades each answer on its own merits (JUDGE_SYSTEM_PROMPT)",
    _prepare_judge, _score_judge, needs_endpoint=True,
))


# ===========================================================================
# 2. llm_judge_reference -- the same judge, shown the dataset's own answer
# ===========================================================================


def _prepare_judge_reference(conf, pending, context=None):
    prepared = _prepare_judge(conf, pending, context)
    prepared.references = _prepare_references(conf, "llm_judge_reference")
    prepared.notes.append(
        "reference answers: %d, from %s"
        % (len(prepared.references["by_position"]),
           generate_runs.training_set_path(conf.get("TRAINING_SET"))))
    return prepared


def _score_judge_reference(item, prepared):
    """Grade against the dataset's answer to the same question.

    The reference is what the LoRAs were trained to produce, so this is the
    evaluator that can see *style*: a judge grading on merit alone will happily
    reward a helpful prose answer from a blend that was supposed to rhyme.

    A prompt with no reference is graded on merit instead rather than failed --
    the answer is still an answer, and dropping it would quietly shrink the
    eval set for that individual and make its fitness incomparable with the
    rest.
    """
    reference = reference_for(item, prepared)
    if not reference:
        return _score_judge(item, prepared)
    prompt = prepared.conf.get("JUDGE_REFERENCE_SYSTEM_PROMPT")
    content = ("QUESTION:\n%s\n\nREFERENCE ANSWER:\n%s\n\nANSWER:\n%s"
               % (item["question"], reference, item["answer"]))
    return ask_judge(prompt, content, prepared.settings)


register(Evaluator(
    "llm_judge_reference",
    "a judge model compares each answer with the dataset's own answer to the "
    "same question (JUDGE_REFERENCE_SYSTEM_PROMPT)",
    _prepare_judge_reference, _score_judge_reference,
    wants_reference=True, needs_endpoint=True,
))


# ===========================================================================
# 3. llm_judge_baseline -- the same judge, shown what the base model said
# ===========================================================================


def _prepare_judge_baseline(conf, pending, context=None):
    """The judge, plus the base model's own answer to every pending question.

    The control comes out of the `baselines` table, and is produced -- once --
    only if something is missing from it. That is the whole economy of this
    evaluator: the first sweep on a base model pays a model load for it, every
    sweep after it pays nothing, and adding prompts to the eval set costs only
    the new ones.

    Producing it needs the database and the run folder, which is why this is
    the one evaluator handed the step's Context.
    """
    prepared = _prepare_judge(conf, pending, context)
    if not _needs_grading(pending):
        # Nothing to grade means nothing to compare, so neither the judge nor
        # the base model is troubled -- the same bargain _prepare_judge strikes.
        prepared.notes.append("baseline: not needed -- no answer needs grading")
        return prepared

    if context is None or getattr(context, "conn", None) is None:
        raise SystemExit(
            "EVALUATOR = 'llm_judge_baseline' needs the sweep's database to read "
            "and fill its cache of base-model answers, and this evaluate step was "
            "not given one."
        )

    questions = [row["question"] for row in pending
                 if (row["answer"] or "").strip()]
    prepared.baselines = baseline_run.ensure(
        context.conn, conf, questions, context.run_dir,
        keep_script=getattr(context.options, "keep_scripts", False))

    model = baseline_run.model_key(conf)
    covered = sum(1 for question in questions
                  if baseline_run.answer_for(question, prepared.baselines))
    prepared.notes.append(
        "baseline: %d of %d answer(s) have the base model's own reply to compare "
        "against, cached under %r" % (covered, len(questions), model))
    return prepared


def _score_judge_baseline(item, prepared):
    """Grade the blend against the model it was built on.

    Not "is this answer good" but "did the adapters earn their place", which is
    the question the search is actually asking. 0.5 is the middle of the rubric
    and means the blend changed nothing worth having, so an individual whose
    fitness lands there is the base model with extra steps.

    A missing control fails this one exchange rather than falling back to
    grading on merit: a merit score and an improvement score are not the same
    number, and averaging the two into one fitness would quietly reward
    whichever individuals happened to lose their baseline.
    """
    base = baseline_run.answer_for(item["question"], prepared.baselines)
    if not base:
        raise ValueError("no cached base-model answer for this question")
    prompt = prepared.conf.get("JUDGE_BASELINE_SYSTEM_PROMPT")
    content = ("QUESTION:\n%s\n\nBASE ANSWER:\n%s\n\nTUNED ANSWER:\n%s"
               % (item["question"], base, item["answer"]))
    return ask_judge(prompt, content, prepared.settings)


register(Evaluator(
    "llm_judge_baseline",
    "a judge model compares each answer with what the base model itself said, "
    "and scores the improvement (JUDGE_BASELINE_SYSTEM_PROMPT) -- 0.5 is no "
    "change; the base answers are cached in the database",
    _prepare_judge_baseline, _score_judge_baseline,
    needs_endpoint=True, wants_baseline=True,
))


# ===========================================================================
# 4. similarity -- token overlap with the dataset's answer, no model at all
# ===========================================================================


_WORD = re.compile(r"[\w']+", re.UNICODE)


def _tokens(text, case_sensitive):
    text = text if case_sensitive else text.lower()
    return _WORD.findall(text)


def _token_f1(answer, reference, case_sensitive):
    """Bag-of-words F1, counting repeats -- the SQuAD-style overlap score."""
    got, want = _tokens(answer, case_sensitive), _tokens(reference, case_sensitive)
    if not got or not want:
        return 0.0, 0.0, 0.0
    shared = 0
    pool = list(want)
    for token in got:
        if token in pool:
            pool.remove(token)
            shared += 1
    if not shared:
        return 0.0, 0.0, 0.0
    precision, recall = shared / len(got), shared / len(want)
    return 2 * precision * recall / (precision + recall), precision, recall


def _prepare_similarity(conf, pending, context=None):
    metric = conf.get("SIMILARITY_METRIC", "token_f1")
    if metric not in ("token_f1", "sequence", "containment"):
        raise SystemExit("SIMILARITY_METRIC must be 'token_f1', 'sequence' or "
                         "'containment', not %r" % metric)
    references = _prepare_references(conf, "similarity")
    return Prepared(conf, "similarity:" + metric, references=references,
                    notes=["similarity: %s against %d reference answer(s), no judge "
                           "contacted" % (metric, len(references["by_position"]))])


def _score_similarity(item, prepared):
    """How much of the dataset's answer this answer reproduces, 0..1.

    Cheap, deterministic and offline -- the whole eval half of a sweep costs
    nothing and repeats exactly, which is what makes it worth having next to a
    judge whose scores wobble. It measures agreement with one particular
    answer, though, not quality: a better answer than the dataset's scores
    badly, and that is the trade being made.
    """
    reference = reference_for(item, prepared)
    if not reference:
        raise ValueError("no reference answer for this prompt")
    case_sensitive = bool(prepared.conf.get("SIMILARITY_CASE_SENSITIVE", False))
    metric = prepared.conf.get("SIMILARITY_METRIC", "token_f1")
    answer = item["answer"]

    if metric == "sequence":
        left = answer if case_sensitive else answer.lower()
        right = reference if case_sensitive else reference.lower()
        score = difflib.SequenceMatcher(None, left, right).ratio()
        return score, "character overlap with the reference %.2f" % score
    if metric == "containment":
        got = set(_tokens(answer, case_sensitive))
        want = set(_tokens(reference, case_sensitive))
        score = (len(got & want) / len(want)) if want else 0.0
        return score, "%d/%d reference words present" % (len(got & want), len(want))

    score, precision, recall = _token_f1(answer, reference, case_sensitive)
    return score, "token f1 %.2f (p %.2f, r %.2f)" % (score, precision, recall)


register(Evaluator(
    "similarity",
    "token or character overlap with the dataset's own answer -- no judge, "
    "deterministic (SIMILARITY_METRIC)",
    _prepare_similarity, _score_similarity, wants_reference=True,
))


# ===========================================================================
# 5. heuristic -- local checkable properties, no model at all
# ===========================================================================


def _repetition(words):
    """0..1, how little this answer repeats itself.

    A blend that has collapsed says the same thing over and over -- the failure
    mode a judge is expensive to detect and a distinct-word ratio is free to.
    Short answers are exempted: five words cannot help repeating.
    """
    if len(words) < 10:
        return 1.0
    return len(set(words)) / len(words)


def _prepare_heuristic(conf, pending, context=None):
    for name in ("HEURISTIC_REQUIRE", "HEURISTIC_FORBID"):
        pattern = conf.get(name)
        if pattern:
            try:
                re.compile(pattern)
            except re.error as error:
                raise SystemExit("%s is not a valid regular expression: %s" % (name, error))
    return Prepared(conf, "heuristic",
                    notes=["heuristic: local checks only, no judge contacted"])


def _score_heuristic(item, prepared):
    """Score from properties that can be checked without asking anyone.

    Four equally weighted checks -- length, repetition, a pattern the answer
    must match, a pattern it must not -- averaged over the ones that apply.
    This is not a measure of whether an answer is *good*; it is a measure of
    whether it is malformed, which is the half of quality a judge charges the
    most to notice. Use it to smoke-test the pipeline, to pair with the mocked
    template, or to score a task whose rule really is checkable -- an
    all-uppercase adapter, say, with HEURISTIC_REQUIRE.
    """
    conf = prepared.conf
    answer = item["answer"]
    words = _WORD.findall(answer.lower())
    parts, faults = [], []

    low = conf.get("HEURISTIC_MIN_WORDS", 8)
    high = conf.get("HEURISTIC_MAX_WORDS", 400)
    if len(words) < low:
        length = len(words) / low if low else 1.0
        faults.append("short (%d words)" % len(words))
    elif high and len(words) > high:
        length = max(0.0, 1.0 - (len(words) - high) / float(high))
        faults.append("long (%d words)" % len(words))
    else:
        length = 1.0
    parts.append(length)

    repetition = _repetition(words)
    parts.append(repetition)
    if repetition < 0.5:
        faults.append("repetitive (%.0f%% distinct)" % (repetition * 100))

    required = conf.get("HEURISTIC_REQUIRE")
    if required:
        hit = bool(re.search(required, answer, re.MULTILINE))
        parts.append(1.0 if hit else 0.0)
        if not hit:
            faults.append("missing required pattern")

    forbidden = conf.get("HEURISTIC_FORBID")
    if forbidden:
        hit = bool(re.search(forbidden, answer, re.MULTILINE))
        parts.append(0.0 if hit else 1.0)
        if hit:
            faults.append("matched forbidden pattern")

    score = sum(parts) / len(parts)
    return score, ", ".join(faults) if faults else "well formed"


register(Evaluator(
    "heuristic",
    "local checks only -- length, repetition, required/forbidden patterns "
    "(HEURISTIC_*); no judge, deterministic",
    _prepare_heuristic, _score_heuristic,
))


# ===========================================================================
# 6. panel -- several judges, aggregated
# ===========================================================================


def _aggregate(scores, how):
    if how == "median":
        return statistics.median(scores)
    if how == "min":
        return min(scores)
    if how == "max":
        return max(scores)
    return sum(scores) / len(scores)


def _prepare_panel(conf, pending, context=None):
    how = conf.get("PANEL_AGGREGATE", "mean")
    if how not in ("mean", "median", "min", "max"):
        raise SystemExit("PANEL_AGGREGATE must be 'mean', 'median', 'min' or "
                         "'max', not %r" % how)
    base_url = conf.get("PANEL_BASE_URL") or conf.get("JUDGE_BASE_URL")
    models = list(conf.get("PANEL_MODELS") or [])
    grading = _needs_grading(pending)
    if grading and not models:
        # Nothing named: fall back to whatever the endpoint has loaded, so a
        # panel of one still runs rather than failing on an empty list.
        models = [discover_model(base_url, API_KEY, conf.get("JUDGE_TIMEOUT", 300))]

    members = [endpoint_settings(conf, model=model, base_url=base_url)
               for model in models]
    references = None
    if conf.get("PANEL_USE_REFERENCE", False):
        references = _prepare_references(conf, "panel")

    label = "panel:" + ",".join(models) if models else "panel"
    note = ("panel: %s at %s, aggregated by %s"
            % (", ".join(models), base_url, how) if grading else
            "panel: not contacted -- no answer needs grading")
    prepared = Prepared(conf, label[:200], references=references, notes=[note])
    prepared.settings = {"members": members, "aggregate": how}
    return prepared


def _score_panel(item, prepared):
    """Ask every member, aggregate what came back.

    A member that fails is dropped rather than fatal: a panel that loses one
    model still has a score, and losing the whole exchange because one endpoint
    hiccupped would cost the individual an answer its rivals kept. Only a panel
    where *nobody* answered fails, which main.py counts like any other failure.
    """
    conf = prepared.conf
    reference = reference_for(item, prepared) if prepared.references else None
    if reference:
        prompt = conf.get("JUDGE_REFERENCE_SYSTEM_PROMPT")
        content = ("QUESTION:\n%s\n\nREFERENCE ANSWER:\n%s\n\nANSWER:\n%s"
                   % (item["question"], reference, item["answer"]))
    else:
        prompt = conf.get("JUDGE_SYSTEM_PROMPT")
        content = "QUESTION:\n%s\n\nANSWER:\n%s" % (item["question"], item["answer"])

    scores, reasons, errors = [], [], []
    for member in prepared.settings["members"]:
        try:
            score, reason = ask_judge(prompt, content, member)
        except (RuntimeError, ValueError) as error:
            errors.append("%s: %s" % (member["model"], error))
            continue
        scores.append(score)
        if reason:
            reasons.append(reason)

    if not scores:
        raise RuntimeError("no panel member scored this answer (%s)"
                           % "; ".join(errors)[:200])

    final = _aggregate(scores, prepared.settings["aggregate"])
    spread = "/".join("%.2f" % score for score in scores)
    reason = "%s of %s" % (prepared.settings["aggregate"], spread)
    if reasons:
        reason += " -- " + reasons[0]
    return final, reason


register(Evaluator(
    "panel",
    "several judge models score each answer and the scores are aggregated "
    "(PANEL_MODELS, PANEL_AGGREGATE) -- less noise, N times the cost",
    _prepare_panel, _score_panel, needs_endpoint=True,
))
