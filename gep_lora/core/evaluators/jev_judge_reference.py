"""
gep_lora/core/evaluators/jev_judge_reference.py - The reference comparison, graded by Jev
instead of an LLM.

Same question as llm_judge_reference: given the dataset's own answer to a
question, does the blend's answer match its manner? The transport is what
differs. Jev (typesafe.ai's "System One" model) is not an LLM asked to produce
prose and have a number picked out of it -- it takes a `state` and a set of
typed `questions` and returns a constrained decision: a probability
distribution over predefined options, plus a confidence. There is no free-text
reply to parse and no reason in the model's own words, because that is the
trade Jev makes for being orders of magnitude cheaper and faster than a judge
model call.

This module asks a single "score" question -- Jev's primitive for rating
against an ordered set of described levels -- with the same five anchors
llm_judge_reference's rubric uses (JEV_REFERENCE_LEVELS below), and turns the
returned probability distribution into a quality by taking the
probability-weighted mean of those anchors. That is deliberate rather than
reading Jev's own `score` field directly: the API reference does not pin down
whether that field is 0- or 1-indexed, while the meaning of "put P(useless) on
0.0 and P(excellent) on 1.0" does not depend on it.

Everything here is this evaluator's own -- the HTTP call, the settings it
reads (JEV_*), the rubric -- because it is the only evaluator that speaks to
Jev. If a second one arrives (a merit-only "jev_judge", say), pull the shared
parts into gep_lora/core/evaluators/common.py the way the llm_judge family does; until then,
duplicating that machinery here would be guessing at a shape nothing needs yet.
"""

import json
import os
import time
import urllib.error
import urllib.request

from gep_lora.core.blends import generate_runs

from gep_lora.core.evaluators import common


# Read from the environment, never from settings: a sweep writes its settings
# into the database, and a bearer token has no business there. This is the
# same environment variable name typesafe.ai's own SDK reads.
API_KEY = os.environ.get("TYPESAFE_API_KEY", "")

# The TypeSafe evaluation endpoint. JEV_BASE_URL overrides the host; the path
# below is fixed by the API rather than configurable per sweep.
DEFAULT_BASE_URL = "https://api.typesafe.ai/v1"


# The rubric, as an ordered "score" question: worst first, matching the level
# order the answer's probability distribution is indexed by. The descriptions
# are llm_judge_reference's own five anchors, so the two evaluators select on
# the same idea of "matches the reference's manner" and a sweep can be
# switched from one to the other without changing what a 0.7 means.
#
# (label, description sent to Jev as that level's criteria, quality anchor)
JEV_REFERENCE_LEVELS = (
    ("useless", "Incoherent, empty, or entirely off topic.", 0.0),
    ("poor", "Answers the question, but in nothing like the reference's manner.", 0.3),
    ("mixed", "Some of the reference's manner, or the manner without the substance.", 0.5),
    ("good", "Recognisably the reference's manner and substance, with minor slips.", 0.7),
    ("excellent", "The reference's manner, and a sound answer to the question.", 1.0),
)

# The question sent alongside the levels above. Jev's "instructions" field is
# the question itself; the levels' own descriptions (sent as that question's
# "criteria") are what distinguishes one score from the next, so this stays
# short.
JEV_REFERENCE_INSTRUCTIONS = (
    "Rate how well ANSWER matches REFERENCE ANSWER's manner (style, voice, "
    "form, register) and substance (does it actually answer QUESTION), and "
    "whether it is coherent. Do not reward copying REFERENCE ANSWER -- reward "
    "answering in its manner."
)


def jev_settings(conf):
    """The JEV_* block of a sweep's settings, resolved for ask_jev().

    Defaults are spelled out here, the way judge_settings() spells out
    JUDGE_*'s, so a sweep created before one of these knobs existed still
    resolves rather than raising a KeyError on resume.
    """
    return {
        "base_url": conf.get("JEV_BASE_URL") or DEFAULT_BASE_URL,
        "api_key": API_KEY,
        "model": conf.get("JEV_MODEL") or "jev-latest",
        "timeout": conf.get("JEV_TIMEOUT", 30),
        "retries": conf.get("JEV_RETRIES", 2),
        "retry_wait": conf.get("JEV_RETRY_WAIT", 3),
    }


def _post(url, payload, api_key, timeout):
    """POST JSON to the Jev endpoint, return the decoded JSON reply."""
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _quality_from(answer):
    """A Jev "score" answer -> (quality, reason).

    `answer` is `{"type": "score", "score": ..., "legend": {...},
    "probabilities": {"0": p0, ...}, "confidence": ...}`. The probability
    distribution is read against JEV_REFERENCE_LEVELS' own anchors rather than
    trusting the raw `score` field's scale -- see the module docstring -- and
    falls back to treating that field as a level index only when no
    distribution came back at all.
    """
    anchors = [value for _, _, value in JEV_REFERENCE_LEVELS]
    probabilities = answer.get("probabilities") or {}
    weighted, total, top_index, top_prob = 0.0, 0.0, None, -1.0
    for key, prob in probabilities.items():
        try:
            index, prob = int(key), float(prob)
        except (TypeError, ValueError):
            continue
        if 0 <= index < len(anchors):
            weighted += anchors[index] * prob
            total += prob
            if prob > top_prob:
                top_index, top_prob = index, prob

    if total > 0 and top_index is not None:
        quality = weighted / total
        label = JEV_REFERENCE_LEVELS[top_index][0]
    else:
        index = max(0, min(len(anchors) - 1, round(float(answer.get("score", 0.0)))))
        quality, label = anchors[index], JEV_REFERENCE_LEVELS[index][0]

    quality = max(0.0, min(1.0, quality))
    confidence = answer.get("confidence")
    reason = ("jev: %s (confidence %.2f)" % (label, confidence)
              if isinstance(confidence, (int, float)) else "jev: %s" % label)
    return quality, reason


def ask_jev(question, reference, answer, settings):
    """One grading call to Jev. Returns (quality, reason).

    Retries a transient failure (rate limit, overload, dropped connection) up
    to `settings["retries"]` times; a request the endpoint itself rejects
    (missing/invalid key, a validation error) fails once rather than retrying
    a call that will fail identically every time.
    """
    payload = {
        "model": settings["model"],
        "state": {
            "question": question,
            "reference_answer": reference,
            "answer": answer,
        },
        "questions": {
            "manner_match": {
                "type": "score",
                "instructions": JEV_REFERENCE_INSTRUCTIONS,
                "criteria": [description for _, description, _ in JEV_REFERENCE_LEVELS],
            },
        },
    }
    url = settings["base_url"].rstrip("/") + "/systemone"

    last_error = None
    for attempt in range(settings["retries"] + 1):
        try:
            reply = _post(url, payload, settings["api_key"], settings["timeout"])
            return _quality_from(reply["answers"]["manner_match"])
        except urllib.error.HTTPError as error:
            if error.code in (401, 422):
                raise RuntimeError(
                    "jev rejected the request: HTTP %d: %s"
                    % (error.code, error.read().decode("utf-8", "replace")[:200]))
            last_error = error
        except (urllib.error.URLError, OSError, KeyError, ValueError, IndexError) as error:
            last_error = error
        if attempt < settings["retries"]:
            time.sleep(settings["retry_wait"])
    raise RuntimeError("jev unreachable after %d attempts: %s"
                       % (settings["retries"] + 1, last_error))


def prepare(conf, pending, context=None):
    settings = jev_settings(conf)
    grading = common.needs_grading(pending)
    note = ("jev: %s at %s" % (settings["model"], settings["base_url"]) if grading
            else "jev: not contacted -- no answer needs grading")
    prepared = common.Prepared(conf, settings["model"], settings=settings, notes=[note])
    prepared.references = common.prepare_references(conf, "jev_judge_reference")
    prepared.notes.append(
        "reference answers: %d, from %s"
        % (len(prepared.references["by_position"]),
           generate_runs.training_set_path(conf.get("TRAINING_SET"))))
    return prepared


def score(item, prepared):
    """Grade against the dataset's answer to the same question, via Jev.

    Unlike llm_judge_reference, there is no merit-only Jev evaluator to fall
    back to for a prompt with no reference, so such an exchange fails outright
    -- the same choice llm_judge_answers makes for the same reason.
    """
    reference = common.reference_for(item, prepared)
    if not reference:
        raise ValueError(
            "jev_judge_reference needs the dataset's own answer, and this "
            "exchange has none")
    return ask_jev(item["question"], reference, item["answer"], prepared.settings)


common.register(common.Evaluator(
    "jev_judge_reference",
    "Jev (typesafe.ai) rates each answer's match to the dataset's own answer "
    "to the same question, on a fixed scale, instead of an LLM judge",
    prepare, score, wants_reference=True, needs_judge=True,
    via_judge_backend=False,
))
