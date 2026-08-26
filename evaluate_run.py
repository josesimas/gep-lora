"""
evaluate_run.py - Score an answer with a judge model.

The process step stores the question/answer pairs a blended model produced.
This module grades them: it sends each pair to a *different* model with a
grading system prompt, and returns the score main.py writes back onto that
exchange:

    question   "Help me organize my desktop."
    answer     "Before we lay anything out, ..."
    quality    0.65
    reason     "asks a useful clarifying question but gives no concrete step"

Quality runs 0.0 to 1.0, where 1.0 is the best answer and 0.0 the worst. That is
the number a fitness function selects on.

The judge is reached over the OpenAI-compatible /v1/chat/completions API, which
LMStudio, OpenAI, OpenRouter, vLLM and most gateways all speak. It defaults to a
local LMStudio instance; point BASE_URL and API_KEY at a cloud endpoint to use a
hosted model instead. Every parameter is at the top of this file, and every one
of them is recorded into a sweep when it starts -- SYSTEM_PROMPT included, since
it is the rubric the whole search selects on.

Scoring is resumable: an exchange that already has a quality is left alone
unless the evaluate step is run with --force, so an interrupted sweep can simply
be re-run. A sweep generated from template_code_mocked.py arrives already scored
and never reaches the judge at all.
"""

import json
import os
import re
import time
import urllib.error
import urllib.request

# ===========================================================================
# The judge model. Everything you need to change lives in this block.
# ===========================================================================

# Where the judge lives. Default is the local LMStudio instance; its API is
# OpenAI-compatible, so a cloud endpoint is a drop-in replacement:
#   OpenAI      https://api.openai.com/v1
#   OpenRouter  https://openrouter.ai/api/v1
#   vLLM        http://<host>:8000/v1
BASE_URL = "http://172.22.208.1:1234/v1"

# Sent as "Authorization: Bearer <key>". LMStudio ignores it; cloud endpoints
# require it. Falls back to the JUDGE_API_KEY environment variable so a real key
# never has to be written into this file.
API_KEY = os.environ.get("JUDGE_API_KEY", "")

# Which model does the grading. None asks the endpoint what it has loaded, which
# is what you want with LMStudio; name it explicitly for a cloud model.
MODEL = None

# Grading should be repeatable, so keep the temperature at zero.
TEMPERATURE = 0.0

# The judge emits a short JSON object, but reasoning models spend tokens
# thinking first and return an empty message if they run out mid-thought, so
# this needs far more headroom than the answer itself requires.
MAX_TOKENS = 2000

# Seconds to wait for one grading call, and how many times to retry a call that
# fails for a transient reason (connection dropped, 5xx, rate limit).
TIMEOUT = 300
RETRIES = 2
RETRY_WAIT = 3

# Ask for a JSON object back. Set to None for an endpoint that rejects the
# parameter -- the prompt asks for JSON anyway, and a 400 falls back to that.
RESPONSE_FORMAT = {"type": "json_object"}

# How the judge is told to grade. This is the rubric the whole search selects on,
# so it is worth tuning deliberately.
SYSTEM_PROMPT = """\
You are grading the quality of a single answer given by an AI planning coach.

You will be shown the QUESTION a user asked and the ANSWER the coach gave.
Judge the answer only, on how well it serves the person who asked.

Consider:
- Relevance: does it address what was actually asked?
- Usefulness: could the person act on it, or are they left stuck?
- Specificity: concrete and grounded rather than vague filler.
- Coherence: well formed and consistent, free of contradictions, repetition,
  broken grammar or nonsense.
- Appropriateness: sensible length and tone. Asking one focused clarifying
  question is fine when the request genuinely needs it; deflecting every
  request without helping is not.

Score from 0.0 to 1.0:
  1.0  excellent - directly useful, specific, clear
  0.7  good - helpful, minor weaknesses
  0.5  mixed - partly useful, vague or padded
  0.3  poor - barely addresses the question
  0.0  useless - incoherent, empty, or entirely off topic

Reply with JSON and nothing else, with the score FIRST:
{"quality": <number between 0 and 1>, "reason": "<at most 12 words>"}
"""

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
            "loaded and its server started? Point BASE_URL at a different endpoint "
            "to use another one."
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


def judge(question, answer, settings):
    """Score one question/answer pair. Returns (quality, reason)."""
    payload = {
        "model": settings["model"],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": "QUESTION:\n%s\n\nANSWER:\n%s" % (question, answer)},
        ],
    }
    if RESPONSE_FORMAT:
        payload["response_format"] = RESPONSE_FORMAT

    url = settings["base_url"].rstrip("/") + "/chat/completions"
    last_error = None
    for attempt in range(RETRIES + 1):
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

        if attempt < RETRIES:
            time.sleep(RETRY_WAIT)
    raise RuntimeError("judge unreachable after %d attempts: %s" % (RETRIES + 1, last_error))
