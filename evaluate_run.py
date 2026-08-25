"""
evaluate_run.py - Score every answer in the run transcripts with a judge model.

process_run.py writes run/output_result_NNN.json: the chromosome, the weight
draw, and the question/answer pairs the blended model produced. This reads each
of those, sends every question/answer pair to a *different* model with a grading
system prompt, and writes the score back into that exchange:

    {
      "question": "Help me organize my desktop.",
      "answer":   "Before we lay anything out, ...",
      "quality":  0.65,
      "reason":   "asks a useful clarifying question but gives no concrete step"
    }

Quality runs 0.0 to 1.0, where 1.0 is the best answer and 0.0 the worst. That is
the number a fitness function selects on.

The judge is reached over the OpenAI-compatible /v1/chat/completions API, which
LMStudio, OpenAI, OpenRouter, vLLM and most gateways all speak. It defaults to a
local LMStudio instance; point BASE_URL and API_KEY at a cloud endpoint to use a
hosted model instead. Every parameter is at the top of this file.

Scoring is resumable: an exchange that already has a "quality" is left alone
unless you pass --force, so an interrupted run can simply be re-run.

Usage:
    python evaluate_run.py                       # score everything not yet scored
    python evaluate_run.py --force               # re-score, overwriting existing scores
    python evaluate_run.py --limit 1             # one transcript, to check the setup
    python evaluate_run.py --model gpt-4o-mini --base-url https://api.openai.com/v1
"""

import argparse
import glob
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))

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

# The transcripts to score.
RESULT_GLOB = "output_result_*.json"

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
            "loaded and its server started? Set --base-url for a different endpoint."
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


def has_unscored(path):
    """Does this transcript still hold an answer without a quality?

    Asked before the judge is contacted at all: a sweep generated from
    template_code_mocked.py arrives already scored, and having to start a judge
    just to be told there is nothing to grade would undo the point of it.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return True                             # unreadable: let score_file report it
    exchanges = data["exchanges"] if isinstance(data, dict) else data
    return any("quality" not in exchange for exchange in exchanges)


def score_file(path, settings, force):
    """Score every exchange in one transcript and save it. Returns counts."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)

    # Tolerate the older shape, where the file was a bare list of exchanges.
    exchanges = data["exchanges"] if isinstance(data, dict) else data

    scored = skipped = failed = 0
    for number, exchange in enumerate(exchanges, 1):
        if "quality" in exchange and not force:
            skipped += 1
            continue

        answer = exchange.get("answer", "")
        if not answer.strip():
            # Nothing to grade: an unanswered question is worth nothing, and
            # asking the judge about an empty string just wastes a call.
            exchange["quality"] = 0.0
            exchange["reason"] = "no answer given"
            scored += 1
            print("    [%d] 0.00  (no answer)" % number)
            continue

        try:
            quality, reason = judge(exchange["question"], answer, settings)
            exchange["quality"] = round(quality, 3)
            exchange["reason"] = reason
            scored += 1
            print("    [%d] %.2f  %s" % (number, exchange["quality"], reason))
        except RuntimeError as error:
            failed += 1
            print("    [%d] FAILED  %s" % (number, error))
        except ValueError as error:
            failed += 1
            print("    [%d] FAILED  %s" % (number, error))

    # Save after each file, so an interrupted run keeps the work already done.
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return scored, skipped, failed


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Score the answers in each run transcript with a judge model."
    )
    parser.add_argument("--run-dir", default=os.path.join(_HERE, "run"),
                        help="folder holding the transcripts (default run)")
    parser.add_argument("--base-url", default=BASE_URL,
                        help="OpenAI-compatible endpoint (default %s)" % BASE_URL)
    parser.add_argument("--api-key", default=API_KEY,
                        help="bearer token; also read from JUDGE_API_KEY")
    parser.add_argument("--model", default=MODEL,
                        help="judge model id (default: whatever the endpoint has loaded)")
    parser.add_argument("--timeout", type=int, default=TIMEOUT,
                        help="seconds to allow one grading call (default %d)" % TIMEOUT)
    parser.add_argument("--limit", type=int, default=0,
                        help="score only the first N transcripts (0 = all)")
    parser.add_argument("--force", action="store_true",
                        help="re-score answers that already have a quality")
    args = parser.parse_args(argv)

    run_dir = os.path.abspath(args.run_dir)
    paths = sorted(glob.glob(os.path.join(run_dir, RESULT_GLOB)))
    if not paths:
        raise SystemExit("no %s files in %s. Run process_run.py first."
                         % (RESULT_GLOB, run_dir))
    if args.limit:
        paths = paths[:args.limit]

    # Only reach for the judge if there is grading left to do, so a transcript
    # set that is already complete costs nothing and needs no endpoint up.
    wanted = args.force or any(has_unscored(path) for path in paths)

    settings = {
        "base_url": args.base_url,
        "api_key": args.api_key,
        "timeout": args.timeout,
        "model": args.model or (discover_model(args.base_url, args.api_key, args.timeout)
                                if wanted else None),
    }

    if wanted:
        print("judge: %s at %s" % (settings["model"], settings["base_url"]))
    else:
        print("judge: not contacted -- every answer already has a quality")
    print("scoring %d transcript(s)%s\n"
          % (len(paths), " (--force: re-scoring)" if args.force else ""))

    totals = [0, 0, 0]
    started = time.time()
    for path in paths:
        print(os.path.basename(path))
        counts = score_file(path, settings, args.force)
        totals = [running + new for running, new in zip(totals, counts)]

    scored, skipped, failed = totals
    print("\nscored %d, skipped %d already done, %d failed, in %.1fs"
          % (scored, skipped, failed, time.time() - started))

    qualities = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        exchanges = data["exchanges"] if isinstance(data, dict) else data
        qualities += [e["quality"] for e in exchanges if "quality" in e]
    if qualities:
        print("quality across %d answers: min %.2f, max %.2f, mean %.2f"
              % (len(qualities), min(qualities), max(qualities),
                 sum(qualities) / len(qualities)))

    if failed and not scored:
        raise SystemExit("nothing could be scored -- check the judge endpoint")
    return 0


if __name__ == "__main__":
    sys.exit(main())
