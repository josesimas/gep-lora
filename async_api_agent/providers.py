"""
providers.py - One chat call to whichever model the agent talks through.

Standard library only, like the rest of the async API: the server runs under
any Python 3, and a provider is two HTTP shapes, not an SDK each.

    "openai"     POST {base_url}/chat/completions, GET {base_url}/models --
                 LM Studio, Ollama, OpenAI, Gemini, Mistral and OpenRouter all
                 speak it
    "anthropic"  POST {base_url}/messages, GET {base_url}/models, with the
                 x-api-key and anthropic-version headers
    "scripted"   no call at all; the agent's fallback wording answers

`resolve()` turns what the page asked for ({provider, model, base_url}) into
everything a call needs, `chat()` makes one, and `list_models()` is what the
page's model box offers. Anything that goes wrong is a ProviderError whose
text says what and where, which the agent turns into a fallback and a note.
"""

import json
import os
import re
import urllib.error
import urllib.request
from urllib.parse import urlparse

from async_api_agent import settings

ANTHROPIC_VERSION = "2023-06-01"

# A reasoning model served over an OpenAI-compatible endpoint (a Qwen3 in LM
# Studio, say) may put its thinking in the reply itself.
_THINKING = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


class ProviderError(Exception):
    """A call that did not produce a reply, in words for the person."""


def _judge_base_url():
    from config import settings as config
    return (config.JUDGE_BASE_URL or "").rstrip("/")


def _entry(name):
    entry = settings.PROVIDERS.get(name)
    if entry is None:
        raise ProviderError("unknown provider %r; there are: %s"
                            % (name, ", ".join(sorted(settings.PROVIDERS))))
    return entry


def _key(entry):
    return os.environ.get(entry["key_env"], "") if entry.get("key_env") else ""


def describe():
    """Every provider, as the page's picker shows it. Never a key -- only
    whether the server has one."""
    out = []
    for name, entry in settings.PROVIDERS.items():
        base_url = entry["base_url"] or (_judge_base_url() if entry["kind"] == "openai" else None)
        has_key = bool(_key(entry))
        needs_key = bool(entry.get("key_env")) and not entry.get("local")
        out.append({"name": name, "label": entry["label"], "kind": entry["kind"],
                    "base_url": base_url, "key_env": entry.get("key_env"),
                    "has_key": has_key, "needs_key": needs_key,
                    "ready": has_key or not needs_key,
                    "model": entry.get("model"), "local": bool(entry.get("local"))})
    return out


def resolve(choice=None):
    """What the page asked for, or the defaults. -> {name, kind, label,
    base_url, api_key, model, temperature}. The model may still be None for a
    local server; chat() asks it then."""
    choice = choice if isinstance(choice, dict) else {}
    name = choice.get("provider") or settings.PROVIDER
    entry = _entry(name)
    configured = entry["base_url"] or (_judge_base_url() if entry["kind"] == "openai" else "")
    base_url = (choice.get("base_url") or "").strip().rstrip("/")
    api_key = _key(entry)
    if base_url and base_url != configured:
        # Only a local provider may be pointed elsewhere, and never with a key:
        # a key goes to the URL it was issued for, not to whatever a page says.
        if not entry.get("local"):
            raise ProviderError("%s cannot be pointed at another URL" % entry["label"])
        parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ProviderError("base_url must be an http(s) URL, not %r" % base_url)
        api_key = ""
    else:
        base_url = configured
    model = (choice.get("model") or "").strip() or None
    if model is None and name == settings.PROVIDER:
        model = settings.MODEL
    if model is None:
        model = entry.get("model")
    if entry["kind"] != "scripted" and entry.get("key_env") and not entry.get("local") \
            and not api_key:
        raise ProviderError("%s needs an API key: set %s in the environment the API "
                            "server runs in" % (entry["label"], entry["key_env"]))
    return {"name": name, "kind": entry["kind"], "label": entry["label"],
            "base_url": base_url, "api_key": api_key, "model": model,
            "temperature": bool(entry.get("temperature"))}


def _request(url, payload=None, headers=None, timeout=None):
    """-> the JSON reply. Raises ProviderError naming the endpoint."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST" if data else "GET",
                                     headers=dict({"Content-Type": "application/json"},
                                                  **(headers or {})))
    try:
        with urllib.request.urlopen(request, timeout=timeout or settings.TIMEOUT) as reply:
            return json.loads(reply.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(body).get("error")
            if isinstance(detail, dict):
                detail = detail.get("message") or json.dumps(detail)
        except (ValueError, AttributeError):
            detail = body[:300]
        raise ProviderError("%s answered %d: %s" % (url, error.code, detail or error.reason),
                            ) from error
    except (urllib.error.URLError, OSError) as error:
        raise ProviderError("cannot reach %s (%s)" % (url, getattr(error, "reason", error)))
    except ValueError:
        raise ProviderError("%s did not answer in JSON" % url)


def _headers(resolved):
    if resolved["kind"] == "anthropic":
        return {"x-api-key": resolved["api_key"], "anthropic-version": ANTHROPIC_VERSION}
    return {"Authorization": "Bearer " + resolved["api_key"]} if resolved["api_key"] else {}


def list_models(resolved, timeout=None):
    """The chat models the provider lists, in its order. -> [id]."""
    if resolved["kind"] == "scripted":
        return ["scripted"]
    reply = _request(resolved["base_url"] + "/models", headers=_headers(resolved),
                     timeout=timeout or settings.MODELS_TIMEOUT)
    listed = reply.get("data") if isinstance(reply, dict) else None
    ids = [entry.get("id") for entry in listed or [] if isinstance(entry, dict)]
    # Embedding models are listed beside chat ones by LM Studio and Ollama.
    return [one for one in ids if isinstance(one, str) and "embed" not in one.lower()]


def model_for(resolved):
    """The model a call goes to: the one asked for, else the first a local
    server lists -- the judges' discover_model(), for the agent."""
    if resolved["model"]:
        return resolved["model"]
    models = list_models(resolved)
    if not models:
        raise ProviderError("%s lists no chat models; load one first" % resolved["base_url"])
    resolved["model"] = models[0]
    return models[0]


def _turns(messages):
    """The conversation as the APIs take it: user first, roles alternating,
    nothing empty. The page's history opens with the agent's welcome, which
    no API accepts as a first turn."""
    out = []
    for message in messages or []:
        role = message.get("role")
        text = str(message.get("content") or "").strip()
        if role not in ("user", "assistant") or not text:
            continue
        if not out and role == "assistant":
            continue
        if out and out[-1]["role"] == role:
            out[-1]["content"] += "\n\n" + text
        else:
            out.append({"role": role, "content": text})
    return out


def chat(resolved, system, messages):
    """One reply. -> (text, model). Raises ProviderError."""
    if resolved["kind"] == "scripted":
        raise ProviderError("no model is configured")
    turns = _turns(messages)
    if not turns:
        raise ProviderError("nothing to send")
    model = model_for(resolved)
    if resolved["kind"] == "anthropic":
        return _anthropic(resolved, model, system, turns), model
    return _openai(resolved, model, system, turns), model


def _openai(resolved, model, system, turns):
    payload = {"model": model, "messages": [{"role": "system", "content": system}] + turns}
    # OpenAI's own reasoning models take max_completion_tokens and refuse
    # max_tokens; every other server here takes max_tokens.
    payload["max_completion_tokens" if resolved["name"] == "openai" else "max_tokens"] = \
        settings.MAX_TOKENS
    if resolved["temperature"]:
        payload["temperature"] = settings.TEMPERATURE
    reply = _request(resolved["base_url"] + "/chat/completions", payload, _headers(resolved))
    try:
        text = reply["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError, AttributeError):
        raise ProviderError("%s sent a reply with no message in it" % resolved["base_url"])
    text = _THINKING.sub("", text).strip()
    if not text:
        raise ProviderError("%s replied with nothing (a reasoning model out of tokens?)" % model)
    return text


def _anthropic(resolved, model, system, turns):
    payload = {"model": model, "max_tokens": settings.ANTHROPIC_MAX_TOKENS,
               "system": system, "messages": turns}
    if settings.ANTHROPIC_EFFORT:
        payload["output_config"] = {"effort": settings.ANTHROPIC_EFFORT}
    url = resolved["base_url"] + "/messages"
    try:
        reply = _request(url, payload, _headers(resolved))
    except ProviderError as error:
        # An older model (Haiku 4.5) refuses effort; the reply is worth more
        # than the setting.
        if "output_config" not in payload or "400" not in str(error):
            raise
        payload.pop("output_config")
        reply = _request(url, payload, _headers(resolved))
    if reply.get("stop_reason") == "refusal":
        raise ProviderError("%s declined to answer" % model)
    # Thinking blocks come first on the newest models; only text is the reply.
    text = "".join(block.get("text", "") for block in reply.get("content") or []
                   if isinstance(block, dict) and block.get("type") == "text").strip()
    if not text:
        raise ProviderError("%s replied with no text" % model)
    return text
