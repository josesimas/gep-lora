"""
providers.py - One chat call to whichever model the agent talks through.

Standard library only, like the rest of the async API: the server runs under
any Python 3, and a provider is two HTTP shapes, not an SDK each.

    "openai"     POST {base_url}/chat/completions, GET {base_url}/models --
                 LM Studio, Ollama, OpenAI, Gemini, Mistral and OpenRouter all
                 speak it
    "anthropic"  POST {base_url}/messages, GET {base_url}/models, with the
                 x-api-key and anthropic-version headers
    "responses"  POST {base_url}/responses, OpenAI's Responses API -- only as
                 a gateway's per-model wire (OpenCode Go's Grok and GPT)
    "scripted"   no call at all; the agent's fallback wording answers

A provider's `kind` is its wire unless its `wires` name the model: OpenCode Go
is one key and one URL in front of all three formats, chosen by model id
(`wire()`), and its models are listed the `kind` way.

`resolve()` turns what the page asked for ({provider, model, base_url,
conversation}) into everything a call needs, `chat()` makes one, `converse()`
makes one that may call tools (the chat box's, tools.py), and `list_models()`
is what the page's model box offers. Anything that goes wrong is a ProviderError whose
text says what and where, which the agent turns into a fallback and a note.
"""

import json
import os
import re
import urllib.error
import urllib.request
import uuid
from urllib.parse import urlparse

from async_api_agent import guide_defaults
from async_api_agent import settings

ANTHROPIC_VERSION = "2023-06-01"

# Sent with every request. Cloudflare turns urllib's own "Python-urllib/3.x"
# away with a 403 (error 1010) in front of OpenCode Go, and may in front of
# any hosted provider.
USER_AGENT = "gep-lora-agent/1.0"

# What a conversation id from the page may be: a UUID, or the page's
# fallback -- nothing a header could be made to carry more than an id in.
_CONVERSATION = re.compile(r"[A-Za-z0-9_-]{8,64}")

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
    base_url, api_key, model, temperature, wires, session}. The model may still
    be None for a local server; chat() asks it then. `session` is the header a
    provider wants the conversation's id in, and the id: the page's own, else
    one made up for this request, so every round of one chat carries the
    same."""
    choice = choice if isinstance(choice, dict) else {}
    name = choice.get("provider") or guide_defaults.value("PROVIDER")
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
    if model is None and name == guide_defaults.value("PROVIDER"):
        model = guide_defaults.value("MODEL")
    if model is None:
        model = entry.get("model")
    if entry["kind"] != "scripted" and entry.get("key_env") and not entry.get("local") \
            and not api_key:
        raise ProviderError("%s needs an API key: set %s in the environment the API "
                            "server runs in" % (entry["label"], entry["key_env"]))
    session = None
    if entry.get("session_header"):
        conversation = str(choice.get("conversation") or "")
        # A header value: the page's id if it looks like one, never raw text.
        if not _CONVERSATION.fullmatch(conversation):
            conversation = uuid.uuid4().hex
        session = (entry["session_header"], conversation)
    thinking = choice.get("thinking")
    if not isinstance(thinking, bool):
        thinking = guide_defaults.value("THINKING")
    return {"name": name, "kind": entry["kind"], "label": entry["label"],
            "base_url": base_url, "api_key": api_key, "model": model,
            "temperature": bool(entry.get("temperature")), "wires": entry.get("wires") or {},
            "session": session, "thinking": thinking}


def _request(url, payload=None, headers=None, timeout=None):
    """-> the JSON reply. Raises ProviderError naming the endpoint."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST" if data else "GET",
                                     headers=dict({"Content-Type": "application/json",
                                                   "User-Agent": USER_AGENT},
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


def thinking_fields(wire_name, thinking):
    """What a request in `wire_name` carries to switch thinking on (True) or
    off (False); nothing for None, the model's own habit.

    The OpenAI format's is `reasoning_effort`, the one control LM Studio was
    found to honour ("none" took a reasoning model from 251 tokens to 4;
    chat_template_kwargs, `reasoning` and /no_think changed nothing), and what
    OpenAI, Gemini and Ollama take. Anthropic's is its `thinking` block, the
    Responses API's its `reasoning` effort."""
    if thinking is None:
        return {}
    if wire_name == "anthropic":
        return {"thinking": {"type": "adaptive"} if thinking else {"type": "disabled"}}
    if wire_name == "responses":
        return {"reasoning": {"effort": "medium" if thinking else "none"}}
    return {"reasoning_effort": "medium" if thinking else "none"}


def _asking(url, payload, headers, optional):
    """_request(), and once more without the `optional` fields if the
    provider refuses the request (400/422) while it carries any: an older
    model that knows no effort or thinking switch still has a reply worth
    more than the setting."""
    try:
        return _request(url, payload, headers)
    except ProviderError as error:
        dropped = [key for key in optional if key in payload]
        if not dropped or not re.search(r" answered (400|422):", str(error)):
            raise
        return _request(url, {key: value for key, value in payload.items()
                              if key not in dropped}, headers)


def wire(resolved, model):
    """The format a call to `model` goes in: the first of the provider's
    `wires` whose prefixes the model id starts with, else its `kind`."""
    lowered = (model or "").lower()
    for name, prefixes in resolved.get("wires", {}).items():
        if any(lowered.startswith(prefix.lower()) for prefix in prefixes):
            return name
    return resolved["kind"]


def _headers(resolved, wire_name=None):
    wire_name = wire_name or resolved["kind"]
    bearer = {"Authorization": "Bearer " + resolved["api_key"]} if resolved["api_key"] else {}
    if wire_name == "anthropic":
        headers = {"x-api-key": resolved["api_key"], "anthropic-version": ANTHROPIC_VERSION}
        # A gateway speaking Anthropic's format for someone else's model may
        # read its own bearer rather than x-api-key; Anthropic gets its own two.
        if resolved["kind"] != "anthropic":
            headers.update(bearer)
    else:
        headers = bearer
    if resolved.get("session"):
        header, conversation = resolved["session"]
        headers[header] = conversation
    return headers


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
    reply = _POSTS[wire(resolved, model)](resolved, model, system, turns)
    if not reply["text"]:
        raise ProviderError("%s replied with nothing (a reasoning model out of tokens?)" % model)
    return reply["text"], model


def converse(resolved, system, history, exchange, tools):
    """One turn of a conversation that may call tools. -> {text, calls, raw,
    model}; `calls` is [{id, name, arguments}] and empty when the model is
    done. Raises ProviderError.

    `history` is the plain conversation so far ([{role, content}]).
    `exchange` is this message's own turns, provider-neutral: the user's
    message, then per round the assistant's reply ({"role": "assistant",
    "content", "calls", "raw"}) and a {"role": "tool", "id", "name",
    "content"} per call it made -- each provider's shape is made from these
    here, so the agent's loop is written once. `tools` is [(name,
    description, JSON schema)].
    """
    if resolved["kind"] == "scripted":
        raise ProviderError("no model is configured")
    turns = _turns(list(history or []) + exchange[:1])
    if not turns:
        raise ProviderError("nothing to send")
    model = model_for(resolved)
    wire_name = wire(resolved, model)
    if wire_name == "anthropic":
        messages = turns + _anthropic_rest(exchange[1:])
        spec = [{"name": name, "description": about, "input_schema": schema}
                for name, about, schema in tools]
    elif wire_name == "responses":
        messages = turns + _responses_rest(exchange[1:])
        spec = [{"type": "function", "name": name, "description": about, "parameters": schema}
                for name, about, schema in tools]
    else:
        messages = turns + _openai_rest(exchange[1:])
        spec = [{"type": "function",
                 "function": {"name": name, "description": about, "parameters": schema}}
                for name, about, schema in tools]
    reply = _POSTS[wire_name](resolved, model, system, messages, spec)
    reply["model"] = model
    return reply


def _openai_rest(exchange):
    out = []
    for turn in exchange:
        if turn["role"] == "assistant":
            out.append({"role": "assistant", "content": turn.get("content") or "",
                        "tool_calls": [{"id": call["id"], "type": "function",
                                        "function": {"name": call["name"],
                                                     "arguments": json.dumps(call["arguments"])}}
                                       for call in turn.get("calls") or []]})
        elif turn["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": turn["id"], "name": turn["name"],
                        "content": turn["content"]})
    return out


def _anthropic_rest(exchange):
    """Anthropic's shape: the assistant's content blocks exactly as they came
    (thinking included -- they must go back unchanged), then one user turn
    holding a tool_result per call."""
    out = []
    for turn in exchange:
        if turn["role"] == "assistant":
            out.append({"role": "assistant", "content": turn.get("raw") or turn.get("content")})
        elif turn["role"] == "tool":
            result = {"type": "tool_result", "tool_use_id": turn["id"], "content": turn["content"]}
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(result)
            else:
                out.append({"role": "user", "content": [result]})
    return out


def _responses_rest(exchange):
    """The Responses API's shape: a function_call item per call and a
    function_call_output per result. The calls go back without their item
    ids, and the reasoning items not at all -- an id names a reasoning item
    the gateway need not have kept, and a call sent without it is refused."""
    out = []
    for turn in exchange:
        if turn["role"] == "assistant":
            if turn.get("content"):
                out.append({"role": "assistant", "content": turn["content"]})
            out.extend({"type": "function_call", "call_id": call["id"], "name": call["name"],
                        "arguments": json.dumps(call["arguments"])}
                       for call in turn.get("calls") or [])
        elif turn["role"] == "tool":
            out.append({"type": "function_call_output", "call_id": turn["id"],
                        "output": turn["content"]})
    return out


def _openai_post(resolved, model, system, turns, tools=None):
    payload = {"model": model, "messages": [{"role": "system", "content": system}] + turns}
    # OpenAI's own reasoning models take max_completion_tokens and refuse
    # max_tokens; every other server here takes max_tokens.
    payload["max_completion_tokens" if resolved["name"] == "openai" else "max_tokens"] = \
        settings.MAX_TOKENS
    if resolved["temperature"]:
        payload["temperature"] = settings.TEMPERATURE
    if tools:
        payload["tools"] = tools
    extra = thinking_fields("openai", resolved.get("thinking"))
    payload.update(extra)
    reply = _asking(resolved["base_url"] + "/chat/completions", payload,
                    _headers(resolved, "openai"), extra)
    try:
        message = reply["choices"][0]["message"]
        text = message.get("content") or ""
    except (KeyError, IndexError, TypeError, AttributeError):
        raise ProviderError("%s sent a reply with no message in it" % resolved["base_url"])
    calls = []
    for number, call in enumerate(message.get("tool_calls") or [], 1):
        function = call.get("function") or {}
        raw = function.get("arguments") or "{}"
        try:
            arguments = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except (ValueError, TypeError):
            arguments = {"__unreadable__": str(raw)[:200]}
        calls.append({"id": call.get("id") or "call_%d" % number,
                      "name": function.get("name") or "", "arguments": arguments})
    return {"text": _THINKING.sub("", text).strip(), "calls": calls, "raw": None}


def _anthropic_post(resolved, model, system, messages, tools=None):
    payload = {"model": model, "max_tokens": settings.ANTHROPIC_MAX_TOKENS,
               "system": system, "messages": messages}
    if tools:
        payload["tools"] = tools
    # Effort is Claude's own; another maker's model behind the same format
    # would only refuse it and cost a retry.
    if settings.ANTHROPIC_EFFORT and resolved["kind"] == "anthropic":
        payload["output_config"] = {"effort": settings.ANTHROPIC_EFFORT}
    payload.update(thinking_fields("anthropic", resolved.get("thinking")))
    # An older model (Haiku 4.5) refuses effort, and one may not know adaptive
    # thinking; the reply is worth more than either setting.
    reply = _asking(resolved["base_url"] + "/messages", payload,
                    _headers(resolved, "anthropic"), ("output_config", "thinking"))
    if reply.get("stop_reason") == "refusal":
        raise ProviderError("%s declined to answer" % model)
    blocks = [block for block in reply.get("content") or [] if isinstance(block, dict)]
    # Thinking blocks come first on the newest models; only text is the reply.
    text = "".join(block.get("text", "") for block in blocks if block.get("type") == "text")
    calls = [{"id": block.get("id"), "name": block.get("name"),
              "arguments": block.get("input") or {}}
             for block in blocks if block.get("type") == "tool_use"]
    return {"text": text.strip(), "calls": calls, "raw": blocks}


def _responses_post(resolved, model, system, messages, tools=None):
    # No temperature: the reasoning models this wire carries refuse it.
    payload = {"model": model, "instructions": system, "input": messages,
               "max_output_tokens": settings.MAX_TOKENS}
    if tools:
        payload["tools"] = tools
    extra = thinking_fields("responses", resolved.get("thinking"))
    payload.update(extra)
    reply = _asking(resolved["base_url"] + "/responses", payload,
                    _headers(resolved, "responses"), extra)
    output = reply.get("output") if isinstance(reply, dict) else None
    if not isinstance(output, list):
        raise ProviderError("%s sent a reply with no output in it" % resolved["base_url"])
    items = [item for item in output if isinstance(item, dict)]
    parts = [part for item in items if item.get("type") == "message"
             for part in item.get("content") or [] if isinstance(part, dict)]
    # Reasoning items come first; only the message's text is the reply.
    text = "".join(part.get("text", "") for part in parts if part.get("type") == "output_text")
    if not text and any(part.get("type") == "refusal" for part in parts):
        raise ProviderError("%s declined to answer" % model)
    calls = []
    for number, item in enumerate((one for one in items if one.get("type") == "function_call"), 1):
        raw = item.get("arguments") or "{}"
        try:
            arguments = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except (ValueError, TypeError):
            arguments = {"__unreadable__": str(raw)[:200]}
        calls.append({"id": item.get("call_id") or item.get("id") or "call_%d" % number,
                      "name": item.get("name") or "", "arguments": arguments})
    return {"text": _THINKING.sub("", text).strip(), "calls": calls, "raw": items}


# One per wire: (resolved, model, system, messages, tools=None) -> {text, calls, raw}.
_POSTS = {"openai": _openai_post, "anthropic": _anthropic_post, "responses": _responses_post}
