"""
analysis.py - A dataset the person pasted, uploaded or picked, read and measured.

The agent's facts about a dataset come from here, never from the model: the
model is shown these numbers and a few samples, and asked to put them into
words. `analyse()` is the one entry point:

    analyse(text, name=None, selection=None)
        -> {usable, format, records, total_records, selection, selection_text,
            lines, stats, samples, problems, histogram, keywords}

It accepts every shape a person is likely to have to hand, and normalises all
of them into what a LoRA trains on -- one JSON object per line with a
`messages` list (what train.dataset_lines() and create_lora.py read):

  * JSON Lines of {"messages": [...]}           datasets/*.json, as trained on
  * a JSON array of those records
  * JSON Lines / an array of {"prompt"/"question"/"instruction"/"input",
    "response"/"answer"/"output"/"completion"} pairs
  * CSV (or TSV) with a question column and an answer column -- by name when
    the header says which, else the first two

`lines` is the normalised dataset -- the part of it `selection` keeps
(selection.py), all of it by default -- ready to go to POST /loras as it is;
every number describes that part, and `total_records` says of how many.
Records that cannot become a conversation with an assistant turn are counted
and skipped rather than failing the whole dataset, and `problems` says so.
"""

import csv
import io
import json
import re
import statistics

from async_api_agent import selection as picking
from async_api_agent import settings

# Column and field names that mean "what the user said" and "the answer".
USER_KEYS = ("prompt", "question", "instruction", "input", "user", "query",
             "conversation", "dialogue", "text_in", "source")
ASSISTANT_KEYS = ("response", "answer", "output", "completion", "assistant", "reply",
                  "summary", "target", "text_out")

# Words too common to say anything about what a dataset is about.
STOPWORDS = set("""
a about after all also am an and any are as at be because been before being but by can
could did do does doing don't down each even for from get got had has have having he her
here hers him his how i i'm if in into is it it's its just let like me more most my no
nor not now of off on once one only or other our out over own please same she should so
some such than that that's the their them then there these they this those through to
too under until up us very was we were what when where which while who why will with
would you you're your yours yes okay ok thank thanks hello hi
""".split())

_WORD = re.compile(r"[A-Za-zÀ-ɏ][A-Za-zÀ-ɏ'-]+")


class DatasetError(ValueError):
    """What is wrong with a dataset as a whole, in words for the person."""


def _words(text):
    return len(str(text or "").split())


def _turn(record, role):
    for message in record.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == role:
            return message.get("content")
    return None


def _from_pair(item):
    """A {prompt, answer}-shaped object as a conversation, or None."""
    lowered = {str(key).lower(): value for key, value in item.items()}
    ask = next((lowered[key] for key in USER_KEYS if lowered.get(key)), None)
    answer = next((lowered[key] for key in ASSISTANT_KEYS if lowered.get(key)), None)
    if ask is None or answer is None:
        return None
    messages = []
    system = lowered.get("system")
    if isinstance(system, str) and system.strip():
        messages.append({"role": "system", "content": system.strip()})
    messages += [{"role": "user", "content": str(ask).strip()},
                 {"role": "assistant", "content": str(answer).strip()}]
    return {"messages": messages}


def _record(item):
    """One parsed item as a conversation. -> (record or None, why not)."""
    if isinstance(item, str):
        return None, "a bare prompt with no answer to learn"
    if not isinstance(item, dict):
        return None, "not a JSON object"
    if "messages" in item:
        messages = item.get("messages")
        if not isinstance(messages, list):
            return None, "'messages' is not a list"
        turns = [m for m in messages if isinstance(m, dict) and m.get("role")
                 and isinstance(m.get("content"), str)]
        if not any(m["role"] == "assistant" and m["content"].strip() for m in turns):
            return None, "no assistant turn to learn"
        if not any(m["role"] == "user" for m in turns):
            return None, "no user turn"
        return {"messages": [{"role": m["role"], "content": m["content"]} for m in turns]}, None
    pair = _from_pair(item)
    if pair is None:
        return None, "no question/answer fields it could recognise"
    return pair, None


def _csv(text):
    """Rows of a CSV/TSV as conversations. -> (records, skipped reasons, format)."""
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
    except csv.Error:
        dialect = csv.excel
    rows = [row for row in csv.reader(io.StringIO(text), dialect) if any(cell.strip() for cell in row)]
    if not rows:
        return [], [], "csv"
    header = [cell.strip().lower() for cell in rows[0]]
    user_col = next((header.index(key) for key in USER_KEYS if key in header), None)
    answer_col = next((header.index(key) for key in ASSISTANT_KEYS if key in header), None)
    body = rows[1:]
    if user_col is None or answer_col is None or user_col == answer_col:
        # No header naming them: the first two columns, and the first row is
        # data unless it looks like a header (short, no sentence in it).
        user_col, answer_col = 0, 1
        if not all(len(cell) < 40 and not cell.endswith((".", "?", "!")) for cell in rows[0][:2]):
            body = rows
    records, skipped = [], []
    for row in body:
        if len(row) <= max(user_col, answer_col) or not row[user_col].strip() \
                or not row[answer_col].strip():
            skipped.append("a row without both a question and an answer")
            continue
        records.append({"messages": [{"role": "user", "content": row[user_col].strip()},
                                     {"role": "assistant", "content": row[answer_col].strip()}]})
    return records, skipped, "csv"


def parse(text):
    """Any accepted shape -> (records, skipped reasons, format name)."""
    body = (text or "").lstrip("﻿").strip()
    if not body:
        raise DatasetError("the dataset is empty")
    if body.startswith("["):
        try:
            items = json.loads(body)
        except ValueError as error:
            raise DatasetError("it starts like a JSON array but is not valid JSON (%s)" % error)
        if not isinstance(items, list):
            raise DatasetError("a JSON array of records was expected")
        fmt = "json-array"
    else:
        lines = [line.strip() for line in body.splitlines() if line.strip()]
        if lines[0].startswith("{"):
            items, fmt = [], "jsonl"
            for number, line in enumerate(lines, 1):
                try:
                    items.append(json.loads(line))
                except ValueError:
                    if number == 1 and len(lines) > 1 and not lines[1].startswith("{"):
                        raise DatasetError("it starts like JSON but its lines are not "
                                           "one JSON object each")
                    items.append(line)
        else:
            return _csv(body)
    records, skipped = [], []
    for item in items:
        record, why = _record(item)
        if record is None:
            skipped.append(why)
        else:
            records.append(record)
    return records, skipped, fmt


def _spread(values):
    if not values:
        return {"min": 0, "median": 0, "max": 0, "mean": 0}
    return {"min": min(values), "median": int(statistics.median(values)),
            "max": max(values), "mean": round(statistics.fmean(values), 1)}


def histogram(values, bins=12):
    """Counts of `values` in equal-width bins. -> [{low, high, count}]."""
    if not values:
        return []
    low, high = min(values), max(values)
    if low == high:
        return [{"low": low, "high": high, "count": len(values)}]
    width = (high - low) / float(bins)
    counts = [0] * bins
    for value in values:
        counts[min(bins - 1, int((value - low) / width))] += 1
    return [{"low": round(low + i * width), "high": round(low + (i + 1) * width), "count": count}
            for i, count in enumerate(counts)]


def keywords(records, top=12):
    """The words that say what a dataset is about. -> [{word, count}]."""
    counts = {}
    for record in records:
        for message in record["messages"]:
            if message["role"] == "system":
                continue
            for word in _WORD.findall(message["content"].lower()):
                if len(word) > 3 and word not in STOPWORDS:
                    counts[word] = counts.get(word, 0) + 1
    ranked = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:top]
    return [{"word": word, "count": count} for word, count in ranked]


def _clip(text, limit):
    text = str(text or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def analyse(text, name=None, selection=None):
    """Read, normalise and measure one dataset -- the part of it `selection`
    keeps (selection.py), when there is one. -> the facts (see the module)."""
    records, skipped, fmt = parse(text)
    total = len(records)
    chosen = picking.clean(selection)
    records = picking.apply(records, chosen)
    user_words = [_words(_turn(r, "user")) for r in records]
    assistant_words = [_words(_turn(r, "assistant")) for r in records]
    turns = [len(r["messages"]) for r in records]
    systems = {_turn(r, "system") for r in records if _turn(r, "system")}
    seen, duplicates = set(), 0
    for record in records:
        key = json.dumps(record["messages"], sort_keys=True)
        duplicates += key in seen
        seen.add(key)
    lines = [json.dumps(record, ensure_ascii=False) for record in records]

    problems = []
    if not records and total:
        problems.append("The part of it chosen (%s) holds no records; ask for another part, "
                        "or for all of it." % picking.describe(chosen, 0, total))
    elif not records:
        problems.append("None of its %d record(s) is a conversation with an answer to learn%s."
                        % (len(skipped), " (%s)" % skipped[0] if skipped else ""))
    elif skipped:
        reasons = sorted(set(skipped))
        problems.append("%d record(s) were skipped: %s." % (len(skipped), "; ".join(reasons[:3])))
    if duplicates:
        problems.append("%d record(s) repeat another exactly." % duplicates)
    if records and len(records) < settings.FEW_RECORDS:
        problems.append("Only %d conversation(s): small for a LoRA." % len(records))

    step = max(1, len(records) // settings.SAMPLE_RECORDS) if records else 1
    samples = [{"user": _clip(_turn(r, "user"), settings.SAMPLE_CHARS),
                "assistant": _clip(_turn(r, "assistant"), settings.SAMPLE_CHARS)}
               for r in records[::step][:settings.SAMPLE_RECORDS]]

    return {
        "usable": bool(records),
        "name": name,
        "format": fmt,
        "records": len(records),
        "total_records": total,
        "selection": chosen,
        "selection_text": picking.describe(chosen, len(records), total),
        "lines": lines,
        "stats": {
            "records": len(records),
            "skipped": len(skipped),
            "duplicates": duplicates,
            "turns": _spread(turns),
            "multi_turn": sum(1 for count in turns if count > 3),
            "system_prompts": len(systems),
            "user_words": _spread(user_words),
            "assistant_words": _spread(assistant_words),
            "characters": sum(len(line) for line in lines),
        },
        "samples": samples,
        "problems": problems,
        "histogram": histogram(assistant_words),
        "keywords": keywords(records),
    }
