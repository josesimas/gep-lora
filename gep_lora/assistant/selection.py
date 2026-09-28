"""
selection.py - Which part of a dataset the LoRAs are trained on.

A selection is a small dict the page keeps in its session and the agent's
tools change -- never a copy of the data. The dataset stays what the person
gave; every step that reads it (analysis, the estimate, the plan) applies the
selection to it on the way, so "only the first 20" is one key rather than a
second dataset to keep in step with the first.

    {"first": 20}                          the first 20 records
    {"last": 10} / {"start": 5, "end": 30} the last 10 / records 5..30 (from 1)
    {"percent": 50}                        the first half
    {"sample": 25, "seed": 7}              25 at random, repeatably
    {"contains": ["fever"]}                only records mentioning any of these
    {"excludes": ["cough"]}                none mentioning any of these
    {"min_answer_words": 5, "max_answer_words": 40}
    {"drop_duplicates": true}

The order they apply in is fixed, so a selection reads the same whichever
order it was asked for in: the filters first (duplicates, words, answer
length), then the position (first / last / range / percent) among what is
left, then the random sample.
"""

import json
import random

POSITION = ("first", "last", "start", "end", "percent")
FILTERS = ("contains", "excludes", "min_answer_words", "max_answer_words", "drop_duplicates")
KEYS = POSITION + FILTERS + ("sample", "seed")


class SelectionError(ValueError):
    """A selection that cannot be applied, in words for the person."""


def _count(name, value, low=1, high=10 ** 7):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value):
        raise SelectionError("%s must be a whole number" % name)
    value = int(value)
    if not low <= value <= high:
        raise SelectionError("%s must be between %d and %d" % (name, low, high))
    return value


def _words(name, value):
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",")]
    if not isinstance(value, list) or not all(isinstance(one, str) for one in value):
        raise SelectionError("%s must be a word or a list of words" % name)
    value = [one.strip() for one in value if one.strip()]
    if not value:
        raise SelectionError("%s names no words" % name)
    return value[:20]


def clean(selection):
    """A selection checked and tidied. -> dict (empty means everything)."""
    if selection is None:
        return {}
    if not isinstance(selection, dict):
        raise SelectionError("a selection is an object")
    unknown = sorted(set(selection) - set(KEYS))
    if unknown:
        raise SelectionError("unknown selection key(s): %s" % ", ".join(unknown))
    out = {}
    for key, value in selection.items():
        if value is None or value is False:
            continue
        if key in ("first", "last", "start", "end", "sample", "min_answer_words",
                   "max_answer_words"):
            out[key] = _count(key, value, 0 if key.endswith("words") else 1)
        elif key == "seed":
            out[key] = _count(key, value, 0, 2 ** 31 - 1)
        elif key == "percent":
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not 0 < value <= 100:
                raise SelectionError("percent must be above 0 and at most 100")
            out[key] = float(value)
        elif key in ("contains", "excludes"):
            out[key] = _words(key, value)
        elif key == "drop_duplicates":
            out[key] = bool(value)
    if len([key for key in ("first", "last", "percent") if key in out]) > 1 or \
            (("start" in out or "end" in out) and any(key in out for key in ("first", "last",
                                                                            "percent"))):
        raise SelectionError("choose one of first, last, a range (start/end) or percent")
    if "start" in out and "end" in out and out["end"] < out["start"]:
        raise SelectionError("the range ends (%d) before it starts (%d)"
                             % (out["end"], out["start"]))
    if "min_answer_words" in out and "max_answer_words" in out \
            and out["max_answer_words"] < out["min_answer_words"]:
        raise SelectionError("max_answer_words is below min_answer_words")
    if "seed" in out and "sample" not in out:
        out.pop("seed")
    return out


def merge(selection, change):
    """`selection` with `change` applied. A position key replaces the other
    position keys; None clears a key. -> the cleaned result."""
    out = dict(selection or {})
    change = dict(change or {})
    if any(change.get(key) is not None for key in POSITION):
        for key in POSITION:
            out.pop(key, None)
    for key, value in change.items():
        if value is None:
            out.pop(key, None)
        else:
            out[key] = value
    return clean(out)


def _turn(record, role):
    for message in record.get("messages") or []:
        if message.get("role") == role:
            return message.get("content") or ""
    return ""


def _text(record):
    return " ".join(message.get("content") or "" for message in record.get("messages") or []
                    if message.get("role") != "system").lower()


def indexed(records, selection):
    """The records a selection keeps, with their numbers in the whole
    dataset (from 1). -> [(number, record)]."""
    selection = clean(selection)
    kept = list(enumerate(records, 1))
    if selection.get("drop_duplicates"):
        seen, unique = set(), []
        for number, record in kept:
            key = json.dumps(record.get("messages"), sort_keys=True)
            if key not in seen:
                seen.add(key)
                unique.append((number, record))
        kept = unique
    if "contains" in selection:
        words = [one.lower() for one in selection["contains"]]
        kept = [(n, r) for n, r in kept if any(word in _text(r) for word in words)]
    if "excludes" in selection:
        words = [one.lower() for one in selection["excludes"]]
        kept = [(n, r) for n, r in kept if not any(word in _text(r) for word in words)]
    if "min_answer_words" in selection:
        kept = [(n, r) for n, r in kept
                if len(_turn(r, "assistant").split()) >= selection["min_answer_words"]]
    if "max_answer_words" in selection:
        kept = [(n, r) for n, r in kept
                if len(_turn(r, "assistant").split()) <= selection["max_answer_words"]]
    if "first" in selection:
        kept = kept[:selection["first"]]
    elif "last" in selection:
        kept = kept[-selection["last"]:]
    elif "percent" in selection:
        kept = kept[:max(1, int(round(len(kept) * selection["percent"] / 100.0)))] if kept else []
    elif "start" in selection or "end" in selection:
        start = selection.get("start", 1)
        end = selection.get("end", len(records))
        kept = [(n, r) for n, r in kept if start <= n <= end]
    if "sample" in selection and selection["sample"] < len(kept):
        rng = random.Random(selection.get("seed", 0))
        chosen = sorted(rng.sample(range(len(kept)), selection["sample"]))
        kept = [kept[index] for index in chosen]
    return kept


def apply(records, selection):
    """The records a selection keeps, in their order. -> [record]."""
    return [record for _, record in indexed(records, selection)]


def describe(selection, kept, total):
    """A selection in words: 'the first 20 records, without duplicates -- 20 of 60'."""
    selection = clean(selection)
    if not selection:
        return "all %d record(s)" % total
    parts = []
    if "first" in selection:
        parts.append("the first %d" % selection["first"])
    elif "last" in selection:
        parts.append("the last %d" % selection["last"])
    elif "percent" in selection:
        parts.append("the first %g%%" % selection["percent"])
    elif "start" in selection or "end" in selection:
        parts.append("records %d to %s" % (selection.get("start", 1),
                                           selection.get("end", "the end")))
    if "sample" in selection:
        parts.append("%d picked at random (seed %d)" % (selection["sample"],
                                                          selection.get("seed", 0)))
    if "contains" in selection:
        parts.append("mentioning %s" % " or ".join(repr(w) for w in selection["contains"]))
    if "excludes" in selection:
        parts.append("not mentioning %s" % " or ".join(repr(w) for w in selection["excludes"]))
    if "min_answer_words" in selection or "max_answer_words" in selection:
        parts.append("answers of %s to %s words" % (selection.get("min_answer_words", 0),
                                                     selection.get("max_answer_words", "any")))
    if selection.get("drop_duplicates"):
        parts.append("without duplicates")
    if not any(key in selection for key in POSITION + ("sample",)):
        parts[0] = "the records " + parts[0].replace("without duplicates", "with no duplicates")
    return "%s — %d of %d record(s)" % (", ".join(parts), kept, total)
