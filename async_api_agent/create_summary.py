"""
create_summary.py - The journey so far, as a short illustrated story.

POST /agent/summary asks for it when the person presses **Summary** on the
guide page, or asks for one in the chat (the show_summary tool). The page
shows it in an overlay and prints it to a PDF.

    gather()     the facts: what the page knows (the dataset's numbers, the
                 conversation) beside what the API holds of the person's own
                 LoRAs, search, tests, verification and live blend
    journey()    the six chapters, each done or not -- the page draws them as
                 the illustration at the top
    charts()     the charts the facts support, as data the page draws; the
                 model places them by id ([[chart:ID]])
    fallback()   the story in built-in words, when no model answers
    create()     all of it: {title, markdown, by, fallback, note, charts,
                 journey, facts, created_at}; a chart the story did not
                 place is marked so, and the page draws it at the end

The bargain is the rest of the guide's: the numbers are computed here and
the model only tells them as a story, told what it may draw with
(prompts.SUMMARY -- the one place to tweak what it writes). The charts are
data rather than pictures from the model, so every number on them is one of
ours, and a summary written without a model has the same charts and tables.

Everything is read as the user: a LoRA, job, verification or deployment of
someone else's is left out as though it were not asked for, and a search
whose results are gone just leaves its chapters undone.
"""

import json
import re
import time

from adapters import catalog as lora_catalog
from async_api_agent import blending
from async_api_agent import planner
from async_api_agent import prompts
from async_api_agent import providers
from async_api_agent import release
from async_api_agent import settings

CHAPTERS = ("dataset", "training", "search", "testing", "verification", "live")

# A chart id the model may write: [[chart:training_loss]].
CHART = re.compile(r"\[\[chart:([a-z_]+)\]\]")


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _int_id(value):
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _clip(text, limit):
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _score(value):
    return "%.3f" % value if _number(value) else "–"


# --- the facts ------------------------------------------------------------------


def _dataset(raw):
    """The page's analysis of the dataset, cut to what a summary tells.
    Only numbers and short labels survive: it is the page's, not ours."""
    if not isinstance(raw, dict) or not _number(raw.get("records")):
        return None
    stats = raw.get("stats") if isinstance(raw.get("stats"), dict) else {}

    def median(name):
        spread = stats.get(name)
        return spread.get("median") if isinstance(spread, dict) and _number(
            spread.get("median")) else None

    keywords = [one.get("word") if isinstance(one, dict) else one
                for one in (raw.get("keywords") or [])[:6]]
    bins = [{"low": one["low"], "high": one["high"], "count": one["count"]}
            for one in (raw.get("histogram") or [])
            if isinstance(one, dict) and all(_number(one.get(k)) for k in ("low", "high", "count"))]
    return {"name": _clip(raw.get("name"), 80) or None, "records": int(raw["records"]),
            "total_records": raw.get("total_records") if _number(raw.get("total_records")) else None,
            "part": _clip(raw.get("selection_text"), 120) or None,
            "format": _clip(raw.get("format"), 30) or None,
            "user_words": median("user_words"), "assistant_words": median("assistant_words"),
            "keywords": [_clip(word, 30) for word in keywords if isinstance(word, str)],
            "problems": [_clip(one, 160) for one in (raw.get("problems") or [])[:3]
                         if isinstance(one, str)],
            "histogram": bins[:40]}


def _training(catalog, user, ids):
    loras = []
    for lora_id in ids:
        row = catalog.get(lora_id)
        if row is None or row["owner"] != user["name"]:
            continue
        one = lora_catalog.as_dict(row)
        loras.append({"id": one["id"], "name": one["name"], "rank": one["rank"],
                      "status": one["status"], "final_loss": one["final_loss"],
                      "steps": one["steps"], "seconds": one["seconds"],
                      "time": planner.human(one["seconds"]) if one["seconds"] else None,
                      "mock": bool(one["mock"]),
                      "sample": _clip(one["sample"], 300) or None,
                      "error": _clip(one["error"], 200) or None})
    return {"loras": loras} if loras else None


def _search(registry, catalog, user, job_id):
    job = registry.job(job_id, user["id"])
    if job is None:
        return None, None
    found = blending.outcome(registry, job, user, catalog)
    best = found.get("best")
    search = {"job": job_id, "label": job["label"], "status": found["status"],
              "error": _clip(found.get("error"), 200) or None,
              "generations": found.get("generations"), "population": found.get("population"),
              "evaluator": found.get("evaluator"), "blocked": found.get("blocked"),
              "loras": sorted(set((found.get("slots") or {}).values())),
              "history": found.get("history") or [],
              "best": None if best is None else {
                  "number": best["number"], "formula": best["formula"],
                  "fitness": best["fitness"]},
              "mock": bool(found.get("mock"))}
    return search, job


def _testing(registry, catalog, user, job):
    try:
        found = release.blends(registry, catalog, user, job)
    except release.ReleaseError:
        return None
    if not found["testing"]["tested"]:
        return None
    top = [{"number": one["number"], "formula": one["formula"], "loras": one["loras"],
            "search_score": one["quality"], "tested_score": one["tested"]}
           for one in found["blends"] if one["state"] != "BAD" and one["tested"] is not None]
    return {"questions": found["testing"]["records"], "tested": found["testing"]["tested"],
            "blends": found["testing"]["blends"], "recommended": found["recommended"],
            "top": top[:settings.SUMMARY_TOP_BLENDS]}


def _verification(registry, catalog, user, verification_id):
    row = registry.verification(verification_id, user["id"])
    if row is None:
        return None
    facts = release.verification_outcome(registry, catalog, user, row)
    report = facts.get("report")
    out = {"individual": facts["individual"], "status": facts["status"],
           "error": _clip(facts.get("error"), 200) or None, "report": None}
    if report is not None:
        out["report"] = {"questions": report["questions"], "source": report["questions_from"],
                         "blend_mean": report["blend_mean"], "formula": report["formula"],
                         "against": [{"lora": one["lora"], "mean": one["mean"],
                                      "wins": one["wins"], "ties": one["ties"],
                                      "losses": one["losses"], "blend_is": one["blend_is"]}
                                     for one in report["against"]]}
    return out


def _live(registry, user, deployment_id):
    """The live blend -- which, and on what; never its token."""
    row = registry.deployment(deployment_id, user["id"])
    if row is None:
        return None
    spec = json.loads(row["spec"] or "{}")
    return {"individual": row["number"], "base_model": spec.get("base_model"),
            "engine": spec.get("engine"), "live": not row["revoked_at"]}


def _conversation(raw):
    """The latest turns of the conversation, as data for the story to draw on."""
    turns = []
    for one in (raw if isinstance(raw, list) else [])[-settings.SUMMARY_TURNS:]:
        if not isinstance(one, dict) or not isinstance(one.get("text"), str):
            continue
        who = {"me": "person", "agent": "guide", "event": "event"}.get(one.get("who"))
        if who:
            turns.append({"who": who, "text": _clip(one["text"], settings.SUMMARY_TURN_CHARS)})
    return turns


def gather(registry, catalog, user, body):
    """Everything a summary may tell, from the page's request. -> facts."""
    body = body or {}
    ids = [one for one in body.get("loras") or [] if _int_id(one) is not None]
    facts = {"stage": _clip(body.get("stage"), 30) or None, "mock": bool(body.get("mock")),
             "dataset": _dataset(body.get("dataset")),
             "training": _training(catalog, user, ids),
             "search": None, "testing": None, "verification": None, "live": None,
             "conversation": _conversation(body.get("transcript"))}
    job = None
    if _int_id(body.get("job")) is not None:
        facts["search"], job = _search(registry, catalog, user, body["job"])
    if job is not None:
        facts["testing"] = _testing(registry, catalog, user, job)
    if _int_id(body.get("verification")) is not None:
        facts["verification"] = _verification(registry, catalog, user, body["verification"])
    if _int_id(body.get("deployment")) is not None:
        facts["live"] = _live(registry, user, body["deployment"])
    training = facts["training"] or {}
    facts["mock"] = facts["mock"] or any(one["mock"] for one in training.get("loras", [])) \
        or bool((facts["search"] or {}).get("mock"))
    return facts


# --- the pictures -----------------------------------------------------------------


def journey(facts):
    """The six chapters, each {key, title, done, detail}, in order."""
    dataset, training = facts["dataset"], facts["training"]
    search, testing = facts["search"], facts["testing"]
    verification, live = facts["verification"], facts["live"]
    ready = [one for one in (training or {}).get("loras", [])
             if one["status"] == lora_catalog.READY]
    best = (search or {}).get("best")
    report = (verification or {}).get("report")
    details = {
        "dataset": "%d conversations" % dataset["records"] if dataset else None,
        "training": ("%d of %d ready" % (len(ready), len(training["loras"]))) if training else None,
        "search": ("best #%d · %s" % (best["number"], _score(best["fitness"])) if best
                   else search["status"]) if search else None,
        "testing": ("#%s held up best" % testing["recommended"]) if testing else None,
        "verification": ("blend %s" % _score(report["blend_mean"]) if report
                         else verification["status"]) if verification else None,
        "live": ("#%d live" % live["individual"] if live["live"] else "taken down")
        if live else None,
    }
    done = {"dataset": bool(dataset), "training": bool(training),
            "search": bool(search), "testing": bool(testing),
            "verification": bool(verification), "live": bool(live)}
    return [{"key": key, "title": prompts.SUMMARY_CHAPTERS[key], "done": done[key],
             "detail": details[key] if done[key] else None} for key in CHAPTERS]


def charts(facts):
    """The charts the facts support, as data. -> [{id, kind, title, caption, ...}]."""
    out = []
    dataset, training = facts["dataset"], facts["training"]
    if dataset and dataset["histogram"]:
        out.append({"id": "answer_lengths", "kind": "histogram", "chapter": "dataset",
                    "title": "How long the answers are",
                    "caption": "words per answer in your dataset",
                    "bins": dataset["histogram"]})
    losses = [one for one in (training or {}).get("loras", []) if _number(one["final_loss"])]
    if losses:
        best = min(losses, key=lambda one: one["final_loss"])
        out.append({"id": "training_loss", "kind": "bars", "chapter": "training",
                    "title": "Final loss of each LoRA",
                    "caption": "lower is better: how far off its answers still were",
                    "rows": [{"label": "%s (rank %s)" % (one["name"], one["rank"]),
                              "value": one["final_loss"], "highlight": one is best}
                             for one in losses], "max": None, "digits": 3})
    history = [one for one in ((facts["search"] or {}).get("history") or [])
               if _number(one.get("best"))]
    if len(history) > 1:
        out.append({"id": "search_progress", "kind": "line", "chapter": "search",
                    "title": "The search, round by round",
                    "caption": "the best and the average blend's score in each round (0 to 1)",
                    "x_label": "round",
                    "series": [{"name": "best", "points": [[one["generation"], one["best"]]
                                                           for one in history]},
                               {"name": "average", "points": [[one["generation"], one["mean"]]
                                                              for one in history
                                                              if _number(one.get("mean"))]}]})
    testing = facts["testing"]
    if testing and testing["top"]:
        out.append({"id": "tested_blends", "kind": "bars", "chapter": "testing",
                    "title": "The top blends, before and after testing",
                    "caption": "the search's score beside the score on new questions (0 to 1)",
                    "legend": ["search", "tested"], "max": 1, "digits": 2,
                    "rows": [{"label": "#%d" % one["number"], "value": one["search_score"],
                              "value2": one["tested_score"],
                              "highlight": one["number"] == testing["recommended"]}
                             for one in testing["top"]]})
    report = (facts["verification"] or {}).get("report")
    if report and report["against"]:
        out.append({"id": "verification", "kind": "bars", "chapter": "verification",
                    "title": "The blend beside each of its LoRAs",
                    "caption": "average score on the same questions (0 to 1)",
                    "max": 1, "digits": 3,
                    "rows": [{"label": "blend #%d" % facts["verification"]["individual"],
                              "value": report["blend_mean"], "highlight": True}] +
                            [{"label": "%s alone" % one["lora"], "value": one["mean"]}
                             for one in report["against"]]})
    return out


# --- the words ----------------------------------------------------------------------


def _table(head, rows):
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(" --- " for _ in head) + "|"]
    lines += ["| " + " | ".join(str(cell).replace("|", "/") for cell in row) + " |"
              for row in rows]
    return "\n".join(lines)


def _chart(shown, key):
    return ["", "[[chart:%s]]" % key] if key in shown else []


def fallback(facts, pictures, chapters):
    """The summary in built-in words, from the same facts. -> Markdown."""
    shown = {one["id"] for one in pictures}
    dataset, training = facts["dataset"], facts["training"]
    search, testing = facts["search"], facts["testing"]
    verification, live = facts["verification"], facts["live"]
    name = (dataset or {}).get("name")
    lines = ["# " + (prompts.SUMMARY_TITLE_NAMED.format(name=re.sub(r"\.\w+$", "", name))
                     if name else prompts.SUMMARY_TITLE)]
    if not any(one["done"] for one in chapters):
        return "\n".join(lines + ["", prompts.SUMMARY_NOTHING])
    if facts["mock"]:
        lines += ["", prompts.SUMMARY_MOCK]
    learned = []
    words = prompts.SUMMARY_CHAPTERS

    if dataset:
        lines += ["", "## " + words["dataset"], "", prompts.SUMMARY_DATASET.format(
            records=dataset["records"], name=" (%s)" % dataset["name"] if dataset["name"] else "",
            user_words=dataset["user_words"] if dataset["user_words"] is not None else "?",
            assistant_words=(dataset["assistant_words"]
                             if dataset["assistant_words"] is not None else "?"))]
        lines += _chart(shown, "answer_lengths")
    if training:
        loras = training["loras"]
        ready = [one for one in loras if one["status"] == lora_catalog.READY]
        failed = len(loras) - len(ready)
        losses = [one for one in ready if _number(one["final_loss"])]
        extra = ", %d not" % failed if failed else ""
        if losses:
            best = min(losses, key=lambda one: one["final_loss"])
            text = prompts.SUMMARY_TRAINING.format(count=len(loras), ready=len(ready),
                                                   failed=extra, best_loss=_score(best["final_loss"]),
                                                   best_name=best["name"])
            if len(losses) > 1 and not facts["mock"]:
                learned.append("**%s** (rank %s) learned your data most closely."
                               % (best["name"], best["rank"]))
        else:
            text = prompts.SUMMARY_TRAINING_NO_LOSS.format(count=len(loras), ready=len(ready),
                                                           failed=extra)
        lines += ["", "## " + words["training"], "", text]
        lines += _chart(shown, "training_loss")
        lines += ["", _table(["LoRA", "Rank", "Final loss", "Time"],
                             [[one["name"], one["rank"], _score(one["final_loss"]),
                               one["time"] or "–"] for one in loras[:5]])]
    if search:
        best = search["best"]
        lines += ["", "## " + words["search"], ""]
        if search["status"] == "done" and best:
            scores = [one["best"] for one in search["history"] if _number(one.get("best"))]
            trend = ""
            if len(scores) > 1 and scores[-1] > scores[0]:
                trend = " The best score rose from %.2f to %.2f as it went." % (scores[0], scores[-1])
                learned.append("The search improved on its first round: blending paid off.")
            lines.append(prompts.SUMMARY_SEARCH.format(
                generations=search["generations"], population=search["population"],
                number=best["number"], fitness=_score(best["fitness"]), formula=best["formula"],
                trend=trend))
        else:
            lines.append(prompts.SUMMARY_SEARCH_UNFINISHED.format(
                status=search["status"],
                best="; the best so far is **#%d**, scoring **%s**" % (
                    best["number"], _score(best["fitness"])) if best else ""))
        lines += _chart(shown, "search_progress")
    if testing:
        lines += ["", "## " + words["testing"], "", prompts.SUMMARY_TESTING.format(
            questions=testing["questions"], tested=testing["tested"],
            recommended=testing["recommended"])]
        lines += _chart(shown, "tested_blends")
        lines += ["", _table(["Blend", "LoRAs", "Search", "Tested"],
                             [["#%d" % one["number"], ", ".join(one["loras"]) or "–",
                               _score(one["search_score"]), _score(one["tested_score"])]
                              for one in testing["top"]])]
        top = testing["top"][0] if testing["top"] else None
        if top and _number(top["tested_score"]) and _number(top["search_score"]):
            learned.append("On new questions the best blend scored %s against %s in the search"
                           " — %s." % (_score(top["tested_score"]), _score(top["search_score"]),
                                       "it generalises" if top["tested_score"] >= top["search_score"]
                                       else "the search's score flattered it a little"))
    if verification:
        report = verification["report"]
        lines += ["", "## " + words["verification"], ""]
        if report:
            lines.append(prompts.SUMMARY_VERIFICATION.format(
                individual=verification["individual"], blend=_score(report["blend_mean"]),
                questions=report["questions"], source=report["source"]))
            lines += _chart(shown, "verification")
            lines += ["", _table(["Against", "Won", "Tied", "Lost", "The blend is"],
                                 [[one["lora"], one["wins"], one["ties"], one["losses"],
                                   one["blend_is"]] for one in report["against"]])]
            if report["against"] and all(one["blend_is"] == "better" for one in report["against"]):
                learned.append(prompts.FALLBACK_VERIFY_BETTER)
            else:
                learned.append(prompts.FALLBACK_VERIFY_MIXED)
        else:
            lines.append(prompts.SUMMARY_VERIFICATION_UNFINISHED.format(
                individual=verification["individual"], status=verification["status"]))
    if live:
        lines += ["", "## " + words["live"], "",
                  (prompts.SUMMARY_LIVE if live["live"] else prompts.SUMMARY_LIVE_DOWN).format(
                      individual=live["individual"], base_model=live["base_model"] or "?")]
    if learned and not facts["mock"]:
        lines += ["", "## " + prompts.SUMMARY_LEARNED, ""] + ["- " + one for one in learned]
    upcoming = next((one["key"] for one in chapters if not one["done"]), None)
    lines += ["", "## " + prompts.SUMMARY_NEXT, "", prompts.SUMMARY_NEXT_STEPS[upcoming]]
    return "\n".join(lines)


def _model_facts(facts, pictures, chapters):
    """What the model is shown: the facts, the chapters and what each chart
    shows -- not the charts' points, which it is not to repeat."""
    shown = dict(facts)
    if shown["dataset"]:
        shown["dataset"] = {k: v for k, v in shown["dataset"].items() if k != "histogram"}
    return dict(shown, journey=chapters,
                charts=[{"id": one["id"], "chapter": one["chapter"], "title": one["title"],
                         "shows": one["caption"]} for one in pictures])


def title_of(markdown):
    found = re.search(r"^#\s+(.+)$", markdown or "", re.M)
    return found.group(1).strip() if found else prompts.SUMMARY_TITLE


def create(registry, catalog, user, body, choice=None):
    """The summary. -> {title, markdown, by, fallback, note, charts, journey,
    facts, created_at}."""
    facts = gather(registry, catalog, user, body)
    chapters = journey(facts)
    pictures = charts(facts)
    built = fallback(facts, pictures, chapters)
    out = {"markdown": built, "by": "built-in wording", "fallback": True, "note": None}
    if any(one["done"] for one in chapters):
        try:
            resolved = providers.resolve(choice)
            if resolved["kind"] != "scripted":
                ask = ("FACTS:\n" + json.dumps(_model_facts(facts, pictures, chapters), indent=1,
                                               ensure_ascii=False, default=str)
                       + "\n\nWrite the summary now.")
                text, model = providers.chat(resolved, prompts.SUMMARY,
                                             [{"role": "user", "content": ask}])
                out = {"markdown": clean(text, pictures), "fallback": False, "note": None,
                       "by": "%s · %s" % (resolved["label"], model)}
        except providers.ProviderError as error:
            out["note"] = prompts.FALLBACK_NOTE.format(reason=error)
    # A chart the story left out is still drawn, after it: the page shows
    # each one where it is placed, and the rest at the end.
    placed = set(CHART.findall(out["markdown"]))
    facts.pop("conversation")
    return dict(out, title=title_of(out["markdown"]), journey=chapters, facts=facts,
                charts=[dict(one, placed=one["id"] in placed) for one in pictures],
                created_at=time.strftime("%Y-%m-%d %H:%M"))


def clean(text, pictures):
    """The model's Markdown, fenced code unwrapped, a chart it named that the
    page cannot draw removed, and a chart written inside a sentence lifted
    onto a line of its own after its paragraph (lift())."""
    text = re.sub(r"^```(?:markdown|md)?\s*\n(.*?)\n```\s*$", r"\1", text.strip(), flags=re.S)
    titles = {one["id"]: one["title"] for one in pictures}
    seen = set()

    def keep(match):
        # A chart drawn already, or one there is none of, is not drawn: alone
        # on its line it goes, inside a sentence it leaves words behind.
        key, start, end = match.group(1), match.start(), match.end()
        if key in titles and key not in seen:
            seen.add(key)
            return match.group(0)
        alone = not text[text.rfind("\n", 0, start) + 1:start].strip() and \
            not text[end:(text.find("\n", end) + 1 or len(text) + 1) - 1].strip()
        if alone or key not in titles:
            return "" if alone else "the chart"
        before = text[text.rfind("\n", 0, start) + 1:start].rstrip()
        title = titles[key]
        if before and not re.search(r"[.!?:]$", before):
            title = title[:1].lower() + title[1:]
        return "*%s* (above)" % title
    return lift(CHART.sub(keep, text), titles).strip()


def lift(text, titles):
    """Every [[chart:ID]] on a line of its own, since the page draws only
    those. One inside a sentence ("As [[chart:x]] shows, ...") leaves the
    chart's title and "(below)" in its place; one ending a line just moves.
    Either way the chart is drawn after the paragraph."""
    out, waiting = [], []

    def release():
        if waiting:
            out.extend([""] + ["[[chart:%s]]" % key for key in waiting] + [""])
            waiting.clear()

    for line in text.split("\n"):
        if not line.strip():
            release()
            out.append("")
            continue
        if CHART.fullmatch(line.strip()) or not CHART.search(line):
            out.append(line)
            continue

        def words(match, line=line):
            waiting.append(match.group(1))
            after = CHART.sub("", line[match.end():])
            if not re.sub(r"[\s.,;:!?)]", "", after):
                return ""                              # ends the line: it only moves
            title = titles.get(match.group(1), "the chart")
            before = CHART.sub("", line[:match.start()]).rstrip()
            if before and not re.search(r"[.!?:]$", before):
                title = title[:1].lower() + title[1:]
            return "*%s* (below)" % title
        said = CHART.sub(words, line)
        said = re.sub(r"\s+([.,;:!?])", r"\1", re.sub(r" {2,}", " ", said)).rstrip()
        if said.strip():
            out.append(said)
    release()
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out))
