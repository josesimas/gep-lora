"""
release.py - After the search: testing its blends, verifying one, putting one live.

The third part of the guide, and the same bargain as the other two: this
module turns what the chat has set up into the bodies of requests the API
already takes, and reads back what they produced. Nothing here queues,
verifies or deploys anything -- the page sends each body and watches the
result through the API's own endpoints:

    testing       POST /jobs/{id}/test      every blend of the search, asked the
                                            questions it never saw (the job's
                                            testing split); watched through
                                            GET /jobs/{id}/status
    verification  POST /jobs/{id}/verify    the blend the person picks, beside
                                            each LoRA it is made of, alone, on
                                            the validation split -- or another
                                            split, a demo dataset or one of
                                            their LoRAs' own data; watched
                                            through GET /verifications/{id}
    going live    POST /jobs/{id}/live      the blend the person picks, behind
                                            a token; asked through POST /infer

What the person decides lives in the session's `release` part
(planner.session_of checks it on the way in):

    {"job":        id | None     the search this is about -- a job of theirs
     "individual": n | None     the blend picked ("the best"), by number
     "questions":  {"split": name} | {"file": name} | {"lora": id} | None
                                what a verification asks; None is the
                                validation split, else testing, else training
     "count":      n | None}    how many of them; None is the split's own
                                size, or VERIFY_QUESTIONS of a file

**A verification is against every LoRA of that blend, once each.** The slots
a chromosome names are the places it uses; with fewer LoRAs than places one
LoRA fills several places, and asking it the same questions once per place
would only repeat the same answers. So the slots sent are the first place of
each distinct LoRA the blend uses (`lora_slots`).
"""

import json
import os

from adapters import catalog as lora_catalog
from async_api import results
from async_api import verify
from async_api_agent import blending
from async_api_agent import guide_defaults
from async_api_agent import settings

# The splits a verification may ask, the first one a job holds being the default:
# validation was kept apart from both the search and the testing step.
SPLITS = ("validation", "testing", "training")


class ReleaseError(ValueError):
    """Why this cannot be done, in words for the person."""


# --- the session's release part -------------------------------------------------


def _id(name, value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReleaseError("%s must be a whole number" % name)
    return value


def release_of(raw):
    """The session's release part, checked. -> the full part, defaults filled."""
    raw = raw if isinstance(raw, dict) else {}
    questions = raw.get("questions")
    if questions is not None:
        ok = isinstance(questions, dict) and len(questions) == 1 and (
            questions.get("split") in SPLITS
            or (isinstance(questions.get("file"), str) and questions["file"].strip())
            or (isinstance(questions.get("lora"), int) and not isinstance(questions["lora"], bool)))
        if not ok:
            raise ReleaseError('the verification\'s questions must be {"split": %s}, '
                               '{"file": name} or {"lora": id}' % " | ".join(SPLITS))
    count = raw.get("count")
    if count is not None and (isinstance(count, bool) or not isinstance(count, int)
                              or not 1 <= count <= settings.MAX_VERIFY_QUESTIONS):
        raise ReleaseError("the verification's questions must be from 1 to %d"
                           % settings.MAX_VERIFY_QUESTIONS)
    return {"job": _id("job", raw.get("job")), "individual": _id("individual", raw.get("individual")),
            "questions": questions, "count": count}


# --- the search's blends ----------------------------------------------------------


def own_job(registry, user, job_id):
    """The user's own job. -> the row. Another user's is "no search", as a
    missing one is."""
    job = registry.job(job_id, user["id"]) if isinstance(job_id, int) else None
    if job is None:
        raise ReleaseError("no search %s of yours" % (job_id,))
    return job


def lora_slots(chromosome, conf):
    """The slots to verify a blend against: the first place of each distinct
    LoRA its chromosome uses, in slot order. -> ["L1", "L2"]."""
    folders = conf.get("LORA_SLOTS") or {}
    seen, out = set(), []
    for slot in verify.blend_slots(chromosome):
        where = os.path.normcase(lora_catalog.absolute(str(folders.get(slot, slot))))
        if where not in seen:
            seen.add(where)
            out.append(slot)
    return out


def blends(registry, catalog, user, job):
    """Every blend of a search, beside what testing said of it. -> facts.

    {"blends": [...], "testing": {...}, "splits": [...], "mock", "names",
    "recommended"}: each blend its number, chromosome and formula, the score
    the search gave it (quality, fitness), what testing gave it (tested,
    None if it was not tested) and the LoRAs it is made of. Ordered by the
    tested score where there is one -- the search picked its scores' owners
    for their training answers, so questions it never saw say more -- then
    by the search's; a blend that cannot be built comes last.
    """
    database = registry.database(job)
    try:
        found = results.detail(database, job["run_id"])
    except results.NoResults:
        raise ReleaseError("that search's results are gone; its run was deleted")
    conf = found["settings"]
    names = blending.slot_names(conf, catalog, user)
    tested = {}
    for row in found["testing"]["individuals"]:
        if row.get("verdict") == "ok" and row.get("quality") is not None:
            tested[row["number"]] = row          # the last pass on it wins
    out = []
    for one in found["population"]:
        if not one["chromosome"]:
            continue
        detail = results.individual(database, job["run_id"], one["number"]) or {}
        weights = (detail.get("execution") or {}).get("weights")
        test = tested.get(one["number"])
        used = [] if one["state"] == "BAD" else lora_slots(one["chromosome"], conf)
        out.append({"number": one["number"], "chromosome": one["chromosome"],
                    "formula": blending.formula(one["chromosome"], names, weights),
                    "state": one["state"], "fitness": one["fitness"], "quality": one["quality"],
                    "tested": None if test is None else test["quality"],
                    "tested_answers": None if test is None else test.get("answers"),
                    "slots": used, "loras": [names.get(slot, slot) for slot in used]})
    out.sort(key=lambda one: (one["state"] == "BAD", one["tested"] is None,
                              -(one["tested"] or 0.0), -(one["quality"] or 0.0),
                              one["number"]))
    held = {entry["split"]: entry["records"] for entry in found["datasets"]}
    runnable = [one for one in out if one["state"] != "BAD"]
    return {"job": job["id"], "label": job["label"], "status": job["status"],
            "blends": out, "names": names,
            "mock": conf.get("TEMPLATE") == blending.MOCKED_TEMPLATE,
            "evaluator": conf.get("EVALUATOR"),
            "testing": {"records": held.get("testing", 0), "tested": len(tested),
                        "blends": len(runnable)},
            "splits": {split: held[split] for split in SPLITS if held.get(split)},
            "recommended": runnable[0]["number"] if runnable else None}


def pick(found, release):
    """The blend the person picked, else the recommended one. -> the entry."""
    by_number = {one["number"]: one for one in found["blends"]}
    wanted = release.get("individual")
    if wanted is None:
        wanted = found["recommended"]
    one = by_number.get(wanted)
    if one is None:
        raise ReleaseError("that search holds no blend #%s" % (wanted,))
    if one["state"] == "BAD":
        raise ReleaseError("blend #%d cannot be built -- PEFT cannot mix LoRAs of different "
                           "ranks that way -- so pick another" % one["number"])
    return one


# --- the requests -----------------------------------------------------------------


def verify_request(found, release, lora_name=None):
    """The POST /jobs/{id}/verify body for the picked blend, and what it is.

    -> {body, blend, against, questions_from, count}. `lora_name` turns a
    {"lora": id} source into words; the server checks the LoRA is theirs.
    """
    one = pick(found, release)
    if not one["slots"]:
        raise ReleaseError("blend #%d names no LoRA to compare it with" % one["number"])
    body = {"individual": one["number"], "slots": list(one["slots"])}
    questions, count = release.get("questions"), release.get("count")
    if questions is None or questions.get("split"):
        split = (questions or {}).get("split") or next(iter(found["splits"]), None)
        if split is None:
            raise ReleaseError("that search holds no questions to verify on; pick a demo "
                               "dataset for them")
        if split not in found["splits"]:
            raise ReleaseError("that search holds no %s questions; it has: %s"
                               % (split, ", ".join(found["splits"]) or "none"))
        body["split"] = split
        where = "its %s questions" % split
        count = count or found["splits"][split]
    else:
        body["dataset"] = dict(questions)
        where = (questions["file"] if questions.get("file")
                 else "%s's training data" % (lora_name or "LoRA %d" % questions["lora"]))
        count = count or guide_defaults.value("VERIFY_QUESTIONS")
    body["count"] = count
    return {"body": body, "blend": one, "against": list(one["loras"]),
            "questions_from": where, "count": count}


def verification_outcome(registry, catalog, user, row):
    """What a verification came to, from the user's own row. -> facts."""
    job = registry.job(row["job_id"])
    report = verify.report(registry.verification_folder(job, row["id"]))
    facts = {"id": row["id"], "job": row["job_id"], "status": row["status"],
             "error": row["error"], "individual": row["number"],
             "chromosome": row["chromosome"], "report": None}
    if report is None:
        return facts
    try:
        conf = results.detail(registry.database(job), job["run_id"])["settings"]
    except results.NoResults:
        conf = {}
    names = blending.slot_names(conf, catalog, user)
    meta = report.get("meta") or {}
    rows = {one["key"]: one for one in report.get("summary") or []}
    blend = rows.get("blend") or {}
    against = []
    for key, one in sorted(rows.items()):
        if key == "blend":
            continue
        verdict = ("better" if one["wins"] > one["losses"] and one["p"] < 0.05 else
                   "worse" if one["losses"] > one["wins"] and one["p"] < 0.05 else "no clear difference")
        against.append({"slot": key, "lora": names.get(key, key), "mean": one["mean"],
                        "wins": one["wins"], "ties": one["ties"], "losses": one["losses"],
                        "delta": one["delta"], "p": one["p"], "blend_is": verdict})
    facts["report"] = {
        "questions": meta.get("questions"), "evaluator": meta.get("evaluator"),
        "judge": meta.get("judge"), "split": meta.get("split"),
        "questions_from": _label(row) or meta.get("dataset"),
        "blend_mean": blend.get("mean"), "graded": blend.get("graded"),
        "formula": blending.formula(row["chromosome"], names),
        "against": against,
        "mock": meta.get("evaluator") == "generated"}
    return facts


def _label(row):
    """What a verification asked, in words: its dataset's label or its split."""
    options = json.loads(row["options"] or "{}")
    return options.get("dataset_label") or (
        "its %s questions" % options["split"] if options.get("split") else None)
