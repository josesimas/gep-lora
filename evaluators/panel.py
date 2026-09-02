"""
evaluators/panel.py - Several judge models, aggregated.

Less noise per score, N times the cost. PANEL_MODELS are the members, all served
by one endpoint (PANEL_BASE_URL, or JUDGE_BASE_URL); everything else about a
member -- temperature, token budget, timeouts, the rubric -- comes from the
JUDGE_* settings, so a panel is several models grading identically rather than
several differently configured judges. PANEL_AGGREGATE folds their scores into
one, and PANEL_USE_REFERENCE decides which of the two shared rubrics they grade
under.
"""

import statistics

from evaluators import common


def _aggregate(scores, how):
    if how == "median":
        return statistics.median(scores)
    if how == "min":
        return min(scores)
    if how == "max":
        return max(scores)
    return sum(scores) / len(scores)


def prepare(conf, pending, context=None):
    how = conf.get("PANEL_AGGREGATE", "mean")
    if how not in ("mean", "median", "min", "max"):
        raise SystemExit("PANEL_AGGREGATE must be 'mean', 'median', 'min' or "
                         "'max', not %r" % how)
    base_url = conf.get("PANEL_BASE_URL") or conf.get("JUDGE_BASE_URL")
    models = list(conf.get("PANEL_MODELS") or [])
    grading = common.needs_grading(pending)
    if grading and not models:
        # Nothing named: fall back to whatever the endpoint has loaded, so a
        # panel of one still runs rather than failing on an empty list.
        models = [common.discover_model(base_url, common.API_KEY,
                                        conf.get("JUDGE_TIMEOUT", 300))]

    members = [common.endpoint_settings(conf, model=model, base_url=base_url)
               for model in models]
    references = None
    if conf.get("PANEL_USE_REFERENCE", False):
        references = common.prepare_references(conf, "panel")

    label = "panel:" + ",".join(models) if models else "panel"
    note = ("panel: %s at %s, aggregated by %s"
            % (", ".join(models), base_url, how) if grading else
            "panel: not contacted -- no answer needs grading")
    prepared = common.Prepared(conf, label[:200], references=references, notes=[note])
    prepared.settings = {"members": members, "aggregate": how}
    return prepared


def score(item, prepared):
    """Ask every member, aggregate what came back.

    A member that fails is dropped rather than fatal: a panel that loses one
    model still has a score, and losing the whole exchange because one endpoint
    hiccupped would cost the individual an answer its rivals kept. Only a panel
    where *nobody* answered fails, which main.py counts like any other failure.
    """
    conf = prepared.conf
    reference = common.reference_for(item, prepared) if prepared.references else None
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
            value, reason = common.ask_judge(prompt, content, member)
        except (RuntimeError, ValueError) as error:
            errors.append("%s: %s" % (member["model"], error))
            continue
        scores.append(value)
        if reason:
            reasons.append(reason)

    if not scores:
        raise RuntimeError("no panel member scored this answer (%s)"
                           % "; ".join(errors)[:200])

    final = _aggregate(scores, prepared.settings["aggregate"])
    spread = "/".join("%.2f" % value for value in scores)
    reason = "%s of %s" % (prepared.settings["aggregate"], spread)
    if reasons:
        reason += " -- " + reasons[0]
    return final, reason


common.register(common.Evaluator(
    "panel",
    "several judge models score each answer and the scores are aggregated "
    "(PANEL_MODELS, PANEL_AGGREGATE) -- less noise, N times the cost",
    prepare, score, needs_endpoint=True,
))
