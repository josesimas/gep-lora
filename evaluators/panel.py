"""
evaluators/panel.py - Several judge models, aggregated.

Less noise per score, N times the cost. PANEL_MODELS are the members, all
reached the same way -- one endpoint (PANEL_BASE_URL, or JUDGE_BASE_URL), or one
machine when JUDGE_BACKEND is "unsloth"; everything else about a member --
temperature, token budget, timeouts, the rubric -- comes from the JUDGE_*
settings, so a panel is several models grading identically rather than several
differently configured judges. PANEL_AGGREGATE folds their scores into one, and
PANEL_USE_REFERENCE decides which of the two shared rubrics they grade under.

**On the unsloth backend a panel is N models resident at once**, since each
member is asked about every answer in turn and unloading between them would
reload the whole panel per answer. That is the same bargain PROCESS_RUN_BATCH_SIZE
strikes with base models, and prepare() prints the count so the cost is on the
console before it is on the card.
"""

import statistics

from evaluators import common


# This evaluator's own copy of the merit rubric, used for an item with no
# reference answer (or when PANEL_USE_REFERENCE is off). A copy rather than an
# import from llm_judge: a panel is a different instrument -- several models,
# aggregated -- and the point of keeping the text here is that it can be tuned
# for that without changing what the single-judge evaluator selects on.
#
# It is worth knowing they start out identical, so a difference between the two
# files is always deliberate. As in llm_judge, a sweep's stored value still wins
# when it has one.
JUDGE_SYSTEM_PROMPT = """\
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


# This evaluator's own copy of the reference rubric, used when
# PANEL_USE_REFERENCE is on and the item has a reference answer. A copy for the
# same reason as the merit rubric above: a panel is several models aggregated,
# and its rubrics should be tunable without moving what llm_judge_reference
# selects on. Both copies start out identical to the originals, so a difference
# is always deliberate.
JUDGE_REFERENCE_SYSTEM_PROMPT = """\
You are grading a single answer produced by a fine-tuned AI model.

You will be shown:
  QUESTION          what the user asked
  REFERENCE ANSWER  how the model's training data answers that question
  ANSWER            what the model actually replied

The REFERENCE ANSWER is an example of the intended behaviour, not the only
correct answer. Do not reward copying it, and do not punish different wording,
different examples or different details.

Judge the ANSWER on:
- Manner: does it answer in the same style, voice, form and register as the
  reference? This matters most -- it is what the model was trained for.
- Substance: does it actually answer the QUESTION, as the reference does?
- Coherence: well formed and consistent, free of contradictions, repetition,
  broken grammar or nonsense.

Score from 0.0 to 1.0:
  1.0  excellent - same manner as the reference, and a sound answer
  0.7  good - recognisably the same manner, minor slips
  0.5  mixed - part of the manner, or the manner without the substance
  0.3  poor - answers, but in nothing like the intended manner
  0.0  useless - incoherent, empty, or entirely off topic

Reply with JSON and nothing else, with the score FIRST:
{"quality": <number between 0 and 1>, "reason": "<at most 12 words>"}
"""


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
        # Nothing named: fall back to the one model the judge would have used --
        # whatever the endpoint has loaded, or a refusal from the local backend,
        # which has nothing to ask -- so a panel of one still runs rather than
        # failing on an empty list.
        one = common.judge_settings(conf, base_url=base_url)
        models = [common.resolve_model(one, grading)]

    members = [common.judge_settings(conf, model=model, base_url=base_url)
               for model in models]
    references = None
    if conf.get("PANEL_USE_REFERENCE", False):
        references = common.prepare_references(conf, "panel")

    label = "panel:" + ",".join(models) if models else "panel"
    if not grading:
        note = "panel: not contacted -- no answer needs grading"
    elif members[0]["backend"] == common.UNSLOTH:
        note = ("panel: %s, loaded here with unsloth -- %d model(s) resident at "
                "once, aggregated by %s" % (", ".join(models), len(models), how))
    else:
        note = ("panel: %s at %s, aggregated by %s"
                % (", ".join(models), base_url, how))
    prepared = common.Prepared(conf, label[:200], references=references, notes=[note])
    prepared.settings = {"members": members, "aggregate": how}
    return prepared


def score(item, prepared):
    """Ask every member, aggregate what came back.

    A member that fails is dropped rather than fatal: a panel that loses one
    model still has a score, and losing the whole exchange because one endpoint
    hiccupped -- or one local generate() ran out of memory -- would cost the
    individual an answer its rivals kept. Only a panel where *nobody* answered
    fails, which start_run.py counts like any other failure.
    """
    conf = prepared.conf
    reference = common.reference_for(item, prepared) if prepared.references else None
    if reference:
        prompt = (conf.get("JUDGE_REFERENCE_SYSTEM_PROMPT")
                  or JUDGE_REFERENCE_SYSTEM_PROMPT)
        content = ("QUESTION:\n%s\n\nREFERENCE ANSWER:\n%s\n\nANSWER:\n%s"
                   % (item["question"], reference, item["answer"]))
    else:
        prompt = conf.get("JUDGE_SYSTEM_PROMPT") or JUDGE_SYSTEM_PROMPT
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
    prepare, score, needs_judge=True,
))
