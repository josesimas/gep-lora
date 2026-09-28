"""
gep_lora/core/evaluators/composite.py - Several evaluators, their scores combined into one.

`panel` is several *models* grading under one rubric; this is several
*evaluators*, each grading the way it always does, folded together. A judge's
opinion and a checkable property are different instruments that fail in
different ways -- a judge is charmed by a fluent answer to the wrong question, a
length band cannot tell fluent from nonsense -- and a composite lets a sweep
select on both at once:

    COMPOSITE_EVALUATORS = ["llm_judge_reference", ["heuristic", 0.5]]
    COMPOSITE_AGGREGATE  = "geometric"
    COMPOSITE_FLOOR      = 0.2

Each member is a registered evaluator, named alone (weight 1) or with a weight,
as ["name", weight] or {"name": ..., "weight": ...} -- the JSON shapes, since
this setting reaches a sweep through --set and the API as often as through
settings.py. Every member reads the same sweep settings it would read on its
own, so a member is configured exactly as if it were EVALUATOR: JUDGE_* for the
judges, HEURISTIC_* for heuristic, and so on. One member twice is refused --
that is what a weight is for -- and so is a composite inside a composite, which
would read its own members out of the same one setting and never finish.

**How the scores become one** is COMPOSITE_AGGREGATE:

    mean           weighted arithmetic mean. The default; each member moves
                   the result in proportion to its weight.
    median         weighted median. Ignores one member that is out on its own,
                   which is what a wobbly judge beside two steady scorers is.
    min            the pessimistic reading: an answer is as good as its worst
                   member says. Weights ignored.
    max            the optimistic reading: good by any one measure is good.
                   Weights ignored. Rarely what a search wants to select on.
    geometric      weighted geometric mean. A low member drags the result down
                   much harder than in the mean, and a 0 from any member is 0 --
                   "good on every count" rather than "good on average".
    harmonic       weighted harmonic mean, the F1 of the family: harsher on
                   imbalance still. Also 0 when any member is 0.
    trimmed_mean   drop the single highest and lowest score, weighted mean of
                   the rest. Needs three members to trim; fewer is the mean.

For scores on the same 0..1 scale the order is always
min <= harmonic <= geometric <= mean <= max, so moving along it is choosing how
much one bad member should cost.

**COMPOSITE_FLOOR is a veto** that applies before any of those: a member scoring
below it makes the whole answer 0.0, whatever the others said. It is how a
checkable rule -- a malformed answer, a forbidden pattern -- stops being one
voice among several and becomes a gate. None turns it off.

**A member that fails** (its score() raised) fails the whole exchange by
default, COMPOSITE_ON_FAILURE = "fail": a mean over whichever members happened
to answer is a different quantity from the mean over all of them, and averaging
the two into one fitness would reward whichever individuals lost their harshest
member. "skip" combines what came back instead -- panel's bargain -- and only an
exchange no member scored fails.

Every member is prepared, once, in the order named, and handed the step's
Context, so llm_judge_baseline as a member still fills its cache. Whether the
composite asks a judge -- which decides the abandon rule and whether a lora
server pool has to make room -- is whether any member does (`judges()`), since a
composite of local scorers costs nothing to finish.
"""

import math

from gep_lora.core.evaluators import common


NAME = "composite"

AGGREGATES = ("mean", "median", "min", "max", "geometric", "harmonic",
              "trimmed_mean")
ON_FAILURE = ("fail", "skip")

# One line each, for a form offering the choice (the async API's /settings).
AGGREGATE_DESCRIPTIONS = {
    "mean": "weighted mean: each member moves the result in proportion to its weight",
    "median": "weighted median: ignores one member that is out on its own",
    "min": "the worst member's score: good only if every member agrees (weights ignored)",
    "max": "the best member's score: good by any one measure (weights ignored)",
    "geometric": "weighted geometric mean: a low member costs far more than in the mean, and a 0 is 0",
    "harmonic": "weighted harmonic mean: harsher on imbalance still, and a 0 is 0",
    "trimmed_mean": "drop the highest and lowest score, mean of the rest (3+ members)",
}
ON_FAILURE_DESCRIPTIONS = {
    "fail": "one member failing fails the answer -- every score is over all members",
    "skip": "combine the members that did answer; fail only if none did",
}

# A member's reason is kept short in the combined one, which is written into
# exchanges.reason beside every other member's.
REASON_CHARS = 80


# ===========================================================================
# The arithmetic
# ===========================================================================


def _mean(pairs):
    total = sum(weight for _value, weight in pairs)
    return sum(value * weight for value, weight in pairs) / total


def _median(pairs):
    """Weighted median. With equal weights, exactly statistics.median."""
    pairs = sorted(pairs)
    half = sum(weight for _value, weight in pairs) / 2.0
    running = 0.0
    for index, (value, weight) in enumerate(pairs):
        running += weight
        if math.isclose(running, half) and index + 1 < len(pairs):
            # The halfway mark falls exactly between two scores: the middle of
            # them, the way an even-length median is.
            return (value + pairs[index + 1][0]) / 2.0
        if running > half:
            return value
    return pairs[-1][0]


def _geometric(pairs):
    if any(value <= 0.0 for value, _weight in pairs):
        return 0.0
    total = sum(weight for _value, weight in pairs)
    return math.exp(sum(weight * math.log(value) for value, weight in pairs) / total)


def _harmonic(pairs):
    if any(value <= 0.0 for value, _weight in pairs):
        return 0.0
    total = sum(weight for _value, weight in pairs)
    return total / sum(weight / value for value, weight in pairs)


def _trimmed_mean(pairs):
    if len(pairs) < 3:
        return _mean(pairs)
    return _mean(sorted(pairs)[1:-1])


def aggregate(pairs, how):
    """Fold [(score, weight), ...] into one score. -> float in 0..1."""
    if how == "min":
        value = min(value for value, _weight in pairs)
    elif how == "max":
        value = max(value for value, _weight in pairs)
    elif how == "median":
        value = _median(pairs)
    elif how == "geometric":
        value = _geometric(pairs)
    elif how == "harmonic":
        value = _harmonic(pairs)
    elif how == "trimmed_mean":
        value = _trimmed_mean(pairs)
    else:
        value = _mean(pairs)
    # Rounding in the logs and exps can land a hair outside the scale.
    return min(1.0, max(0.0, value))


# ===========================================================================
# The members
# ===========================================================================


def members(conf):
    """COMPOSITE_EVALUATORS, checked. -> [(Evaluator, weight), ...].

    Raises SystemExit naming what is wrong: a composite that cannot say what
    it combines has nothing to score with, and that is a setting to fix rather
    than a hundred exchanges to fail one at a time.
    """
    named = conf.get("COMPOSITE_EVALUATORS") or []
    if isinstance(named, (str, dict)):
        named = [named]
    if not named:
        raise SystemExit(
            "EVALUATOR = 'composite' combines the evaluators COMPOSITE_EVALUATORS "
            "names, and it names none. There are: %s"
            % ", ".join(name for name, _ in common.available() if name != NAME))

    out, seen = [], set()
    for entry in named:
        if isinstance(entry, str):
            name, weight = entry, 1.0
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            name, weight = entry
        elif isinstance(entry, dict) and "name" in entry and set(entry) <= {"name", "weight"}:
            name, weight = entry["name"], entry.get("weight", 1.0)
        else:
            raise SystemExit(
                "COMPOSITE_EVALUATORS entries are a name, [name, weight] or "
                "{\"name\": ..., \"weight\": ...}, not %r" % (entry,))
        if name == NAME:
            raise SystemExit("COMPOSITE_EVALUATORS cannot name 'composite' itself: "
                             "every composite reads the same one setting")
        if name in seen:
            raise SystemExit("COMPOSITE_EVALUATORS names %r twice; give it a "
                             "weight instead" % name)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) \
                or not math.isfinite(weight) or weight <= 0:
            raise SystemExit("COMPOSITE_EVALUATORS: the weight of %r must be a "
                             "number above 0, not %r" % (name, weight))
        seen.add(name)
        out.append((common.get(name), float(weight)))
    return out


def judges(conf):
    """Whether any member asks a model. Unreadable settings count as yes."""
    try:
        return any(evaluator.asks_judge(conf) for evaluator, _ in members(conf))
    except SystemExit:
        # The same pessimism start_run.wants_the_card() has: prepare() will
        # report the setting, and until then assume the card is wanted.
        return True


def _settings(conf):
    how = conf.get("COMPOSITE_AGGREGATE") or "mean"
    if how not in AGGREGATES:
        raise SystemExit("COMPOSITE_AGGREGATE must be one of %s, not %r"
                         % (", ".join(AGGREGATES), how))
    floor = conf.get("COMPOSITE_FLOOR")
    if floor is not None and (isinstance(floor, bool)
                              or not isinstance(floor, (int, float))
                              or not 0.0 <= floor <= 1.0):
        raise SystemExit("COMPOSITE_FLOOR must be None or a number from 0 to 1, "
                         "not %r" % (floor,))
    on_failure = conf.get("COMPOSITE_ON_FAILURE") or "fail"
    if on_failure not in ON_FAILURE:
        raise SystemExit("COMPOSITE_ON_FAILURE must be %s, not %r"
                         % (" or ".join(repr(one) for one in ON_FAILURE), on_failure))
    return how, floor, on_failure


def check(conf):
    """Refuse settings no composite could score under. Raises SystemExit."""
    _settings(conf)
    for evaluator, _weight in members(conf):
        if evaluator.check:
            evaluator.check(conf)


# ===========================================================================
# The evaluator
# ===========================================================================


def _weight_text(weight):
    return "" if weight == 1.0 else "*%g" % weight


def prepare(conf, pending, context=None):
    how, floor, on_failure = _settings(conf)
    chosen = members(conf)

    prepared_members, notes = [], []
    for evaluator, weight in chosen:
        # Each member prepares exactly as it would as EVALUATOR, context and
        # all, and says so in its own words under the composite's.
        one = evaluator.prepare(conf, pending, context)
        prepared_members.append((evaluator, weight, one))
        notes.extend("  %s" % note for note in one.notes)

    if how in ("min", "max") and any(weight != 1.0 for _, weight in chosen):
        notes.append("  (weights are ignored by %s)" % how)
    names = ",".join(evaluator.name + _weight_text(weight) for evaluator, weight in chosen)
    head = "composite: %s of %s" % (how, names)
    if floor is not None:
        head += ", any member below %g vetoes to 0" % floor
    head += ", a failed member %s" % ("fails the answer" if on_failure == "fail"
                                      else "is skipped")
    label = "composite:%s(%s)" % (how, ",".join(
        "%s=%s" % (evaluator.name, one.label) if one.label != evaluator.name
        else evaluator.name for evaluator, _weight, one in prepared_members))

    prepared = common.Prepared(conf, label[:200], notes=[head] + notes)
    prepared.settings = {"members": prepared_members, "aggregate": how,
                         "floor": floor, "on_failure": on_failure}
    return prepared


def score(item, prepared):
    """Ask every member, combine what came back.

    The reason names every member's score -- and the start of its own reason --
    so a combined number can be read back into the opinions it came from.
    """
    settings = prepared.settings
    pairs, parts, errors = [], [], []
    for evaluator, weight, one in settings["members"]:
        try:
            value, reason = evaluator.score(item, one)
        except (RuntimeError, ValueError) as error:
            if settings["on_failure"] == "fail":
                raise RuntimeError("%s failed: %s" % (evaluator.name, error))
            errors.append(evaluator.name)
            continue
        value = min(1.0, max(0.0, float(value)))
        pairs.append((value, weight))
        text = "%s %.2f" % (evaluator.name, value)
        if reason:
            text += " (%s)" % reason[:REASON_CHARS]
        parts.append((evaluator.name, value, text))

    if not pairs:
        raise RuntimeError("no composite member scored this answer (%s)"
                           % ", ".join(errors))

    detail = "; ".join(text for _name, _value, text in parts)
    if errors:
        detail += "; skipped %s" % ", ".join(errors)

    floor = settings["floor"]
    if floor is not None:
        below = [(name, value) for name, value, _text in parts if value < floor]
        if below:
            return 0.0, "vetoed: %s below %g -- %s" % (
                ", ".join("%s %.2f" % one for one in below), floor, detail)

    how = settings["aggregate"]
    return aggregate(pairs, how), "%s of %s" % (how, detail)


common.register(common.Evaluator(
    NAME,
    "several evaluators score each answer and their scores are combined "
    "(COMPOSITE_EVALUATORS, COMPOSITE_AGGREGATE: mean, median, min, max, "
    "geometric, harmonic, trimmed_mean; COMPOSITE_FLOOR vetoes)",
    prepare, score, needs_judge=True, judges=judges, check=check,
))
