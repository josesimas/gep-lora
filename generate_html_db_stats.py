"""
generate_html_db_stats.py - one stored sweep, as a page you can open.

`store.py --show` prints a sweep; this writes the same sweep out as a single
self-contained HTML file, with the charts a column of numbers cannot draw: how
fitness moved generation by generation, how the population is spread, what the
best blend actually answered. It is a *view* of the database, derived and
disposable -- nothing here writes to the sweep, and re-running it after another
generation simply produces a newer page.

    python generate_html_db_stats.py run_db/gep.sqlite3
    python generate_html_db_stats.py run_db/gep.sqlite3 --run 2 --open

The page is written **beside the database** (`gep_run1_stats.html` next to
`gep.sqlite3`), because that is where the thing it describes lives and a report
that has wandered off from its database says nothing about which sweep it was.

It leads with the sweep's best individual -- the one thing a search exists to
produce -- drawn as a tree, with its weight draw, its per-question scores, its
transcript and the script that earned them, and then widens out to the
population, the search's own history, the testing pass, the dataset and the
settings the whole thing ran under.

Everything is inlined: no CDN, no fonts to fetch, no JSON beside it. A page
that needs the network is a page that stops working the moment it is copied
somewhere, and these are meant to be mailed, archived and opened offline. The
charts are hand-drawn SVG for the same reason, and they take their colours from
the page's CSS variables so light and dark are one drawing rather than two.

Read-only, and it says so: a path that is not already a database is an error
rather than an empty sweep, since store.connect() would happily create one.
"""

import argparse
import html
import json
import os
import re
import sys
import webbrowser

import store
from generate_population import ARITY, UNARY_OPS, decode, levels

_HERE = os.path.dirname(os.path.abspath(__file__))


# --- reading the sweep -----------------------------------------------------


def read_sweep(conn, run_id):
    """Everything one page needs, in one dict, read through store.py.

    Gathered up front rather than queried from inside the renderers: the page
    is one snapshot of a sweep, and a section that went back to the database
    for itself could describe a different moment than the section above it.
    """
    run = store.get_run(conn, run_id)
    if run is None:
        raise SystemExit("no run %s in %s" % (run_id, conn.path))

    people = store.individuals(conn, run_id)
    quality = {row["number"]: row for row in store.quality_rows(conn, run_id)}
    executions = {row["number"]: store.latest_execution(conn, row["id"])
                  for row in people}

    # force=True is "every answer of the latest execution", scored or not --
    # the step uses it to re-grade, this uses it to read.
    answers = {}
    for row in store.exchanges_to_score(conn, run_id, force=True):
        answers.setdefault(row["number"], []).append(row)

    history = store.fitness_by_generation(conn, run_id)
    champions = {entry["generation"]: store.best_of_generation(conn, run_id,
                                                               entry["generation"])
                 for entry in history}
    frames = generations(conn, run_id, history, champions)

    return {
        "run": run,
        "db_path": conn.path,
        "settings": store.get_settings(conn, run_id),
        "splits": store.dataset_summary(conn, run_id),
        "samples": {split["split"]: store.dataset(conn, run_id, split["split"])[:5]
                    for split in store.dataset_summary(conn, run_id)},
        "people": people,
        "quality": quality,
        "ranks": slot_ranks(people),
        "executions": executions,
        "answers": answers,
        "history": history,
        "champions": champions,
        "frames": frames,
        "best": choose_best(people, quality),
        "tested": store.test_quality(conn, run_id),
        "test_rows": {row["id"]: row for row in store.test_results(conn, run_id)},
        "test_summary": store.test_summary(conn, run_id),
        "totals": totals(conn, run_id),
    }


# One line of a generated script's build order, which is the only place a
# stored sweep writes down what rank a slot was:
#
#     n1_L2      = L2 @ w3                           rank 16
#
# generate_runs.build_order_block() is what writes it, and writes nothing else
# in that shape.
_LEAF_RANK = re.compile(r"=\s*(L\d)\s*@\s*w\d\s+rank\s+(\d+)")


def slot_ranks(people):
    """{slot: rank}, as the sweep's own scripts recorded it.

    Read out of `script_source` rather than off the adapters on disk, for the
    reason every step reads its sweep's stored settings rather than
    settings.py: the rank worth showing is the one the individual was *built
    with*, and `LORA_SLOTS` may have been repointed at a different adapter --
    or the adapters may not be on this machine at all -- since the sweep ran.
    The five slots are not assumed to share a rank; each is taken from a line
    that names it.

    Silent about a slot no script mentions. A sweep whose individuals have
    never been through `runs` has recorded no ranks, and the page then draws
    the leaves as it did before -- an invented rank would be worse than none.
    """
    found = {}
    for row in people:
        for slot, rank in _LEAF_RANK.findall(row["script_source"] or ""):
            found.setdefault(slot, int(rank))
    return found


def generations(conn, run_id, history, champions):
    """The population as it stood at each recorded generation, oldest first.

    `fitness_history` is the only place a sweep says what it *used to be* --
    every row keeps the chromosome, the state and the fitness as they were then
    -- so the animations replay those rows rather than reconstructing a past out
    of the present population, which mutation and selection have both moved on
    from. One frame per generation, each carrying its own membership: the
    population holds its size while the individuals in it turn over, so a frame
    is the same width as the last one and made of partly different bars -- one
    that did not exist yet is simply absent from the earlier frames, and one the
    cull took is absent from the later ones. The history keeps its rows for the
    generations an individual lived through, which is why a cull can be read
    here at all.
    """
    members = {}
    for row in store.fitness_history(conn, run_id):
        members.setdefault(row["generation"], []).append(row)
    frames = []
    for entry in history:
        champion = champions.get(entry["generation"])
        frames.append({
            "generation": entry["generation"],
            "recorded_at": entry["recorded_at"],
            "population": entry["population"],
            "best": entry["best"], "mean": entry["mean"], "worst": entry["worst"],
            "chromosome": champion["chromosome"] if champion else None,
            "rows": [{"number": row["number"], "fitness": row["fitness"],
                      "chromosome": row["chromosome"], "state": row["state"]}
                     for row in sorted(members.get(entry["generation"], []),
                                       key=lambda row: row["number"])]})
    return frames


def choose_best(people, quality):
    """The individual the page leads with.

    `is_best` when elitism has run, since that is the sweep's own answer to the
    question and the one the next generation is built around. Failing that the
    highest measured quality, lowest number breaking a tie -- elitism's rule,
    applied here so a sweep that stopped before the elitism step still leads
    with the individual it would have elected.
    """
    elected = [row for row in people if row["is_best"]]
    if elected:
        return elected[0]

    def score(row):
        measured = quality.get(row["number"])
        return (measured["quality"] if measured and measured["quality"] is not None
                else -1.0)

    ranked = sorted(people, key=lambda row: (-score(row), row["number"]))
    return ranked[0] if ranked and score(ranked[0]) >= 0 else None


def totals(conn, run_id):
    """The handful of counts the tiles need, over the whole sweep.

    Over every execution rather than the latest of each, because these describe
    what the sweep *did* -- how many times a script ran, how long that cost --
    which is not the same question as how the population stands now.
    """
    row = conn.execute(
        "SELECT COUNT(*) AS runs, COALESCE(SUM(e.seconds), 0) AS seconds,"
        "       SUM(CASE WHEN e.verdict = 'ok' THEN 1 ELSE 0 END) AS ok"
        "  FROM executions e JOIN individuals i ON i.id = e.individual_id"
        " WHERE i.run_id = ?", (run_id,)).fetchone()
    scored = conn.execute(
        "SELECT COUNT(*) AS answers,"
        "       SUM(CASE WHEN x.quality IS NULL THEN 0 ELSE 1 END) AS scored,"
        "       AVG(x.quality) AS mean"
        "  FROM exchanges x JOIN executions e ON e.id = x.execution_id"
        "  JOIN individuals i ON i.id = e.individual_id"
        " WHERE i.run_id = ?", (run_id,)).fetchone()
    return {"executions": row["runs"], "ok": row["ok"] or 0,
            "seconds": row["seconds"] or 0.0, "answers": scored["answers"],
            "scored": scored["scored"] or 0, "mean": scored["mean"]}


def all_qualities(conn, run_id):
    """Every score this sweep's judge ever gave, for the histogram."""
    return [row["quality"] for row in conn.execute(
        "SELECT x.quality FROM exchanges x"
        "  JOIN executions e ON e.id = x.execution_id"
        "  JOIN individuals i ON i.id = e.individual_id"
        " WHERE i.run_id = ? AND x.quality IS NOT NULL", (run_id,))]


# --- small formatting helpers ---------------------------------------------


def esc(value):
    """Text, safe to drop into the page. None becomes a dash, not 'None'."""
    return html.escape("-" if value is None else str(value))


def num(value, places=3):
    return "-" if value is None else ("%.*f" % (places, value))


def clock(seconds):
    """Seconds as something readable: 94.4s, 4m 21s, 1h 12m."""
    if seconds is None:
        return "-"
    seconds = float(seconds)
    if seconds < 90:
        return "%.1fs" % seconds
    if seconds < 3600:
        return "%dm %02ds" % (seconds // 60, seconds % 60)
    return "%dh %02dm" % (seconds // 3600, (seconds % 3600) // 60)


def band(value):
    """Which colour band a 0..1 score falls in. None is 'not judged'."""
    if value is None:
        return "s-none"
    if value >= 0.8:
        return "s-high"
    if value >= 0.5:
        return "s-mid"
    if value > 0:
        return "s-low"
    return "s-zero"


def meter(value, maximum=1.0, tone=None):
    """A bar in a table cell. Width only -- the number goes beside it.

    `tone` overrides the score bands, which mean nothing outside a quality: a
    weight of 0.3 is not a bad weight, and colouring it like a bad score would
    make the page say something the sweep never said.
    """
    if value is None:
        return '<span class="meter empty"></span>'
    width = 0.0 if not maximum else max(0.0, min(1.0, value / maximum)) * 100.0
    return ('<span class="meter %s"><i style="width:%.1f%%"></i></span>'
            % (band(value) if tone is None else tone, width))


def table(headers, body, sortable=True):
    """A table in its own scroll box.

    Every table here has a column that will not wrap -- a chromosome, a path, a
    seed -- so without the box a wide one widens the whole page and every other
    section scrolls sideways with it.
    """
    return ('<div class="tablewrap"><table class="data%s"><thead><tr>%s</tr>'
            '</thead><tbody>%s</tbody></table></div>'
            % (" sortable" if sortable else "", headers, body))


def shorten(text, limit=90):
    text = "" if text is None else str(text)
    return text if len(text) <= limit else text[:limit - 1] + "…"


# --- charts ----------------------------------------------------------------
#
# Hand-drawn SVG, sized in a viewBox so the page can scale them, and coloured
# through CSS variables so one drawing serves both themes.


def _axis(peak, divisions=(4, 5)):
    """A count axis that lands on round numbers. -> (top, divisions).

    The smallest top at or above `peak` that a nice step divides exactly, so a
    histogram of 50 answers is labelled 0/10/20/30/40/50 rather than 0/12/25/38.
    """
    steps = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000]
    best = None
    for count in divisions:
        for step in steps:
            top = step * count
            if top >= peak and (best is None or top < best[0]):
                best = (top, count)
                break
    return best or (max(1, peak), divisions[0])


def _grid(left, right, top, bottom, steps=5, places=2):
    """Horizontal gridlines and their labels across a 0..1 y axis."""
    parts = []
    for index in range(steps + 1):
        value = index / float(steps)
        y = bottom - value * (bottom - top)
        parts.append('<line class="grid" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"/>'
                     % (left, y, right, y))
        parts.append('<text class="tick" x="%.1f" y="%.1f" text-anchor="end">%.*f</text>'
                     % (left - 8, y + 4, places, value))
    return "".join(parts)


def history_chart(history, champions):
    """Fitness by generation: the spread as a band, the mean as a line.

    The band is what says whether the search is working -- a population whose
    worst is climbing has spread its gains, one where only the best moves has
    found a lucky individual and nothing else.

    Drawn whole, then handed to the page's script twice over: the plot geometry
    goes out as `data-plot` so the selected individual's own line can be laid
    over exactly these axes without a second copy of the arithmetic, and the
    plotted group sits behind a clip rectangle so the search can be replayed by
    widening it. Both are additions to a finished drawing -- with no script at
    all the chart is complete and the clip is already full width.
    """
    if not history:
        return ""
    width, height = 760, 300
    left, right, top, bottom = 52, width - 20, 24, height - 44
    span = right - left

    def x_at(index):
        if len(history) == 1:
            return (left + right) / 2.0
        return left + span * index / float(len(history) - 1)

    def y_at(value):
        return bottom - max(0.0, min(1.0, value or 0.0)) * (bottom - top)

    xs = [x_at(i) for i in range(len(history))]
    best = [y_at(entry["best"]) for entry in history]
    mean = [y_at(entry["mean"]) for entry in history]
    worst = [y_at(entry["worst"]) for entry in history]

    parts = [_grid(left, right, top, bottom)]

    # Everything that is "the search happening" goes inside the clip; the axes,
    # the labels and the legend stay outside it, because they are the frame the
    # replay happens in rather than part of the replay.
    parts.append('<defs><clipPath id="history-reveal">'
                 '<rect id="history-reveal-rect" x="%.1f" y="0" width="%.1f" '
                 'height="%d"/></clipPath></defs>'
                 % (left - 6, span + 12, height))
    parts.append('<g class="plotted" clip-path="url(#history-reveal)">')

    band_points = (["%.1f,%.1f" % (x, y) for x, y in zip(xs, best)] +
                   ["%.1f,%.1f" % (x, y) for x, y in zip(reversed(xs), reversed(worst))])
    parts.append('<polygon class="spread" points="%s"/>' % " ".join(band_points))

    for name, ys in (("worst", worst), ("best", best), ("mean", mean)):
        points = " ".join("%.1f,%.1f" % (x, y) for x, y in zip(xs, ys))
        parts.append('<polyline class="line %s" points="%s"/>' % (name, points))

    # Filled by the page's script for whichever individual is selected; empty,
    # and so invisible, until something selects one.
    parts.append('<polyline class="line picked" id="history-picked" points=""/>'
                 '<g id="history-picked-dots"></g>')

    for index, entry in enumerate(history):
        champion = champions.get(entry["generation"])
        tip = "gen %d - best %.3f, mean %.3f, worst %.3f, %d individuals%s" % (
            entry["generation"], entry["best"] or 0.0, entry["mean"] or 0.0,
            entry["worst"] or 0.0, entry["population"],
            "\n" + champion["chromosome"] if champion else "")
        parts.append('<g class="node"><title>%s</title>' % esc(tip))
        parts.append('<circle class="dot best" cx="%.1f" cy="%.1f" r="4.5"/>'
                     % (xs[index], best[index]))
        parts.append('<circle class="dot mean" cx="%.1f" cy="%.1f" r="4.5"/>'
                     % (xs[index], mean[index]))
        parts.append('<circle class="hit" cx="%.1f" cy="%.1f" r="18"/></g>'
                     % (xs[index], (top + bottom) / 2.0))
    parts.append('</g>')

    # The playhead the replay drags across the plot. Parked and hidden until it.
    parts.append('<line class="playhead is-off" id="history-playhead" '
                 'x1="%.1f" y1="%d" x2="%.1f" y2="%.1f"/>'
                 % (left, top - 12, left, bottom + 4))

    for index, entry in enumerate(history):
        parts.append('<text class="tick" x="%.1f" y="%.1f" text-anchor="middle">'
                     'gen %d</text>' % (xs[index], bottom + 20, entry["generation"]))
        parts.append('<text class="tick faint" x="%.1f" y="%.1f" text-anchor="middle">'
                     'n=%d</text>' % (xs[index], bottom + 34, entry["population"]))

    # The fourth entry belongs to whichever individual is selected, so it is
    # drawn empty and hidden; the script gives it a name when there is one.
    legend = [("best", "best", ""), ("mean", "mean", ""), ("worst", "worst", ""),
              ("picked", "", ' id="history-picked-key"')]
    for index, (cls, label, extra) in enumerate(legend):
        cx = right - 290 + index * 72
        parts.append('<g class="legend-key%s"%s>'
                     '<line class="line %s" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"/>'
                     '<text class="tick" x="%.1f" y="%.1f">%s</text></g>'
                     % (" is-off" if cls == "picked" else "", extra, cls,
                        cx, top - 8, cx + 22, top - 8,
                        cx + 28, top - 4, esc(label)))

    plot = json.dumps({"left": left, "right": right, "top": top, "bottom": bottom,
                       "xs": [round(x, 1) for x in xs],
                       "generations": [entry["generation"] for entry in history]})
    return ('<svg class="chart" id="history-chart" data-plot=\'%s\' '
            'viewBox="0 0 %d %d" role="img" '
            'aria-label="fitness by generation">%s</svg>'
            % (esc(plot), width, height, "".join(parts)))


def population_chart(people, quality):
    """One horizontal bar per individual: the mean score its latest run earned.

    Measured quality rather than the stored fitness column, because mutation
    clears that column and a chart of the population should still show what the
    population has actually done.
    """
    rows = []
    for row in people:
        measured = quality.get(row["number"])
        rows.append((row["number"], row["chromosome"], row["state"],
                     measured["quality"] if measured else None))
    if not rows:
        return ""

    width = 760
    left, right, top = 210, width - 56, 20
    step, bar = 26, 15
    height = top + len(rows) * step + 26
    parts = []
    for index in range(6):
        value = index / 5.0
        x = left + value * (right - left)
        parts.append('<line class="grid" x1="%.1f" y1="%d" x2="%.1f" y2="%d"/>'
                     % (x, top - 6, x, top + len(rows) * step))
        parts.append('<text class="tick" x="%.1f" y="%d" text-anchor="middle">%.1f</text>'
                     % (x, height - 8, value))

    for index, (number, chromosome, state, value) in enumerate(rows):
        y = top + index * step
        parts.append('<text class="rowlabel" x="10" y="%.1f">#%d</text>'
                     % (y + bar - 3, number))
        parts.append('<text class="rowchrom" x="%d" y="%.1f" text-anchor="end">%s</text>'
                     % (left - 10, y + bar - 3, esc(shorten(chromosome, 24))))
        parts.append('<rect class="track" x="%d" y="%.1f" width="%.1f" height="%d" rx="4"/>'
                     % (left, y, right - left, bar))
        if value is None:
            parts.append('<text class="tick faint" x="%.1f" y="%.1f">%s</text>'
                         % (left + 6, y + bar - 3,
                            "blocked" if state == "BAD" else "not scored"))
            continue
        length = max(2.0, (right - left) * max(0.0, min(1.0, value)))
        parts.append('<g><title>#%d  %s  %.3f</title>'
                     '<rect class="bar %s" x="%d" y="%.1f" width="%.1f" height="%d" rx="4"/>'
                     '</g>' % (number, esc(chromosome), value, band(value),
                               left, y, length, bar))
        parts.append('<text class="value" x="%.1f" y="%.1f">%.3f</text>'
                     % (left + length + 7, y + bar - 3, value))

    return ('<svg class="chart" viewBox="0 0 %d %d" role="img" '
            'aria-label="quality per individual">%s</svg>'
            % (width, height, "".join(parts)))


def evolution_chart(frames):
    """The population, generation by generation, as one animatable drawing.

    Every bar the sweep will ever need is drawn here once, at the last
    generation, and the page's script only ever changes widths, labels and
    classes -- so the animation is this chart being re-labelled rather than a
    second chart drawn in JavaScript, and the still picture you get with no
    script at all is the finished population.

    A row exists for every individual any generation held, so the growth
    selection causes shows as bars arriving rather than as the chart resizing
    under the reader.
    """
    if not frames:
        return ""
    numbers = sorted({row["number"] for frame in frames for row in frame["rows"]})
    if not numbers:
        return ""
    last = {row["number"]: row for row in frames[-1]["rows"]}

    width = 760
    left, right, top = 210, width - 56, 20
    step, bar = 24, 14
    height = top + len(numbers) * step + 26
    parts = []
    for index in range(6):
        value = index / 5.0
        x = left + value * (right - left)
        parts.append('<line class="grid" x1="%.1f" y1="%d" x2="%.1f" y2="%d"/>'
                     % (x, top - 6, x, top + len(numbers) * step))
        parts.append('<text class="tick" x="%.1f" y="%d" text-anchor="middle">%.1f</text>'
                     % (x, height - 8, value))

    for index, number in enumerate(numbers):
        y = top + index * step
        row = last.get(number)
        value = row["fitness"] if row else None
        length = 0.0 if value is None else (right - left) * max(0.0, min(1.0, value))
        parts.append('<text class="rowlabel" x="10" y="%.1f">#%d</text>'
                     % (y + bar - 3, number))
        parts.append('<text class="rowchrom" data-role="chromosome" data-number="%d" '
                     'x="%d" y="%.1f" text-anchor="end">%s</text>'
                     % (number, left - 10, y + bar - 3,
                        esc(shorten(row["chromosome"], 24)) if row else ""))
        parts.append('<rect class="track" x="%d" y="%.1f" width="%.1f" height="%d" '
                     'rx="4"/>' % (left, y, right - left, bar))
        parts.append('<rect class="bar %s" data-role="bar" data-number="%d" x="%d" '
                     'y="%.1f" width="%.1f" height="%d" rx="4"/>'
                     % (band(value), number, left, y, length, bar))
        parts.append('<text class="value" data-role="value" data-number="%d" '
                     'x="%.1f" y="%.1f">%s</text>'
                     % (number, left + length + 7, y + bar - 3, num(value)))

    geometry = json.dumps({"left": left, "right": right, "top": top,
                           "step": step, "bar": bar, "numbers": numbers})
    return ('<svg class="chart" id="evolution-chart" data-geometry=\'%s\' '
            'viewBox="0 0 %d %d" role="img" '
            'aria-label="the population, generation by generation">%s</svg>'
            % (esc(geometry), width, height, "".join(parts)))


def answers_chart(rows):
    """One vertical bar per question of a transcript, in the order asked."""
    if not rows:
        return ""
    width, height = 760, 240
    left, right, top, bottom = 52, width - 16, 20, height - 34
    slot = (right - left) / float(len(rows))
    bar = min(38.0, slot * 0.72)
    parts = [_grid(left, right, top, bottom)]
    for index, row in enumerate(rows):
        value = row["quality"]
        x = left + slot * index + (slot - bar) / 2.0
        tip = "Q%d  %s\n%s" % (row["position"], num(value),
                              shorten(row["question"], 160))
        if value is None:
            parts.append('<g><title>%s</title><rect class="bar s-none" x="%.1f" '
                         'y="%.1f" width="%.1f" height="6" rx="3"/></g>'
                         % (esc(tip), x, bottom - 6, bar))
        else:
            tall = max(3.0, (bottom - top) * max(0.0, min(1.0, value)))
            parts.append('<g><title>%s</title><rect class="bar %s" x="%.1f" y="%.1f" '
                         'width="%.1f" height="%.1f" rx="3"/></g>'
                         % (esc(tip), band(value), x, bottom - tall, bar, tall))
        if len(rows) <= 30:
            parts.append('<text class="tick faint" x="%.1f" y="%d" '
                         'text-anchor="middle">%d</text>'
                         % (x + bar / 2.0, bottom + 18, row["position"]))
    parts.append('<text class="tick faint" x="%.1f" y="%d" text-anchor="middle">'
                 'question, in the order asked</text>'
                 % ((left + right) / 2.0, height - 6))
    return ('<svg class="chart" viewBox="0 0 %d %d" role="img" '
            'aria-label="score per question">%s</svg>' % (width, height, "".join(parts)))


def histogram_chart(values):
    """Ten buckets of every score the judge gave, across the whole sweep."""
    if not values:
        return ""
    buckets = [0] * 10
    for value in values:
        buckets[min(9, max(0, int(value * 10 - (1e-9 if value >= 1.0 else 0))))] += 1
    peak, divisions = _axis(max(buckets) or 1)

    width, height = 760, 220
    left, right, top, bottom = 52, width - 16, 20, height - 40
    slot = (right - left) / 10.0
    bar = slot * 0.78
    parts = []
    for index in range(divisions + 1):
        value = peak * index / float(divisions)
        y = bottom - (bottom - top) * index / float(divisions)
        parts.append('<line class="grid" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"/>'
                     % (left, y, right, y))
        parts.append('<text class="tick" x="%.1f" y="%.1f" text-anchor="end">%d</text>'
                     % (left - 8, y + 4, round(value)))
    for index, count in enumerate(buckets):
        x = left + slot * index + (slot - bar) / 2.0
        tall = max(1.0, (bottom - top) * count / float(peak)) if count else 0.0
        low, high = index / 10.0, (index + 1) / 10.0
        if count:
            parts.append('<g><title>%.1f to %.1f: %d answer(s)</title>'
                         '<rect class="bar %s" x="%.1f" y="%.1f" width="%.1f" '
                         'height="%.1f" rx="3"/></g>'
                         % (low, high, count, band((low + high) / 2.0),
                            x, bottom - tall, bar, tall))
            parts.append('<text class="value" x="%.1f" y="%.1f" text-anchor="middle">'
                         '%d</text>' % (x + bar / 2.0, bottom - tall - 6, count))
        parts.append('<text class="tick faint" x="%.1f" y="%d" text-anchor="middle">'
                     '%.1f</text>' % (x + bar / 2.0, bottom + 18, low))
    parts.append('<text class="tick faint" x="%.1f" y="%d" text-anchor="middle">'
                 'score awarded</text>' % ((left + right) / 2.0, height - 6))
    return ('<svg class="chart" viewBox="0 0 %d %d" role="img" '
            'aria-label="distribution of scores">%s</svg>'
            % (width, height, "".join(parts)))


def testing_chart(tested):
    """Training beside testing, per tested individual, with the gap between.

    The two bars are the whole point of a testing pass: a blend selected for
    the questions rather than for the job shows as a long training bar over a
    short testing one.
    """
    rows = [row for row in tested if row["quality"] is not None]
    if not rows:
        return ""
    width = 760
    left, right, top = 176, width - 60, 30
    step, bar = 40, 13
    height = top + len(rows) * step + 26
    parts = []
    for index in range(6):
        value = index / 5.0
        x = left + value * (right - left)
        parts.append('<line class="grid" x1="%.1f" y1="%d" x2="%.1f" y2="%d"/>'
                     % (x, top - 8, x, top + len(rows) * step - 8))
        parts.append('<text class="tick" x="%.1f" y="%d" text-anchor="middle">%.1f</text>'
                     % (x, height - 8, value))
    for index, cls, label in ((0, "training", "training"), (1, "testing", "testing")):
        cx = right - 200 + index * 100
        parts.append('<rect class="bar %s" x="%.1f" y="%d" width="16" height="10" rx="3"/>'
                     % (cls, cx, top - 24))
        parts.append('<text class="tick" x="%.1f" y="%d">%s</text>'
                     % (cx + 22, top - 15, label))

    for index, row in enumerate(rows):
        y = top + index * step
        parts.append('<text class="rowlabel" x="10" y="%.1f">#%d</text>'
                     % (y + 14, row["number"]))
        parts.append('<text class="rowchrom" x="%d" y="%.1f" text-anchor="end">%s</text>'
                     % (left - 10, y + 14, esc(shorten(row["chromosome"], 16))))
        for offset, cls, value in ((0, "training", row["selected_on"]),
                                   (bar + 3, "testing", row["quality"])):
            value = value or 0.0
            length = max(2.0, (right - left) * max(0.0, min(1.0, value)))
            parts.append('<g><title>#%d %s %.3f</title>'
                         '<rect class="bar %s" x="%d" y="%.1f" width="%.1f" '
                         'height="%d" rx="3"/></g>'
                         % (row["number"], cls, value, cls, left, y + offset, length, bar))
            parts.append('<text class="value" x="%.1f" y="%.1f">%.3f</text>'
                         % (left + length + 7, y + offset + bar - 2, value))
    return ('<svg class="chart" viewBox="0 0 %d %d" role="img" '
            'aria-label="training against testing quality">%s</svg>'
            % (width, height, "".join(parts)))


# --- the blend, drawn ------------------------------------------------------


def tree_svg(chromosome, weights=None, ranks=None):
    """The chromosome as the blend it describes.

    An L* node and the w* below it are drawn as one leaf: the pair is a single
    fact -- this adapter, at this weight -- and splitting them across two rows
    would make the picture a drawing of the encoding rather than of the blend.
    The operators above them are the folds, in the order PEFT would apply them.

    The leaf carries the slot's rank beside its weight, because the rank is
    what decides whether the folds above it can run at all: cat sums the ranks
    it meets, svd takes the larger, and linear refuses two that differ. Reading
    the ranks off the bottom row is reading the constraint the whole tree is
    built under.
    """
    try:
        root, _ = decode(chromosome)
    except ValueError as error:
        return '<p class="warn">cannot draw this chromosome: %s</p>' % esc(error)

    ranks = ranks or {}
    leaf_step, level_step = 158.0, 92.0
    node_w, node_h, leaf_h = 74.0, 38.0, 46.0
    leaf_w = 124.0
    margin_x, margin_y = 24.0, 16.0

    placed, counter, depth_seen = [], [0], [0]

    def visit(node, depth):
        """-> the record for this node. Leaves take the next slot along the
        bottom; a fold sits centred over the children it folds."""
        depth_seen[0] = max(depth_seen[0], depth)
        if node.symbol in UNARY_OPS:
            x = margin_x + counter[0] * leaf_step + leaf_step / 2.0
            counter[0] += 1
            variable = node.children[0].symbol if node.children else None
            entry = {"kind": "leaf", "symbol": node.symbol, "x": x, "depth": depth,
                     "variable": variable, "weight": (weights or {}).get(variable),
                     "children": []}
        else:
            kids = [visit(child, depth + 1) for child in node.children]
            entry = {"kind": "op", "symbol": node.symbol, "depth": depth,
                     "x": (sum(kid["x"] for kid in kids) / float(len(kids))
                           if kids else margin_x),
                     "variable": None, "weight": None, "children": kids}
        placed.append(entry)
        return entry

    visit(root, 0)

    width = margin_x * 2 + max(1, counter[0]) * leaf_step
    height = margin_y * 2 + depth_seen[0] * level_step + leaf_h + 16

    def y_of(entry):
        return margin_y + entry["depth"] * level_step

    edges, nodes = [], []
    for entry in placed:
        y = y_of(entry)
        if entry["kind"] == "op":
            for child in entry["children"]:
                start_y, end_y = y + node_h, y_of(child)
                middle = (start_y + end_y) / 2.0
                edges.append('<path class="edge" d="M%.1f %.1f C %.1f %.1f, %.1f %.1f, '
                             '%.1f %.1f"/>'
                             % (entry["x"], start_y, entry["x"], middle,
                                child["x"], middle, child["x"], end_y))
            nodes.append(
                '<g class="tnode op op-%s"><title>%s</title>'
                '<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="10"/>'
                '<text x="%.1f" y="%.1f" text-anchor="middle">%s</text></g>'
                % (entry["symbol"].lower(), esc(_fold_note(entry["symbol"])),
                   entry["x"] - node_w / 2.0, y, node_w, node_h,
                   entry["x"], y + 25, esc(entry["symbol"])))
        else:
            rank = ranks.get(entry["symbol"])
            label = entry["variable"] or "?"
            if entry["weight"] is not None:
                label = "%s = %.4f" % (label, entry["weight"])
            if rank is not None:
                label = "%s · r%d" % (label, rank)
            spelt = "%s at %s%s" % (
                entry["symbol"], entry["variable"] or "?",
                "" if entry["weight"] is None else " = %.4f" % entry["weight"])
            if rank is not None:
                spelt += ", rank %d" % rank
            nodes.append(
                '<g class="tnode leaf leaf-%s"><title>%s</title>'
                '<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="10"/>'
                '<text class="slot" x="%.1f" y="%.1f" text-anchor="middle">%s</text>'
                '<text class="wt" x="%.1f" y="%.1f" text-anchor="middle">%s</text></g>'
                % (entry["symbol"].lower(), esc(spelt),
                   entry["x"] - leaf_w / 2.0, y, leaf_w, leaf_h,
                   entry["x"], y + 21, esc(entry["symbol"]),
                   entry["x"], y + 37, esc(label)))

    return ('<div class="treewrap"><svg class="tree" viewBox="0 0 %.0f %.0f" '
            'width="%.0f" role="img" aria-label="the blend this chromosome builds">'
            '%s%s</svg></div>'
            % (width, height, width, "".join(edges), "".join(nodes)))


def _fold_note(symbol):
    """What PEFT does at this node, and what it does to the rank."""
    return {"CAT": "cat - concatenates; the rank is the sum of its inputs",
            "SVD": "svd - re-decomposes; the rank is the larger input",
            "LIN": "linear - weighted sum; both inputs must share a rank"}.get(
                symbol, symbol)


def karva_rows(chromosome):
    """The level-order rows, the way plan.txt draws them."""
    try:
        root, used = decode(chromosome)
    except ValueError:
        return ""
    rows = ['<div class="karva-row"><span class="lvl">%d</span>%s</div>'
            % (index, "".join('<code class="sym k-%s">%s</code>'
                              % (_symbol_class(symbol), esc(symbol)) for symbol in row))
            for index, row in enumerate(levels(root))]
    tail = chromosome.split(".")[used:]
    if tail:
        rows.append('<div class="karva-row"><span class="lvl">tail</span>%s</div>'
                    % "".join('<code class="sym k-tail">%s</code>' % esc(symbol)
                              for symbol in tail))
    return '<div class="karva">%s</div>' % "".join(rows)


def _symbol_class(symbol):
    if ARITY.get(symbol) == 2:
        return "op"
    return "leaf" if ARITY.get(symbol) == 1 else "var"


# --- page sections ---------------------------------------------------------


def card(body):
    """A chart in its own panel -- or nothing at all, when there is no chart.

    An empty panel reads as something that failed to load; a section with one
    fewer panel reads as a sweep that has not got there yet.
    """
    return '<div class="card">%s</div>' % body if body else ""


def tile(label, value, note=None, wordy=False):
    """One number in the summary row. `wordy` is for the ones holding a name
    rather than a number -- a model id at 26px would run off the card."""
    return ('<div class="tile"><span class="tile-label">%s</span>'
            '<span class="tile-value%s">%s</span>'
            '<span class="tile-note">%s</span></div>'
            % (esc(label), " wordy" if wordy else "", value,
               esc(note) if note else ""))


def section_overview(data):
    run, conf, totals_ = data["run"], data["settings"], data["totals"]
    people, quality = data["people"], data["quality"]
    ok = sum(1 for row in people if row["state"] == "ok")
    blocked = sum(1 for row in people if row["state"] == "BAD")
    measured = [row["quality"] for row in quality.values()
                if row["quality"] is not None]
    last = data["history"][-1] if data["history"] else None

    tiles = [
        tile("generations", "%d" % len(data["history"]),
             "recorded by the fitness step"),
        tile("population", "%d" % len(people),
             "%d ok, %d blocked by the rank rule" % (ok, blocked)),
        tile("best quality", num(max(measured) if measured else None),
             "over %d individual(s) with a score" % len(measured)),
        tile("mean quality", num(last["mean"] if last else
                                 (sum(measured) / len(measured) if measured else None)),
             "latest generation" if last else "across the population"),
        tile("model runs", "%d" % totals_["executions"],
             "%d ok, %s of compute" % (totals_["ok"], clock(totals_["seconds"]))),
        tile("answers", "%d" % totals_["answers"],
             "%d judged, mean %s" % (totals_["scored"], num(totals_["mean"]))),
        tile("evaluator", esc(conf.get("EVALUATOR", "-")),
             "the rubric the whole search optimised toward", wordy=True),
        tile("template", esc(run["template"]),
             "mocked - plumbing only" if "mocked" in (run["template"] or "")
             else "the real base-model script", wordy=True),
    ]

    facts = [("database", data["db_path"]),
             ("run", "#%d%s" % (run["id"], "  " + run["label"] if run["label"] else "")),
             ("created", run["created_at"]),
             ("status", run["status"]),
             ("commit", run["git_commit"]),
             ("interpreter", run["interpreter"]),
             ("base model", conf.get("BASE_MODEL")),
             ("judge", conf.get("JUDGE_MODEL") or conf.get("JUDGE_BASE_URL"))]

    return ('<section id="overview"><h2>Overview</h2>'
            '<div class="tiles">%s</div>'
            '<dl class="facts">%s</dl></section>'
            % ("".join(tiles),
               "".join("<div><dt>%s</dt><dd>%s</dd></div>" % (esc(key), esc(value))
                       for key, value in facts)))


def picker(data, name):
    """The combobox that says which individual the page is showing.

    One per section that follows the selection, all carrying the same options
    and kept in step by the page's script, so choosing an individual anywhere
    chooses it everywhere. Its own value is the truth about what is on screen;
    the panels below merely show and hide.
    """
    options = []
    for row in data["people"]:
        measured = data["quality"].get(row["number"])
        value = measured["quality"] if measured else None
        options.append(
            '<option value="%d"%s>#%-3d  %s  %s%s</option>'
            % (row["number"], " selected" if data["best"] is not None
               and row["number"] == data["best"]["number"] else "",
               row["number"], num(value), shorten(row["chromosome"], 34),
               "  ★" if row["is_best"] else
               ("  (blocked)" if row["state"] == "BAD" else "")))
    return ('<label class="picker"><span>Individual</span>'
            '<select data-sync="individual" id="%s">%s</select></label>'
            % (esc(name), "".join(options)))


def section_individuals(data):
    """The lead: everything known about one individual, with a way to switch.

    Every individual gets a complete panel and all but one start hidden, rather
    than the page holding a dataset and building a panel on demand. It costs
    bytes and buys the two things a report wants: the panels are the same
    server-rendered HTML whichever one you are looking at, and every one of them
    is there in the file whether or not any script ever runs.
    """
    if not data["people"]:
        return ('<section id="individual"><h2>Individuals</h2>'
                '<p class="warn">This sweep has no population yet - run '
                '<code>population</code>.</p></section>')

    seen = {}
    chosen = data["best"]["number"] if data["best"] else data["people"][0]["number"]
    panels = "".join(individual_panel(data, row, seen, chosen)
                     for row in data["people"])
    lead = ('The best individual is selected; the rest of the population is one '
            'choice away. What follows changes with it, and so does its line on '
            'the search chart below.')
    return ('<section id="individual"><div class="section-head">'
            '<h2>Individual</h2>%s</div><p class="muted lead">%s</p>'
            '<div class="panels" data-default="%d">%s</div></section>'
            % (picker(data, "picker-individual"), lead, chosen, panels))


def individual_panel(data, best, seen, chosen):
    """One individual, whole: the panel the picker shows and hides.

    `seen` maps a script source to the individual that already carried it, so
    the copies selection appends -- which inherit `script_source` verbatim --
    point at that one instead of repeating eighteen kilobytes of identical
    Python per copy. Saying "identical to #4's" is also the truer statement.
    """
    number = best["number"]
    measured = data["quality"].get(number)
    execution = data["executions"].get(number)
    rows = data["answers"].get(number, [])
    weights = json.loads(execution["weights"]) if execution and execution["weights"] else {}
    conf = data["settings"]
    slots = conf.get("LORA_SLOTS") or {}
    if best["is_best"]:
        standing = "elected by the elitism step - this sweep's best"
    elif data["best"] is not None and data["best"]["number"] == number:
        standing = "highest measured quality (elitism has not run)"
    else:
        standing = "one of the population"

    scored = [row["quality"] for row in rows if row["quality"] is not None]
    tested = [row for row in data["tested"] if row["number"] == number]

    head = ('<div class="champ-head">'
            '<div class="champ-score"><span class="big %s">%s</span>'
            '<span class="champ-cap">mean quality</span></div>'
            '<div class="champ-id"><h3>Individual #%d%s</h3>'
            '<code class="chromosome">%s</code>'
            '<p class="muted">%s</p></div></div>'
            % (band(measured["quality"] if measured else None),
               num(measured["quality"] if measured else None), number,
               ' <span class="crown" title="the elite">&#9733;</span>'
               if best["is_best"] else "",
               esc(best["chromosome"]), esc(standing)))

    stats = [("state", best["state"]),
             ("final rank", best["rank"]),
             ("script", best["script_name"]),
             ("weight seed", best["weight_seed"]),
             ("stored fitness", num(best["fitness"])),
             ("verdict", execution["verdict"] if execution else None),
             ("ran for", clock(execution["seconds"]) if execution else None),
             ("answers", "%d asked, %d judged" % (len(rows), len(scored))),
             ("best answer", num(max(scored) if scored else None)),
             ("worst answer", num(min(scored) if scored else None))]

    used = sorted({row["variable"] for row in _leaves(best["chromosome"])} - {None})
    weight_rows = []
    for name in sorted(weights) or []:
        value = weights[name]
        weight_rows.append(
            '<div class="wrow %s"><span class="wname">%s</span>%s'
            '<span class="wval">%.4f</span></div>'
            % ("on" if name in used else "off", esc(name),
               meter(value, tone="neutral"), value))

    slot_rows = []
    for leaf in _leaves(best["chromosome"]):
        drawn = weights.get(leaf["variable"])
        slot_rows.append('<tr><td><code>%s</code></td><td><code>%s</code></td>'
                         '<td class="numeric">%s</td><td class="numeric">%s</td>'
                         '<td class="path">%s</td></tr>'
                         % (esc(leaf["symbol"]), esc(leaf["variable"]),
                            num(drawn, 4), esc(data["ranks"].get(leaf["symbol"])),
                            esc(slots.get(leaf["symbol"], "-"))))

    transcript = "".join(
        '<details class="qa"><summary><span class="pill %s">%s</span>'
        '<span class="qnum">Q%d</span><span class="qtext">%s</span></summary>'
        '<div class="qa-body"><h4>Question</h4><p>%s</p>'
        '<h4>Answer</h4><p>%s</p><h4>Why that score</h4>'
        '<p class="muted">%s</p></div></details>'
        % (band(row["quality"]), num(row["quality"], 2), row["position"],
           esc(shorten(row["question"], 110)), esc(row["question"]),
           esc(row["answer"] or "(nothing came back)"),
           esc(row["reason"] or "not judged"))
        for row in rows)

    # What the numbers above do not say for themselves. A leading section that
    # shows a dash and explains nothing is worse than one that says why.
    notices = []
    if best["state"] == "BAD":
        notices.append(
            "<strong>Blocked by the rank rule.</strong> A <code>LIN</code> node in "
            "this tree folds two adapters of different rank, which PEFT's "
            "<code>linear</code> combination cannot do, so <code>process</code> "
            "skips it unless asked for <code>--include-blocked</code>.")
    if execution is None:
        notices.append(
            "<strong>Never run.</strong> Nothing has been executed for this "
            "individual, so the tree below is what it would build rather than "
            "what it did.")
    elif best["has_changed"]:
        notices.append(
            "<strong>Mutated since it was scored.</strong> The chromosome above is "
            "the one this individual holds now; the transcript and the script "
            "below belong to the one it held when it last ran. A finished sweep "
            "ends in mutation, so this is the normal state of one - re-run "
            "<code>trees runs process evaluate</code> to score what it is now.")

    notices_html = "".join('<div class="callout warn-note">%s</div>' % text
                           for text in notices)

    testing_note = ""
    if tested:
        entry = tested[0]
        delta = ((entry["quality"] or 0.0) - (entry["selected_on"] or 0.0)
                 if entry["quality"] is not None else None)
        testing_note += (
            '<div class="callout"><strong>Held out:</strong> on '
            '<code>%s</code> this blend scored <strong>%s</strong> against the '
            '<strong>%s</strong> it was selected on - a change of '
            '<span class="%s">%s</span>.</div>'
            % (esc(os.path.basename(entry["dataset"])), num(entry["quality"]),
               num(entry["selected_on"]),
               "delta up" if (delta or 0) >= 0 else "delta down",
               "-" if delta is None else "%+.3f" % delta))

    source = best["script_source"] or ""
    extras = []
    if source and source in seen:
        extras.append('<p class="muted">Its script is byte-for-byte the script of '
                      '<a href="#individual" data-show="%d">individual #%d</a> - '
                      'selection copies <code>script_source</code> verbatim, so a '
                      'copy carries its parent\'s script until <code>runs</code> '
                      're-renders it.</p>' % (seen[source], seen[source]))
    elif source:
        seen[source] = number
        extras.append('<details class="code"><summary>The script that earned this '
                      'score (%d lines)</summary><pre><code>%s</code></pre></details>'
                      % (source.count("\n") + 1, esc(source)))
    if execution and execution["stderr"]:
        extras.append('<details class="code"><summary>stderr</summary>'
                      '<pre><code>%s</code></pre></details>'
                      % esc(_tail(execution["stderr"])))

    return ('<div class="panel champion" data-number="%d"%s>'
            '%s%s'
            '<div class="grid-2">'
            '<div class="card"><h3>The blend</h3>%s'
            '<p class="muted">Read bottom-up: each leaf attaches one adapter at one '
            'weight and its own rank (<code>r</code>), and each fold above it is a '
            'PEFT combination - <code>CAT</code> sums the ranks it meets, '
            '<code>SVD</code> takes the larger, and <code>LIN</code> requires two '
            'that match.</p>%s</div>'
            '<div class="card"><h3>At a glance</h3><dl class="facts tight">%s</dl>'
            '<h3>The weight draw</h3><div class="weights">%s</div>'
            '<p class="muted">All five are drawn from the individual\'s own seed; '
            'the dimmed ones are not referenced by this tree.</p></div>'
            '</div>'
            '%s'
            '<div class="card"><h3>Adapters this blend attaches</h3>%s</div>'
            '<div class="card"><h3>Score per question</h3>%s</div>'
            '<div class="card"><h3>The transcript</h3>%s</div>'
            '%s</div>'
            % (number,
               "" if number == chosen else " hidden",
               head, notices_html,
               tree_svg(best["chromosome"], weights, data["ranks"]),
               karva_rows(best["chromosome"]),
               "".join("<div><dt>%s</dt><dd>%s</dd></div>" % (esc(key), esc(value))
                       for key, value in stats),
               "".join(weight_rows) or '<p class="muted">no draw recorded</p>',
               testing_note,
               table('<th>slot</th><th>weight</th><th class="numeric">drawn</th>'
                     '<th class="numeric">rank</th><th>path</th>',
                     "".join(slot_rows), sortable=False),
               answers_chart(rows) or '<p class="muted">nothing answered yet</p>',
               transcript or '<p class="muted">no transcript stored</p>',
               "".join(extras)))


def _leaves(chromosome):
    """The L* nodes of a chromosome, left to right, with the w* under each."""
    try:
        root, _ = decode(chromosome)
    except ValueError:
        return []
    found = []

    def visit(node):
        if node.symbol in UNARY_OPS:
            found.append({"symbol": node.symbol,
                          "variable": node.children[0].symbol if node.children else None})
            return
        for child in node.children:
            visit(child)

    visit(root)
    return found


def _tail(text, limit=6000):
    text = text or ""
    return text if len(text) <= limit else "... (trimmed)\n" + text[-limit:]


def section_history(data):
    history, champions = data["history"], data["champions"]
    if not history:
        return ('<section id="history"><h2>The search</h2>'
                '<p class="muted">No generation has been recorded yet - the '
                '<code>fitness</code> step writes that history.</p></section>')
    rows = "".join(
        '<tr><td class="numeric">%d</td><td>%s</td><td class="numeric">%d</td>'
        '<td class="numeric">%s</td><td class="numeric">%s</td>'
        '<td class="numeric">%s</td><td class="numeric">%d</td>'
        '<td><code>%s</code></td></tr>'
        % (entry["generation"], esc(entry["recorded_at"]), entry["population"],
           num(entry["best"]), num(entry["mean"]), num(entry["worst"]),
           entry["scored"] or 0,
           esc(champions[entry["generation"]]["chromosome"])
           if champions.get(entry["generation"]) else "-")
        for entry in history)
    controls = ('<div class="controls">'
                '<button type="button" class="action" id="play-search">'
                '&#9654; Replay the search</button>'
                '%s<span class="caption" id="history-caption">%d generation(s) '
                'recorded</span></div>'
                % (picker(data, "picker-search"), len(history)))
    return ('<section id="history"><h2>The search</h2>%s%s%s'
            '<p class="muted">Each row is what the population looked like when the '
            'fitness step last ran over it. Selection appends, so a growing '
            'population is the generations passing. The fourth line is whichever '
            'individual is selected, generation by generation - a line that stops '
            'is an individual mutation cleared the fitness of, and one that starts '
            'late is a copy selection appended.</p></section>'
            % (controls, card(history_chart(history, champions)),
               table('<th>gen</th><th>recorded</th><th class="numeric">pop</th>'
                     '<th class="numeric">best</th><th class="numeric">mean</th>'
                     '<th class="numeric">worst</th><th class="numeric">scored</th>'
                     '<th>fittest chromosome</th>', rows)))


def section_population(data):
    people, quality, executions = data["people"], data["quality"], data["executions"]
    if not people:
        return ""
    rows = []
    for row in people:
        measured = quality.get(row["number"])
        execution = executions.get(row["number"])
        value = measured["quality"] if measured else None
        flags = []
        if row["is_best"]:
            flags.append('<span class="flag best">best</span>')
        if row["state"] == "BAD":
            flags.append('<span class="flag bad">blocked</span>')
        if row["has_changed"]:
            flags.append('<span class="flag changed">mutated</span>')
        rows.append(
            '<tr><td class="numeric">%d</td><td>%s</td><td><code>%s</code></td>'
            '<td class="numeric">%s</td><td>%s</td>'
            '<td class="bar-cell"><div class="bar-row">%s'
            '<span class="numeric">%s</span></div></td>'
            '<td class="numeric">%s</td><td class="numeric">%s</td>'
            '<td class="numeric">%s</td><td class="numeric">%s</td></tr>'
            % (row["number"], "".join(flags) or "&nbsp;", esc(row["chromosome"]),
               esc(row["rank"]), esc(execution["verdict"] if execution else None),
               meter(value), num(value),
               num(row["fitness"]),
               esc((measured["answers"] if measured else 0) or 0),
               clock(execution["seconds"]) if execution else "-",
               esc(row["weight_seed"])))
    note = ('<p class="muted"><strong>quality</strong> is the mean over the latest '
            'execution\'s answers; <strong>fitness</strong> is the column the search '
            'reads. They part company on purpose - mutation clears the fitness of an '
            'individual whose chromosome it rewrote, because that score was earned by '
            'a blend it no longer describes.</p>')
    frames = data["frames"]
    evolution = ""
    if frames:
        evolution = (
            '<div class="card"><h3>Generation by generation</h3>'
            '<div class="controls">'
            '<button type="button" class="action" id="play-evolution">'
            '&#9654; Play the evolution</button>'
            '<label class="scrub"><span>generation</span>'
            '<input type="range" id="evolution-scrub" min="1" max="%d" value="%d" '
            'step="1"></label>'
            '<span class="caption" id="evolution-caption"></span></div>%s'
            '<p class="muted">Replayed from <code>fitness_history</code>, which '
            'keeps each individual\'s chromosome and score <em>as they were</em> '
            'in that generation. A bar arriving is selection appending a copy; a '
            'bar collapsing to nothing is mutation clearing a fitness that was '
            'earned by a chromosome the individual no longer holds.</p></div>'
            % (len(frames), len(frames), evolution_chart(frames)))

    return ('<section id="population"><h2>Population</h2>%s%s%s%s</section>'
            % (card(population_chart(people, quality)), evolution,
               table('<th class="numeric">#</th><th>flags</th><th>chromosome</th>'
                     '<th class="numeric">rank</th><th>verdict</th><th>quality</th>'
                     '<th class="numeric">fitness</th><th class="numeric">answers</th>'
                     '<th class="numeric">ran for</th>'
                     '<th class="numeric">weight seed</th>', "".join(rows)),
               note))


def section_judging(data, values):
    if not values:
        return ""
    return ('<section id="judging"><h2>How the judge scored</h2>'
            '%s'
            '<p class="muted">Every score this sweep\'s <code>%s</code> evaluator '
            'gave, across %d answer(s). A wall at 0.0 is worth a look: the evaluate '
            'step writes zeros unasked once an individual opens badly enough, which '
            'is how a hopeless blend stops costing tokens.</p></section>'
            % (card(histogram_chart(values)), esc(data["settings"].get("EVALUATOR", "-")),
               len(values)))


def section_testing(data):
    tested, summary = data["tested"], data["test_summary"]
    if not tested:
        return ""
    rows = []
    for row in tested:
        delta = ((row["quality"] or 0.0) - (row["selected_on"] or 0.0)
                 if row["quality"] is not None else None)
        rows.append(
            '<tr><td class="numeric">%d</td><td><code>%s</code></td>'
            '<td>%s</td><td class="numeric">%s</td><td class="numeric">%s</td>'
            '<td class="numeric %s">%s</td><td class="numeric">%s</td>'
            '<td>%s</td></tr>'
            % (row["number"], esc(row["chromosome"]), esc(row["verdict"]),
               num(row["selected_on"]), num(row["quality"]),
               "delta up" if (delta or 0) >= 0 else "delta down",
               "-" if delta is None else "%+.3f" % delta,
               esc(row["answers"]), esc(row["evaluator"])))
    passes = "".join(
        '<tr><td>%s</td><td class="numeric">%d</td><td class="numeric">%d</td>'
        '<td class="numeric">%d</td><td class="numeric">%s</td><td>%s</td></tr>'
        % (esc(entry["last"]), entry["tested"], entry["ok"], entry["answers"],
           num(entry["quality"]), esc(entry["dataset"]))
        for entry in summary)
    return ('<section id="testing"><h2>Held-out testing</h2>'
            '%s%s<h3>Passes</h3>%s'
            '<p class="muted">The training column is the score that got the '
            'individual picked; the testing column is what it earned on questions '
            'the search never saw. The gap between them is the only thing that says '
            'whether a blend was selected for the questions or for the job.</p>'
            '</section>'
            % (card(testing_chart(tested)),
               table('<th class="numeric">#</th><th>chromosome</th><th>verdict</th>'
                     '<th class="numeric">training</th><th class="numeric">testing</th>'
                     '<th class="numeric">delta</th><th class="numeric">answers</th>'
                     '<th>evaluator</th>', "".join(rows)),
               table('<th>last</th><th class="numeric">tested</th>'
                     '<th class="numeric">ok</th><th class="numeric">answers</th>'
                     '<th class="numeric">quality</th><th>dataset</th>',
                     passes, sortable=False)))


def section_dataset(data):
    splits, samples = data["splits"], data["samples"]
    if not splits:
        return ""
    rows = "".join(
        '<tr><td><span class="flag %s">%s</span></td><td class="numeric">%d</td>'
        '<td class="numeric">%d</td><td class="path">%s</td></tr>'
        % (entry["split"], entry["split"], entry["records"], entry["references"],
           esc(entry["source"]))
        for entry in splits)
    previews = []
    for entry in splits:
        items = samples.get(entry["split"]) or []
        if not items:
            continue
        previews.append(
            '<details class="qa"><summary><span class="flag %s">%s</span>'
            '<span class="qtext">first %d of %d record(s)</span></summary>'
            '<div class="qa-body">%s</div></details>'
            % (entry["split"], entry["split"], len(items), entry["records"],
               "".join('<h4>%d. Question</h4><p>%s</p>'
                       '<h4>Reference</h4><p class="muted">%s</p>'
                       % (row["position"], esc(row["question"]),
                          esc(row["reference"] or "(none in the file)"))
                       for row in items)))
    return ('<section id="dataset"><h2>Dataset</h2>%s'
            '<p class="muted">Stored whole and uncapped at sweep creation, so the '
            'questions this sweep was built beside are here even after the files '
            'move on. <code>TRAINING_COUNT</code> is how many an individual was '
            'actually judged on.</p>%s</section>'
            % (table('<th>split</th><th class="numeric">records</th>'
                     '<th class="numeric">with a reference</th><th>source</th>',
                     rows, sortable=False),
               "".join(previews)))


def section_settings(data):
    conf = data["settings"]
    rows = []
    for key in sorted(conf):
        value = conf[key]
        text = value if isinstance(value, str) else json.dumps(value)
        if len(text) > 120:
            body = ('<details class="inline"><summary>%s…</summary>'
                    '<pre><code>%s</code></pre></details>'
                    % (esc(text[:100]), esc(text)))
        else:
            body = "<code>%s</code>" % esc(text)
        rows.append('<tr><td class="key">%s</td><td>%s</td></tr>' % (esc(key), body))
    return ('<section id="settings"><h2>Settings</h2>'
            '<p class="muted">What this sweep was created with, read back from its '
            'own settings table - not from <code>settings.py</code> as it stands '
            'now. That is what makes a resumed sweep still be the same sweep.</p>'
            '%s</section>'
            % table("<th>setting</th><th>value</th>", "".join(rows)))


# --- the page --------------------------------------------------------------


CSS = """
:root {
  color-scheme: light dark;
  --bg: #f5f6fa; --bg-2: #ffffff; --ink: #12141c; --ink-2: #5a6076;
  --line: #e2e5ee; --line-2: #cfd4e2;
  --accent: #4b5ee4; --accent-2: #7b5ce4; --accent-soft: #eceefc;
  --high: #1f9d63; --mid: #d99a1e; --low: #e0722f; --zero: #d24b4b;
  --none: #b3b8c8; --track: #eceef5;
  --cat: #4b5ee4; --svd: #7b5ce4; --lin: #2596a8;
  --l1: #2f7fd6; --l2: #1f9d63; --l3: #d99a1e; --l4: #d2568c; --l5: #7b5ce4;
  --shadow: 0 1px 2px rgba(16,20,40,.06), 0 8px 24px rgba(16,20,40,.06);
  --radius: 14px;
  --mono: ui-monospace, "Cascadia Code", "SF Mono", Consolas, monospace;
}
html[data-theme="dark"] {
  --bg: #0e1017; --bg-2: #161a24; --ink: #e7e9f2; --ink-2: #9aa1b8;
  --line: #242a38; --line-2: #313849;
  --accent: #8b9bff; --accent-2: #b18cff; --accent-soft: #1c2135;
  --high: #45c98a; --mid: #e5b455; --low: #ef8f57; --zero: #ef6b6b;
  --none: #4d5468; --track: #1e2331;
  --cat: #8b9bff; --svd: #b18cff; --lin: #4fc4d6;
  --l1: #6fb0ff; --l2: #45c98a; --l3: #e5b455; --l4: #ef85b6; --l5: #b18cff;
  --shadow: 0 1px 2px rgba(0,0,0,.4), 0 10px 30px rgba(0,0,0,.35);
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  -webkit-font-smoothing: antialiased;
}
a { color: var(--accent); }
code, pre { font-family: var(--mono); }

header.top {
  padding: 40px 32px 28px; border-bottom: 1px solid var(--line);
  background:
    radial-gradient(1100px 340px at 12% -20%, var(--accent-soft), transparent 70%),
    var(--bg-2);
}
.top-inner { max-width: 1160px; margin: 0 auto; display: flex;
  align-items: flex-start; gap: 24px; flex-wrap: wrap; }
.top h1 { margin: 0 0 6px; font-size: 30px; letter-spacing: -.02em; }
.top .sub { margin: 0; color: var(--ink-2); font-size: 14px; }
.top .sub code { color: var(--ink); }
.spacer { flex: 1 1 120px; }
.badge {
  display: inline-flex; align-items: center; gap: 7px; padding: 5px 12px;
  border-radius: 999px; font-size: 12.5px; font-weight: 600;
  background: var(--accent-soft); color: var(--accent); border: 1px solid var(--line-2);
}
.badge.done { background: color-mix(in srgb, var(--high) 14%, transparent);
  color: var(--high); }
.badge.failed { background: color-mix(in srgb, var(--zero) 14%, transparent);
  color: var(--zero); }
button.theme {
  border: 1px solid var(--line-2); background: var(--bg-2); color: var(--ink-2);
  border-radius: 10px; padding: 8px 12px; cursor: pointer; font-size: 13px;
}
button.theme:hover { color: var(--ink); border-color: var(--accent); }

nav.jump {
  position: sticky; top: 0; z-index: 5; background: color-mix(in srgb, var(--bg) 86%, transparent);
  backdrop-filter: blur(10px); border-bottom: 1px solid var(--line);
}
nav.jump ul { max-width: 1160px; margin: 0 auto; padding: 0 32px; list-style: none;
  display: flex; gap: 4px; overflow-x: auto; }
nav.jump a { display: block; padding: 12px 12px; font-size: 13.5px; font-weight: 550;
  color: var(--ink-2); text-decoration: none; border-bottom: 2px solid transparent; }
nav.jump a:hover { color: var(--ink); border-bottom-color: var(--accent); }

main { max-width: 1160px; margin: 0 auto; padding: 8px 32px 80px; }
section { padding: 34px 0 8px; scroll-margin-top: 56px; }
section h2 { font-size: 21px; letter-spacing: -.01em; margin: 0 0 18px;
  padding-bottom: 10px; border-bottom: 1px solid var(--line); }
section h3 { font-size: 14px; text-transform: uppercase; letter-spacing: .07em;
  color: var(--ink-2); margin: 0 0 12px; }
h4 { font-size: 12px; text-transform: uppercase; letter-spacing: .07em;
  color: var(--ink-2); margin: 14px 0 4px; }
.muted { color: var(--ink-2); font-size: 13.5px; }
.warn { color: var(--zero); }

.card { background: var(--bg-2); border: 1px solid var(--line);
  border-radius: var(--radius); padding: 20px; box-shadow: var(--shadow);
  margin-bottom: 18px; overflow: hidden; }
.grid-2 { display: grid; grid-template-columns: 1.55fr 1fr; gap: 18px; }
@media (max-width: 900px) { .grid-2 { grid-template-columns: 1fr; } }

.tiles { display: grid; gap: 14px; margin-bottom: 20px;
  grid-template-columns: repeat(auto-fit, minmax(178px, 1fr)); }
.tile { background: var(--bg-2); border: 1px solid var(--line);
  border-radius: var(--radius); padding: 15px 16px; box-shadow: var(--shadow); }
.tile-label { display: block; font-size: 11.5px; text-transform: uppercase;
  letter-spacing: .08em; color: var(--ink-2); }
.tile-value { display: block; font-size: 26px; font-weight: 640;
  letter-spacing: -.02em; margin: 4px 0 2px; overflow-wrap: anywhere; }
.tile-value.wordy { font-size: 17px; font-family: var(--mono); font-weight: 600;
  letter-spacing: 0; line-height: 1.35; }
.tile-note { display: block; font-size: 12px; color: var(--ink-2); }

dl.facts { display: grid; gap: 1px; margin: 0; background: var(--line);
  border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
dl.facts > div { display: grid; grid-template-columns: 150px 1fr; gap: 12px;
  background: var(--bg-2); padding: 9px 14px; }
dl.facts dt { color: var(--ink-2); font-size: 13px; }
dl.facts dd { margin: 0; font-size: 13.5px; word-break: break-word;
  font-family: var(--mono); }
dl.facts.tight > div { grid-template-columns: 120px 1fr; padding: 7px 12px; }

.is-off { display: none !important; }
.panel[hidden] { display: none; }
.section-head { display: flex; align-items: flex-end; gap: 18px; flex-wrap: wrap;
  margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid var(--line); }
.section-head h2 { border: 0; margin: 0; padding: 0; flex: 1 1 auto; }
p.lead { margin: 0 0 16px; max-width: 70ch; }

.picker { display: inline-flex; align-items: center; gap: 9px; font-size: 12px;
  text-transform: uppercase; letter-spacing: .07em; color: var(--ink-2); }
.picker select {
  font: 13px/1.3 var(--mono); color: var(--ink); background: var(--bg-2);
  border: 1px solid var(--line-2); border-radius: 9px; padding: 7px 10px;
  max-width: 380px; cursor: pointer; text-transform: none; letter-spacing: 0; }
.picker select:hover { border-color: var(--accent); }
.picker select:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }

.controls { display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
  margin-bottom: 14px; }
button.action {
  display: inline-flex; align-items: center; gap: 7px; cursor: pointer;
  font: 600 13px/1 system-ui, sans-serif; color: #fff; background: var(--accent);
  border: 1px solid var(--accent); border-radius: 9px; padding: 9px 14px; }
button.action:hover { filter: brightness(1.08); }
button.action:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.scrub { display: inline-flex; align-items: center; gap: 9px; font-size: 12px;
  text-transform: uppercase; letter-spacing: .07em; color: var(--ink-2); }
.scrub input[type="range"] { width: 150px; accent-color: var(--accent); cursor: pointer; }
.caption { font: 12.5px/1.4 var(--mono); color: var(--ink-2); }

.champion { }
.champ-head { display: flex; align-items: center; gap: 26px; flex-wrap: wrap;
  background: var(--bg-2); border: 1px solid var(--line); border-radius: var(--radius);
  padding: 22px 26px; box-shadow: var(--shadow); margin-bottom: 18px;
  background-image: linear-gradient(105deg, var(--accent-soft), transparent 55%); }
.champ-score { text-align: center; min-width: 130px; }
.big { display: block; font-size: 52px; font-weight: 700; letter-spacing: -.03em;
  line-height: 1.05; }
.champ-cap { font-size: 11.5px; text-transform: uppercase; letter-spacing: .09em;
  color: var(--ink-2); }
.champ-id h3 { margin: 0 0 8px; font-size: 20px; text-transform: none;
  letter-spacing: -.01em; color: var(--ink); }
.chromosome { display: inline-block; font-size: 14px; padding: 6px 11px;
  border-radius: 8px; background: var(--track); border: 1px solid var(--line-2);
  word-break: break-all; }
.crown { color: var(--mid); }
.callout { background: var(--bg-2); border: 1px solid var(--line);
  border-left: 3px solid var(--accent); border-radius: 10px; padding: 13px 16px;
  margin-bottom: 12px; font-size: 14px; }
.callout.warn-note { border-left-color: var(--mid); }
.delta.up { color: var(--high); font-weight: 600; }
.delta.down { color: var(--zero); font-weight: 600; }

.treewrap { overflow-x: auto; padding: 4px 0 10px; }
svg.tree { display: block; max-width: 100%; height: auto; }
svg.tree .edge { fill: none; stroke: var(--line-2); stroke-width: 1.6; }
svg.tree .tnode rect { stroke-width: 1.4; }
svg.tree text { font-family: var(--mono); font-size: 13px; fill: #fff; }
svg.tree .leaf text.slot { font-size: 13px; font-weight: 600; }
svg.tree .leaf text.wt { font-size: 10.5px; opacity: .85; }
svg.tree .op-cat rect { fill: var(--cat); stroke: var(--cat); }
svg.tree .op-svd rect { fill: var(--svd); stroke: var(--svd); }
svg.tree .op-lin rect { fill: var(--lin); stroke: var(--lin); }
svg.tree .leaf-l1 rect { fill: var(--l1); stroke: var(--l1); }
svg.tree .leaf-l2 rect { fill: var(--l2); stroke: var(--l2); }
svg.tree .leaf-l3 rect { fill: var(--l3); stroke: var(--l3); }
svg.tree .leaf-l4 rect { fill: var(--l4); stroke: var(--l4); }
svg.tree .leaf-l5 rect { fill: var(--l5); stroke: var(--l5); }

.karva { margin-top: 10px; display: grid; gap: 5px; }
.karva-row { display: flex; align-items: center; gap: 5px; flex-wrap: wrap; }
.karva .lvl { width: 34px; font-size: 11px; color: var(--ink-2); text-align: right;
  margin-right: 6px; }
.sym { font-size: 12px; padding: 2px 7px; border-radius: 6px; border: 1px solid var(--line-2);
  background: var(--track); }
.sym.k-op { color: var(--cat); border-color: color-mix(in srgb, var(--cat) 45%, transparent); }
.sym.k-leaf { color: var(--l2); border-color: color-mix(in srgb, var(--l2) 45%, transparent); }
.sym.k-var { color: var(--ink-2); }
.sym.k-tail { color: var(--zero); }

.weights { display: grid; gap: 7px; margin-bottom: 8px; }
.wrow { display: grid; grid-template-columns: 32px 1fr 60px; align-items: center;
  gap: 10px; font-size: 13px; }
.wrow.off { opacity: .38; }
.wname { font-family: var(--mono); color: var(--ink-2); }
.wval { font-family: var(--mono); text-align: right; }

.meter { display: inline-block; width: 100%; min-width: 70px; height: 9px;
  background: var(--track); border-radius: 999px; overflow: hidden;
  vertical-align: middle; }
.meter i { display: block; height: 100%; border-radius: 999px; background: var(--accent); }
.meter.s-high i { background: var(--high); } .meter.s-mid i { background: var(--mid); }
.meter.s-low i { background: var(--low); }  .meter.s-zero i { background: var(--zero); }
.meter.neutral i { background: var(--accent); }
.meter.empty { background: var(--track); }

svg.chart { display: block; width: 100%; height: auto; }
svg.chart .grid { stroke: var(--line); stroke-width: 1; }
svg.chart .tick { fill: var(--ink-2); font-size: 11px;
  font-family: system-ui, sans-serif; }
svg.chart .tick.faint { opacity: .7; font-size: 10.5px; }
svg.chart .rowlabel { fill: var(--ink); font-size: 12px; font-weight: 600;
  font-family: var(--mono); }
svg.chart .rowchrom { fill: var(--ink-2); font-size: 10.5px; font-family: var(--mono); }
svg.chart .value { fill: var(--ink-2); font-size: 11px; font-family: var(--mono); }
svg.chart .track { fill: var(--track); }
svg.chart .bar { fill: var(--accent); }
svg.chart .bar.s-high { fill: var(--high); } svg.chart .bar.s-mid { fill: var(--mid); }
svg.chart .bar.s-low { fill: var(--low); }  svg.chart .bar.s-zero { fill: var(--zero); }
svg.chart .bar.s-none { fill: var(--none); }
svg.chart .bar.training { fill: var(--accent); }
svg.chart .bar.testing { fill: var(--accent-2); }
svg.chart .spread { fill: var(--accent); opacity: .12; }
svg.chart .line { fill: none; stroke-width: 2.4; stroke-linejoin: round;
  stroke-linecap: round; }
svg.chart .line.best { stroke: var(--high); }
svg.chart .line.mean { stroke: var(--accent); }
svg.chart .line.worst { stroke: var(--none); stroke-dasharray: 5 4; stroke-width: 1.8; }
svg.chart .line.picked { stroke: var(--accent-2); stroke-width: 3;
  stroke-dasharray: 1 6; stroke-linecap: round; }
svg.chart .dot { stroke: var(--bg-2); stroke-width: 2; }
svg.chart .dot.best { fill: var(--high); } svg.chart .dot.mean { fill: var(--accent); }
svg.chart .dot.picked { fill: var(--accent-2); }
svg.chart .hit { fill: transparent; }
svg.chart .playhead { stroke: var(--accent-2); stroke-width: 1.5;
  stroke-dasharray: 3 3; }

table.data { width: 100%; border-collapse: collapse; background: var(--bg-2);
  border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden;
  font-size: 13.5px; box-shadow: var(--shadow); margin-bottom: 14px;
  display: table; }
table.data th { text-align: left; font-size: 11.5px; text-transform: uppercase;
  letter-spacing: .06em; color: var(--ink-2); font-weight: 600;
  padding: 11px 13px; border-bottom: 1px solid var(--line); white-space: nowrap; }
table.data td { padding: 9px 13px; border-bottom: 1px solid var(--line);
  vertical-align: middle; }
table.data tbody tr:last-child td { border-bottom: 0; }
table.data tbody tr:hover { background: var(--accent-soft); }
table.data .numeric { text-align: right; font-family: var(--mono); white-space: nowrap; }
table.data .key { font-family: var(--mono); color: var(--ink-2); white-space: nowrap; }
table.data .path { font-family: var(--mono); font-size: 12px; color: var(--ink-2);
  word-break: break-all; }
table.data code { font-size: 12.5px; }
table.data .bar-cell { min-width: 170px; }
.bar-row { display: flex; align-items: center; gap: 9px; }
.bar-row .numeric { font-size: 12.5px; min-width: 42px; }
table.sortable th { cursor: pointer; user-select: none; }
table.sortable th:hover { color: var(--accent); }
.tablewrap { overflow-x: auto; margin-bottom: 14px; border-radius: var(--radius); }
.tablewrap table.data { margin-bottom: 0; }

.flag { display: inline-block; font-size: 10.5px; font-weight: 700; padding: 2px 7px;
  border-radius: 999px; text-transform: uppercase; letter-spacing: .05em;
  margin-right: 4px; border: 1px solid transparent; }
.flag.best { background: color-mix(in srgb, var(--mid) 18%, transparent); color: var(--mid); }
.flag.bad { background: color-mix(in srgb, var(--zero) 15%, transparent); color: var(--zero); }
.flag.changed { background: var(--accent-soft); color: var(--accent); }
.flag.training { background: var(--accent-soft); color: var(--accent); }
.flag.validation { background: color-mix(in srgb, var(--mid) 16%, transparent); color: var(--mid); }
.flag.testing { background: color-mix(in srgb, var(--accent-2) 16%, transparent);
  color: var(--accent-2); }
.pill { display: inline-block; min-width: 42px; text-align: center; font-size: 12px;
  font-weight: 700; font-family: var(--mono); padding: 3px 8px; border-radius: 7px;
  background: var(--track); }
.pill.s-high { background: color-mix(in srgb, var(--high) 16%, transparent); color: var(--high); }
.pill.s-mid { background: color-mix(in srgb, var(--mid) 16%, transparent); color: var(--mid); }
.pill.s-low { background: color-mix(in srgb, var(--low) 16%, transparent); color: var(--low); }
.pill.s-zero { background: color-mix(in srgb, var(--zero) 16%, transparent); color: var(--zero); }
.pill.s-none { color: var(--ink-2); }

details.qa { border: 1px solid var(--line); border-radius: 10px; margin-bottom: 7px;
  background: var(--bg-2); overflow: hidden; }
details.qa summary { display: flex; align-items: center; gap: 11px; padding: 9px 13px;
  cursor: pointer; font-size: 13.5px; list-style: none; }
details.qa summary::-webkit-details-marker { display: none; }
details.qa summary:hover { background: var(--accent-soft); }
details.qa[open] summary { border-bottom: 1px solid var(--line); }
.qnum { font-family: var(--mono); color: var(--ink-2); font-size: 12px; min-width: 30px; }
.qtext { color: var(--ink-2); overflow: hidden; text-overflow: ellipsis;
  white-space: nowrap; }
.qa-body { padding: 4px 16px 16px; }
.qa-body p { margin: 0; white-space: pre-wrap; font-size: 13.5px; }
details.code { border: 1px solid var(--line); border-radius: 10px; background: var(--bg-2);
  margin-bottom: 12px; }
details.code summary { padding: 11px 15px; cursor: pointer; font-size: 13.5px;
  color: var(--ink-2); }
details.code summary:hover { color: var(--ink); }
details.code pre { margin: 0; padding: 0 15px 15px; overflow-x: auto; font-size: 12px;
  line-height: 1.5; }
details.inline summary { cursor: pointer; font-family: var(--mono); font-size: 12.5px;
  color: var(--ink-2); }
details.inline pre { margin: 6px 0 0; padding: 10px; background: var(--track);
  border-radius: 8px; overflow-x: auto; font-size: 12px; white-space: pre-wrap; }

footer { max-width: 1160px; margin: 0 auto; padding: 26px 32px 50px;
  color: var(--ink-2); font-size: 12.5px; border-top: 1px solid var(--line); }
@media print {
  nav.jump, button.theme { display: none; }
  details { open: true; }
  .card, table.data, .tile { box-shadow: none; }
}
"""

SCRIPT = """
(function () {
  var root = document.documentElement;
  var stored = null;
  try { stored = localStorage.getItem('gep-stats-theme'); } catch (e) {}
  if (stored) { root.setAttribute('data-theme', stored); }
  else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
    root.setAttribute('data-theme', 'dark');
  } else { root.setAttribute('data-theme', 'light'); }

  var toggle = document.getElementById('theme');
  function label() {
    toggle.textContent = root.getAttribute('data-theme') === 'dark' ? 'Light' : 'Dark';
  }
  if (toggle) {
    label();
    toggle.addEventListener('click', function () {
      var next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('gep-stats-theme', next); } catch (e) {}
      label();
    });
  }

  // Click a heading to sort. Numbers sort as numbers, everything else as text;
  // a second click reverses. A cell with no value ('-', blank) always sinks to
  // the bottom, either way round -- reversing the order of the rows that have
  // an answer should not float the ones that have none to the top.
  function value(cell) {
    var text = cell.textContent.trim().replace(/[+,]/g, '');
    if (text === '' || text === '-') { return { missing: true }; }
    var asNumber = parseFloat(text);
    return isNaN(asNumber) ? { text: text.toLowerCase() } : { number: asNumber };
  }
  function compare(left, right, down) {
    if (left.missing || right.missing) {
      return left.missing && right.missing ? 0 : (left.missing ? 1 : -1);
    }
    var a = 'number' in left ? left.number : left.text;
    var b = 'number' in right ? right.number : right.text;
    if (a < b) { return down ? -1 : 1; }
    if (a > b) { return down ? 1 : -1; }
    return 0;
  }
  // --- the sweep, as data ---------------------------------------------------
  var block = document.getElementById('gep-data');
  var DATA = block ? JSON.parse(block.textContent) : { frames: [], selected: null };
  var FRAMES = DATA.frames || [];
  var SVGNS = 'http://www.w3.org/2000/svg';

  function at(frame, number) {
    for (var i = 0; i < frame.rows.length; i++) {
      if (frame.rows[i].number === number) { return frame.rows[i]; }
    }
    return null;
  }
  function clamp01(value) { return Math.max(0, Math.min(1, value || 0)); }
  function band(value) {
    if (value === null || value === undefined) { return 's-none'; }
    if (value >= 0.8) { return 's-high'; }
    if (value >= 0.5) { return 's-mid'; }
    return value > 0 ? 's-low' : 's-zero';
  }
  function ease(t) { return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t; }

  // --- which individual the page is showing ---------------------------------
  var pickers = document.querySelectorAll('select[data-sync="individual"]');
  var panels = document.querySelectorAll('.panel[data-number]');
  var historyCaption = document.getElementById('history-caption');
  var resting = historyCaption ? historyCaption.textContent : '';
  var searchRunning = null;

  function drawPicked(number) {
    var chart = document.getElementById('history-chart');
    if (!chart) { return; }
    var plot = JSON.parse(chart.dataset.plot);
    var line = document.getElementById('history-picked');
    var dots = document.getElementById('history-picked-dots');
    var key = document.getElementById('history-picked-key');
    var points = [];
    while (dots.firstChild) { dots.removeChild(dots.firstChild); }
    FRAMES.forEach(function (frame, index) {
      var row = at(frame, number);
      if (!row || row.fitness === null) { return; }
      var x = plot.xs[index];
      var y = plot.bottom - clamp01(row.fitness) * (plot.bottom - plot.top);
      points.push(x.toFixed(1) + ',' + y.toFixed(1));
      var dot = document.createElementNS(SVGNS, 'circle');
      dot.setAttribute('class', 'dot picked');
      dot.setAttribute('cx', x.toFixed(1));
      dot.setAttribute('cy', y.toFixed(1));
      dot.setAttribute('r', '4');
      var tip = document.createElementNS(SVGNS, 'title');
      tip.textContent = '#' + number + ' - gen ' + frame.generation + ': '
        + row.fitness.toFixed(3) + '\\n' + row.chromosome;
      dot.appendChild(tip);
      dots.appendChild(dot);
    });
    line.setAttribute('points', points.join(' '));
    key.setAttribute('class', points.length ? 'legend-key' : 'legend-key is-off');
    key.querySelector('text').textContent = '#' + number;
    // An individual selection appended after the last fitness snapshot has no
    // line at all, which on its own looks like a bug rather than a fact.
    if (historyCaption && !searchRunning) {
      historyCaption.textContent = points.length ? resting
        : ('#' + number + ' is not in any recorded generation yet');
    }
  }

  function show(number) {
    Array.prototype.forEach.call(panels, function (panel) {
      panel.hidden = panel.dataset.number !== String(number);
    });
    Array.prototype.forEach.call(pickers, function (picker) {
      if (picker.value !== String(number)) { picker.value = String(number); }
    });
    drawPicked(number);
  }

  Array.prototype.forEach.call(pickers, function (picker) {
    picker.addEventListener('change', function () { show(Number(picker.value)); });
  });
  // "identical to the script of #4" jumps to that individual rather than
  // scrolling to a panel that is not on screen.
  Array.prototype.forEach.call(document.querySelectorAll('[data-show]'),
    function (link) {
      link.addEventListener('click', function () { show(Number(link.dataset.show)); });
    });
  if (DATA.selected !== null && DATA.selected !== undefined) { show(DATA.selected); }

  // --- replaying the search -------------------------------------------------
  // The chart is already drawn; the replay only widens the clip over it and
  // drags a playhead, so what you watch appear is the finished drawing.
  var searchButton = document.getElementById('play-search');
  if (searchButton && FRAMES.length) {
    var reveal = document.getElementById('history-reveal-rect');
    var head = document.getElementById('history-playhead');

    function settle() {
      var plot = JSON.parse(document.getElementById('history-chart').dataset.plot);
      reveal.setAttribute('x', (plot.left - 6).toFixed(1));
      reveal.setAttribute('width', (plot.right - plot.left + 12).toFixed(1));
      head.setAttribute('class', 'playhead is-off');
      searchRunning = null;
      historyCaption.textContent = resting;
      searchButton.innerHTML = '&#9654; Replay the search';
      drawPicked(Number(pickers[0] ? pickers[0].value : DATA.selected));
    }

    searchButton.addEventListener('click', function () {
      if (searchRunning) { cancelAnimationFrame(searchRunning); settle(); return; }
      var plot = JSON.parse(document.getElementById('history-chart').dataset.plot);
      var span = plot.right - plot.left;
      var total = Math.min(9000, Math.max(1600, FRAMES.length * 900));
      var started = null;
      searchButton.innerHTML = '&#9632; Stop';
      head.setAttribute('class', 'playhead');
      searchRunning = requestAnimationFrame(function step(now) {
        if (started === null) { started = now; }
        var t = Math.min(1, (now - started) / total);
        var x = plot.left + span * t;
        reveal.setAttribute('x', (plot.left - 6).toFixed(1));
        reveal.setAttribute('width', (x - plot.left + 6).toFixed(1));
        head.setAttribute('x1', x.toFixed(1));
        head.setAttribute('x2', x.toFixed(1));
        var reached = 0;
        for (var i = 0; i < plot.xs.length; i++) {
          if (plot.xs[i] <= x + 0.5) { reached = i; }
        }
        var frame = FRAMES[reached];
        historyCaption.textContent = 'generation ' + frame.generation + '/' + FRAMES.length
          + '  \\u00b7  best ' + (frame.best || 0).toFixed(3)
          + '  \\u00b7  mean ' + (frame.mean || 0).toFixed(3)
          + '  \\u00b7  ' + (frame.population) + ' individuals'
          + (frame.chromosome ? '  \\u00b7  ' + frame.chromosome : '');
        if (t < 1) { searchRunning = requestAnimationFrame(step); }
        else { settle(); }
      });
    });
  }

  // --- replaying the population --------------------------------------------
  // Same idea one level down: every bar is already in the page, and a frame is
  // a set of widths and labels to move them to.
  var evolutionButton = document.getElementById('play-evolution');
  var evolutionChart = document.getElementById('evolution-chart');
  if (evolutionChart && FRAMES.length) {
    var geometry = JSON.parse(evolutionChart.dataset.geometry);
    var scrub = document.getElementById('evolution-scrub');
    var evoCaption = document.getElementById('evolution-caption');
    var bars = {}, values = {}, chromosomes = {};
    geometry.numbers.forEach(function (number) {
      bars[number] = evolutionChart.querySelector('[data-role="bar"][data-number="'
        + number + '"]');
      values[number] = evolutionChart.querySelector('[data-role="value"][data-number="'
        + number + '"]');
      chromosomes[number] = evolutionChart.querySelector(
        '[data-role="chromosome"][data-number="' + number + '"]');
    });
    var width = geometry.right - geometry.left;
    var playing = null;

    function paint(index, progress) {
      var frame = FRAMES[index];
      var previous = index > 0 ? FRAMES[index - 1] : null;
      geometry.numbers.forEach(function (number) {
        var now = at(frame, number);
        var was = previous ? at(previous, number) : null;
        var to = now && now.fitness !== null ? now.fitness : 0;
        var from = was && was.fitness !== null ? was.fitness : 0;
        var value = from + (to - from) * progress;
        var length = width * clamp01(value);
        bars[number].setAttribute('width', length.toFixed(1));
        bars[number].setAttribute('class', 'bar ' + band(now ? now.fitness : null));
        values[number].setAttribute('x', (geometry.left + length + 7).toFixed(1));
        values[number].textContent = now
          ? (now.fitness === null ? '-' : value.toFixed(3)) : '';
        chromosomes[number].textContent = now
          ? (now.chromosome.length > 24 ? now.chromosome.slice(0, 23) + '\\u2026'
                                        : now.chromosome)
          : '';
        chromosomes[number].setAttribute('class', now ? 'rowchrom' : 'rowchrom is-off');
      });
      evoCaption.textContent = 'generation ' + frame.generation + '/' + FRAMES.length
        + '  \\u00b7  ' + frame.population + ' individuals'
        + '  \\u00b7  best ' + (frame.best || 0).toFixed(3)
        + '  \\u00b7  mean ' + (frame.mean || 0).toFixed(3)
        + '  \\u00b7  ' + frame.recorded_at;
      if (scrub && Number(scrub.value) !== index + 1) { scrub.value = String(index + 1); }
    }

    function stopEvolution() {
      if (playing) { cancelAnimationFrame(playing); playing = null; }
      evolutionButton.innerHTML = '&#9654; Play the evolution';
    }

    if (scrub) {
      scrub.addEventListener('input', function () {
        stopEvolution();
        paint(Math.min(FRAMES.length - 1, Math.max(0, Number(scrub.value) - 1)), 1);
      });
    }
    if (evolutionButton) {
      evolutionButton.addEventListener('click', function () {
        if (playing) { stopEvolution(); return; }
        var tween = 750, hold = 450, each = tween + hold;
        var started = null;
        evolutionButton.innerHTML = '&#9632; Stop';
        playing = requestAnimationFrame(function step(now) {
          if (started === null) { started = now; }
          var elapsed = now - started;
          var index = Math.min(FRAMES.length - 1, Math.floor(elapsed / each));
          var local = Math.min(1, (elapsed - index * each) / tween);
          paint(index, ease(local));
          if (elapsed < FRAMES.length * each) { playing = requestAnimationFrame(step); }
          else { paint(FRAMES.length - 1, 1); stopEvolution(); }
        });
      });
    }
    paint(FRAMES.length - 1, 1);
  }

  Array.prototype.forEach.call(document.querySelectorAll('table.sortable'), function (table) {
    var heads = table.querySelectorAll('thead th');
    Array.prototype.forEach.call(heads, function (head, index) {
      head.addEventListener('click', function () {
        var body = table.tBodies[0];
        var rows = Array.prototype.slice.call(body.rows);
        var down = head.dataset.down !== 'yes';
        rows.sort(function (a, b) {
          return compare(value(a.cells[index]), value(b.cells[index]), down);
        });
        Array.prototype.forEach.call(heads, function (other) { other.dataset.down = ''; });
        head.dataset.down = down ? 'yes' : 'no';
        rows.forEach(function (row) { body.appendChild(row); });
      });
    });
  });
})();
"""


NAV = (("overview", "Overview"), ("individual", "Individual"), ("history", "The search"),
       ("population", "Population"), ("judging", "Judging"), ("testing", "Testing"),
       ("dataset", "Dataset"), ("settings", "Settings"))


def payload(data):
    """What the page's script needs, as JSON in a data block rather than code.

    Only the two things a drawing cannot carry: the generations, to replay, and
    which individual to start on. Everything else on the page is already HTML.
    Escaping `<` keeps a chromosome or a dataset path from ever closing the
    script element it sits in.
    """
    body = json.dumps({
        "frames": data["frames"],
        "selected": (data["best"]["number"] if data["best"]
                     else (data["people"][0]["number"] if data["people"] else None)),
    }, separators=(",", ":"))
    return ('<script type="application/json" id="gep-data">%s</script>'
            % body.replace("<", "\\u003c"))


def render(data, values):
    """The whole page as one string."""
    run = data["run"]
    sections = [section_overview(data), section_individuals(data), section_history(data),
                section_population(data), section_judging(data, values),
                section_testing(data), section_dataset(data), section_settings(data)]
    present = {name for name, _ in NAV
               if any(('id="%s"' % name) in part for part in sections)}
    nav = "".join('<li><a href="#%s">%s</a></li>' % (name, esc(label))
                  for name, label in NAV if name in present)
    title = "GEP sweep #%d - %s" % (run["id"], run["created_at"])

    return ("""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>%s</title>
<style>%s</style>
</head>
<body>
<header class="top"><div class="top-inner">
  <div>
    <h1>GEP sweep #%d%s</h1>
    <p class="sub">%s &middot; template <code>%s</code> &middot; commit <code>%s</code><br>
      <code>%s</code></p>
  </div>
  <div class="spacer"></div>
  <span class="badge %s">%s</span>
  <button class="theme" id="theme" type="button">Dark</button>
</div></header>
<nav class="jump"><ul>%s</ul></nav>
<main>%s</main>
<footer>Generated by <code>generate_html_db_stats.py</code> from
  <code>%s</code>. A view of the sweep, derived from the database &mdash; never the
  sweep itself.</footer>
%s
<script>%s</script>
</body>
</html>
""" % (esc(title), CSS, run["id"],
       " &middot; " + esc(run["label"]) if run["label"] else "",
       esc(run["created_at"]), esc(run["template"]), esc(run["git_commit"] or "?"),
       esc(data["db_path"]), esc(run["status"]), esc(run["status"]), nav,
       "".join(sections), esc(data["db_path"]), payload(data), SCRIPT))


def default_output(db_path, run_id):
    """`gep.sqlite3` + run 2 -> `gep_run2_stats.html`, in the same folder."""
    folder = os.path.dirname(os.path.abspath(db_path))
    stem = os.path.splitext(os.path.basename(db_path))[0]
    return os.path.join(folder, "%s_run%d_stats.html" % (stem, run_id))


def write_report(conn, run_id, out_path=None):
    """Render one sweep to `out_path` (or beside the database). -> the path."""
    data = read_sweep(conn, run_id)
    page = render(data, all_qualities(conn, run_id))
    path = out_path or default_output(conn.path, run_id)
    folder = os.path.dirname(os.path.abspath(path))
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(page)
    return path


def locate(db_path):
    """The database as typed, or beside the repo -- but it must already exist.

    store.connect() creates what it cannot open, which is right for a pipeline
    that starts sweeps and wrong for a reader: a mistyped path would otherwise
    produce a brand new empty database and a report about nothing.
    """
    candidates = [db_path]
    if not os.path.isabs(db_path):
        candidates.append(os.path.join(_HERE, db_path))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise SystemExit("no database at %s (looked in %s)"
                     % (db_path, " and ".join(os.path.abspath(item)
                                              for item in candidates)))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Write one stored sweep out as an HTML page, beside its database.")
    parser.add_argument("db", nargs="?", default=None,
                        help="the database to read (default: settings.DB_PATH)")
    parser.add_argument("--run", type=int, default=0,
                        help="which sweep (0, the default, is the most recent)")
    parser.add_argument("--out", default=None,
                        help="write here instead of beside the database")
    parser.add_argument("--open", dest="open_it", action="store_true",
                        help="open the page in a browser when it is written")
    args = parser.parse_args(argv)

    if args.db is None:
        import settings as _settings
        args.db = _settings.DB_PATH
    conn = store.connect(locate(args.db))

    run_id = store.latest_run(conn) if args.run == 0 else args.run
    if run_id is None:
        raise SystemExit("%s holds no runs yet" % conn.path)

    path = write_report(conn, run_id, args.out)
    print("wrote run %d to %s" % (run_id, path))
    if args.open_it:
        webbrowser.open("file://" + path.replace(os.sep, "/"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
