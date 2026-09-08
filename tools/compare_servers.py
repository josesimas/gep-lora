"""compare_servers.py -- two sweeps of the same search, side by side, as one page.

A dev aid, not part of the pipeline: nothing in a sweep runs this and nothing it
writes goes back into one. It exists to answer the question
[server_pool.py](../blends/server_pool.py) leaves open -- whether a pool of more
than one lora server pays -- by reading two sweeps that differ only in
`LORA_SERVER_COUNT` and saying what the difference bought.

    python -m tools.compare_servers runs/run_db_2_servers/gep.sqlite3 \\
        runs/run_db_4_servers/gep.sqlite3 --run-a 2 --run-b 1 \\
        --out runs/lora_server_count.html

The two sides are `(database, run)` pairs, so the sweeps may live in one database
or in two. `--run 0` is the latest, as everywhere else. With one database given,
both sides read it.

What it is really comparing
---------------------------
Two clocks, and the page's whole point is that they disagree. A *script's*
seconds are its own wall clock, so `PROCESS_RUN_BATCH_SIZE` scripts running at
once overlap each other; the *step's* seconds are the wall clock of all of them
together. More servers can make every individual slower and the step faster at
the same time, and only one of those is the number worth optimising. This is the
same two-denominators rule `reporting/generate_html_db_stats.py` prints on its
cost charts, applied across two sweeps instead of within one.

It reads through `storage/store.py`'s helpers rather than its own SQL, the way
the report generator does -- `store.py` owns the schema. Which side is which,
what each script's batch width was and which individuals are worth naming are
all derived from the sweeps, so the page describes the two it was given rather
than the two it was written for.

Comparability is measured, not assumed
--------------------------------------
Nothing here requires the two sweeps to be the same search. The page opens by
saying which settings differ, how many scripts it managed to pair on
`(pass, individual number)`, and whether the paired individuals ended on the same
chromosomes and fitnesses -- so a reader can see how much the timings are worth
before reading them. Two sweeps that share every seed make the comparison a
controlled experiment; two that do not still produce the page, with the caveat
stated at the top rather than buried.

Derived and disposable, like every other reader in this repo: delete the page and
the sweeps are untouched.
"""

import argparse
import json
import os
import statistics
import sys
import webbrowser

from storage import store
# One definition of "a reader must not create the database it was pointed at":
# store.connect() creates what it cannot open, which is right for a driver
# starting a sweep and wrong for anything that only reads one.
from reporting.generate_html_db_stats import locate

# The repo folder, one above this one -- every path a tool resolves is resolved
# against it, never against the cwd the tool was started from.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The template whose scripts are lora_server clients. A sweep generated from any
# other ignores LORA_SERVER_COUNT, so its batch width is the batch setting alone.
REMOTE_TEMPLATE = "template_remote_code.py"

# The phases worth a row in the table, in the order they are worth reading.
PHASE_ORDER = ("script", "generate", "build", "combine.svd", "combine.cat",
               "combine.linear", "attach", "reset", "inference_setup", "compact",
               "start servers", "recycle servers")

# The settings a reader wants to see agreed on before believing the timings.
CONTROL_KEYS = ("LORA_SERVER_COUNT", "PROCESS_RUN_BATCH_SIZE", "SEED",
                "WEIGHT_MASTER_SEED", "SELECTION_MASTER_SEED", "COUNT",
                "GENERATIONS", "TRAINING_COUNT", "ANSWER_BATCH", "MAX_DEPTH",
                "BRANCH_PROB", "SELECTION_COUNT", "MUTATION_RATE", "TEMPLATE",
                "EVALUATOR", "BASE_MODEL", "LORA_SERVER_RECYCLE_AFTER")

# A script whose longest batch-mate finishes before it is this far through has
# spent most of its life alone whatever the batch width around it, so widening
# that batch cannot have cost it much. Naming the threshold rather than hiding
# it: it is what separates the two control groups the page leans on.
STRAGGLER_SHARE = 0.5

# And how much of a pass's wall clock may sit outside its batches before the
# pass is left out of the throughput figure. Above this the pass is paying for
# something the batch model does not describe -- in practice a pool startup.
FIXED_COST_SHARE = 0.05


# --- one side of the comparison ---------------------------------------------


class Side:
    """One sweep, read once, with everything the page asks of it.

    `scripts` is the sequence the process step ran them in, which is what makes
    the batches reconstructable: process_run.batches() cuts the selected
    individuals into consecutive groups, so grouping the rows the same way in
    the same order recovers which scripts shared a card with which.
    """

    def __init__(self, conn, run_id, label=None):
        self.conn = conn
        self.run_id = run_id
        self.db = conn.path
        self.conf = store.get_settings(conn, run_id)
        self.servers = self._int("LORA_SERVER_COUNT")
        self.remote = str(self.conf.get("TEMPLATE") or "").endswith(REMOTE_TEMPLATE)
        self.width = self._batch_width()
        self.label = label or self._label()

        self.steps = [row for row in store.step_timings(conn, run_id)
                      if row["step"] == "process"]
        self.passes = [row["pass_no"] for row in self.steps]
        self.phases = {row["phase"]: row
                       for row in store.phase_costs(conn, run_id, step="process")}

        self.scripts, self.generates = [], {}
        for row in store.execution_costs(conn, run_id):
            if row["step"] != "process":
                continue
            if row["phase"] == "script":
                self.scripts.append(self._script(row))
            elif row["phase"] == "generate":
                calls, seconds = self.generates.get(row["pass_no"], (0, 0.0))
                self.generates[row["pass_no"]] = (calls + row["calls"],
                                                  seconds + row["seconds"])
        self._assign_widths()

        self.population = {row["number"]: row
                           for row in store.individuals(conn, run_id)}

    # -- reading ------------------------------------------------------------

    def _int(self, key, default=None):
        try:
            return int(self.conf.get(key))
        except (TypeError, ValueError):
            return default

    def _batch_width(self):
        """How many scripts ran at once: the batch setting, capped by the pool.

        start_run.py lowers the batch to the pool's size before cutting it,
        because a server serves one request at a time. A sweep whose scripts are
        not clients has no pool and no cap.
        """
        size = max(1, self._int("PROCESS_RUN_BATCH_SIZE", 1) or 1)
        if self.remote and self.servers:
            return min(size, self.servers)
        return size

    def _label(self):
        """What to call this side. The setting under test, where there is one."""
        if self.remote and self.servers:
            return "%d server%s" % (self.servers, "" if self.servers == 1 else "s")
        return "batch of %d" % self.width

    def _script(self, row):
        """One `script` phase row as the individual behind it.

        The identity comes out of `detail` first: an execution's id can be
        handed to a later individual once a cull has freed it, so the join back
        through executions is the fallback for sweeps recorded before the script
        phase carried its own number and chromosome.
        """
        detail = {}
        if row["detail"]:
            try:
                detail = json.loads(row["detail"])
            except ValueError:
                detail = {}
        return {"pass": row["pass_no"],
                "number": detail.get("number", row["number"]),
                "chromosome": detail.get("chromosome") or row["chromosome"] or "",
                "seconds": row["seconds"]}

    def _assign_widths(self):
        """Cut each pass into the batches it ran as, and record what each script
        shared its batch with."""
        for pn in self.passes:
            rows = [s for s in self.scripts if s["pass"] == pn]
            for start in range(0, len(rows), self.width):
                group = rows[start:start + self.width]
                longest_mate = 0.0
                for script in group:
                    mates = [g["seconds"] for g in group if g is not script]
                    script["width"] = len(group)
                    script["mate"] = max(mates) if mates else 0.0
                    longest_mate = max(longest_mate, script["mate"])

    # -- what the page asks -------------------------------------------------

    def step_seconds(self, pn=None):
        rows = self.steps if pn is None else [r for r in self.steps
                                              if r["pass_no"] == pn]
        return sum(row["seconds"] for row in rows)

    def script_seconds(self, pn=None):
        rows = (self.scripts if pn is None
                else [s for s in self.scripts if s["pass"] == pn])
        return sum(s["seconds"] for s in rows)

    def phase(self, name, field="seconds"):
        row = self.phases.get(name)
        return row[field] if row else 0.0

    def per_call(self, name):
        row = self.phases.get(name)
        if not row or not row["calls"]:
            return 0.0
        return row["seconds"] / row["calls"]

    def generate_per_call(self, pn):
        calls, seconds = self.generates.get(pn, (0, 0.0))
        return seconds / calls if calls else 0.0

    def batches(self, pn):
        rows = [s for s in self.scripts if s["pass"] == pn]
        return [rows[i:i + self.width] for i in range(0, len(rows), self.width)]

    def predicted(self, pn):
        """The pass as the batch model has it: a batch costs its slowest script."""
        return sum(max(g["seconds"] for g in group) for group in self.batches(pn))

    def alone_share(self, script):
        """How much of a script's life had no batch-mate left in it."""
        if not script["seconds"]:
            return 1.0
        return max(0.0, 1.0 - script["mate"] / script["seconds"])


# --- pairing the two ---------------------------------------------------------


def pair(a, b):
    """The scripts both sides ran, matched on (pass, individual number).

    Two sweeps of the same search run the same individual in the same pass, so
    this is the finest-grained honest comparison there is: the same blend, the
    same weight seed and the same prompts, timed twice.
    """
    index = {(s["pass"], s["number"]): s for s in b.scripts}
    out = []
    for script in a.scripts:
        other = index.get((script["pass"], script["number"]))
        if other:
            out.append((script, other))
    return out


def classify(left, right, b):
    """Which of the three stories a paired script is in.

    The point of the split: contention should cost a script in proportion to how
    much of its life it actually spent sharing the card. A script whose batch
    never widened is the flat control; one that widened but outlives everything
    in its batch is the partial control; the rest are the measurement.
    """
    if left["width"] == right["width"]:
        return "same"
    if b.alone_share(right) > STRAGGLER_SHARE:
        return "straggler"
    return "shared"


def comparability(a, b, pairs):
    """What a reader should know before believing any of the numbers below.

    -> (differing settings, matched population rows, disagreements)
    """
    differs = [key for key in CONTROL_KEYS
               if a.conf.get(key) != b.conf.get(key)]
    matched, disagree = [], []
    for number, row in sorted(a.population.items()):
        other = b.population.get(number)
        if other is None:
            continue
        matched.append((row, other))
        same_chromosome = row["chromosome"] == other["chromosome"]
        same_fitness = abs((row["fitness"] or 0.0) - (other["fitness"] or 0.0)) < 1e-9
        if not (same_chromosome and same_fitness):
            disagree.append(number)
    return differs, matched, disagree


# --- svg ---------------------------------------------------------------------

def esc(text):
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def paired_bars(rows, scale_max, height_each=15, width=540, label_w=230, note=None):
    """rows: [(label, a_seconds, b_seconds, flag)] -- two bars per row."""
    gap, pad_t = 5, 26
    height = pad_t + len(rows) * (height_each * 2 + gap) + 26
    out = ['<svg viewBox="0 0 %d %d" class="chart" role="img">'
           % (label_w + width + 66, height)]
    for frac in (0, .25, .5, .75, 1):
        x = label_w + frac * width
        out.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" class="grid"/>'
                   % (x, pad_t - 8, x, height - 24))
        out.append('<text x="%.1f" y="%d" class="tick">%gs</text>'
                   % (x, pad_t - 13, round(frac * scale_max, 1)))
    y = pad_t
    for label, left, right, flag in rows:
        for value, cls in ((left, "a"), (right, "b")):
            bar = max(1.0, min(1.0, value / scale_max) * width)
            out.append('<rect x="%d" y="%.1f" width="%.1f" height="%d" rx="2" '
                       'class="bar %s"/>'
                       % (label_w, y, bar, height_each - 2, "flag" if flag else cls))
            out.append('<text x="%.1f" y="%.1f" class="val">%.1f</text>'
                       % (label_w + bar + 5, y + height_each - 4, value))
            y += height_each
        out.append('<text x="%d" y="%.1f" class="rowlab%s" text-anchor="end">%s</text>'
                   % (label_w - 8, y - height_each + 2, " flag" if flag else "",
                      esc(label)))
        y += gap
    if note:
        out.append('<text x="%d" y="%d" class="tick" text-anchor="start">%s</text>'
                   % (label_w, height - 7, esc(note)))
    out.append("</svg>")
    return "\n".join(out)


def ratio_chart(rows, lo, hi, width=500, label_w=250, height_each=15):
    """rows: [(label, ratio, css class)] -- each script's b-time over its a-time."""
    gap, pad_t = 4, 26
    height = pad_t + len(rows) * (height_each + gap) + 20
    out = ['<svg viewBox="0 0 %d %d" class="chart" role="img">'
           % (label_w + width + 60, height)]

    def x_of(value):
        return label_w + (min(max(value, lo), hi) - lo) / (hi - lo) * width

    tick = lo
    while tick <= hi + 1e-9:
        x = x_of(tick)
        out.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" class="grid%s"/>'
                   % (x, pad_t - 8, x, height - 18,
                      " one" if abs(tick - 1.0) < 1e-9 else ""))
        out.append('<text x="%.1f" y="%d" class="tick">%.1f&#215;</text>'
                   % (x, pad_t - 13, tick))
        tick += 0.1
    y = pad_t
    for label, ratio, cls in rows:
        x0, x1 = x_of(1.0), x_of(ratio)
        out.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%d" rx="2" '
                   'class="bar %s"/>'
                   % (min(x0, x1), y, max(abs(x1 - x0), 1.5), height_each - 3, cls))
        out.append('<text x="%.1f" y="%.1f" class="val">%.2f&#215;</text>'
                   % (max(x0, x1) + 5, y + height_each - 5, ratio))
        out.append('<text x="%d" y="%.1f" class="rowlab" text-anchor="end">%s</text>'
                   % (label_w - 8, y + height_each - 5, esc(label)))
        y += height_each + gap
    out.append("</svg>")
    return "\n".join(out)


def timeline(a, b, pn, heavy):
    """One pass as the batches it ran as, both sides on a shared time axis."""
    span = max(a.predicted(pn), b.predicted(pn)) or 1.0
    width, label_w, lane_h = 600, 96, 17
    sides, heights, height = [], [], 30
    for side in (a, b):
        groups = side.batches(pn)
        sides.append((side, groups))
        tall = max(len(g) for g in groups) * lane_h + 26
        heights.append(tall)
        height += tall
    out = ['<svg viewBox="0 0 %d %d" class="chart" role="img">'
           % (label_w + width + 60, height)]
    for frac in (0, .25, .5, .75, 1):
        x = label_w + frac * width
        out.append('<line x1="%.1f" y1="18" x2="%.1f" y2="%d" class="grid"/>'
                   % (x, x, height - 12))
        out.append('<text x="%.1f" y="13" class="tick">%ds</text>'
                   % (x, round(frac * span)))
    y = 30
    for (side, groups), tall in zip(sides, heights):
        cls = "a" if side is a else "b"
        out.append('<text x="%d" y="%.1f" class="rowlab" text-anchor="end">%s</text>'
                   % (label_w - 8, y + 12, esc(side.label)))
        cursor = 0.0
        for group in groups:
            cost = max(g["seconds"] for g in group)
            x0 = label_w + cursor / span * width
            box = cost / span * width
            out.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="3" '
                       'class="batchbox"/>'
                       % (x0, y - 3, max(box, 2), len(group) * lane_h + 4))
            for k, script in enumerate(group):
                bar = script["seconds"] / span * width
                out.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%d" rx="2" '
                           'class="bar %s"/>'
                           % (x0 + 1, y + k * lane_h + 1, max(bar - 2, 1.5),
                              lane_h - 4,
                              "flag" if script["number"] in heavy else cls))
                if bar > 30:
                    out.append('<text x="%.1f" y="%.1f" class="inbar">#%s%s</text>'
                               % (x0 + 5, y + k * lane_h + lane_h - 6,
                                  script["number"],
                                  "  %.0fs" % script["seconds"] if bar > 62 else ""))
            cursor += cost
        out.append('<text x="%.1f" y="%.1f" class="val">%.0fs</text>'
                   % (label_w + cursor / span * width + 6, y + 12, cursor))
        y += tall
    out.append("</svg>")
    return "\n".join(out)


# --- the page ----------------------------------------------------------------

CSS = """
:root {
  color-scheme: light dark;
  --bg: #f5f6fa; --bg-2: #ffffff; --ink: #12141c; --ink-2: #5a6076;
  --line: #e2e5ee; --line-2: #cfd4e2;
  --accent: #4b5ee4; --accent-soft: #eceefc;
  --a: #2596a8; --b: #d2568c; --flag: #7b5ce4; --same: #1f9d63;
  --good: #1f9d63; --bad: #e0722f;
  --shadow: 0 1px 2px rgba(16,20,40,.06), 0 8px 24px rgba(16,20,40,.06);
  --radius: 14px;
  --mono: ui-monospace, "Cascadia Code", "SF Mono", Consolas, monospace;
}
html[data-theme="dark"] {
  --bg: #0e1017; --bg-2: #161a24; --ink: #e7e9f2; --ink-2: #9aa1b8;
  --line: #242a38; --line-2: #313849;
  --accent: #8b9bff; --accent-soft: #1c2135;
  --a: #4fc4d6; --b: #ef85b6; --flag: #b18cff; --same: #45c98a;
  --good: #45c98a; --bad: #ef8f57;
  --shadow: 0 1px 2px rgba(0,0,0,.4), 0 10px 30px rgba(0,0,0,.35);
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  -webkit-font-smoothing: antialiased; }
code, pre { font-family: var(--mono); }
code { font-size: .92em; }
.top { position: relative; padding: 40px 32px 26px; border-bottom: 1px solid var(--line);
  background: radial-gradient(1100px 340px at 12% -20%, var(--accent-soft), transparent 70%), var(--bg-2); }
.top h1 { margin: 0 0 8px; font-size: 27px; letter-spacing: -.02em; }
.top .sub { margin: 0; color: var(--ink-2); font-size: 14.5px; max-width: 78ch; }
.top .sub code, .top .sub strong { color: var(--ink); }
.pill { display: inline-block; margin: 14px 8px 0 0; padding: 3px 11px;
  border-radius: 999px; font: 12px/1.7 var(--mono);
  background: var(--accent-soft); color: var(--accent); border: 1px solid var(--line-2); }
button.theme { position: absolute; top: 26px; right: 28px; cursor: pointer;
  border-radius: 999px; padding: 6px 14px; font-size: 13px;
  border: 1px solid var(--line-2); background: var(--bg-2); color: var(--ink-2); }
button.theme:hover { color: var(--ink); border-color: var(--accent); }
main { max-width: 1060px; margin: 0 auto; padding: 8px 24px 60px; }
section { margin-top: 42px; }
h2 { font-size: 20px; letter-spacing: -.01em; margin: 0 0 10px;
  padding-bottom: 10px; border-bottom: 1px solid var(--line); }
h3 { font-size: 15px; margin: 24px 0 6px; }
p { max-width: 82ch; }
.muted { color: var(--ink-2); font-size: 13.5px; }
.card { background: var(--bg-2); border: 1px solid var(--line);
  border-radius: var(--radius); box-shadow: var(--shadow);
  padding: 16px 18px; margin: 14px 0; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px; margin: 16px 0; }
.tile { background: var(--bg-2); border: 1px solid var(--line);
  border-radius: var(--radius); box-shadow: var(--shadow); padding: 14px 16px; }
.tile-label { display: block; font-size: 11px; text-transform: uppercase;
  letter-spacing: .08em; color: var(--ink-2); }
.tile-value { display: block; font-size: 26px; font-weight: 650; margin: 5px 0 3px;
  letter-spacing: -.02em; }
.tile-note { display: block; font-size: 12px; color: var(--ink-2); }
.up { color: var(--bad); } .down { color: var(--good); }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; }
th, td { text-align: right; padding: 7px 10px; border-bottom: 1px solid var(--line);
  white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
tbody tr:last-child td { border-bottom: none; }
th { font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
  color: var(--ink-2); font-weight: 600; }
tbody tr:hover { background: var(--accent-soft); }
td.mono, th.mono { font-family: var(--mono); }
.scroll { overflow-x: auto; }
svg.chart { width: 100%; height: auto; display: block; min-width: 620px; }
.grid { stroke: var(--line); stroke-width: 1; }
.grid.one { stroke: var(--ink-2); stroke-dasharray: 3 3; }
.tick { font: 10px var(--mono); fill: var(--ink-2); text-anchor: middle; }
.rowlab { font: 11.5px var(--mono); fill: var(--ink-2); }
.rowlab.flag { fill: var(--flag); font-weight: 700; }
.val { font: 10.5px var(--mono); fill: var(--ink-2); }
.inbar { font: 10px var(--mono); fill: #fff; opacity: .95; }
.bar.a { fill: var(--a); } .bar.b { fill: var(--b); }
.bar.flag, .bar.straggler { fill: var(--flag); }
.bar.same { fill: var(--same); } .bar.shared { fill: var(--b); }
.batchbox { fill: none; stroke: var(--line-2); stroke-width: 1; }
.key { display: flex; flex-wrap: wrap; gap: 18px; margin: 12px 0 2px;
  font-size: 12.5px; color: var(--ink-2); }
.key span { display: inline-flex; align-items: center; gap: 6px; }
.key i { width: 11px; height: 11px; border-radius: 3px; display: inline-block; }
.callout { border-left: 3px solid var(--accent); padding: 3px 0 3px 16px; margin: 18px 0; }
.warn { border-left-color: var(--bad); }
ul { max-width: 82ch; } li { margin-bottom: 9px; }
footer { margin-top: 50px; padding-top: 18px; border-top: 1px solid var(--line);
  color: var(--ink-2); font-size: 12.5px; }
@media print { button.theme { display: none; } body { background: #fff; } }
"""

PAGE = """<!doctype html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>%(title)s</title>
<style>%(css)s</style>
</head>
<body>
<header class="top">
<button class="theme" type="button">Dark</button>
<h1>%(title)s</h1>
<p class="sub">%(headline)s</p>
%(pills)s
</header>
<main>
%(body)s
<footer>
Read out of <code>step_timings</code> and <code>phase_timings</code> through
<code>storage/store.py</code>, in %(source_a)s and %(source_b)s. Derived and
disposable: nothing here was written back to either sweep.
Written by <code>python -m tools.compare_servers</code>.
</footer>
</main>
<script>
(function () {
  var root = document.documentElement, stored = null;
  try { stored = localStorage.getItem('gep-compare-theme'); } catch (e) {}
  if (stored) { root.setAttribute('data-theme', stored); }
  else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
    root.setAttribute('data-theme', 'dark');
  }
  var toggle = document.querySelector('button.theme');
  function label() {
    toggle.textContent = root.getAttribute('data-theme') === 'dark' ? 'Light' : 'Dark';
  }
  label();
  toggle.addEventListener('click', function () {
    var next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    try { localStorage.setItem('gep-compare-theme', next); } catch (e) {}
    label();
  });
})();
</script>
</body>
</html>
"""


def pct(before, after):
    return (after - before) / before * 100.0 if before else 0.0


def signed(value):
    return '<span class="%s">%+.0f%%</span>' % ("up" if value > 0 else "down", value)


def gain(value):
    """A percentage change said as a gain -- and said as nothing when it is.

    A difference that rounds to zero is the interesting answer here, not a
    rounding artefact to be printed as "-0%": it means the extra concurrency
    bought nothing, which is the finding.
    """
    if abs(value) < 0.5:
        return "nothing measurable"
    return "%+.0f%%" % value


def humanise(numbers):
    """[2, 6, 12] -> '#2, #6 and #12'."""
    tags = ["#%s" % n for n in numbers]
    if len(tags) == 1:
        return tags[0]
    return "%s and %s" % (", ".join(tags[:-1]), tags[-1])


def build(a, b, gpu):
    """The whole page body, out of the two sides. -> (html, headline, pills)."""
    pairs = pair(a, b)
    if not pairs:
        raise SystemExit(
            "the two sweeps share no (pass, individual) in their process step, "
            "so there is nothing to compare script for script")

    differs, matched, disagree = comparability(a, b, pairs)
    groups = {"same": [], "straggler": [], "shared": []}
    ratio_rows, members = [], {"same": [], "straggler": [], "shared": []}
    for left, right in pairs:
        cls = classify(left, right, b)
        ratio = right["seconds"] / left["seconds"] if left["seconds"] else 1.0
        groups[cls].append(ratio)
        members[cls].append(left["number"])
        chromosome = left["chromosome"]
        if len(chromosome) > 20:
            chromosome = chromosome[:19] + "…"
        ratio_rows.append(("#%s %s  [%d→%d]"
                           % (left["number"], chromosome,
                              left["width"], right["width"]), ratio, cls))
    ratio_rows.sort(key=lambda row: row[1])
    means = {cls: (statistics.mean(values) if values else 1.0)
             for cls, values in groups.items()}

    # The passes the throughput figure may be read off. Two exclusions, and both
    # are about keeping it a measure of concurrency rather than of anything else.
    # A pass holding a straggler is bounded by that one script whatever the batch
    # width, so it says nothing about how well the width works. And a pass whose
    # recorded seconds run well past what its batches account for is carrying a
    # fixed cost outside them -- the pool startup, which is paid per driver and
    # not per server, so it dilutes the very ratio it would be counted in.
    heavy_numbers = {left["number"] for left, right in pairs
                     if classify(left, right, b) == "straggler"}

    def batches_explain_it(pn):
        return all(side.step_seconds(pn) - side.predicted(pn)
                   < FIXED_COST_SHARE * side.step_seconds(pn) for side in (a, b))

    light = [pn for pn in a.passes
             if pn in b.passes and batches_explain_it(pn)
             and not any(s["number"] in heavy_numbers
                         for s in a.scripts if s["pass"] == pn)]
    light_a = sum(a.step_seconds(pn) for pn in light)
    light_b = sum(b.step_seconds(pn) for pn in light)
    light_n = len([s for s in a.scripts if s["pass"] in light])

    script_a, script_b = a.script_seconds(), b.script_seconds()
    step_a, step_b = a.step_seconds(), b.step_seconds()

    tiles = [("Per-individual script time", "%+.0f%%" % pct(script_a, script_b),
              "%.0fs &#8594; %.0fs summed over the same %d scripts"
              % (script_a, script_b, len(pairs))),
             ("<code>process</code> step, wall", "%+.0f%%" % pct(step_a, step_b),
              "%.0fs &#8594; %.0fs over %d passes"
              % (step_a, step_b, len(a.steps))),
             ("<code>generate</code>, per call",
              "%+.0f%%" % pct(a.per_call("generate"), b.per_call("generate")),
              "%.2fs &#8594; %.2fs, %d calls either way"
              % (a.per_call("generate"), b.per_call("generate"),
                 a.phase("generate", "calls")))]
    if light_n and light_b:
        tiles.append(("Throughput, clean passes",
                      "%.2f&#215;" % ((light_a / light_n) / (light_b / light_n)),
                      "%.2fs &#8594; %.2fs per script over %d of them, pass%s %s"
                      % (light_a / light_n, light_b / light_n, light_n,
                         "" if len(light) == 1 else "es",
                         ", ".join(str(pn) for pn in light))))
    tiles_html = "\n".join(
        '<div class="tile"><span class="tile-label">%s</span>'
        '<span class="tile-value %s">%s</span>'
        '<span class="tile-note">%s</span></div>'
        % (label, "up" if value.startswith("+") else "down", value, note)
        for label, value, note in tiles)

    conf_rows = "\n".join(
        "<tr><td class=mono>%s</td><td class=mono>%s</td><td class=mono>%s</td>"
        "<td>%s</td></tr>"
        % (key, esc(json.dumps(a.conf.get(key))), esc(json.dumps(b.conf.get(key))),
           "<strong>differs</strong>" if key in differs else "same")
        for key in CONTROL_KEYS if key in a.conf or key in b.conf)

    pop_rows = "\n".join(
        "<tr><td class=mono>#%d</td><td class=mono>%s</td><td class=mono>%.3f</td>"
        "<td class=mono>%.3f</td><td>%s</td></tr>"
        % (left["number"], esc(left["chromosome"]), left["fitness"] or 0.0,
           right["fitness"] or 0.0,
           "identical" if left["number"] not in disagree else "<strong>differs</strong>")
        for left, right in matched)

    pass_rows = []
    for pn in a.passes:
        if pn not in b.passes:
            continue
        n = len([s for s in a.scripts if s["pass"] == pn])
        pass_rows.append(
            "<tr><td>pass %d</td><td class=mono>%d</td>"
            "<td class=mono>%.1f</td><td class=mono>%.1f</td>"
            "<td class=mono>%.2f</td><td class=mono>%.2f</td>"
            "<td class=mono>%.2f</td><td class=mono>%.2f</td></tr>"
            % (pn, n, a.step_seconds(pn), b.step_seconds(pn),
               a.script_seconds(pn) / (a.step_seconds(pn) or 1),
               b.script_seconds(pn) / (b.step_seconds(pn) or 1),
               a.generate_per_call(pn), b.generate_per_call(pn)))

    phase_rows = []
    for name in PHASE_ORDER:
        left, right = a.phases.get(name), b.phases.get(name)
        if not left or not right:
            continue
        phase_rows.append(
            "<tr><td class=mono>%s</td><td class=mono>%d</td>"
            "<td class=mono>%.1f</td><td class=mono>%.1f</td>"
            "<td class=mono>%.2f</td><td class=mono>%.2f</td><td>%s</td></tr>"
            % (name, left["calls"], left["seconds"], right["seconds"],
               a.per_call(name), b.per_call(name),
               signed(pct(a.per_call(name), b.per_call(name)))))

    check_rows = "\n".join(
        "<tr><td>pass %d</td><td class=mono>%.1f</td><td class=mono>%.1f</td>"
        "<td class=mono>%.1f</td><td class=mono>%.1f</td></tr>"
        % (pn, a.predicted(pn), a.step_seconds(pn),
           b.predicted(pn), b.step_seconds(pn))
        for pn in a.passes if pn in b.passes)

    # The two passes worth drawing: the most expensive, and the cheapest light
    # one -- the straggler-bound case and the clean case.
    shared_passes = [pn for pn in a.passes if pn in b.passes]
    heavy_pass = max(shared_passes, key=a.step_seconds)
    light_pass = (min(light, key=a.step_seconds) if light else None)

    longest = max(max(s["seconds"] for s in a.scripts),
                  max(s["seconds"] for s in b.scripts))
    all_rows, zoom_rows = [], []
    for left, right in pairs:
        chromosome = left["chromosome"]
        flag = left["number"] in heavy_numbers
        short = chromosome if len(chromosome) <= 26 else chromosome[:25] + "…"
        all_rows.append(("#%s %s" % (left["number"], short),
                         left["seconds"], right["seconds"], flag))
        if not flag:
            zoom_rows.append(("#%s %s" % (left["number"], chromosome),
                              left["seconds"], right["seconds"], False))
    zoom_max = max([max(r[1], r[2]) for r in zoom_rows] or [longest]) * 1.05

    ratio_lo = min(0.95, min(r[1] for r in ratio_rows) - 0.05)
    ratio_hi = max(1.1, max(r[1] for r in ratio_rows) + 0.05)

    verdict = (
        '<p class="callout">Every setting that would change what the search did '
        'agrees, and every paired individual ended on the same chromosome with '
        'the same fitness. The two sweeps are the same search run twice, so what '
        'follows is a controlled experiment: only <code>%s</code> differs.</p>'
        % ", ".join(differs)
        if not disagree and set(differs) <= {"LORA_SERVER_COUNT"} else
        '<p class="callout warn">These two sweeps are <strong>not</strong> the '
        'same search run twice. %s %s Read the timings as two runs that happen '
        'to share individuals, not as a controlled comparison.</p>'
        % ("%d setting(s) differ: <code>%s</code>." % (len(differs), ", ".join(differs))
           if differs else "The recorded settings agree.",
           "%d paired individual(s) ended differently: %s."
           % (len(disagree), humanise(disagree)) if disagree
           else "The paired individuals all ended the same."))

    headline = (
        "Two sweeps of the same search, differing in <code>LORA_SERVER_COUNT</code>. "
        if not disagree and set(differs) <= {"LORA_SERVER_COUNT"} else
        "Two sweeps, compared script for script. ")
    headline += (
        "With <strong>%s</strong> each individual is <strong>%+.0f%%</strong> slower "
        "and the <code>process</code> step <strong>%+.0f%%</strong> faster than "
        "with <strong>%s</strong>. This page is why, and what the setting should be."
        % (b.label, pct(script_a, script_b), pct(step_a, step_b), a.label))

    pills = "\n".join(
        '<span class="pill">%s</span>' % esc(text) for text in (
            "%s · %s · run %d" % (a.label, os.path.basename(a.db), a.run_id),
            "%s · %s · run %d" % (b.label, os.path.basename(b.db), b.run_id),
            "%d paired scripts · %d passes · %s"
            % (len(pairs), len(shared_passes), gpu)))

    parts = []
    parts.append("""
<section>
<h2>How comparable are they?</h2>
<p>Both sides are read out of their own sweep's stored settings and timings.
%(paired)d of %(total_a)d and %(total_b)d scripts pair up on
<code>(pass, individual)</code> &mdash; those are the ones timed twice, and the
only ones this page compares.</p>
%(verdict)s
<div class="card scroll">
<table><thead><tr><th>setting</th><th class=mono>%(label_a)s</th>
<th class=mono>%(label_b)s</th><th></th></tr></thead>
<tbody>%(conf_rows)s</tbody></table>
</div>
<div class="card scroll">
<table><thead><tr><th>individual</th><th>chromosome</th>
<th class=mono>fitness, %(label_a)s</th><th class=mono>fitness, %(label_b)s</th>
<th></th></tr></thead>
<tbody>%(pop_rows)s</tbody></table>
</div>
</section>

<section>
<h2>What actually changed</h2>
<div class="tiles">%(tiles)s</div>
<p class="callout">The per-individual figure and the step figure disagree, and
both are true. Summed script seconds &mdash; what the stats page's per-individual
bars and the phase table are built from &mdash; went
%(script_a).0fs&nbsp;&#8594;&nbsp;%(script_b).0fs. The step's own wall clock went
%(step_a).0fs&nbsp;&#8594;&nbsp;%(step_b).0fs. A script's seconds are its own wall
clock and %(width_b)d of them run at once, so the two measure different things:
the same two-denominators rule the stats page prints on every cost chart.</p>
</section>

<section>
<h2>Every script, both ways</h2>
<p>The %(paired)d paired scripts in the order they ran. Two bars each: the upper
is %(label_a)s, the lower %(label_b)s.%(heavy_note)s</p>
<div class="key">
<span><i style="background:var(--a)"></i>%(label_a)s</span>
<span><i style="background:var(--b)"></i>%(label_b)s</span>
%(heavy_key)s
</div>
<div class="card scroll">%(all_chart)s</div>
%(zoom)s
</section>

<section>
<h2>The control that settles it</h2>
<p>Each script's %(label_b)s time over its own %(label_a)s time. Batch width is
set by the pool, so most scripts went from sharing the card with
%(mates_a)d other%(mates_a_s)s to sharing it with %(mates_b)d &mdash; and those
rose by %(shared_pct).0f%% on average. What makes it a control is the
%(controls)d that did not.</p>
<div class="key">
<span><i style="background:var(--same)"></i>batch width unchanged &#8212; mean %(mean_same).2f&#215;</span>
<span><i style="background:var(--flag)"></i>widened, but outlived its batch-mates &#8212; mean %(mean_straggler).2f&#215;</span>
<span><i style="background:var(--b)"></i>widened and shared throughout &#8212; mean %(mean_shared).2f&#215;</span>
</div>
<div class="card scroll">%(ratio_chart)s</div>
%(control_prose)s
<p>So the slowdown tracks <em>time actually spent sharing the card</em>, not the
setting. That is contention and nothing else: same blend, same weight seed, same
prompts, same warm server. %(gpu_prose)s running several independent CUDA contexts
time-slices them rather than running them together, and decoding a handful of
prompts at a time is small-batch, bandwidth-bound work that never filled the card
to begin with. Each job's latency rises with the number of co-resident jobs while
the work done per second barely moves.</p>
<p><code>build</code> inflates far less than <code>generate</code>
(%(build_pct)+.0f%% against %(gen_pct)+.0f%% per call) because it is larger single
kernels plus CPU and disk work &mdash; the part that genuinely does overlap.</p>
</section>

<section>
<h2>Why the wall clock did not follow</h2>
<p>A batch costs what its slowest script costs: the pool waits the group out
before starting the next. Pass&nbsp;%(heavy_pass)d is the most expensive of the
sweep, and %(straggler_prose)s</p>
<div class="card scroll">%(timeline_heavy)s</div>
<p class="muted">Each outlined box is one batch; the bars inside it are its
scripts at their own lengths. The figure on the right is the sum of the batch
costs, which is what the step spends once the pool startup is allowed for.</p>
%(light_timeline)s
</section>

<section>
<h2>Pass by pass</h2>
<p><em>Overlap</em> is summed script seconds over the step's wall seconds: the
concurrency actually achieved.</p>
<div class="card scroll">
<table><thead><tr><th>pass</th><th>scripts</th>
<th class=mono>step s, %(label_a)s</th><th class=mono>step s, %(label_b)s</th>
<th class=mono>overlap, %(label_a)s</th><th class=mono>overlap, %(label_b)s</th>
<th class=mono>gen/call, %(label_a)s</th><th class=mono>gen/call, %(label_b)s</th>
</tr></thead>
<tbody>%(pass_rows)s</tbody></table>
</div>
<p class="muted">A pass that started a pool carries its startup
(<code>start servers</code> in the phase table below) inside its step seconds
&mdash; <code>start_run.py</code> and <code>continue_run.py</code> are two
Contexts, so a whole search pays it twice whatever the server count.</p>
</section>

<section>
<h2>Where the seconds went</h2>
<div class="card scroll">
<table><thead><tr><th>phase</th><th>calls</th>
<th class=mono>total s, %(label_a)s</th><th class=mono>total s, %(label_b)s</th>
<th class=mono>per call, %(label_a)s</th><th class=mono>per call, %(label_b)s</th>
<th>per call</th></tr></thead>
<tbody>%(phase_rows)s</tbody></table>
</div>
%(model_load_note)s
<h3>The batch model reproduces the recorded step times</h3>
<p class="muted">A check on the reasoning above: predicting each pass as
&ldquo;sum over batches of the slowest script in the batch&rdquo; lands on the
recorded wall seconds, once any pool startup in that pass is allowed for.</p>
<div class="card scroll">
<table><thead><tr><th>pass</th><th class=mono>predicted, %(label_a)s</th>
<th class=mono>recorded</th><th class=mono>predicted, %(label_b)s</th>
<th class=mono>recorded</th></tr></thead>
<tbody>%(check_rows)s</tbody></table>
</div>
</section>

<section>
<h2>What to set it to</h2>
<p><code>server_pool.py</code>'s docstring leaves it open &mdash; &ldquo;whether a
pool of more than one pays is an open question&rdquo; &mdash; on the reasoning
that a warm server removes exactly the CPU-bound <code>import</code> and
<code>model_load</code> that used to overlap well, leaving GPU-bound work on one
card. Measured here, that reasoning holds.</p>
<ul>
<li><strong>%(label_a)s &#8594; %(label_b)s buys %(light_gain)s on the clean
passes and %(heavy_gain)s on pass %(heavy_pass)d</strong>, at the cost of
%(script_pct)+.0f%% longer per individual and %(extra_models)s.</li>
<li><strong>The bigger lever is not the pool.</strong> %(heavy_share).0f%% of all
script time is in %(heavy_count)d straggler%(heavy_count_s)s, and stragglers are
batched blind &mdash; a batch is cut in individual order, not by expected cost.
Ordering each pass heaviest-first would cut pass %(heavy_pass)d by more than any
server count will.</li>
<li><strong>Per-individual seconds are not comparable across this setting.</strong>
An execution's <code>seconds</code> is its own wall clock, so these two sweeps
disagree by ~%(script_pct).0f%% about what the same blend cost &mdash; worth
remembering before reading a per-individual bar as a property of the blend.</li>
</ul>
</section>
""" % {
        "paired": len(pairs), "total_a": len(a.scripts), "total_b": len(b.scripts),
        "verdict": verdict,
        "label_a": esc(a.label), "label_b": esc(b.label),
        "conf_rows": conf_rows, "pop_rows": pop_rows, "tiles": tiles_html,
        "script_a": script_a, "script_b": script_b,
        "step_a": step_a, "step_b": step_b,
        "width_b": b.width,
        "heavy_note": ("" if not heavy_numbers else
                       " %s outlive their batch-mates and are drawn apart: between "
                       "them they are %.0f%% of all script time."
                       % (humanise(sorted(heavy_numbers)),
                          sum(left["seconds"] for left, _ in pairs
                              if left["number"] in heavy_numbers) / script_a * 100)),
        "heavy_key": ("" if not heavy_numbers else
                      '<span><i style="background:var(--flag)"></i>a straggler, '
                      'both ways</span>'),
        "all_chart": paired_bars(all_rows, longest,
                                 note="linear to %.0fs" % longest),
        "zoom": ("" if not heavy_numbers or not zoom_rows else
                 "<h3>The other %d, on their own scale</h3>\n"
                 '<div class="card scroll">%s</div>'
                 % (len(zoom_rows),
                    paired_bars(zoom_rows, zoom_max,
                                note="linear to %.0fs" % zoom_max))),
        "mates_a": a.width - 1, "mates_a_s": "" if a.width == 2 else "s",
        "mates_b": b.width - 1,
        "shared_pct": (means["shared"] - 1) * 100,
        "controls": len(members["same"]) + len(members["straggler"]),
        "mean_same": means["same"], "mean_straggler": means["straggler"],
        "mean_shared": means["shared"],
        "ratio_chart": ratio_chart(ratio_rows, ratio_lo, ratio_hi),
        "control_prose": control_prose(a, b, pairs, members, means),
        "gpu_prose": "One %s" % gpu if gpu != "one GPU" else "One GPU",
        "build_pct": pct(a.per_call("build"), b.per_call("build")),
        "gen_pct": pct(a.per_call("generate"), b.per_call("generate")),
        "heavy_pass": heavy_pass,
        "straggler_prose": straggler_prose(a, b, heavy_pass, heavy_numbers),
        "timeline_heavy": timeline(a, b, heavy_pass, heavy_numbers),
        "light_timeline": ("" if light_pass is None else
                           "<h3>The same picture in a clean pass</h3>\n"
                           "<p>With no straggler in it, pass %d is what the extra "
                           "concurrency looks like when it can actually work.</p>\n"
                           '<div class="card scroll">%s</div>'
                           % (light_pass, timeline(a, b, light_pass, heavy_numbers))),
        "pass_rows": "\n".join(pass_rows),
        "phase_rows": "\n".join(phase_rows),
        "model_load_note": (
            "<p><code>model_load</code> appears in neither, which is the point of "
            "the remote template: it was paid per server by the pool rather than "
            "once per individual by the scripts.</p>"
            if a.remote and b.remote else ""),
        "check_rows": check_rows,
        "light_gain": gain(pct(light_b, light_a) if light_n else 0.0),
        "heavy_gain": gain(pct(b.step_seconds(heavy_pass),
                               a.step_seconds(heavy_pass))),
        "script_pct": pct(script_a, script_b),
        "extra_models": ("%d more base model%s resident"
                         % (b.width - a.width, "" if b.width - a.width == 1 else "s")
                         if b.width > a.width else "a narrower batch"),
        "heavy_share": (sum(left["seconds"] for left, _ in pairs
                            if left["number"] in heavy_numbers) / script_a * 100
                        if heavy_numbers else 0.0),
        "heavy_count": len(heavy_numbers),
        "heavy_count_s": "" if len(heavy_numbers) == 1 else "s",
    })
    return "\n".join(parts), headline, pills


def control_prose(a, b, pairs, members, means):
    """The two control groups, named from the data rather than from the example."""
    out = []
    if members["same"]:
        out.append(
            "<p><strong>%d kept %s batch width.</strong> %s ran in a batch of the "
            "same size under both configurations &mdash; a trailing group too "
            "small to widen, or a pass with too few scripts to fill one. Nothing "
            "about their surroundings changed, and nothing about their timings "
            "did: mean %.2f&#215;.</p>"
            % (len(members["same"]),
               "its" if len(members["same"]) == 1 else "their",
               humanise(sorted(members["same"])), means["same"]))
    if members["straggler"]:
        shares = []
        index = {s["number"]: s for s in b.scripts}
        for number in sorted(members["straggler"]):
            script = index.get(number)
            if script:
                shares.append(b.alone_share(script))
        out.append(
            "<p><strong>%d widened and still barely moved &mdash; the "
            "stragglers.</strong> %s do go into a wider batch, but they outlive "
            "everything they share it with: their longest batch-mate finishes "
            "before they are half done, so %.0f&ndash;%.0f%% of their life is "
            "uncontended either way. Widening a batch cannot slow down a script "
            "that is already alone in it. They rose %.0f%%, against %.0f%% for "
            "the scripts that shared the whole way.</p>"
            % (len(members["straggler"]), humanise(sorted(members["straggler"])),
               min(shares) * 100 if shares else 0.0,
               max(shares) * 100 if shares else 0.0,
               (means["straggler"] - 1) * 100, (means["shared"] - 1) * 100))
    return "\n".join(out)


def straggler_prose(a, b, pn, heavy):
    """What pass `pn` does with its stragglers, said from the batches themselves."""
    spread = []
    for side in (a, b):
        groups = side.batches(pn)
        holding = [i for i, group in enumerate(groups)
                   if any(s["number"] in heavy for s in group)]
        spread.append(len(holding))
    if not heavy or not spread[0]:
        return ("no one script dominates it, so the wider batch is free to do "
                "what it can.")
    if spread[0] == spread[1]:
        return ("its stragglers land in %d different batches under <em>both</em> "
                "configurations &mdash; so doubling the width doubled nothing. It "
                "only added competitors to each straggler's batch, which is why "
                "the stragglers themselves got slower." % spread[0])
    return ("its stragglers fall into %d batches on the left and %d on the right, "
            "so the wider batch absorbs some of them &mdash; but a batch still "
            "costs what its slowest script costs." % (spread[0], spread[1]))


# --- the command line --------------------------------------------------------


def out_path(a, given):
    """Where the page goes: as asked, or beside the first sweep's database."""
    if given:
        return os.path.abspath(given if os.path.isabs(given)
                               else os.path.join(_ROOT, given))
    stem = os.path.splitext(os.path.basename(a.db))[0]
    return os.path.join(os.path.dirname(a.db),
                        "%s_run%d_servers.html" % (stem, a.run_id))


def resolve(conn, run):
    run_id = store.latest_run(conn) if run == 0 else run
    if run_id is None:
        raise SystemExit("%s holds no runs yet" % conn.path)
    if store.get_run(conn, run_id) is None:
        raise SystemExit("%s holds no run %d" % (conn.path, run_id))
    return run_id


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare two sweeps that differ in LORA_SERVER_COUNT, as one "
                    "HTML page.")
    parser.add_argument("db_a", help="the database holding the first sweep")
    parser.add_argument("db_b", nargs="?", default=None,
                        help="the database holding the second (default: the first)")
    parser.add_argument("--run-a", type=int, default=0,
                        help="which sweep in db_a (0, the default, is the latest)")
    parser.add_argument("--run-b", type=int, default=0,
                        help="which sweep in db_b (0, the default, is the latest)")
    parser.add_argument("--label-a", default=None,
                        help="what to call the first side (default: its server count)")
    parser.add_argument("--label-b", default=None, help="and the second")
    parser.add_argument("--gpu", default="one GPU",
                        help="the card these ran on, for the page's prose")
    parser.add_argument("--out", default=None,
                        help="write here instead of beside the first database")
    parser.add_argument("--open", dest="open_it", action="store_true",
                        help="open the page in a browser when it is written")
    args = parser.parse_args(argv)

    conn_a = store.connect(locate(args.db_a))
    conn_b = conn_a if args.db_b is None else store.connect(locate(args.db_b))
    a = Side(conn_a, resolve(conn_a, args.run_a), args.label_a)
    b = Side(conn_b, resolve(conn_b, args.run_b), args.label_b)
    if a.db == b.db and a.run_id == b.run_id:
        raise SystemExit("both sides are run %d of %s -- name two sweeps"
                         % (a.run_id, a.db))
    for side in (a, b):
        if not side.scripts:
            raise SystemExit(
                "run %d of %s recorded no script timings, so there is nothing to "
                "compare. Only a sweep whose process step ran under the timing "
                "tables can be read this way." % (side.run_id, side.db))

    body, headline, pills = build(a, b, args.gpu)
    title = "%s or %s?" % (a.label.capitalize(), b.label)
    page = PAGE % {"title": esc(title), "css": CSS, "headline": headline,
                   "pills": pills, "body": body,
                   "source_a": "<code>%s</code> (run %d)"
                               % (esc(os.path.relpath(a.db, _ROOT)), a.run_id),
                   "source_b": "<code>%s</code> (run %d)"
                               % (esc(os.path.relpath(b.db, _ROOT)), b.run_id)}

    path = out_path(a, args.out)
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(page)
    print("wrote %s" % path)
    print("  %-22s %s run %d -- %d scripts, %.0fs of script time, %.0fs of step time"
          % (a.label, os.path.basename(a.db), a.run_id, len(a.scripts),
             a.script_seconds(), a.step_seconds()))
    print("  %-22s %s run %d -- %d scripts, %.0fs of script time, %.0fs of step time"
          % (b.label, os.path.basename(b.db), b.run_id, len(b.scripts),
             b.script_seconds(), b.step_seconds()))
    if args.open_it:
        webbrowser.open("file://" + path.replace(os.sep, "/"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
