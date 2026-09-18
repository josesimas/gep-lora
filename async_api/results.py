"""
results.py - A job's sweep database, read back as JSON.

Reads through store.py's helpers, the way `python -m storage.store --show` and
the HTML report do, and writes nothing. Every function takes a path and opens
the database only if it is already there: store.connect() creates what it
cannot open, and a job whose files were deleted should read as "no results",
not as an empty sweep.
"""

import json
import os

from storage import store

# The settings worth putting in a job's summary; the detail carries all of them.
SUMMARY_SETTINGS = ("TEMPLATE", "BASE_MODEL", "CHAT_TEMPLATE", "EVALUATOR",
                    "COMPOSITE_EVALUATORS", "COMPOSITE_AGGREGATE", "COUNT",
                    "GENERATIONS", "TRAINING_COUNT", "SELECTION_COUNT", "MUTATION_RATE")


class NoResults(LookupError):
    """The job's database is not on disk."""


def _open(db_path):
    if not db_path or not os.path.exists(db_path):
        raise NoResults(db_path)
    return store.connect(db_path)


def _dict(row):
    return None if row is None else {key: row[key] for key in row.keys()}


def _weights(text):
    try:
        return json.loads(text) if text else None
    except ValueError:
        return None


def progress(db_path, run_id):
    """How far the search has got. -> {generations, expected, steps, last_step}."""
    try:
        conn = _open(db_path)
    except NoResults:
        return None
    try:
        conf = store.get_settings(conn, run_id)
        generations = len(store.fitness_by_generation(conn, run_id))
        steps = store.step_timings(conn, run_id)
        last = steps[-1] if steps else None
        tested = store.test_summary(conn, run_id)
        return {"generations_scored": generations,
                "generations_expected": 1 + int(conf.get("GENERATIONS") or 0),
                "steps_run": len(steps),
                "last_step": None if last is None else {
                    "step": last["step"], "generation": last["generation"],
                    "status": last["status"], "seconds": last["seconds"],
                    "started_at": last["started_at"]},
                "tested": bool(tested)}
    finally:
        conn.close()


def snapshot(db_path, destination):
    """Copy a job's database to `destination`, consistently. -> its size in bytes.

    Through sqlite's backup rather than a file copy, so a job the worker is
    still writing to comes out as the database it was at one moment, not a file
    caught half way through a transaction. The destination is opened through
    store.connect() -- the backup replaces everything in it, schema included.
    """
    source = _open(db_path)
    try:
        target = store.connect(destination)
        try:
            source.backup(target)
        finally:
            target.close()
    finally:
        source.close()
    return os.path.getsize(destination)


def run_status(db_path, run_id):
    """The sweep's own status column (open | done | failed), or None."""
    try:
        conn = _open(db_path)
    except NoResults:
        return None
    try:
        run = store.get_run(conn, run_id)
        return run["status"] if run else None
    finally:
        conn.close()


def population(conn, run_id):
    quality = {row["number"]: row for row in store.quality_rows(conn, run_id)}
    out = []
    for row in store.individuals(conn, run_id):
        view = quality.get(row["number"])
        out.append({"number": row["number"], "chromosome": row["chromosome"],
                    "state": row["state"], "rank": row["rank"],
                    "fitness": row["fitness"], "is_best": bool(row["is_best"]),
                    "has_changed": bool(row["has_changed"]),
                    "quality": view["quality"] if view else None,
                    "answers": view["answers"] if view else 0,
                    "verdict": view["verdict"] if view else None})
    return out


def best(individuals):
    """The fittest runnable individual: elitism's rule, applied to fitness now.

    Not the `is_best` flag. A finished sweep's last generation stops after
    `fitness`, so the flag is still the one elitism set a generation earlier,
    and the population has been scored again since.
    """
    runnable = [one for one in individuals if one["state"] != "BAD" and one["fitness"]]
    if not runnable:
        return None
    return sorted(runnable, key=lambda one: (-(one["fitness"] or 0.0), one["number"]))[0]


def detail(db_path, run_id):
    """Everything a job produced, as one JSON-ready dict."""
    conn = _open(db_path)
    try:
        run = store.get_run(conn, run_id)
        if run is None:
            raise NoResults(db_path)
        conf = store.get_settings(conn, run_id)
        individuals = population(conn, run_id)
        history = []
        for entry in store.fitness_by_generation(conn, run_id):
            top = store.best_of_generation(conn, run_id, entry["generation"])
            history.append(dict(_dict(entry), best_chromosome=top["chromosome"] if top else None))
        testing = [_dict(row) for row in store.test_quality(conn, run_id)]
        return {
            "run": _dict(run),
            "settings": conf,
            "datasets": store.dataset_summary(conn, run_id),
            "best": best(individuals),
            "population": individuals,
            "fitness_history": history,
            "testing": {"summary": store.test_summary(conn, run_id), "individuals": testing},
            "costs": [_dict(row) for row in store.step_costs(conn, run_id)],
        }
    finally:
        conn.close()


def summary(db_path, run_id):
    """The short form a job list carries: the best individual and a few settings."""
    try:
        conn = _open(db_path)
    except NoResults:
        return None
    try:
        conf = store.get_settings(conn, run_id)
        top = best(population(conn, run_id))
        return {"settings": {name: conf.get(name) for name in SUMMARY_SETTINGS if name in conf},
                "best": None if top is None else {
                    key: top[key] for key in ("number", "chromosome", "fitness", "quality")}}
    finally:
        conn.close()


def individual(db_path, run_id, number):
    """One individual with its latest transcript, or None."""
    conn = _open(db_path)
    try:
        rows = [row for row in store.individuals(conn, run_id) if row["number"] == number]
        if not rows:
            return None
        row = rows[0]
        execution = store.latest_execution(conn, row["id"])
        out = {key: row[key] for key in ("number", "chromosome", "tree", "state", "rank",
                                         "weight_seed", "fitness", "is_best", "has_changed")}
        out["is_best"] = bool(out["is_best"])
        out["execution"] = None
        if execution is not None:
            exchanges = conn.execute(
                "SELECT position, question, answer, quality, reason, judge_model"
                "  FROM exchanges WHERE execution_id = ? ORDER BY position",
                (execution["id"],)).fetchall()
            out["execution"] = {
                "started_at": execution["started_at"], "seconds": execution["seconds"],
                "verdict": execution["verdict"], "exit_code": execution["exit_code"],
                "weights": _weights(execution["weights"]),
                "exchanges": [_dict(one) for one in exchanges]}
        out["testing"] = [
            {"dataset": test["dataset"], "verdict": test["verdict"],
             "quality": test["quality"], "selected_on": test["selected_on"],
             "exchanges": json.loads(test["exchanges"] or "[]")}
            for test in store.test_results(conn, run_id) if test["number"] == number]
        return out
    finally:
        conn.close()
