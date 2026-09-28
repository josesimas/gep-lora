"""
facade.py - The async API's features as plain Python, for every UI.

`App(registry, cache, catalog)` is the state the features share, and each
method is one thing a user may do -- `App.submit_job(user, body)`,
`App.start_verification(user, job_id, body)`, `App.set_live(user, job_id, body)`
-- taking the user it is done as and returning (status, payload). The web app
(gep_lora/apps/web/server.py) routes a request to one of them; a desktop app,
a command line or a test calls the same method and gets the same checks: that
the job, LoRA or verification is the user's own (another user's is a 404, as a
missing one is), that its state allows what is asked, and what it is queued as.

Why (status, payload) rather than just the payload: the outcome has more than
one success (201 made, 202 accepted but still running) and the refusals differ
in kind, so a caller needs the code whatever it speaks. The codes are HTTP's
numbers because they are a vocabulary every UI already has:

    200 done    201 made    202 accepted, still happening
    400 the request is wrong      401 no such key or token
    403 not the API's to do       404 not found, or not yours
    409 the state does not allow it    413 too big
    502 a service it asked failed      503 not available here

A refusal is a ServiceError carrying one of them and a message, and nothing
here knows about sockets, headers or pages. A download is a FileReply: a file
the caller sends and then removes.
"""

import functools
import json
import os
import re
import shutil
import tempfile
from urllib.parse import urlparse

from gep_lora.core.adapters import catalog as lora_catalog
from gep_lora.core.config import settings as config
from gep_lora.core.search.generate_population import slot_key
from gep_lora.service import drawn
from gep_lora.service import evaluate
from gep_lora.service import golive
from gep_lora.service import inference
from gep_lora.service import registry as reg
from gep_lora.service import results
from gep_lora.service import settings
from gep_lora.service import submit
from gep_lora.service import testpass
from gep_lora.service import train
from gep_lora.service import verify


class FileReply:
    """A download: a file on disk, sent as an attachment and then removed."""

    def __init__(self, path, filename, content_type="application/octet-stream"):
        self.path = path
        self.filename = filename
        self.content_type = content_type


class ServiceError(Exception):
    """A refusal: `status` says what kind (see above), the message says why,
    and `payload` is what a caller shows -- {"error": message, **extra}."""

    def __init__(self, status, message, **extra):
        super().__init__(message)
        self.status = status
        self.payload = dict(extra, error=message)


def job_json(app, job, with_summary=False):
    out = {key: job[key] for key in ("id", "label", "status", "created_at", "started_at",
                                     "finished_at", "run_id", "exit_code", "error")}
    out["options"] = json.loads(job["options"] or "{}")
    out["cancel_requested"] = bool(job["cancel_requested"])
    # What the worker does with it when it is queued, and what it can be asked
    # to do next -- so a client need not know the rules to draw its buttons.
    out["task"] = job["task"]
    out["resumable"] = job["status"] in reg.RESUMABLE
    out["can_evaluate"] = can_requeue(job, reg.EVALUATE)
    out["can_test"] = can_requeue(job, reg.TEST)
    if job["status"] == reg.QUEUED:
        out["queue_position"] = app.registry.queue_position(job["id"])
    if with_summary and job["status"] != reg.DELETED:
        out["summary"] = results.summary(app.registry.database(job), job["run_id"])
    return out


def can_requeue(job, task):
    """May this job be queued again for `task`? A drawn blend never may: no
    search made it, so there is nothing to resume, evaluate or test."""
    return job["status"] in reg.REQUEUE_FROM[task] and job["task"] != reg.BLEND


def refuse_drawn(job, what):
    if job["task"] == reg.BLEND:
        raise ServiceError(409, "job %d is a blend drawn by hand, not a search; there is "
                            "nothing to %s -- verify it instead" % (job["id"], what))


def verification_json(row, report=None):
    out = {key: row[key] for key in ("id", "job_id", "status", "created_at",
                                     "started_at", "finished_at", "number",
                                     "chromosome", "exit_code", "error")}
    out["options"] = json.loads(row["options"] or "{}")
    # The dataset is said by its label; where it sits on the server is not the
    # client's business.
    out["options"].pop("dataset", None)
    if report is not None:
        out["report"] = report
    return out


def training_json(registry, row):
    out = {key: row[key] for key in ("id", "name", "status", "created_at", "started_at",
                                     "finished_at", "exit_code", "error")}
    out["options"] = json.loads(row["options"] or "{}")
    out["cancel_requested"] = bool(row["cancel_requested"])
    if row["status"] == reg.QUEUED:
        out["queue_position"] = registry.training_queue_position(row["id"])
    return out


def default_slots():
    """{folder: [slot]} for the server's own LORA_SLOTS -- the command line's
    set, which no job submitted here may use (submit.own_slots). Kept only so
    a LoRA that is one of them cannot be deleted from under the command line."""
    out = {}
    for slot, where in sorted(config.LORA_SLOTS.items(), key=lambda pair: slot_key(pair[0])):
        out.setdefault(os.path.normcase(lora_catalog.absolute(where)), []).append(slot)
    return out


def training_of(registry, row):
    """The training that made a catalogue row, or None.

    Matched on the folder as well as the id: the two live in different files,
    and a row forgotten and another catalogued in its place must never inherit
    a training -- nor with it who may delete the folder.
    """
    training = registry.training_for_lora(row["id"])
    if training is None or training["folder"] != row["folder"]:
        return None
    return training


def lora_json(app, row, user, detail=False):
    """One of the user's catalogue rows as the API shows it: the row, the
    training that made it here if one did, and what may be done with it."""
    out = lora_catalog.as_dict(row)
    training = training_of(app.registry, row)
    if training is not None and training["user_id"] != user["id"]:
        training = None
    in_flight = row["status"] in lora_catalog.IN_FLIGHT
    trained_here = training is not None
    out["trained_here"] = trained_here
    out["training"] = training_json(app.registry, training) if trained_here else None
    out["progress"] = train.progress(row["folder"]) if in_flight else None
    out["can_cancel"] = trained_here and in_flight
    # Only what the API made can the API delete: a folder found on disk, or
    # trained by hand, is left to whoever put it there, owner or not.
    out["can_delete"] = trained_here and not in_flight and training["status"] in reg.FINISHED
    if detail:
        out["history"] = lora_catalog.history(row["folder"])
    return out


def deployment_json(row):
    spec = json.loads(row["spec"])
    return {"id": row["id"], "job_id": row["job_id"], "target": row["target"],
            "individual": row["number"], "chromosome": row["chromosome"],
            "fitness": spec.get("fitness"), "base_model": spec.get("base_model"),
            "engine": spec.get("engine"), "created_at": row["created_at"],
            "revoked_at": row["revoked_at"]}


class App:
    """The state every request shares: the registry, the targets, the cache."""

    def __init__(self, registry=None, cache=None, catalog=None):
        self.registry = registry or reg.Registry()
        self.cache = cache or inference.ModelCache()
        self.targets = golive.targets(self.cache)
        # The LoRAs are a table of the registry's own file.
        self.catalog = catalog or self.registry.catalog

    # --- helpers -----------------------------------------------------------

    def own_job(self, user, job_id):
        job = self.registry.job(job_id, user["id"])
        if job is None:
            raise ServiceError(404, "no job %d" % job_id)
        return job

    def unset(self, deployments):
        for row in deployments:
            target = self.targets.get(row["target"])
            if target is not None:
                target.retire(row["id"], json.loads(row["spec"]))
        return self.registry.revoke([row["id"] for row in deployments])

    # --- jobs --------------------------------------------------------------

    def submit_job(self, user, body):
        try:
            job = submit.submit(self.registry, user, body)
        except submit.SubmissionError as error:
            raise ServiceError(400, str(error))
        return 201, {"job": job_json(self, job)}

    def list_runs(self, user):
        """Every search of the user's, and what came after it, in one read.

        Only their own: the jobs, verifications and deployments are all read
        by the user's id, and a slot's LoRA is named only when it is one of
        their catalogue rows. The folders behind the slots stay here."""
        verifications, live = {}, {}
        for row in self.registry.verifications(user_id=user["id"]):
            verifications.setdefault(row["job_id"], []).append(verification_json(row))
        for row in self.registry.deployments(user_id=user["id"]):
            live.setdefault(row["job_id"], []).append(deployment_json(row))
        runs = []
        for job in self.registry.jobs(user["id"]):
            out = job_json(self, job, with_summary=True)
            summary = out.get("summary") or {}
            names = self.catalog.slot_names(summary.pop("slots", None), user["name"])
            out["loras"] = sorted(set(names.values()))
            out["verifications"] = verifications.get(job["id"], [])
            out["live"] = live.get(job["id"], [])
            runs.append(out)
        return 200, {"runs": runs}

    def list_jobs(self, user, query):
        status = (query.get("status") or [None])[0]
        return 200, {"jobs": [job_json(self, job, with_summary=True)
                              for job in self.registry.jobs(user["id"], status)]}

    def job_detail(self, user, job_id):
        job = self.own_job(user, job_id)
        out = {"job": job_json(self, job)}
        try:
            out["results"] = results.detail(self.registry.database(job), job["run_id"])
        except results.NoResults:
            out["results"] = None
        out["live"] = [deployment_json(row)
                       for row in self.registry.deployments(job_id=job_id)]
        return 200, out

    def job_status(self, user, job_id):
        job = self.own_job(user, job_id)
        return 200, {"job": job_json(self, job),
                     "progress": results.progress(self.registry.database(job), job["run_id"])}

    def job_database(self, user, job_id):
        """The job's sweep database, as a snapshot, for download."""
        job = self.own_job(user, job_id)
        handle, path = tempfile.mkstemp(prefix="gep-job%d-" % job_id, suffix=".sqlite3")
        os.close(handle)
        os.remove(path)                  # store.connect() makes it, with the schema
        try:
            results.snapshot(self.registry.database(job), path)
        except results.NoResults:
            raise ServiceError(404, "job %d has no database; its run was deleted" % job_id)
        except BaseException:
            if os.path.exists(path):
                os.remove(path)
            raise
        label = re.sub(r"[^A-Za-z0-9._-]+", "_", job["label"] or "").strip("_")
        name = "job%d%s.sqlite3" % (job_id, "_" + label if label else "")
        return 200, FileReply(path, name, "application/vnd.sqlite3")

    def job_log(self, user, job_id, query):
        job = self.own_job(user, job_id)
        try:
            lines = max(1, int((query.get("lines") or ["200"])[0]))
        except ValueError:
            raise ServiceError(400, "lines must be a whole number")
        path = os.path.join(self.registry.folder(job), "job.log")
        if not os.path.exists(path):
            return 200, {"lines": []}
        with open(path, encoding="utf-8", errors="replace") as handle:
            return 200, {"lines": handle.read().splitlines()[-lines:]}

    def job_individual(self, user, job_id, number):
        job = self.own_job(user, job_id)
        try:
            found = results.individual(self.registry.database(job), job["run_id"], number)
        except results.NoResults:
            found = None
        if found is None:
            raise ServiceError(404, "job %d holds no individual %d" % (job_id, number))
        return 200, {"individual": found}

    def cancel_job(self, user, job_id):
        """Cancel a queued job, or stop a running one.

        A running search is *stopped* rather than ended: the worker kills
        gep_lora/core/pipeline/main.py, and what it had done stays in the sweep database for a resume
        to carry on from. A resume or an evaluation cancelled before the worker
        got to it puts the job back as it was.
        """
        job = self.own_job(user, job_id)
        before = job["status"]
        status = self.registry.request_cancel(job["id"])
        if before not in (reg.QUEUED, reg.PREPARING, reg.RUNNING):
            raise ServiceError(409, "job %d is %s; there is nothing to cancel" % (job_id, status))
        if status == reg.RUNNING:
            note = ("the worker will stop it within a few seconds" +
                    ("" if job["task"] == reg.EVALUATE
                     else "; resume it afterwards to carry on from there"))
        elif job["requeued_from"]:
            note = "taken out of the queue before it started; the job is %s again" % status
        else:
            note = "cancelled before it started"
        return 202 if status == reg.RUNNING else 200, {
            "job": job_json(self, self.registry.job(job_id)), "note": note}

    def _has_database(self, job):
        if not os.path.exists(self.registry.database(job)):
            raise ServiceError(409, "job %d has no database; its run was deleted" % job["id"])

    def rename_job(self, user, job_id, body):
        """Give a job another label -- the registry's, which is what every page
        shows. The sweep's own `runs.label` stays what it was created with."""
        self.own_job(user, job_id)
        label = (body or {}).get("label")
        if label is not None and not isinstance(label, str):
            raise ServiceError(400, "label must be a string")
        label = (label or "").strip() or None
        self.registry.rename_job(job_id, label)
        return 200, {"job": job_json(self, self.own_job(user, job_id))}

    def resume_job(self, user, job_id):
        """Queue a search that did not finish, to carry on from where it got to."""
        job = self.own_job(user, job_id)
        refuse_drawn(job, "resume")
        if job["status"] not in reg.RESUMABLE:
            raise ServiceError(409, "job %d is %s; only a stopped, cancelled or failed job "
                                "can be resumed" % (job_id, job["status"]))
        self._has_database(job)
        queued = self.registry.requeue(job_id, reg.RESUME)
        if queued is None:
            raise ServiceError(409, "job %d changed status while being resumed" % job_id)
        return 200, {"job": job_json(self, queued),
                     "note": "queued; the worker carries it on from where it stopped"}

    def _grading(self, job):
        """What an evaluation of this job's answers is offered."""
        try:
            return evaluate.choices(self.registry.database(job), job["run_id"])
        except results.NoResults:
            raise ServiceError(409, "job %d has no database; its run was deleted" % job["id"])

    def evaluate_form(self, user, job_id):
        job = self.own_job(user, job_id)
        out = self._grading(job)
        out["can_evaluate"] = can_requeue(job, reg.EVALUATE)
        return 200, out

    def start_evaluation(self, user, job_id, body):
        """Queue the job to grade the answers it already holds."""
        job = self.own_job(user, job_id)
        refuse_drawn(job, "evaluate")
        if job["status"] not in reg.REQUEUE_FROM[reg.EVALUATE]:
            raise ServiceError(409, "job %d is %s; wait for it to stop before evaluating "
                                "its answers" % (job_id, job["status"]))
        try:
            options = evaluate.options_for(body, self._grading(job))
        except evaluate.EvaluateError as error:
            raise ServiceError(400, str(error))
        queued = self.registry.requeue(job_id, reg.EVALUATE, options)
        if queued is None:
            raise ServiceError(409, "job %d changed status while being queued" % job_id)
        return 200, {"job": job_json(self, queued),
                     "note": "queued; the worker grades its answers after any job "
                             "ahead of it"}

    def _testing(self, job):
        """What a testing pass of this job is offered."""
        try:
            return testpass.choices(self.registry.database(job), job["run_id"])
        except results.NoResults:
            raise ServiceError(409, "job %d has no database; its run was deleted" % job["id"])

    def test_form(self, user, job_id):
        job = self.own_job(user, job_id)
        out = self._testing(job)
        out["can_test"] = can_requeue(job, reg.TEST)
        return 200, out

    def start_test(self, user, job_id, body):
        """Queue a finished job to put its blends in front of its testing split."""
        job = self.own_job(user, job_id)
        refuse_drawn(job, "test")
        if job["status"] not in reg.REQUEUE_FROM[reg.TEST]:
            raise ServiceError(409, "job %d is %s; only a finished search's blends can be "
                                "tested" % (job_id, job["status"]))
        try:
            options = testpass.options_for(body, self._testing(job))
        except testpass.TestPassError as error:
            raise ServiceError(400, str(error))
        queued = self.registry.requeue(job_id, reg.TEST, options)
        if queued is None:
            raise ServiceError(409, "job %d changed status while being queued" % job_id)
        return 200, {"job": job_json(self, queued),
                     "note": "queued; the worker tests its blends after any job "
                             "ahead of it"}

    def _finished(self, job, what):
        if job["status"] not in reg.FINISHED:
            raise ServiceError(409, "job %d is %s; cancel it before deleting %s"
                           % (job["id"], job["status"], what))

    def delete_run(self, user, job_id):
        job = self.own_job(user, job_id)
        self._finished(job, "its run")
        self.unset(self.registry.deployments(job_id=job_id))
        shutil.rmtree(self.registry.folder(job), ignore_errors=True)
        self.registry.mark_deleted(job_id)
        return 200, {"job": job_json(self, self.registry.job(job_id))}

    def delete_job(self, user, job_id):
        job = self.own_job(user, job_id)
        self._finished(job, "it")
        self.unset(self.registry.deployments(job_id=job_id))
        shutil.rmtree(self.registry.folder(job), ignore_errors=True)
        self.registry.delete_job(job_id)
        return 200, {"deleted": job_id}

    # --- verification ------------------------------------------------------

    def _offered(self, job):
        """What this job's sweep allows a verification to ask for."""
        try:
            return verify.choices(self.registry.database(job), job["run_id"])
        except results.NoResults:
            raise ServiceError(404, "job %d has no database to verify against" % job["id"])

    def verify_form(self, user, job_id):
        job = self.own_job(user, job_id)
        if not os.path.exists(self.registry.database(job)):
            raise ServiceError(404, "job %d has no database; its run was deleted" % job_id)
        out = self._offered(job)
        out["verifications"] = [verification_json(row) for row
                                in self.registry.verifications(job_id=job_id)]
        # Another dataset a verification may ask instead of a split: a shared
        # file by name, or {"lora": id} for one of the user's own LoRAs' data.
        out["datasets"] = self.list_datasets(user)[1]["datasets"]
        out["loras"] = []
        for row in self.catalog.all(owner=user["name"]):
            training = training_of(self.registry, row)
            if training is not None and training["user_id"] != user["id"]:
                training = None
            if train.dataset_path(row, training, self.registry) is not None:
                out["loras"].append({"id": row["id"], "name": row["name"]})
        out["can_verify"] = job["status"] == reg.DONE
        return 200, out

    def start_verification(self, user, job_id, body):
        """Queue one blend of this job against the LoRAs it is made of."""
        job = self.own_job(user, job_id)
        if job["status"] != reg.DONE:
            raise ServiceError(409, "job %d is %s; only a finished job can be verified"
                           % (job_id, job["status"]))
        offered = self._offered(job)
        try:
            number, options = verify.options_for(
                body, offered, functools.partial(self.verify_dataset, user, job))
        except verify.VerifyError as error:
            raise ServiceError(400, str(error))
        chromosome = next(one["chromosome"] for one in offered["individuals"]
                          if one["number"] == number)
        row = self.registry.add_verification(job, number, chromosome, options)
        return 201, {"verification": verification_json(row),
                     "note": "queued; the worker runs it after any queued job"}

    def verify_dataset(self, user, job, value):
        """A verification's `dataset`, as (path, label): {"file": name}, a file
        under SHARED_DATASETS_DIR, {"lora": id}, the data one of the user's
        own LoRAs was trained on -- another user's is refused as a missing one
        is -- or {"text": ..., "name": ...}, a file the page read and sent,
        written into the job's folder so deleting the run takes it too. Raises
        verify.VerifyError."""
        if isinstance(value, dict) and "text" in value and set(value) <= {"text", "name"}:
            return verify.upload(self.registry.folder(job), value.get("text"),
                                 value.get("name"))
        if isinstance(value, dict) and set(value) == {"file"}:
            path = submit.shared_path(value["file"])
            if path is None:
                raise verify.VerifyError("no shared dataset %r" % (value["file"],))
            return path, value["file"]
        lora_id = (value.get("lora") if isinstance(value, dict) and set(value) == {"lora"}
                   else None)
        if not isinstance(lora_id, int) or isinstance(lora_id, bool):
            raise verify.VerifyError('dataset must be {"file": name}, {"lora": id} '
                                     'or {"text": ..., "name": ...}')
        row = self.catalog.get(lora_id)
        if row is None or row["owner"] != user["name"]:
            raise verify.VerifyError("no LoRA %d of yours" % lora_id)
        training = training_of(self.registry, row)
        if training is not None and training["user_id"] != user["id"]:
            training = None
        path = train.dataset_path(row, training, self.registry)
        if path is None:
            raise verify.VerifyError("no copy of %s's training data is kept" % row["name"])
        return path, "%s's training data" % row["name"]

    def own_verification(self, user, verification_id):
        row = self.registry.verification(verification_id, user["id"])
        if row is None:
            raise ServiceError(404, "no verification %d" % verification_id)
        return row

    def verification_detail(self, user, verification_id):
        row = self.own_verification(user, verification_id)
        job = self.registry.job(row["job_id"])
        folder = self.registry.verification_folder(job, row["id"])
        return 200, {"verification": verification_json(row, verify.report(folder))}

    def verification_log(self, user, verification_id, query):
        row = self.own_verification(user, verification_id)
        job = self.registry.job(row["job_id"])
        try:
            lines = max(1, int((query.get("lines") or ["200"])[0]))
        except ValueError:
            raise ServiceError(400, "lines must be a whole number")
        path = os.path.join(self.registry.verification_folder(job, row["id"]), "verify.log")
        if not os.path.exists(path):
            return 200, {"lines": []}
        with open(path, encoding="utf-8", errors="replace") as handle:
            return 200, {"lines": handle.read().splitlines()[-lines:]}

    # --- blends drawn by hand ------------------------------------------------

    def check_drawn(self, user, body):
        """A drawing, read the way the pipeline would read it (drawn.check)."""
        body = body or {}
        try:
            return 200, drawn.check(self.catalog, user, body.get("tree"), body.get("seed"),
                                    body.get("number"), body.get("pin"), body.get("values"))
        except drawn.DrawnError as error:
            raise ServiceError(400, str(error))

    def random_drawn(self, user, body):
        """A random drawing of the user's LoRAs (drawn.random_drawing)."""
        base_model = (body or {}).get("base_model")
        if base_model is not None and not isinstance(base_model, str):
            raise ServiceError(400, "base_model must be a base model's name")
        try:
            return 200, drawn.random_drawing(self.catalog, user, base_model or None)
        except drawn.DrawnError as error:
            raise ServiceError(409, str(error))

    def save_drawn(self, user, job_id, body):
        """A drawing saved into the run of the job it was opened from (drawn.save)."""
        job = self.own_job(user, job_id)
        if job["status"] == reg.DELETED:
            raise ServiceError(404, "job %d's run was deleted" % job_id)
        try:
            return 201, drawn.save(self.registry, self.catalog, user, job, body)
        except drawn.DrawnError as error:
            raise ServiceError(409, str(error))

    def drawn_code(self, user, body):
        """The script a drawing runs as when it is processed (drawn.code)."""
        try:
            return 200, drawn.code(self.registry, self.catalog, user, body)
        except drawn.DrawnError as error:
            raise ServiceError(409, str(error))

    def list_blends(self, user):
        """Every blend of the user's a page can open (drawn.sources)."""
        return 200, {"jobs": drawn.sources(self.registry, self.catalog, user)}

    def open_drawn(self, user, job_id, query):
        """One blend of a job, as the visual guide and the comparison draw it
        again: a drawn blend's own, or a search's (its best by default)."""
        job = self.own_job(user, job_id)
        if job["status"] == reg.DELETED:
            raise ServiceError(404, "job %d's run was deleted" % job_id)
        number = (query.get("individual") or [None])[0]
        if number is not None:
            if not str(number).isdigit():
                raise ServiceError(400, "individual must be a blend's number")
            number = int(number)
        try:
            return 200, drawn.opened(self.registry, self.catalog, user, job, number)
        except drawn.DrawnError as error:
            raise ServiceError(409, str(error))

    def test_drawn(self, user, body):
        """A drawing, stored as a sweep of one and queued to be verified."""
        try:
            # No job yet: drawn.create asks this only for {"lora": id}, and reads
            # a pasted or uploaded dataset itself, so the job's folder is never needed.
            job, row, found = drawn.create(self.registry, self.catalog, user, body,
                                           functools.partial(self.verify_dataset, user, None))
        except drawn.DrawnError as error:
            raise ServiceError(400, str(error))
        return 201, {"job": job_json(self, job), "verification": verification_json(row),
                     "blend": found,
                     "note": "queued; the worker runs it after any queued job"}

    # --- live --------------------------------------------------------------

    def set_live(self, user, job_id, body):
        job = self.own_job(user, job_id)
        if job["status"] != reg.DONE:
            raise ServiceError(409, "job %d is %s; only a finished job can go live"
                           % (job_id, job["status"]))
        body = body or {}
        target_name = body.get("target") or "local"
        target = self.targets.get(target_name)
        if target is None:
            raise ServiceError(400, "unknown target %r; there are: %s"
                           % (target_name, ", ".join(sorted(self.targets))))
        number = body.get("individual")
        if number is not None and (not isinstance(number, int) or isinstance(number, bool)):
            raise ServiceError(400, "individual must be an individual's number")
        try:
            spec = golive.blend_spec(self.registry.database(job), job["run_id"], number)
        except golive.GoLiveError as error:
            raise ServiceError(409, str(error))
        token, row = self.registry.add_deployment(job, target_name, spec["number"],
                                                  spec["chromosome"], spec)
        published = target.publish(row["id"], spec)
        return 201, {"token": token, "deployment": deployment_json(row),
                     "target": published,
                     "note": "the token is shown once; keep it"}

    def job_live(self, user, job_id):
        self.own_job(user, job_id)
        return 200, {"live": [deployment_json(row)
                              for row in self.registry.deployments(job_id=job_id)]}

    def unset_job(self, user, job_id):
        self.own_job(user, job_id)
        return 200, {"unset": self.unset(self.registry.deployments(job_id=job_id))}

    def submission_form(self, user):
        return 200, submit.form()

    def list_datasets(self, user):
        """The shared dataset files a submission may name with {"file": name}."""
        folder = os.path.join(submit._ROOT,
                              settings.SHARED_DATASETS_DIR)
        try:
            names = sorted(name for name in os.listdir(folder)
                           if os.path.isfile(os.path.join(folder, name)))
        except OSError:
            names = []
        return 200, {"datasets": names}

    def judge_models(self, user, query):
        """The chat models a judge endpoint lists, for a form to offer.

        Asked from here rather than by the browser: an endpoint such as LM Studio
        sends no CORS headers, and the one worth listing is the one this machine
        -- where the worker will grade -- can reach, which a browser elsewhere
        may not. Only the model ids come back. $JUDGE_API_KEY goes with the
        request, as it does when a step asks the same endpoint.
        """
        base_url = ((query.get("base_url") or [""])[0].strip()
                    or config.JUDGE_BASE_URL or "")
        if urlparse(base_url).scheme not in ("http", "https") or not urlparse(base_url).netloc:
            raise ServiceError(400, "base_url must be an http(s) URL, not %r" % base_url)
        from gep_lora.core import evaluators                   # the judge transport, loaded when first asked
        try:
            models = evaluators.common.list_models(
                base_url, evaluators.API_KEY, settings.JUDGE_MODELS_TIMEOUT)
        except SystemExit as error:
            raise ServiceError(502, str(error), base_url=base_url)
        return 200, {"base_url": base_url, "models": models}

    def list_live(self, user):
        return 200, {"live": [deployment_json(row)
                              for row in self.registry.deployments(user_id=user["id"])]}

    def unset_one(self, user, deployment_id):
        row = self.registry.deployment(deployment_id, user["id"])
        if row is None or row["revoked_at"]:
            raise ServiceError(404, "no live deployment %d" % deployment_id)
        return 200, {"unset": self.unset([row])}

    # --- LoRAs -------------------------------------------------------------

    def own_lora(self, user, lora_id):
        """-> (the user's catalogue row, the training of theirs that made it,
        or None). Another user's row is a 404, exactly as a missing one is."""
        row = self.catalog.get(lora_id)
        if row is None or row["owner"] != user["name"]:
            raise ServiceError(404, "no LoRA %d" % lora_id)
        training = training_of(self.registry, row)
        return row, (training if training is not None and training["user_id"] == user["id"]
                     else None)

    def list_loras(self, user, query):
        base_model = (query.get("base_model") or [None])[0]
        return 200, {"loras": [lora_json(self, row, user) for row
                               in self.catalog.all(base_model=base_model, owner=user["name"])]}

    def lora_form(self, user):
        return 200, train.form(self.catalog, user)

    def create_lora(self, user, body):
        try:
            training, row = train.submit(self.registry, self.catalog, user, body)
        except train.TrainError as error:
            raise ServiceError(error.status, str(error))
        return 201, {"lora": lora_json(self, row, user),
                     "training": training_json(self.registry, training),
                     "note": "queued; the worker trains it after any queued job"}

    def scan_loras(self, user):
        """Re-read loras/. What is said back is about the user's own rows: a
        folder found for the first time belongs to nobody, and becomes someone's
        only through `python -m gep_lora.core.adapters.catalog own` on the server."""
        mine = {row["name"] for row in self.catalog.all(owner=user["name"])}
        report = self.catalog.scan()
        return 200, {"missing": [name for name in report["missing"] if name in mine],
                     "updated": [name for name in report["updated"] if name in mine],
                     "loras": len(self.catalog.all(owner=user["name"])),
                     "unowned": len(self.catalog.all(owner=None))}

    def lora_detail(self, user, lora_id):
        row, _ = self.own_lora(user, lora_id)
        return 200, {"lora": lora_json(self, row, user, detail=True)}

    def lora_log(self, user, lora_id, query):
        row, training = self.own_lora(user, lora_id)
        try:
            lines = max(1, int((query.get("lines") or ["200"])[0]))
        except ValueError:
            raise ServiceError(400, "lines must be a whole number")
        if training is None:
            return 200, {"lines": [], "note": "this LoRA was not trained through the API, "
                                              "so there is no log of it here"}
        path = os.path.join(self.registry.training_folder(training), "train.log")
        if not os.path.exists(path):
            return 200, {"lines": []}
        with open(path, encoding="utf-8", errors="replace") as handle:
            return 200, {"lines": handle.read().splitlines()[-lines:]}

    def lora_dataset(self, user, lora_id, query):
        """The conversations a LoRA learned."""
        row, training = self.own_lora(user, lora_id)
        try:
            limit = max(1, min(500, int((query.get("limit") or ["20"])[0])))
        except ValueError:
            raise ServiceError(400, "limit must be a whole number")
        path = train.dataset_path(row, training, self.registry)
        if path is None:
            return 200, {"records": [], "total": 0,
                         "note": "no copy of its data is kept with this adapter"}
        return 200, dict(train.preview(path, limit),
                         source=row["dataset"], sha256=row["dataset_sha256"])

    def cancel_lora(self, user, lora_id):
        row, training = self.own_lora(user, lora_id)
        if training is None or row["status"] not in lora_catalog.IN_FLIGHT:
            raise ServiceError(409, "LoRA %d is %s; there is no training of it to stop"
                           % (lora_id, row["status"]))
        status = self.registry.request_training_cancel(training["id"])
        if status == reg.CANCELLED:
            self.catalog.update(lora_id, status=lora_catalog.CANCELLED,
                                error="cancelled before it started")
            note = "cancelled before it started"
        else:
            note = "the worker will stop it within a few seconds"
        return 202 if status == reg.RUNNING else 200, {
            "lora": lora_json(self, self.catalog.get(lora_id), user), "note": note}

    def delete_lora(self, user, lora_id):
        """A LoRA this user trained here: the adapter folder, the Trainer's
        scratch beside it, the training's own folder, and both rows."""
        row, training = self.own_lora(user, lora_id)
        if training is None:
            raise ServiceError(403, "LoRA %d was not trained through the API; its folder "
                                "is left to whoever put it there" % lora_id)
        if row["status"] in lora_catalog.IN_FLIGHT or training["status"] not in reg.FINISHED:
            raise ServiceError(409, "LoRA %d is still training; stop it first" % lora_id)
        slots = default_slots().get(os.path.normcase(lora_catalog.absolute(row["folder"])))
        if slots:
            raise ServiceError(409, "LoRA %d is the server's default %s; point LORA_SLOTS "
                                "elsewhere before deleting it" % (lora_id, ", ".join(slots)))
        folder = lora_catalog.absolute(row["folder"])
        for where in (folder, folder + "_outputs", self.registry.training_folder(training)):
            shutil.rmtree(where, ignore_errors=True)
        self.catalog.remove(lora_id)
        self.registry.delete_training(training["id"])
        return 200, {"deleted": lora_id}

    # --- inference ---------------------------------------------------------

    def open_stream(self, token, body):
        """-> (deployment row, generator already past its load). Raises ServiceError."""
        deployment = self.registry.deployment_for_token(token)
        if deployment is None:
            raise ServiceError(401, "unknown or revoked token")
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ServiceError(400, "prompt must be a non-empty string")
        max_new_tokens = body.get("max_new_tokens", settings.MAX_NEW_TOKENS)
        if (not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool)
                or not 1 <= max_new_tokens <= settings.MAX_NEW_TOKENS_LIMIT):
            raise ServiceError(400, "max_new_tokens must be between 1 and %d"
                           % settings.MAX_NEW_TOKENS_LIMIT)
        target = self.targets.get(deployment["target"])
        if target is None:
            raise ServiceError(503, "target %r is not available on this server"
                           % deployment["target"])
        pieces = target.stream(deployment["id"], json.loads(deployment["spec"]),
                               prompt, max_new_tokens)
        try:
            next(pieces)                   # the load and the build, before any header
        except StopIteration:
            pass
        except Exception as error:         # noqa: BLE001 - the client gets the reason
            pieces.close()
            raise ServiceError(500, "could not load deployment %d: %s: %s"
                           % (deployment["id"], type(error).__name__, error))
        return deployment, pieces
