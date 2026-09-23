"""
server.py - The HTTP surface of the async API.

    python -m async_api.server                    # HOST:PORT from async_api/settings.py
    python -m async_api.server --port 9000

Standard library only (http.server, the way blends/lora_server.py is), so the
API runs under any Python 3 for the job half. The inference half loads a model
in this process for a deployment of a real sweep, so start it with the venv's
python if anything real is going to go live.

Every endpoint but /health and /infer needs `Authorization: Bearer <API key>`
(create one with `python -m async_api.users add <name>`), and a user only ever
sees their own jobs, deployments and LoRAs -- another user's is a 404, not a 403.

Jobs
    POST   /jobs                           submit -> 201 {job}        (see submit.py)
    GET    /jobs[?status=done]             the user's jobs, newest first
    GET    /jobs/{id}                      the job and everything its run produced
    GET    /jobs/{id}/status               status, queue position, progress
    GET    /jobs/{id}/log[?lines=200]      the tail of main.py's console output
    GET    /jobs/{id}/database             the job's sweep database (job<id>_<label>.sqlite3)
    GET    /jobs/{id}/individuals/{n}      one individual and its transcript
    POST   /jobs/{id}/cancel               cancel a queued job, or stop a running one
    POST   /jobs/{id}/resume               queue a stopped, cancelled or failed job
                                           again, to carry on from where it got to
    GET    /jobs/{id}/evaluate             what an evaluation of its answers would do
    POST   /jobs/{id}/evaluate             {"force"?, "judge_backend"?, "judge_model"?,
                                           "judge_base_url"?} -> queue it to grade the
                                           answers it holds (see evaluate.py)
    GET    /jobs/{id}/test                 what a testing pass would do, and the
                                           testing results so far
    POST   /jobs/{id}/test                 {"min_quality"?, "count"?, "limit"?} -> queue a
                                           finished job to test its blends -- every one
                                           that ran, by default -- on its testing split
                                           (see testpass.py)
    DELETE /jobs/{id}/run                  delete what the run produced; keep the job
    DELETE /jobs/{id}                      delete the job and its files

Verification
    GET    /jobs/{id}/verify              what a verification may ask for, and
                                          the job's verifications so far
    POST   /jobs/{id}/verify              {"individual": n?, "evaluator"?, "judge_model"?,
                                          "judge_backend"?, "judge_base_url"?, "split"?,
                                          "count"?, "slots"?, "dataset"?} -> 201 {verification};
                                          dataset is {"file": a shared dataset} or
                                          {"lora": id}, that LoRA's training data
    GET    /verifications/{id}            one verification: its status, and its
                                          report once it has one
    GET    /verifications/{id}/log        the tail of its console output

Going live
    POST   /jobs/{id}/live                 {"individual": n?, "target": "local"?}
                                           -> 201 {token, deployment}; best by default
    GET    /jobs/{id}/live                 the job's live deployments
    DELETE /jobs/{id}/live                 unset every deployment of the job
    GET    /live                           the user's live deployments
    DELETE /live/{deployment id}           unset one

Inference
    POST   /infer                          {"token", "prompt", "max_new_tokens"?}
                                           -> the answer, streamed (chunked text/plain,
                                              or server-sent events with
                                              Accept: text/event-stream)
    GET    /health                         liveness, and what is loaded

LoRAs (the user's own rows of the catalogue a search draws its slots from)
    GET    /loras[?base_model=]            the user's LoRAs
    GET    /loras/form                     what a training may ask for: defaults, choices, the
                                           base models there are LoRAs for or downloads of
    POST   /loras                          {"name", "dataset", "settings"?, "mock"?}
                                           -> 201 {lora, training}; queued like a job (see train.py)
    POST   /loras/scan                     re-read loras/ into the catalogue; reports
                                           on the user's own rows
    GET    /loras/{id}                     one LoRA: its record, loss history and progress
    GET    /loras/{id}/log[?lines=200]     the tail of its training's console
    GET    /loras/{id}/dataset[?limit=20]  the conversations it was trained on
    POST   /loras/{id}/cancel              cancel a queued training, or stop a running one
    DELETE /loras/{id}                     a LoRA you trained here: its folder and its row

Other
    GET    /judge/models[?base_url=]       the chat models a judge endpoint lists (default:
                                           settings.py's JUDGE_BASE_URL), asked from here
    GET    /settings                       settings.py's values and the choices, for a form
    GET    /datasets                       shared dataset files a submission may name
    GET    /  or  /demo                    a test page that drives all of the above

The LoRA agent (async_api_agent/routes.py)
    GET    /agent                          the agent page: a guide that trains LoRAs
    GET    /agent/config, /agent/models    its providers, plan and demo datasets
    POST   /agent/{intro,analyse,wait,plan,started,debrief}
                                           one step of the conversation each; the
                                           page trains through POST /loras above
    POST   /agent/chat                     a typed message, answered and acted on with
                                           the agent's tools
    POST   /agent/blend/{intro,plan,started,debrief}
                                           the second half: the user's own LoRAs blended;
                                           the page submits the plan to POST /jobs above
"""

import argparse
import functools
import json
import os
import re
import shutil
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from async_api import evaluate
from async_api import golive
from async_api import inference
from async_api import registry as reg
from async_api import results
from async_api import settings
from async_api import submit
from async_api import testpass
from async_api import train
from async_api import verify
from async_api_agent import routes as agent_routes
from adapters import catalog as lora_catalog
from config import settings as config


# A page that exercises every endpoint, served at / and /demo. Same origin as
# the API, so it needs no CORS; it holds no secrets of its own.
DEMO_PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo.html")

# The agent page, served at /agent: the same API driven by a guide that walks a
# user through training LoRAs (async_api_agent/). Same origin, same key.
AGENT_PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent-ui.html")


class FileReply:
    """A download: a file on disk, sent as an attachment and then removed."""

    def __init__(self, path, filename, content_type="application/octet-stream"):
        self.path = path
        self.filename = filename
        self.content_type = content_type


class ApiError(Exception):
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
    out["can_evaluate"] = job["status"] in reg.REQUEUE_FROM[reg.EVALUATE]
    out["can_test"] = job["status"] in reg.REQUEUE_FROM[reg.TEST]
    if job["status"] == reg.QUEUED:
        out["queue_position"] = app.registry.queue_position(job["id"])
    if with_summary and job["status"] != reg.DELETED:
        out["summary"] = results.summary(app.registry.database(job), job["run_id"])
    return out


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
    for slot, where in sorted(config.LORA_SLOTS.items()):
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
            raise ApiError(404, "no job %d" % job_id)
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
            raise ApiError(400, str(error))
        return 201, {"job": job_json(self, job)}

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
            raise ApiError(404, "job %d has no database; its run was deleted" % job_id)
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
            raise ApiError(400, "lines must be a whole number")
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
            raise ApiError(404, "job %d holds no individual %d" % (job_id, number))
        return 200, {"individual": found}

    def cancel_job(self, user, job_id):
        """Cancel a queued job, or stop a running one.

        A running search is *stopped* rather than ended: the worker kills
        main.py, and what it had done stays in the sweep database for a resume
        to carry on from. A resume or an evaluation cancelled before the worker
        got to it puts the job back as it was.
        """
        job = self.own_job(user, job_id)
        before = job["status"]
        status = self.registry.request_cancel(job["id"])
        if before not in (reg.QUEUED, reg.PREPARING, reg.RUNNING):
            raise ApiError(409, "job %d is %s; there is nothing to cancel" % (job_id, status))
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
            raise ApiError(409, "job %d has no database; its run was deleted" % job["id"])

    def resume_job(self, user, job_id):
        """Queue a search that did not finish, to carry on from where it got to."""
        job = self.own_job(user, job_id)
        if job["status"] not in reg.RESUMABLE:
            raise ApiError(409, "job %d is %s; only a stopped, cancelled or failed job "
                                "can be resumed" % (job_id, job["status"]))
        self._has_database(job)
        queued = self.registry.requeue(job_id, reg.RESUME)
        if queued is None:
            raise ApiError(409, "job %d changed status while being resumed" % job_id)
        return 200, {"job": job_json(self, queued),
                     "note": "queued; the worker carries it on from where it stopped"}

    def _grading(self, job):
        """What an evaluation of this job's answers is offered."""
        try:
            return evaluate.choices(self.registry.database(job), job["run_id"])
        except results.NoResults:
            raise ApiError(409, "job %d has no database; its run was deleted" % job["id"])

    def evaluate_form(self, user, job_id):
        job = self.own_job(user, job_id)
        out = self._grading(job)
        out["can_evaluate"] = job["status"] in reg.REQUEUE_FROM[reg.EVALUATE]
        return 200, out

    def start_evaluation(self, user, job_id, body):
        """Queue the job to grade the answers it already holds."""
        job = self.own_job(user, job_id)
        if job["status"] not in reg.REQUEUE_FROM[reg.EVALUATE]:
            raise ApiError(409, "job %d is %s; wait for it to stop before evaluating "
                                "its answers" % (job_id, job["status"]))
        try:
            options = evaluate.options_for(body, self._grading(job))
        except evaluate.EvaluateError as error:
            raise ApiError(400, str(error))
        queued = self.registry.requeue(job_id, reg.EVALUATE, options)
        if queued is None:
            raise ApiError(409, "job %d changed status while being queued" % job_id)
        return 200, {"job": job_json(self, queued),
                     "note": "queued; the worker grades its answers after any job "
                             "ahead of it"}

    def _testing(self, job):
        """What a testing pass of this job is offered."""
        try:
            return testpass.choices(self.registry.database(job), job["run_id"])
        except results.NoResults:
            raise ApiError(409, "job %d has no database; its run was deleted" % job["id"])

    def test_form(self, user, job_id):
        job = self.own_job(user, job_id)
        out = self._testing(job)
        out["can_test"] = job["status"] in reg.REQUEUE_FROM[reg.TEST]
        return 200, out

    def start_test(self, user, job_id, body):
        """Queue a finished job to put its blends in front of its testing split."""
        job = self.own_job(user, job_id)
        if job["status"] not in reg.REQUEUE_FROM[reg.TEST]:
            raise ApiError(409, "job %d is %s; only a finished search's blends can be "
                                "tested" % (job_id, job["status"]))
        try:
            options = testpass.options_for(body, self._testing(job))
        except testpass.TestPassError as error:
            raise ApiError(400, str(error))
        queued = self.registry.requeue(job_id, reg.TEST, options)
        if queued is None:
            raise ApiError(409, "job %d changed status while being queued" % job_id)
        return 200, {"job": job_json(self, queued),
                     "note": "queued; the worker tests its blends after any job "
                             "ahead of it"}

    def _finished(self, job, what):
        if job["status"] not in reg.FINISHED:
            raise ApiError(409, "job %d is %s; cancel it before deleting %s"
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
            raise ApiError(404, "job %d has no database to verify against" % job["id"])

    def verify_form(self, user, job_id):
        job = self.own_job(user, job_id)
        if not os.path.exists(self.registry.database(job)):
            raise ApiError(404, "job %d has no database; its run was deleted" % job_id)
        out = self._offered(job)
        out["verifications"] = [verification_json(row) for row
                                in self.registry.verifications(job_id=job_id)]
        # Another dataset a verification may ask instead of a split: a shared
        # file by name, or {"lora": id} for one of the user's own LoRAs' data.
        out["datasets"] = self.list_datasets(user)[1]["datasets"]
        out["can_verify"] = job["status"] == reg.DONE
        return 200, out

    def start_verification(self, user, job_id, body):
        """Queue one blend of this job against the LoRAs it is made of."""
        job = self.own_job(user, job_id)
        if job["status"] != reg.DONE:
            raise ApiError(409, "job %d is %s; only a finished job can be verified"
                           % (job_id, job["status"]))
        offered = self._offered(job)
        try:
            number, options = verify.options_for(
                body, offered, functools.partial(self.verify_dataset, user))
        except verify.VerifyError as error:
            raise ApiError(400, str(error))
        chromosome = next(one["chromosome"] for one in offered["individuals"]
                          if one["number"] == number)
        row = self.registry.add_verification(job, number, chromosome, options)
        return 201, {"verification": verification_json(row),
                     "note": "queued; the worker runs it after any queued job"}

    def verify_dataset(self, user, value):
        """A verification's `dataset`, as (path, label): {"file": name}, a file
        under SHARED_DATASETS_DIR, or {"lora": id}, the data one of the user's
        own LoRAs was trained on -- another user's is refused as a missing one
        is. Raises verify.VerifyError."""
        if isinstance(value, dict) and set(value) == {"file"}:
            path = submit.shared_path(value["file"])
            if path is None:
                raise verify.VerifyError("no shared dataset %r" % (value["file"],))
            return path, value["file"]
        lora_id = (value.get("lora") if isinstance(value, dict) and set(value) == {"lora"}
                   else None)
        if not isinstance(lora_id, int) or isinstance(lora_id, bool):
            raise verify.VerifyError('dataset must be {"file": name} or {"lora": id}')
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
            raise ApiError(404, "no verification %d" % verification_id)
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
            raise ApiError(400, "lines must be a whole number")
        path = os.path.join(self.registry.verification_folder(job, row["id"]), "verify.log")
        if not os.path.exists(path):
            return 200, {"lines": []}
        with open(path, encoding="utf-8", errors="replace") as handle:
            return 200, {"lines": handle.read().splitlines()[-lines:]}

    # --- live --------------------------------------------------------------

    def set_live(self, user, job_id, body):
        job = self.own_job(user, job_id)
        if job["status"] != reg.DONE:
            raise ApiError(409, "job %d is %s; only a finished job can go live"
                           % (job_id, job["status"]))
        body = body or {}
        target_name = body.get("target") or "local"
        target = self.targets.get(target_name)
        if target is None:
            raise ApiError(400, "unknown target %r; there are: %s"
                           % (target_name, ", ".join(sorted(self.targets))))
        number = body.get("individual")
        if number is not None and (not isinstance(number, int) or isinstance(number, bool)):
            raise ApiError(400, "individual must be an individual's number")
        try:
            spec = golive.blend_spec(self.registry.database(job), job["run_id"], number)
        except golive.GoLiveError as error:
            raise ApiError(409, str(error))
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
            raise ApiError(400, "base_url must be an http(s) URL, not %r" % base_url)
        import evaluators                   # the judge transport, loaded when first asked
        try:
            models = evaluators.common.list_models(
                base_url, evaluators.API_KEY, settings.JUDGE_MODELS_TIMEOUT)
        except SystemExit as error:
            raise ApiError(502, str(error), base_url=base_url)
        return 200, {"base_url": base_url, "models": models}

    def list_live(self, user):
        return 200, {"live": [deployment_json(row)
                              for row in self.registry.deployments(user_id=user["id"])]}

    def unset_one(self, user, deployment_id):
        row = self.registry.deployment(deployment_id, user["id"])
        if row is None or row["revoked_at"]:
            raise ApiError(404, "no live deployment %d" % deployment_id)
        return 200, {"unset": self.unset([row])}

    # --- LoRAs -------------------------------------------------------------

    def own_lora(self, user, lora_id):
        """-> (the user's catalogue row, the training of theirs that made it,
        or None). Another user's row is a 404, exactly as a missing one is."""
        row = self.catalog.get(lora_id)
        if row is None or row["owner"] != user["name"]:
            raise ApiError(404, "no LoRA %d" % lora_id)
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
            raise ApiError(error.status, str(error))
        return 201, {"lora": lora_json(self, row, user),
                     "training": training_json(self.registry, training),
                     "note": "queued; the worker trains it after any queued job"}

    def scan_loras(self, user):
        """Re-read loras/. What is said back is about the user's own rows: a
        folder found for the first time belongs to nobody, and becomes someone's
        only through `python -m adapters.catalog own` on the server."""
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
            raise ApiError(400, "lines must be a whole number")
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
            raise ApiError(400, "limit must be a whole number")
        path = train.dataset_path(row, training, self.registry)
        if path is None:
            return 200, {"records": [], "total": 0,
                         "note": "no copy of its data is kept with this adapter"}
        return 200, dict(train.preview(path, limit),
                         source=row["dataset"], sha256=row["dataset_sha256"])

    def cancel_lora(self, user, lora_id):
        row, training = self.own_lora(user, lora_id)
        if training is None or row["status"] not in lora_catalog.IN_FLIGHT:
            raise ApiError(409, "LoRA %d is %s; there is no training of it to stop"
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
            raise ApiError(403, "LoRA %d was not trained through the API; its folder "
                                "is left to whoever put it there" % lora_id)
        if row["status"] in lora_catalog.IN_FLIGHT or training["status"] not in reg.FINISHED:
            raise ApiError(409, "LoRA %d is still training; stop it first" % lora_id)
        slots = default_slots().get(os.path.normcase(lora_catalog.absolute(row["folder"])))
        if slots:
            raise ApiError(409, "LoRA %d is the server's default %s; point LORA_SLOTS "
                                "elsewhere before deleting it" % (lora_id, ", ".join(slots)))
        folder = lora_catalog.absolute(row["folder"])
        for where in (folder, folder + "_outputs", self.registry.training_folder(training)):
            shutil.rmtree(where, ignore_errors=True)
        self.catalog.remove(lora_id)
        self.registry.delete_training(training["id"])
        return 200, {"deleted": lora_id}

    # --- inference ---------------------------------------------------------

    def open_stream(self, token, body):
        """-> (deployment row, generator already past its load). Raises ApiError."""
        deployment = self.registry.deployment_for_token(token)
        if deployment is None:
            raise ApiError(401, "unknown or revoked token")
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ApiError(400, "prompt must be a non-empty string")
        max_new_tokens = body.get("max_new_tokens", settings.MAX_NEW_TOKENS)
        if (not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool)
                or not 1 <= max_new_tokens <= settings.MAX_NEW_TOKENS_LIMIT):
            raise ApiError(400, "max_new_tokens must be between 1 and %d"
                           % settings.MAX_NEW_TOKENS_LIMIT)
        target = self.targets.get(deployment["target"])
        if target is None:
            raise ApiError(503, "target %r is not available on this server"
                           % deployment["target"])
        pieces = target.stream(deployment["id"], json.loads(deployment["spec"]),
                               prompt, max_new_tokens)
        try:
            next(pieces)                   # the load and the build, before any header
        except StopIteration:
            pass
        except Exception as error:         # noqa: BLE001 - the client gets the reason
            pieces.close()
            raise ApiError(500, "could not load deployment %d: %s: %s"
                           % (deployment["id"], type(error).__name__, error))
        return deployment, pieces


ROUTES = [
    ("POST", r"/jobs", "submit_job", ("body",)),
    ("GET", r"/jobs", "list_jobs", ("query",)),
    ("GET", r"/jobs/(\d+)", "job_detail", ()),
    ("GET", r"/jobs/(\d+)/status", "job_status", ()),
    ("GET", r"/jobs/(\d+)/log", "job_log", ("query",)),
    ("GET", r"/jobs/(\d+)/database", "job_database", ()),
    ("GET", r"/jobs/(\d+)/individuals/(\d+)", "job_individual", ()),
    ("POST", r"/jobs/(\d+)/cancel", "cancel_job", ()),
    ("POST", r"/jobs/(\d+)/resume", "resume_job", ()),
    ("GET", r"/jobs/(\d+)/evaluate", "evaluate_form", ()),
    ("POST", r"/jobs/(\d+)/evaluate", "start_evaluation", ("body",)),
    ("GET", r"/jobs/(\d+)/test", "test_form", ()),
    ("POST", r"/jobs/(\d+)/test", "start_test", ("body",)),
    ("DELETE", r"/jobs/(\d+)/run", "delete_run", ()),
    ("DELETE", r"/jobs/(\d+)", "delete_job", ()),
    ("GET", r"/jobs/(\d+)/verify", "verify_form", ()),
    ("POST", r"/jobs/(\d+)/verify", "start_verification", ("body",)),
    ("GET", r"/verifications/(\d+)", "verification_detail", ()),
    ("GET", r"/verifications/(\d+)/log", "verification_log", ("query",)),
    ("POST", r"/jobs/(\d+)/live", "set_live", ("body",)),
    ("GET", r"/jobs/(\d+)/live", "job_live", ()),
    ("DELETE", r"/jobs/(\d+)/live", "unset_job", ()),
    ("GET", r"/settings", "submission_form", ()),
    ("GET", r"/datasets", "list_datasets", ()),
    ("GET", r"/judge/models", "judge_models", ("query",)),
    ("GET", r"/loras", "list_loras", ("query",)),
    ("POST", r"/loras", "create_lora", ("body",)),
    ("GET", r"/loras/form", "lora_form", ()),
    ("POST", r"/loras/scan", "scan_loras", ()),
    ("GET", r"/loras/(\d+)", "lora_detail", ()),
    ("GET", r"/loras/(\d+)/log", "lora_log", ("query",)),
    ("GET", r"/loras/(\d+)/dataset", "lora_dataset", ("query",)),
    ("POST", r"/loras/(\d+)/cancel", "cancel_lora", ()),
    ("DELETE", r"/loras/(\d+)", "delete_lora", ()),
    ("GET", r"/live", "list_live", ()),
    ("DELETE", r"/live/(\d+)", "unset_one", ()),
] + agent_routes.ROUTES          # a function, not an App method name: see _dispatch


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    app = None                                   # bound by make_server()
    quiet = False
    _unread = 0                                  # the request body's bytes not yet read

    def log_message(self, fmt, *args):
        if not self.quiet:
            sys.stderr.write("[api] %s - %s\n" % (self.address_string(), fmt % args))

    # --- plumbing ----------------------------------------------------------

    def _send(self, status, payload):
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        length = self._unread
        if length > settings.MAX_BODY_BYTES:
            raise ApiError(413, "request body over %d bytes" % settings.MAX_BODY_BYTES)
        if not length:
            return {}
        raw = self.rfile.read(length)
        self._unread = 0
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as error:
            raise ApiError(400, "bad JSON: %s" % error)

    def _bearer(self):
        header = self.headers.get("Authorization") or ""
        return header[7:].strip() if header.lower().startswith("bearer ") else None

    def _dispatch(self):
        """One request, and then whatever of its body nobody read.

        The connection is kept alive (HTTP/1.1), so a reply sent before the
        body was read -- an unknown route, a wrong method, a bad key -- would
        leave that body on the socket, to be read as the start of the next
        request ("400 Bad request syntax ('{...}POST /...')"). So it is drained
        here, whatever the route did; one too big to read is not, and the
        connection is closed instead."""
        try:
            self._unread = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._unread, self.close_connection = 0, True
        try:
            self._route()
        finally:
            self._drain()

    def _drain(self):
        left, self._unread = self._unread, 0
        if left > settings.MAX_BODY_BYTES:
            self.close_connection = True
            return
        while left > 0:
            chunk = self.rfile.read(min(left, 1 << 16))
            if not chunk:
                break
            left -= len(chunk)

    def _route(self):
        url = urlparse(self.path)
        path = url.path.rstrip("/") or "/"
        try:
            if path in ("/", "/demo") and self.command == "GET":
                return self._page(DEMO_PAGE)
            if path in ("/agent", "/agent-ui", "/agent-ui.html") and self.command == "GET":
                return self._page(AGENT_PAGE)
            if path == "/health" and self.command == "GET":
                return self._send(200, {"ok": True, "models": self.app.cache.status()})
            if path == "/infer" and self.command == "POST":
                return self._infer()
            matched_path = False
            for method, pattern, name, extras in ROUTES:
                found = re.fullmatch(pattern, path)
                if not found:
                    continue
                matched_path = True
                if method != self.command:
                    continue
                user = self.app.registry.user_for_key(self._bearer())
                if user is None:
                    raise ApiError(401, "missing or unknown API key")
                args = [user] + [int(group) for group in found.groups()]
                if "body" in extras:
                    args.append(self._body())
                if "query" in extras:
                    args.append(parse_qs(url.query))
                handler = (getattr(self.app, name) if isinstance(name, str)
                           else functools.partial(name, self.app))
                status, payload = handler(*args)
                if isinstance(payload, FileReply):
                    return self._file(payload)
                return self._send(status, payload)
            if matched_path:
                raise ApiError(405, "%s is not allowed on %s" % (self.command, path))
            raise ApiError(404, "no such endpoint: %s %s" % (self.command, path))
        except ApiError as error:
            self._send(error.status, error.payload)
        except agent_routes.AgentError as error:
            self._send(error.status, {"error": str(error)})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:                    # noqa: BLE001 - one request fails
            self._send(500, {"error": "%s: %s" % (type(error).__name__, error)})

    do_GET = do_POST = do_DELETE = do_PUT = do_PATCH = _dispatch

    def _file(self, reply):
        try:
            self.send_response(200)
            self.send_header("Content-Type", reply.content_type)
            self.send_header("Content-Length", str(os.path.getsize(reply.path)))
            self.send_header("Content-Disposition", 'attachment; filename="%s"' % reply.filename)
            self.end_headers()
            with open(reply.path, "rb") as handle:
                shutil.copyfileobj(handle, self.wfile, 1024 * 1024)
        finally:
            os.remove(reply.path)

    def _page(self, path):
        with open(path, "rb") as handle:
            body = handle.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # --- streaming ---------------------------------------------------------

    def _chunk(self, text):
        data = text.encode("utf-8")
        if data:
            self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
            self.wfile.flush()

    def _infer(self):
        body = self._body()
        token = body.get("token") or self._bearer()
        deployment, pieces = self.app.open_stream(token, body)
        events = "text/event-stream" in (self.headers.get("Accept") or "")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream" if events
                         else "text/plain; charset=utf-8")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Deployment", str(deployment["id"]))
        self.end_headers()
        try:
            try:
                for piece in pieces:
                    if not piece:
                        continue
                    self._chunk("data: %s\n\n" % json.dumps({"text": piece}) if events
                                else piece)
                if events:
                    self._chunk("event: done\ndata: {}\n\n")
            except (BrokenPipeError, ConnectionResetError):
                raise
            except Exception as error:                # noqa: BLE001 - headers are gone
                message = "%s: %s" % (type(error).__name__, error)
                self._chunk("event: error\ndata: %s\n\n" % json.dumps({"error": message})
                            if events else "\n[error: %s]" % message)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        finally:
            pieces.close()


class Server(ThreadingHTTPServer):
    # http.server sets SO_REUSEADDR, which on Windows lets a second process bind
    # a port another is still listening on -- no error, and connections then go
    # to either. A restart that left the old server running would serve half its
    # requests from the old code. So on Windows a taken port is an error.
    allow_reuse_address = os.name != "nt"


def make_server(app=None, host=None, port=None, quiet=False):
    app = app or App()
    handler = type("BoundHandler", (Handler,), {"app": app, "quiet": quiet})
    server = Server((host or settings.HOST,
                                  settings.PORT if port is None else port), handler)
    server.daemon_threads = True
    server.app = app
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description="The GEP LoRA async API.")
    parser.add_argument("--host", default=settings.HOST)
    parser.add_argument("--port", type=int, default=settings.PORT)
    parser.add_argument("--jobs-dir", default=None,
                        help="the registry folder (default %s)" % settings.JOBS_DIR)
    args = parser.parse_args(argv)
    try:
        server = make_server(App(reg.Registry(args.jobs_dir)), args.host, args.port)
    except OSError as error:
        print("[api] cannot listen on %s:%d (%s) -- is another API server still running? "
              "Stop it first." % (args.host, args.port, error), file=sys.stderr, flush=True)
        return 1
    print("[api] listening on http://%s:%d, jobs in %s"
          % (args.host, server.server_address[1], server.app.registry.root), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.app.cache.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
