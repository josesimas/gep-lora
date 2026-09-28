"""
server.py - The HTTP surface of the async API.

    python -m gep_lora.apps.web.server                    # HOST:PORT from gep_lora/service/settings.py
    python -m gep_lora.apps.web.server --port 9000

Standard library only (http.server, the way gep_lora/core/blends/lora_server.py is), so the
API runs under any Python 3 for the job half. The inference half loads a model
in this process for a deployment of a real sweep, so start it with the venv's
python if anything real is going to go live.

Every endpoint but /health and /infer needs `Authorization: Bearer <API key>`
(create one with `python -m gep_lora.service.users add <name>`), and a user only ever
sees their own jobs, deployments and LoRAs -- another user's is a 404, not a 403.

Jobs
    POST   /jobs                           submit -> 201 {job}        (see submit.py)
    GET    /jobs[?status=done]             the user's jobs, newest first
    GET    /jobs/{id}                      the job and everything its run produced
    GET    /jobs/{id}/status               status, queue position, progress
    GET    /jobs/{id}/log[?lines=200]      the tail of gep_lora/core/pipeline/main.py's console output
    GET    /jobs/{id}/database             the job's sweep database (job<id>_<label>.sqlite3)
    GET    /jobs/{id}/individuals/{n}      one individual and its transcript
    POST   /jobs/{id}/cancel               cancel a queued job, or stop a running one
    POST   /jobs/{id}/label                {"label"} -> rename the job ("" clears it)
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

Blends drawn by hand (drawn.py; the pages are /visual_guide.html and
/blend_comparison.html)
    GET    /blends                        every blend of the user's a page can open,
                                          job by job: searched and drawn alike
    POST   /blends/check                  {tree, seed?, number?, pin?, values?} -> the drawing as a
                                          chromosome, its ranks, whether PEFT can build
                                          it, and what the weights are worth under the
                                          seed for individual `number` (1)
    POST   /blends/random                 {base_model?} -> {tree, seed, check}: a drawing
                                          grown the way a search grows an individual,
                                          over the user's LoRAs on one base model
    POST   /blends/test                   {tree, seed, number?, dataset, count?, label?,
                                          mock?, settings?} -> 201 {job, verification}: a
                                          sweep of that one blend, and a verification
                                          of it beside each of its LoRAs alone on the
                                          dataset -- read back with /verifications/{id}
    POST   /blends/code                   {tree, seed, number?, pin?, job?, count?, mock?}
                                          -> {name, source, exact, note}: the Python the
                                          drawing runs as when processed (drawn.code)
    POST   /blends/{job id}/save           {tree, seed, number?, pin?} -> 201: a drawing
                                          opened from that job and edited, saved into
                                          its run as a brand new individual, with the
                                          weights it was shown pinned (drawn.save)
    GET    /blends/{job id}[?individual=N] one blend of a job as a page draws it again:
                                          its tree, seed, number and latest verification;
                                          a search's best unless N is given

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
    GET    /runs                           every search of the user's, newest first, each with
                                           the LoRAs it blended, how many of its blends were
                                           tested, its verifications and what of it is live
    GET    /settings                       settings.py's values and the choices, for a form
    GET    /datasets                       shared dataset files a submission may name
Pages -- a page's address ends in .html, an endpoint's never does. Each is a
static file of the same name in static/, and every one of them draws the
same top bar from /nav.js. None needs a key to be served; what they show does.
    GET    /  (-> /guide.html)             where a beginner starts
    GET    /guide.html[?job=N]             the LoRA guide: trains LoRAs and blends them
    GET    /visual_guide.html              the visual guide: draw a blend of your LoRAs
                                           as a tree, and test it on a dataset
    GET    /blend_comparison.html          two blends side by side, each opened from a
                                           job or drawn, edited and tested on the same
                                           questions
    GET    /runs.html                      the user's runs (reads GET /runs)
    GET    /settings.html                  appearance, and the user's defaults for the
                                           guide (reads and writes /agent/defaults)
    GET    /console.html[?job=N]           the console: every endpoint above, by hand
    GET    /nav.js                         the top bar every page shares
    GET    /splitter.js                    the resizable left column, width kept per page
    /agent, /demo, /guide_defaults         the old addresses, redirected to the new

The LoRA agent (gep_lora/assistant/facade.py)
    GET    /agent/config, /agent/models    its providers, plan and demo datasets
    POST   /agent/{intro,analyse,wait,plan,started,debrief}
                                           one step of the conversation each; the
                                           page trains through POST /loras above
    POST   /agent/chat                     a typed message, answered and acted on with
                                           the agent's tools
    GET    /agent/defaults                 the user's defaults for a new conversation;
    PUT    /agent/defaults                 saved (DELETE: back to the server's)
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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from gep_lora.apps.web import agent_routes
from gep_lora.assistant.facade import AgentError
from gep_lora.service import registry as reg
from gep_lora.service import settings
from gep_lora.service.facade import App, FileReply, ServiceError

# The name this module has always raised under; the web app's word for it.
ApiError = ServiceError

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# The pages, by address: each is static/<name>, static and same-origin, so
# they need no CORS and hold no secrets; what they show comes from endpoints
# that need the user's key.
#   guide.html     the API driven by a guide that walks a user through training
#                  LoRAs and blending them (gep_lora/assistant/); ?job=N opens that
#                  search of the user's in it
#   visual_guide.html  a blend drawn by hand as a tree of the user's LoRAs, and
#                  tested (drawn.py), with the guide's chat beside it
#   blend_comparison.html  two blends side by side, from any jobs or drawn,
#                  edited and tested on the same questions (drawn.py again,
#                  twice), with its own guide (gep_lora/assistant/compare.py)
#   runs.html      every run of the user's, from GET /runs
#   settings.html  appearance, and what a new guide conversation starts from,
#                  read and saved through /agent/defaults
#   console.html   a page that exercises every endpoint; ?job=N opens that job
#   nav.js         the top bar every page draws
#   code_view.js   the overlay the two drawing pages show a blend's script in
#   splitter.js    the draggable left column of the guide, visual guide, compare and console
PAGES = {name: (os.path.join(HERE, name), "text/html; charset=utf-8")
         for name in ("guide.html", "visual_guide.html", "blend_comparison.html",
                      "runs.html", "settings.html", "console.html")}
PAGES["nav.js"] = (os.path.join(HERE, "nav.js"), "text/javascript; charset=utf-8")
PAGES["code_view.js"] = (os.path.join(HERE, "code_view.js"), "text/javascript; charset=utf-8")
PAGES["splitter.js"] = (os.path.join(HERE, "splitter.js"), "text/javascript; charset=utf-8")

# Where a page used to be, so a bookmark or an old link still lands. The query
# goes along, so /agent?job=3 is /guide.html?job=3.
MOVED = {"/": "/guide.html",
         "/agent": "/guide.html", "/agent-ui": "/guide.html", "/agent-ui.html": "/guide.html",
         "/demo": "/console.html",
         "/guide_defaults": "/settings.html", "/guide_defaults.html": "/settings.html"}

# The pages' themes, served at /themes/<file>: themes.js picks one and each is a
# stylesheet over the pages' own colours. A flat folder of static files, so a
# name is one path segment of a known type and nothing else is served.
THEMES_DIR = os.path.join(HERE, "themes")
THEME_TYPES = {".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8",
               ".svg": "image/svg+xml"}


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
    ("POST", r"/jobs/(\d+)/label", "rename_job", ("body",)),
    ("GET", r"/jobs/(\d+)/evaluate", "evaluate_form", ()),
    ("POST", r"/jobs/(\d+)/evaluate", "start_evaluation", ("body",)),
    ("GET", r"/jobs/(\d+)/test", "test_form", ()),
    ("POST", r"/jobs/(\d+)/test", "start_test", ("body",)),
    ("DELETE", r"/jobs/(\d+)/run", "delete_run", ()),
    ("DELETE", r"/jobs/(\d+)", "delete_job", ()),
    ("GET", r"/jobs/(\d+)/verify", "verify_form", ()),
    ("POST", r"/jobs/(\d+)/verify", "start_verification", ("body",)),
    ("POST", r"/blends/check", "check_drawn", ("body",)),
    ("POST", r"/blends/test", "test_drawn", ("body",)),
    ("POST", r"/blends/random", "random_drawn", ("body",)),
    ("POST", r"/blends/code", "drawn_code", ("body",)),
    ("GET", r"/blends", "list_blends", ()),
    ("GET", r"/blends/(\d+)", "open_drawn", ("query",)),
    ("POST", r"/blends/(\d+)/save", "save_drawn", ("body",)),
    ("GET", r"/verifications/(\d+)", "verification_detail", ()),
    ("GET", r"/verifications/(\d+)/log", "verification_log", ("query",)),
    ("POST", r"/jobs/(\d+)/live", "set_live", ("body",)),
    ("GET", r"/jobs/(\d+)/live", "job_live", ()),
    ("DELETE", r"/jobs/(\d+)/live", "unset_job", ()),
    ("GET", r"/runs", "list_runs", ()),
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
            if self.command == "GET" and path in MOVED:
                return self._redirect(MOVED[path] + ("?" + url.query if url.query else ""))
            if self.command == "GET" and path[1:] in PAGES:
                return self._page(*PAGES[path[1:]])
            theme = re.fullmatch(r"/themes/([A-Za-z0-9_-]+(\.[a-z]+))", path)
            if theme and self.command == "GET" and theme.group(2) in THEME_TYPES:
                return self._page(os.path.join(THEMES_DIR, theme.group(1)), THEME_TYPES[theme.group(2)])
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
        except AgentError as error:
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

    def _redirect(self, location):
        self.send_response(302)          # not 301: a browser would keep that for good
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _page(self, path, content_type="text/html; charset=utf-8"):
        try:
            with open(path, "rb") as handle:
                body = handle.read()
        except FileNotFoundError:
            raise ApiError(404, "no such file")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # Asked again every time: a page and its scripts change together, and a
        # browser holding an older nav.js or code_view.js beside a newer page
        # draws a mixture of the two.
        self.send_header("Cache-Control", "no-cache")
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
