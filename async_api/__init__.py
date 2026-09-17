"""async_api - the search as a web service: submit a sweep, let a worker run
it, put what it found live, and stream answers from it.

    settings.py    the knobs this package reads (not the pipeline's)
    registry.py    jobs.sqlite3: users, jobs in arrival order, live deployments
    submit.py      a submission -> a *prepared* sweep database
    worker.py      the background process: oldest queued job -> main.py
    results.py     a job's sweep database read back as JSON
    golive.py      an individual -> a self-contained blend spec, and the targets
    inference.py   the model cache and the streaming engines
    server.py      the HTTP surface over all of the above
    users.py       creating users and their API keys

See the async API section of README.md.
"""
