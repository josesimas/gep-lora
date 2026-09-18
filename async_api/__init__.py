"""async_api - the search as a web service: submit a sweep, let a worker run
it, put what it found live, and stream answers from it.

    settings.py    the knobs this package reads (not the pipeline's)
    registry.py    api.sqlite: users, jobs in arrival order, trainings, live
                   deployments -- and the LoRAs (adapters/catalog.py's table)
    submit.py      a submission -> a *prepared* sweep database
    worker.py      the background process: oldest queued job -> main.py
                   (to search, resume or evaluate it), then trainings
                   (create_lora.py), then verifications
    results.py     a job's sweep database read back as JSON
    evaluate.py    a stopped job's answers graded again: the form and the flags
    verify.py      a finished job's blend beside the LoRAs it is made of
    train.py       a LoRA trained on the worker: the form, the checks, the command
    golive.py      an individual -> a self-contained blend spec, and the targets
    inference.py   the model cache and the streaming engines
    server.py      the HTTP surface over all of the above
    users.py       creating users and their API keys

See the async API section of README.md.
"""
