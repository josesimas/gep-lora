"""unittests.service - unit tests for the async API: the registry, submitting
a job, the worker, going live, the model cache and the HTTP surface.

Plain Python 3: jobs use the mocked template and adapters that are nothing but
an adapter_config.json, so no GPU, no base model and no judge are needed.

    python -m unittest discover -s unittests -t .
"""
