"""gep_lora - a Gene Expression Programming search over LoRA adapter blends.

Four layers, each importing only the ones below it:

    apps/       the user interfaces: web (the HTTP API and its pages); a
                desktop app next. Each only translates.
    assistant/  the guide: a chat model and its tools, UI-agnostic
    service/    the features, as plain Python: users, jobs, trainings,
                verifications, drawn blends, deployments (service.facade)
    core/       the engine: a sweep, from population to fitness, and the
                LoRAs it blends. Knows nothing of users, jobs or HTTP.

tools/ holds dev aids over core; paths.py is where the repo is.
unittests/test_layers.py holds the rule.
"""
