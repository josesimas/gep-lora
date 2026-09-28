"""assistant - an AI agent that walks a user through training LoRAs and
then combining them, served by the async API beside the demo page.

    settings.py    the knobs: which model talks, where, and what the agent plans
    prompts.py     every system prompt the agent sends, every tool's description,
                   and the wording it falls back to when no model answers -- the
                   one place its words live
    providers.py   one chat call to any of the providers, with or without tools:
                   OpenAI-compatible endpoints (LM Studio, Ollama, OpenAI,
                   Gemini, Mistral, OpenRouter) and Anthropic's Messages API
    analysis.py    a pasted or uploaded dataset read, normalised and measured
    selection.py   which part of the dataset is trained on
    planner.py     the session (what the chat has set up), "how long can you
                   wait" -> epochs, an estimate, and the POST /loras bodies
    blending.py    the second half: the user's own LoRAs combined -- which,
                   the search's size, where its questions come from, the
                   POST /jobs body, and what the search found
    tools.py       what the chat can do: the tools the model is given
    commands.py    the common requests read without a model, as tool calls
    agent.py       the conversation: each step's facts, phrased by the model,
                   and the chat's tool loop
    ui_help.py     "what is this?" for a block of the page, from what it shows
    create_summary.py  the journey so far, as a short illustrated story the
                   page shows in an overlay and prints to a PDF
    routes.py      the /agent endpoints the async API serves

The agent **proposes and the API does**: nothing here trains, queues or stores
anything. The page (gep_lora/apps/web/static/guide.html) takes the plan it is handed and
sends it to the async API's own POST /loras, then watches the LoRAs through
GET /loras/{id} and its log -- the same endpoints the demo page uses. The
blend is the same: a POST /jobs body naming the user's own LoRAs, sent by the
page and watched through GET /jobs/{id}/status. The
chat's tools change the plan -- the session the page keeps -- and hand back
actions (start, stop, another dataset) that the page carries out with those
same calls.

And it **works without a model**. Every step's facts are computed here, not by
the model; the model only puts them into words. A provider that cannot be
reached, has no key, or refuses leaves the step answered in the fallback
wording from prompts.py, with a note saying so, and the chat's common requests
are still carried out (commands.py).
"""
