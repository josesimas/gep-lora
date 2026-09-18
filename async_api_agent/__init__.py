"""async_api_agent - an AI agent that walks a user through training LoRAs,
served by the async API beside the demo page.

    settings.py    the knobs: which model talks, where, and what the agent plans
    prompts.py     every system prompt the agent sends, and the wording it falls
                   back to when no model answers -- the one place its words live
    providers.py   one chat call to any of the providers: OpenAI-compatible
                   endpoints (LM Studio, Ollama, OpenAI, Gemini, Mistral,
                   OpenRouter) and Anthropic's Messages API
    analysis.py    a pasted or uploaded dataset read, normalised and measured
    planner.py     "how long can you wait" -> epochs, an estimate, and the
                   POST /loras bodies that train them
    agent.py       the conversation: each step's facts, phrased by the model
    routes.py      the /agent endpoints the async API serves

The agent **proposes and the API does**: nothing here trains, queues or stores
anything. The page (async_api/agent-ui.html) takes the plan it is handed and
sends it to the async API's own POST /loras, then watches the LoRAs through
GET /loras/{id} and its log -- the same endpoints the demo page uses. The
agent's own endpoints only read, measure and talk.

And it **works without a model**. Every step's facts are computed here, not by
the model; the model only puts them into words. A provider that cannot be
reached, has no key, or refuses leaves the step answered in the fallback
wording from prompts.py, with a note saying so.
"""
