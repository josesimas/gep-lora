"""
agent_routes.py - Where the guide's features sit in the web app: /agent/*.

The handlers are gep_lora/assistant/facade.py's -- every address, and what
each takes and returns, is described there. This is only the table:
(method, path pattern, handler, extras), the shape of server.ROUTES with a
function where the server's own routes name an App method, each run under the
user's saved defaults (facade.theirs).
"""

from gep_lora.assistant import facade
from gep_lora.assistant.facade import theirs

ROUTES = [
    ("GET", r"/agent/config", facade.config, ()),
    ("GET", r"/agent/defaults", facade.defaults, ()),
    ("PUT", r"/agent/defaults", facade.save_defaults, ("body",)),
    ("DELETE", r"/agent/defaults", facade.reset_defaults, ()),
    ("GET", r"/agent/models", facade.models, ("query",)),
    ("POST", r"/agent/intro", facade.intro, ("body",)),
    ("POST", r"/agent/analyse", facade.analyse, ("body",)),
    ("POST", r"/agent/wait", facade.wait, ("body",)),
    ("POST", r"/agent/plan", facade.plan, ("body",)),
    ("POST", r"/agent/started", facade.started, ("body",)),
    ("POST", r"/agent/chat", facade.chat, ("body",)),
    ("POST", r"/agent/debrief", facade.debrief, ("body",)),
    ("GET", r"/agent/help", facade.help_blocks, ()),
    ("POST", r"/agent/help", facade.explain, ("body",)),
    ("POST", r"/agent/blend/intro", facade.blend_intro, ("body",)),
    ("POST", r"/agent/blend/plan", facade.blend_plan, ("body",)),
    ("POST", r"/agent/blend/started", facade.blend_started, ("body",)),
    ("POST", r"/agent/blend/debrief", facade.blend_debrief, ("body",)),
    ("POST", r"/agent/test/started", facade.test_started, ("body",)),
    ("POST", r"/agent/test/debrief", facade.test_debrief, ("body",)),
    ("POST", r"/agent/verify/plan", facade.verify_plan, ("body",)),
    ("POST", r"/agent/verify/debrief", facade.verify_debrief, ("body",)),
    ("POST", r"/agent/live/started", facade.live_started, ("body",)),
    ("POST", r"/agent/summary", facade.summary, ("body",)),
    ("GET", r"/agent/visual/config", facade.visual_config, ()),
    ("POST", r"/agent/visual/intro", facade.visual_intro, ("body",)),
    ("POST", r"/agent/visual/chat", facade.visual_chat, ("body",)),
    ("POST", r"/agent/visual/plan", facade.visual_plan, ("body",)),
    ("POST", r"/agent/visual/debrief", facade.visual_debrief, ("body",)),
    ("GET", r"/agent/compare/config", facade.compare_config, ()),
    ("POST", r"/agent/compare/intro", facade.compare_intro, ("body",)),
    ("POST", r"/agent/compare/chat", facade.compare_chat, ("body",)),
    ("POST", r"/agent/compare/plan", facade.compare_plan, ("body",)),
    ("POST", r"/agent/compare/outcome", facade.compare_outcome, ("body",)),
    ("POST", r"/agent/compare/debrief", facade.compare_debrief, ("body",)),
]
ROUTES = [(method, pattern, theirs(handler), extras) for method, pattern, handler, extras in ROUTES]
