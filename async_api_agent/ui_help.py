"""
ui_help.py - "What is this?" for the blocks of the guide pages.

Every block of async_api/guide.html (and of visual_guide.html, whose keys
start with "visual_", and of blend_comparison.html, "compare_") that can be
explained carries a question mark: the cards on the right and the charts and tables inside them,
the conversation and the step's controls on the left. Pressing one sends
POST /agent/help the block's key and what the block shows at that moment,
and the answer is written into the conversation on the left like any other
message.

    blocks()     every block that can be explained, {key: {where, title}} --
                 GET /agent/help, so the page draws a question mark only
                 where there is an answer behind it
    explain()    one block explained: the model's words over the block's
                 facts, or the block's own description when no model answers

The facts are the same bargain as every other step's (agent.py). What a block
*is* comes from prompts.UI_BLOCKS, never from the page; what it *shows* comes
from the page, as text, and is only ever something to point at -- it is cut
to SHOWN_CHARS and handed over as a fact, not as an instruction. So a block
reads sensibly with no model at all, and a model that answers is told what
the block is before it is told what is on it.
"""

from async_api_agent import agent
from async_api_agent import prompts

# The most of a block's text a request may carry. A log or a long table is
# kept at both ends: the start says what it is, the end what is happening.
SHOWN_CHARS = 3000


class HelpError(Exception):
    """A request for a block the page cannot have drawn."""


def blocks():
    """Every block that can be explained. -> {key: {where, title}}."""
    return {key: {"where": where, "title": title}
            for key, (where, title, _) in prompts.UI_BLOCKS.items()}


def shown_text(value):
    """What the block shows, as one string of at most SHOWN_CHARS."""
    if not isinstance(value, str):
        return ""
    lines = [" ".join(line.split()) for line in value.replace("\r", "").split("\n")]
    text = "\n".join(line for line in lines if line)
    if len(text) <= SHOWN_CHARS:
        return text
    half = (SHOWN_CHARS - 5) // 2
    return text[:half] + "\n…\n" + text[-half:]


def explain(block, title=None, shown=None, stage=None, choice=None, history=None):
    """One block explained. `title` is its heading as drawn, which may carry
    more than the catalogue's (the card's own side note, say); `shown` its
    text; `stage` the step the page is on.

    -> {"block", "title", "message"}, the message as agent.say() gives it.
    Raises HelpError for a block the catalogue does not know.
    """
    if block not in prompts.UI_BLOCKS:
        raise HelpError("there is no block %r to explain" % (block,))
    where, known, about = prompts.UI_BLOCKS[block]
    title = " ".join(title.split())[:120] if isinstance(title, str) and title.strip() else known
    stage = stage if isinstance(stage, str) and stage in prompts.STEP_INSTRUCTIONS else None
    parent = block.split(".", 1)[0] if "." in block else None
    facts = {"block": block, "title": title, "where": where, "about": about,
             "part_of": prompts.UI_BLOCKS[parent][1] if parent in prompts.UI_BLOCKS else None,
             "shown": shown_text(shown), "stage": stage,
             "step_instructions": prompts.STEP_INSTRUCTIONS.get(stage)}
    fallback = prompts.FALLBACK_HELP.format(title=title, about=about)
    # The page's words for the question too (helpQuestion in guide.html).
    asked = "What do I do at “%s”?" if block == "controls" else "What is “%s”?"
    # The visual guide's and the comparison's blocks are explained by the
    # guide of that page.
    step = ("visual_help" if block.startswith("visual_") else
            "compare_help" if block.startswith("compare_") else "help")
    message = agent.say(step, facts, fallback, choice, history, message=asked % title)
    return {"block": block, "title": title, "message": message}
