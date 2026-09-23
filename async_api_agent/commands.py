"""
commands.py - Plain requests turned into tool calls, when no model can.

The chat's tools (tools.py) are normally chosen by the model. When there is no
model -- none configured, no key, unreachable, or one whose server does not
take tools -- the requests people make most still work: this module reads
them with patterns and returns the same calls the model would have made,
which the same Toolbox then runs under the same checks.

    parse("only use the first 20 and drop duplicates", "analysis", names)
    -> [("select_records", {"first": 20, "drop_duplicates": True})]

    parse("blend poem-r8 and poem-r16, 5 generations", "blend", [], ["poem-r8", ...])
    -> [("choose_blend_loras", {"loras": ["poem-r8", "poem-r16"]}),
        ("set_blend_search", {"generations": 5})]

Once the person is combining LoRAs, only the blend's requests are read: "5
epochs" means nothing to a search, and would only be refused. And once the
search is over, only the requests about its blends: test them, pick one
("#7 is the best"), what to verify it on, verify it, put it live.

    parse("verify #7 on the testing questions", "tested")
    -> [("choose_best_blend", {"individual": 7}),
        ("set_verify_questions", {"split": "testing"}),
        ("start_verification", {})]

It is deliberately narrow: a request it does not recognise returns no calls,
and the agent says what it can do instead of guessing.
"""

import re

from async_api_agent import planner

_N = r"(\d+(?:\.\d+)?)"
_WORD = r"[\"'“‘]?([\w-]{2,})[\"'”’]?"
_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
            "eight": 8, "nine": 9, "ten": 10, "twenty": 20, "thirty": 30, "fifty": 50,
            "a hundred": 100}


def _clean(message):
    said = " " + message.lower().replace(",", " , ") + " "
    for word, value in sorted(_NUMBERS.items(), key=lambda pair: -len(pair[0])):
        said = re.sub(r"(?<![a-z])%s(?![a-z])" % word, str(value), said)
    return re.sub(r"\s+", " ", said)


def _selection(said):
    """The select_records arguments a request names. -> dict."""
    out = {}
    found = re.search(r"\b(?:first|top) " + _N + r"\b", said)
    if found:
        out["first"] = int(float(found.group(1)))
    found = re.search(r"\blast " + _N + r"\b", said)
    if found:
        out["last"] = int(float(found.group(1)))
    found = re.search(r"\b(?:records?|rows?|lines?|examples?) " + _N +
                      r" ?(?:to|-|through|until) ?" + _N, said)
    if found:
        out["start"], out["end"] = int(float(found.group(1))), int(float(found.group(2)))
    found = re.search(_N + r" ?(?:%|percent)", said)
    if found:
        out["percent"] = float(found.group(1))
    elif re.search(r"\bhalf (?:of )?(?:the |it|them|records|data)", said):
        out["percent"] = 50.0
    elif re.search(r"\ba quarter\b", said):
        out["percent"] = 25.0
    found = re.search(r"\b(?:random(?:ly)?(?: pick| choose| sample)?|sample(?: of)?) " + _N, said) \
        or re.search(_N + r" (?:records? |examples? )?(?:at random|randomly)", said)
    if found:
        out["sample"] = int(float(found.group(1)))
    found = re.search(r"\b(?:without|exclud\w*|drop|remove|skip|leave out|not) (?:the |any )?"
                      r"(?:records?|ones?|examples?|rows?)? ?(?:that )?(?:mention\w*|contain\w*|"
                      r"about|with the word|with) " + _WORD, said)
    if found and found.group(1) not in ("duplicates", "dupes"):
        out["excludes"] = [found.group(1)]
    else:
        found = re.search(r"\b(?:only|just)?.*?(?:mention\w*|contain\w*|about|with the word) "
                          + _WORD, said)
        if found and found.group(1) not in ("the", "a", "an", "it", "this", "that") \
                and re.search(r"\b(?:only|just|keep|use|train on|records?|ones?)\b", said):
            out["contains"] = [found.group(1)]
    if re.search(r"\b(?:without|remove|drop|no|dedupe|skip) (?:the )?(?:duplicates|dupes|repeats)\b",
                 said) or "deduplicate" in said:
        out["drop_duplicates"] = True
    found = re.search(r"answers? (?:shorter|under|less) than " + _N, said)
    if found:
        out["max_answer_words"] = int(float(found.group(1)))
    found = re.search(r"answers? (?:longer|over|more) than " + _N, said)
    if found:
        out["min_answer_words"] = int(float(found.group(1)))
    return out


def _ranks(said):
    found = re.search(r"\branks? (?:of |=|to )?(\d+(?:(?: ?, ?| and | & | or )\d+)*)", said)
    if found:
        return [int(one) for one in re.findall(r"\d+", found.group(1))]
    found = re.search(r"\b(\d) loras? (?:at|of|with) ranks? (\d+)", said)
    if found:
        return [int(found.group(2))] * int(found.group(1))
    return None


def _options(said):
    out = {}
    found = re.search(r"learning rate (?:of |to |= ?|at )?([\d.]+(?:e-?\d+)?)", said)
    if found:
        out["learning_rate"] = float(found.group(1))
    elif re.search(r"(?:lower|smaller) learning rate", said):
        out["learning_rate"] = 1e-4
    elif re.search(r"(?:higher|bigger) learning rate", said):
        out["learning_rate"] = 4e-4
    for key, pattern in (("alpha", r"\balpha (?:of |to |= ?)?(\d+)"),
                         ("batch_size", r"batch size (?:of |to |= ?)?(\d+)"),
                         ("max_steps", r"(?:max(?:imum)? steps|at most) (?:of |to |= ?)?(\d+)(?: steps)?"),
                         ("max_seq", r"(?:max(?:imum)? )?sequence(?: length)? (?:of |to |= ?)?(\d+)")):
        found = re.search(pattern, said)
        if found:
            out[key] = int(found.group(1))
    found = re.search(r"\b(?:call|name) (?:them|it|the loras?) " + _WORD, said)
    if found:
        out["name"] = found.group(1)
    return out


# The stages in which the person is combining LoRAs, not training them.
BLEND_STAGES = ("blend", "blend_confirm", "blending", "blended")

# The stages after the search: testing its blends, verifying one, going live.
# "blended" is both: the search is over, and another may be planned from it.
RELEASE_STAGES = ("blended", "testing", "tested", "verifying", "verified", "live")

_START = (r"\b(?:start|begin|kick off|launch|run) (?:the )?(?:training|search|blend(?:ing)?)\b|"
          r"\bstart (?:it |them )?now\b|\bgo ahead\b|\blet'?s go\b|^ (?:start|go|begin) ?[.!]? $")


def _named(said, names):
    """The user's LoRA names a request mentions, in the order it mentions them."""
    found = []
    for name in names:
        at = re.search(r"(?<![\w-])%s(?![\w-])" % re.escape(name.lower()), said)
        if at:
            found.append((at.start(), name))
    return [name for _, name in sorted(found)]


def _blend(said, stage, demo_files, names):
    """The blend's calls a plain request asks for. -> [(name, arguments)]."""
    calls = []
    if stage == "blending":
        if re.search(r"\b(?:stop|cancel|abort|halt)\b", said):
            calls.append(("stop_blend", {}))
        return calls
    if re.search(r"\b(?:my|existing) loras\b|\bloras (?:do )?i have\b|\bwhich loras\b", said) \
            and not re.search(r"\b(?:blend|combine|use|only)\b", said):
        calls.append(("list_my_loras", {}))
    mentioned = _named(said, names)
    if mentioned:
        add = bool(re.search(r"\b(?:also|add|as well|too)\b", said))
        calls.append(("choose_blend_loras", dict({"loras": mentioned}, **({"add": True} if add else {}))))
    search = {}
    for key, pattern in (("generations", r"(\d+) ?(?:generations?|rounds?)\b"),
                         ("population", r"(?:(\d+) ?(?:blends|individuals)(?: each| per| a)?\b|"
                                        r"population (?:of |to |= ?)?(\d+))"),
                         ("questions", r"(\d+) ?questions?\b")):
        found = re.search(pattern, said)
        if found:
            search[key] = int(next(group for group in found.groups() if group))
    if search:
        calls.append(("set_blend_search", search))
    found = re.search(r"\b(?:questions?|judge\w*|score\w*)\b.*\b(?:from|on|with) (?:the )?([\w.-]+)"
                      r"(?: demo)? (?:dataset|data|file)\b", said)
    if found:
        match = [name for name in demo_files if found.group(1) in name.lower()]
        if len(match) == 1:
            calls.append(("set_blend_questions", {"file": match[0]}))
    if re.search(r"\b(?:another|different|new|other) dataset\b|\btrain (?:more|new|another)\b", said) \
            and not calls:
        calls.append(("choose_another_dataset", {}))
    if re.search(r"\b(?:turn on|enable|use|do|make it|just) (?:a )?practice(?: run)?\b|"
                 r"practice run on\b", said):
        calls.append(("set_practice_run", {"on": True}))
    if re.search(_START, said):
        calls.append(("start_blend", {}))
    return calls


_BLEND_NUMBER = r"(?:#|\bblend (?:#|number )?|\bindividual (?:#|number )?|\bnumber )(\d+)\b"


def _release(said, stage, demo_files, names):
    """The calls about a finished search's blends. -> [(name, arguments)]."""
    calls = []
    if stage == "testing":
        if re.search(r"\b(?:stop|cancel|abort|halt)\b", said):
            calls.append(("stop_testing", {}))
        return calls
    if stage == "verifying":
        return calls
    if re.search(r"\b(?:test|try) (?:all |every |the |them|those|these)(?:the )?"
                 r"(?:blends?|of them|them)?\b|\b(?:run|start) (?:the )?test(?:s|ing)\b", said) \
            and not re.search(r"\btesting (?:questions|split|set|data)\b", said):
        calls.append(("start_testing", {}))
        return calls
    number = re.search(_BLEND_NUMBER, said)
    live = re.search(r"\b(?:go(?:es)? live|put (?:it |them |that |this |\S+ )?live|deploy|"
                     r"set (?:it |\S+ )?live|make (?:it |\S+ )?live)\b", said)
    verify = re.search(r"\b(?:verify|verification|check (?:it|that|this|blend|#))", said)
    if number and not live:
        calls.append(("choose_best_blend", {"individual": int(number.group(1))}))
    found = re.search(r"\b(?:on|with|using|against) (?:the |its )?(validation|testing|training)"
                      r"(?: questions| split| set| data)?\b", said)
    source = {}
    if found:
        source["split"] = found.group(1)
    else:
        found = re.search(r"\b(?:on|with|using|use) (?:the )?([\w.-]+)(?: demo)? "
                          r"(?:dataset|data|file)\b", said)
        if found:
            match = [name for name in demo_files if found.group(1) in name.lower()]
            if len(match) == 1:
                source["file"] = match[0]
            else:
                mentioned = _named(said, names)
                if mentioned:
                    source["lora"] = mentioned[0]
    found = re.search(r"(\d+) ?questions?\b", said)
    if found:
        source["count"] = int(found.group(1))
    if source and stage != "live":
        calls.append(("set_verify_questions", source))
    if live:
        calls.append(("go_live", {"individual": int(number.group(1))} if number else {}))
    elif verify and stage in ("tested", "verified"):
        calls.append(("start_verification", {}))
    return calls


def parse(message, stage, demo_files=(), lora_names=()):
    """The tool calls a plain request asks for. -> [(name, arguments)].

    `lora_names` are the person's own ready LoRAs, so naming one ("blend
    poem-r8 and poem-r16") chooses it without a model."""
    said = _clean(message)
    if stage in RELEASE_STAGES:
        calls = _release(said, stage, demo_files, lora_names)
        if calls or stage != "blended":
            return calls
    if stage in BLEND_STAGES:
        return _blend(said, stage, demo_files, lora_names)
    calls = []
    if stage != "training" and re.search(
            r"\b(?:blend|combine|merge|mix)\w* (?:them|my loras|the loras|loras|these|those|it)\b|"
            r"\b(?:search|blend) (?:over |across )?(?:my |the )?loras\b", said):
        return [("open_blending", {})] + [call for call in _blend(said, "blend", demo_files,
                                                                    lora_names)
                                          if call[0] not in ("start_blend",)]
    if stage == "training" and re.search(r"\b(?:stop|cancel|abort|halt)\b", said):
        return [("stop_training", {})]
    if re.search(r"\b(?:which|what|list|show)\b.*\b(?:demo )?datasets\b", said):
        calls.append(("list_demo_datasets", {}))
    else:
        found = re.search(r"\b(?:use|try|switch to|load|pick)\b(?: the)? ([\w.-]+)(?: demo)? "
                          r"(?:dataset|data|file)\b", said)
        if found:
            wanted = found.group(1)
            match = [name for name in demo_files if wanted in name.lower()]
            if len(match) == 1:
                calls.append(("use_demo_dataset", {"file": match[0]}))
    if re.search(r"\b(?:my|existing) loras\b|\bloras (?:do )?i have\b", said):
        calls.append(("list_my_loras", {}))
    if re.search(r"\b(?:another|different|new|other) dataset\b", said) and not calls:
        calls.append(("choose_another_dataset", {}))

    if re.search(r"\b(?:all|whole|entire|every)(?: of)? (?:the )?(?:records|dataset|data|rows)\b|"
                 r"\buse everything\b|\bundo the selection\b", said):
        calls.append(("use_all_records", {}))
    else:
        # A request to *see* records is not one to train on them.
        chosen = _selection(re.sub(r"\bshow\b(?: me)?(?: the)? (?:longest|shortest|records?|rows?|"
                                   r"examples?|samples?)(?:(?! and | , )[^.;])*", " ", said))
        if chosen:
            calls.append(("select_records", chosen))

    found = re.search(r"\bshow (?:me )?(?:the )?(longest|shortest)", said)
    if found:
        calls.append(("show_records", {"order": found.group(1)}))
    else:
        found = re.search(r"\bshow (?:me )?(?:records?|rows?|examples?) " + _N +
                          r"(?: ?(?:to|-) ?" + _N + ")?", said)
        if found:
            start = int(float(found.group(1)))
            end = int(float(found.group(2))) if found.group(2) else start
            calls.append(("show_records", {"start": start, "count": max(1, end - start + 1)}))
        elif re.search(r"\bshow (?:me )?(?:some |a few )?(?:records|examples|samples|rows)\b", said):
            calls.append(("show_records", {}))

    ranks = _ranks(said)
    if ranks:
        calls.append(("set_loras", {"ranks": ranks}))
    options = _options(said)
    if options:
        calls.append(("set_training_options", options))

    if re.search(r"\b(?:turn on|enable|use|do|make it|just) (?:a )?practice(?: run)?\b|"
                 r"practice run on\b", said):
        calls.append(("set_practice_run", {"on": True}))
    elif re.search(r"\b(?:turn off|disable|no) (?:the )?practice\b|practice run off\b|"
                   r"\b(?:a real run|train for real|for real|real training)\b", said):
        calls.append(("set_practice_run", {"on": False}))

    found = re.search(_N + r" ?(?:epochs?|passes)\b", said)
    minutes = 0.0
    for pattern, scale in ((_N + r" ?(?:h|hr|hrs|hours?)\b", 60), (_N + r" ?(?:min|mins|minutes?)\b", 1)):
        for match in re.finditer(pattern, said):
            minutes += float(match.group(1)) * scale
    if "half an hour" in said or "half 1 hour" in said:
        minutes = minutes or 30
    if found:
        calls.append(("set_epochs", {"epochs": float(found.group(1))}))
    elif minutes and not re.search(r"\bhow long\b", said):
        calls.append(("set_epochs", {"minutes": minutes}))
    elif stage in ("wait", "confirm") and not calls:
        epochs, _ = planner.read_wait(message, 60.0, 0.0)
        if epochs is not None and not re.search(r"\b(?:minute|hour|second)", said):
            calls.append(("set_epochs", {"epochs": epochs}))

    # Only an unmistakable "start": the word alone also begins "start from record 5".
    if stage != "training" and re.search(_START, said) and not re.search(r"\b(?:search|blend)", said):
        calls.append(("start_training", {}))
    return calls
