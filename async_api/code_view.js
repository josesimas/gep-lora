// The overlay a blend's script is shown in, on the two pages that draw blends
// (visual_guide.html and blend_comparison.html):
//
//     gepCode.show({ title, name, source, note, exact })
//
// `source` is the Python POST /blends/code returns -- the script the blend runs
// as when it is processed -- shown with line numbers and colour, with Copy and
// Save as. Its styles are its own, put first in <head> the way nav.js puts its
// own, and coloured through the pages' variables so a theme and dark mode reach it.
(function () {
  "use strict";

  var STYLE = [
    // A message box: above everything (the top bar included), and pinned to the
    // window itself -- a fixed gap from every edge, whatever the page's height,
    // so its header is always on screen and only its code scrolls. No flex
    // centring and no percentages: with all four insets and auto margins the
    // browser sizes it from the window, and a box wider than max-width centres.
    ".code-shade { position: fixed; top: 0; right: 0; bottom: 0; left: 0; z-index: 1000;",
    "              background: rgba(11, 14, 20, .55); overscroll-behavior: contain; }",
    ".code-box { position: fixed; top: 40px; right: 40px; bottom: 40px; left: 40px; margin: auto;",
    "            max-width: 1100px; box-sizing: border-box; display: flex; flex-direction: column;",
    "            background: var(--panel); color: var(--ink); border: 1px solid var(--line); border-radius: 12px;",
    "            box-shadow: 0 24px 70px rgba(0, 0, 0, .4); overflow: hidden; }",
    "html.code-open, html.code-open body { overflow: hidden !important; }",
    ".code-head, .code-note { flex: none; }",
    ".code-head { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; padding: 12px 16px; border-bottom: 1px solid var(--line); }",
    ".code-head h3 { margin: 0; font-size: 14px; }",
    ".code-head .code-name { font-family: var(--mono); font-size: 12px; color: var(--muted); }",
    ".code-head .code-grow { flex: 1; }",
    ".code-head .code-done { font-size: 12px; color: var(--ok); }",
    ".code-note { margin: 0; padding: 8px 16px; font-size: 12.5px; color: var(--muted); border-bottom: 1px solid var(--line); background: var(--panel-2); }",
    ".code-note b { color: var(--ok); font-weight: 600; }",
    ".code-scroll { overflow: auto; background: var(--code-bg); flex: 1 1 0; min-height: 0; overscroll-behavior: contain; }",
    ".code-table { border-collapse: collapse; font: 12.5px/1.55 var(--mono); width: 100%; }",
    ".code-table td { padding: 0 14px 0 0; border: 0; vertical-align: top; white-space: pre; background: none; }",
    ".code-table td.ln { width: 1%; padding: 0 12px 0 14px; text-align: right; color: var(--code-ln); user-select: none; }",
    ".code-table tr:first-child td { padding-top: 12px; } .code-table tr:last-child td { padding-bottom: 12px; }",
    ".code-table td.src { color: var(--code-ink); }",
    // The colours: the page's own where they fit, a few of their own besides.
    ":root { --code-bg: #f7f8fa; --code-ink: #1f2430; --code-ln: #a3aab6; --tk-kw: #7b3fc4; --tk-str: #1f7a3a;",
    "        --tk-com: #8a929e; --tk-num: #b35309; --tk-fn: #1d5fd1; --tk-bi: #0e7490; --tk-dec: #b0306b; --tk-self: #a23e2a; }",
    "@media (prefers-color-scheme: dark) { :root:not([data-theme=\"light\"]) {",
    "  --code-bg: #0f131a; --code-ink: #d6dce6; --code-ln: #535c6b; --tk-kw: #c792ea; --tk-str: #a5d98c;",
    "  --tk-com: #6d7788; --tk-num: #f7a86b; --tk-fn: #82aaff; --tk-bi: #6bd3e8; --tk-dec: #f38bb8; --tk-self: #f07178; } }",
    ":root[data-theme=\"dark\"] { --code-bg: #0f131a; --code-ink: #d6dce6; --code-ln: #535c6b; --tk-kw: #c792ea; --tk-str: #a5d98c;",
    "  --tk-com: #6d7788; --tk-num: #f7a86b; --tk-fn: #82aaff; --tk-bi: #6bd3e8; --tk-dec: #f38bb8; --tk-self: #f07178; }",
    ".tk-kw { color: var(--tk-kw); font-weight: 600; } .tk-str { color: var(--tk-str); } .tk-com { color: var(--tk-com); font-style: italic; }",
    ".tk-num { color: var(--tk-num); } .tk-fn { color: var(--tk-fn); } .tk-bi { color: var(--tk-bi); } .tk-dec { color: var(--tk-dec); }",
    ".tk-self { color: var(--tk-self); } .tk-const { color: var(--tk-num); font-weight: 600; }",
    "@media (max-width: 640px) { .code-box { top: 12px; right: 12px; bottom: 12px; left: 12px; } }",
  ].join("\n");

  var style = document.createElement("style");
  style.id = "gep-code-style";
  style.textContent = STYLE;
  document.head.insertBefore(style, document.head.firstChild);

  var KEYWORDS = ("False None True and as assert async await break class continue def del elif else except finally for " +
                  "from global if import in is lambda nonlocal not or pass raise return try while with yield match case").split(" ");
  var BUILTINS = ("abs all any bool dict enumerate float format getattr hasattr int isinstance len list map max min next " +
                  "open print range repr reversed round set setattr sorted str sum super tuple type zip Exception " +
                  "ValueError RuntimeError KeyError TypeError SystemExit OSError StopIteration").split(" ");
  var KW = {}, BI = {};
  KEYWORDS.forEach(function (w) { KW[w] = true; });
  BUILTINS.forEach(function (w) { BI[w] = true; });

  function esc(text) {
    return text.replace(/[&<>]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]; });
  }
  function span(cls, text) { return '<span class="' + cls + '">' + esc(text) + "</span>"; }

  // One pass over the whole source, so a triple-quoted string may span lines;
  // then split into lines, closing and reopening a span that crosses one.
  var TOKEN = /(#[^\n]*)|([rRbBuUfF]{0,2}(?:"""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'))|(@[A-Za-z_][\w.]*)|(\b\d[\d_]*(?:\.\d+)?(?:[eE][+-]?\d+)?\b)|([A-Za-z_]\w*)/g;

  function highlight(source) {
    var out = "", last = 0, match, previous = "";
    TOKEN.lastIndex = 0;
    while ((match = TOKEN.exec(source))) {
      out += esc(source.slice(last, match.index));
      last = TOKEN.lastIndex;
      var word = match[5];
      if (match[1]) out += span("tk-com", match[1]);
      else if (match[2]) out += span("tk-str", match[2]);
      else if (match[3]) out += span("tk-dec", match[3]);
      else if (match[4]) out += span("tk-num", match[4]);
      else if (previous === "def" || previous === "class") out += span("tk-fn", word);
      else if (word === "self" || word === "cls") out += span("tk-self", word);
      else if (KW[word]) out += span("tk-kw", word);
      else if (BI[word]) out += span("tk-bi", word);
      else if (/^[A-Z][A-Z0-9_]{2,}$/.test(word)) out += span("tk-const", word);
      else out += esc(word);
      previous = word || "";
    }
    out += esc(source.slice(last));
    // Split into lines, carrying an open span across each line break.
    var lines = [], open = [];
    out.split("\n").forEach(function (line) {
      var text = open.join("") + line;
      var tags = line.match(/<span class="[^"]+">|<\/span>/g) || [];
      tags.forEach(function (tag) { if (tag === "</span>") open.pop(); else open.push(tag); });
      lines.push(text + open.map(function () { return "</span>"; }).join(""));
    });
    return lines;
  }

  function make(tag, attrs, text) {
    var node = document.createElement(tag);
    for (var name in attrs || {}) node.setAttribute(name, attrs[name]);
    if (text) node.textContent = text;
    return node;
  }

  async function copy(text) {
    try { await navigator.clipboard.writeText(text); return true; } catch (_) { /* not allowed here */ }
    var area = make("textarea");
    area.value = text;
    area.style.position = "fixed"; area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    var ok = false;
    try { ok = document.execCommand("copy"); } catch (_) { ok = false; }
    area.remove();
    return ok;
  }

  // A real "Save as" where the browser has one; otherwise a download, which
  // the browser saves where it saves them (or asks, if it is set to).
  async function saveAs(name, text) {
    if (window.showSaveFilePicker) {
      try {
        var handle = await window.showSaveFilePicker({ suggestedName: name,
          types: [{ description: "Python script", accept: { "text/x-python": [".py"] } }] });
        var writable = await handle.createWritable();
        await writable.write(text);
        await writable.close();
        return "saved as " + handle.name;
      } catch (error) {
        if (error && error.name === "AbortError") return null;
        // Anything else (a sandboxed frame, say): fall through to a download.
      }
    }
    var link = make("a", { href: URL.createObjectURL(new Blob([text], { type: "text/x-python" })), download: name });
    document.body.appendChild(link);
    link.click();
    setTimeout(function () { URL.revokeObjectURL(link.href); link.remove(); }, 0);
    return "downloaded " + name;
  }

  var current = null;

  function close() {
    if (!current) return;
    current.shade.remove();
    document.documentElement.classList.remove("code-open");
    document.removeEventListener("keydown", current.keys, true);
    if (current.back && current.back.focus) current.back.focus();
    current = null;
  }

  function show(options) {
    close();
    var name = options.name || "blend.py", source = options.source || "";
    var shade = make("div", { "class": "code-shade", role: "dialog", "aria-modal": "true", "aria-label": options.title || "Code" });
    var box = make("div", { "class": "code-box" });
    var head = make("div", { "class": "code-head" });
    head.appendChild(make("h3", {}, options.title || "Code"));
    head.appendChild(make("span", { "class": "code-name" }, name + " · " + source.replace(/\n$/, "").split("\n").length + " lines"));
    head.appendChild(make("span", { "class": "code-grow" }));
    var done = make("span", { "class": "code-done", "aria-live": "polite" });
    head.appendChild(done);
    var say = function (text) { done.textContent = text || ""; if (text) setTimeout(function () { if (done.textContent === text) done.textContent = ""; }, 2500); };
    var copyButton = make("button", { type: "button", "class": "sm", title: "Copy the whole script" }, "Copy");
    copyButton.addEventListener("click", async function () { say(await copy(source) ? "copied" : "could not copy — select it and copy by hand"); });
    var saveButton = make("button", { type: "button", "class": "sm", title: "Save the script as a .py file" }, "Save as…");
    saveButton.addEventListener("click", async function () { say(await saveAs(name, source)); });
    var closeButton = make("button", { type: "button", "class": "sm primary", title: "Close (Esc)" }, "Close");
    closeButton.addEventListener("click", close);
    head.appendChild(copyButton); head.appendChild(saveButton); head.appendChild(closeButton);
    box.appendChild(head);
    if (options.note) {
      var note = make("p", { "class": "code-note" });
      if (options.exact) { note.appendChild(make("b", {}, "Stored script. ")); }
      note.appendChild(document.createTextNode(options.note));
      box.appendChild(note);
    }
    var scroll = make("div", { "class": "code-scroll", tabindex: "0" });
    var table = make("table", { "class": "code-table" });
    var body = make("tbody");
    highlight(source.replace(/\n$/, "")).forEach(function (line, i) {
      var row = make("tr");
      row.appendChild(make("td", { "class": "ln" }, String(i + 1)));
      var cell = make("td", { "class": "src" });
      cell.innerHTML = line || " ";
      row.appendChild(cell);
      body.appendChild(row);
    });
    table.appendChild(body);
    scroll.appendChild(table);
    box.appendChild(scroll);
    shade.appendChild(box);
    shade.addEventListener("mousedown", function (e) { if (e.target === shade) close(); });
    // The wheel and a finger scroll the code, and nothing else: anywhere outside
    // it -- the shade, the header -- they are swallowed rather than handed on to
    // the page (whose scrolling parts, like the board, are not its ancestors).
    var hold = function (e) { if (!scroll.contains(e.target)) e.preventDefault(); };
    shade.addEventListener("wheel", hold, { passive: false });
    shade.addEventListener("touchmove", hold, { passive: false });
    var keys = function (e) {
      if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); close(); }
      // Ctrl+A selects the code, not the page behind it.
      else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "a" && !/^(INPUT|TEXTAREA)$/.test((document.activeElement || {}).tagName)) {
        e.preventDefault(); e.stopPropagation();
        var range = document.createRange(); range.selectNodeContents(body);
        var selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
      }
      // The page's own keys (Delete, Ctrl+Z) wait while the overlay is up.
      else if (e.key === "Delete" || e.key === "Backspace" || ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z")) e.stopPropagation();
    };
    document.addEventListener("keydown", keys, true);
    document.body.appendChild(shade);
    document.documentElement.classList.add("code-open");
    current = { shade: shade, keys: keys, back: document.activeElement };
    scroll.focus();
  }

  window.gepCode = { show: show, close: close, highlight: highlight };
})();
