// The top bar every page shares, in three blocks: the brand on the left, the
// pages in the middle, and the page's own buttons, the server's health,
// the API key and the theme on the right.
//
// A page asks for it with an empty header and this script right after it:
//
//     <header class="topbar" id="nav" data-page="runs"
//             data-about="every search you have run">
//       <a class="button primary" href="/guide.html?new=1">New search</a>   <- the page's own
//     </header>
//     <script src="/nav.js"></script>
//
// It runs where it is loaded, not on DOMContentLoaded, so #health and
// #changeKey exist before the page's own script looks for them. Whatever the
// header held is the page's own actions: moved, not rebuilt, so its ids and
// listeners stay the page's.
//
// A new page is an entry in PAGES and a route in server.py; a new link on
// every page is an edit here and nowhere else.
(function () {
  "use strict";

  // In the order a beginner uses them: start in the guide, read the results,
  // adjust what the guide starts from, and the raw console last.
  var PAGES = [
    { id: "guide", label: "Guide", href: "/guide.html", job: true,
      about: "Train LoRAs and blend them, with an assistant" },
    { id: "visual", label: "Visual guide", href: "/visual_guide.html",
      about: "Draw a blend of your LoRAs as a tree, and test it" },
    { id: "compare", label: "Compare", href: "/blend_comparison.html",
      about: "Two blends side by side: open, edit and test them on the same questions" },
    { id: "runs", label: "Runs", href: "/runs.html",
      about: "Every search you have run, and what came of it" },
    { id: "settings", label: "Settings", href: "/settings.html",
      about: "What a new guide conversation starts from, and how the pages look" },
    { id: "console", label: "Console", href: "/console.html", job: true, tag: "advanced",
      about: "Every API call, by hand: jobs, LoRAs and inference" },
  ];

  var STYLE = [
    ".topbar { position: sticky; top: 0; z-index: 20; height: var(--top, 56px); box-sizing: border-box;",
    "          display: flex; align-items: center; gap: 12px; padding: 0 18px;",
    "          background: var(--panel); border-bottom: 1px solid var(--line); }",
    ".topbar .brand { display: flex; align-items: center; gap: 10px; min-width: 0; color: inherit; text-decoration: none; }",
    ".topbar .brand svg { width: 26px; height: 26px; flex: none; }",
    ".topbar .brand b { display: block; font-size: 14px; line-height: 1.1; white-space: nowrap; }",
    ".topbar .brand span { display: block; font-size: 11px; color: var(--muted); line-height: 1.2;",
    "                      white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 260px; }",
    ".topbar .viewnav { display: flex; gap: 2px; padding: 3px; flex: none; background: var(--panel-2);",
    "                   border: 1px solid var(--line); border-radius: 10px; }",
    ".topbar .viewnav a { display: inline-flex; align-items: center; gap: 6px; min-height: 28px; padding: 0 12px;",
    "                     border-radius: 7px; color: var(--muted); text-decoration: none; font-size: 13px; white-space: nowrap; }",
    ".topbar .viewnav a:hover { background: var(--panel); color: var(--ink); }",
    ".topbar .viewnav a[aria-current] { background: var(--panel); color: var(--ink); font-weight: 600; box-shadow: var(--shadow); }",
    ".topbar .viewnav .navtag { font-size: 10px; font-weight: 400; letter-spacing: .04em; text-transform: uppercase; color: var(--faint); }",
    // The two sides grow alike from nothing, so the pages sit in the middle of
    // the bar -- until a side's own content is wider than its half, when the
    // pages give way rather than anything being cut off.
    ".topbar .nav-start, .topbar .nav-end { flex: 1 1 0; }",
    ".topbar .nav-start { display: flex; min-width: 0; }",
    ".topbar .nav-end { display: flex; align-items: center; justify-content: flex-end; gap: 12px; }",
    ".topbar .page-actions { display: flex; align-items: center; gap: 8px; }",
    ".topbar .page-actions:empty { display: none; }",
    ".topbar a.button { display: inline-flex; align-items: center; text-decoration: none; white-space: nowrap; }",
    ".topbar .health { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; color: var(--muted);",
    "                  max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }",
    ".topbar .health::before { content: ''; flex: none; width: 8px; height: 8px; border-radius: 50%; background: var(--faint); }",
    ".topbar .health.up::before { background: var(--ok); }",
    ".topbar .health.busy::before { background: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }",
    ".topbar .health.down::before { background: var(--bad); }",
    ".topbar .navkey { min-height: 28px; padding: 0 10px; font: inherit; font-size: 13px; white-space: nowrap; cursor: pointer;",
    "                  color: var(--ink); background: transparent; border: 1px solid var(--line-2); border-radius: 8px; }",
    ".topbar .navkey:hover { background: var(--panel-2); }",
    "@media (max-width: 1100px) { .topbar .brand span { display: none; } }",
    "@media (max-width: 900px) { .topbar .health { display: none; } .topbar .viewnav .navtag { display: none; } }",
    // A phone: the pages on the first row, everything else wraps under them.
    "@media (max-width: 640px) {",
    "  .topbar { height: auto; min-height: var(--top, 56px); flex-wrap: wrap; gap: 8px; padding: 8px 12px; }",
    "  .topbar .nav-start { flex: none; }",
    "  .topbar .brand b { display: none; }",
    "  .topbar .viewnav a { padding: 0 8px; }",
    "  .topbar .nav-end { flex-basis: 100%; justify-content: flex-start; gap: 8px; }",
    "}",
  ].join("\n");

  var LOGO = '<svg viewBox="0 0 26 26" aria-hidden="true">'
    + '<rect width="26" height="26" rx="7" fill="var(--accent)"/>'
    + '<path d="M7 16.5c2.2-5 4.2-7.5 6-7.5s3.8 2.5 6 7.5" stroke="var(--accent-ink)" stroke-width="1.7" fill="none" stroke-linecap="round"/>'
    + '<circle cx="13" cy="9" r="2.3" fill="var(--accent-ink)"/>'
    + '<circle cx="7" cy="17" r="2" fill="var(--accent-ink)"/>'
    + '<circle cx="19" cy="17" r="2" fill="var(--accent-ink)"/></svg>';

  var bar = document.getElementById("nav");
  if (!bar) return;
  var here = bar.getAttribute("data-page");

  // First in <head>, so a page's own rules and a theme's still win over it.
  var style = document.createElement("style");
  style.id = "gep-nav-style";
  style.textContent = STYLE;
  document.head.insertBefore(style, document.head.firstChild);

  function make(tag, attrs, text) {
    var node = document.createElement(tag);
    for (var name in attrs) node.setAttribute(name, attrs[name]);
    if (text) node.textContent = text;
    return node;
  }

  // The page's own buttons, as the page wrote them.
  var actions = make("div", { "class": "page-actions" });
  while (bar.firstChild) actions.appendChild(bar.firstChild);

  var page = PAGES.filter(function (one) { return one.id === here; })[0];
  var brand = make("a", { "class": "brand", href: PAGES[0].href, title: "GEP LoRA: start in the guide" });
  brand.innerHTML = LOGO;
  var words = make("div", {});
  words.appendChild(make("b", {}, "GEP LoRA"));
  words.appendChild(make("span", {}, bar.getAttribute("data-about") || (page ? page.about : "")));
  brand.appendChild(words);

  // The search on screen follows you between the two pages that can open one
  // (?job=N). Read when the link is used, since both pages change the address
  // as a search is opened or closed.
  function jobHref(one) {
    var job = new URLSearchParams(location.search).get("job");
    return one.job && job && /^\d+$/.test(job) ? one.href + "?job=" + job : one.href;
  }
  var links = make("nav", { "class": "viewnav", "aria-label": "Pages" });
  PAGES.forEach(function (one) {
    var link = make("a", { href: one.href, title: one.about, "data-page": one.id }, one.label);
    if (one.tag) link.appendChild(make("span", { "class": "navtag" }, one.tag));
    if (one.id === here) link.setAttribute("aria-current", "page");
    var freshen = function () { link.href = jobHref(one); };
    link.addEventListener("pointerdown", freshen);
    link.addEventListener("focus", freshen);
    link.addEventListener("click", freshen);
    links.appendChild(link);
  });

  var health = make("span", { id: "health", "class": "health", title: "The API server" });
  // The page decides when a key is asked for: it listens for the click and
  // shows its own gate, and shows the button once a key has been accepted.
  var key = make("button", { type: "button", id: "changeKey", "class": "navkey",
                             title: "Use another API key in this browser" }, "Change key");
  key.hidden = true;

  // The right-hand block; themes.js puts its picker at the end of it.
  var end = make("div", { "class": "nav-end" });
  end.appendChild(actions);
  end.appendChild(health);
  end.appendChild(key);
  var start = make("div", { "class": "nav-start" });
  start.appendChild(brand);
  bar.appendChild(start);
  bar.appendChild(links);
  bar.appendChild(end);

  // --- the server's health: no key needed, so every page shows it -------------------------
  var timer = null;
  function refreshHealth() {
    clearTimeout(timer);
    timer = setTimeout(refreshHealth, 10000);
    return fetch("/health").then(function (reply) { return reply.json(); }).then(function (answer) {
      var loaded = (answer.models || []).filter(function (m) { return m.loaded; });
      health.textContent = loaded.length ? "API up · " + loaded.length + " model" + (loaded.length > 1 ? "s" : "") + " loaded"
                                         : "API up";
      health.title = loaded.length
        ? "Loaded: " + loaded.map(function (m) {
            return m.base_model + " (" + m.engine + ", unloads in " + Math.round(m.unloads_in) + "s)";
          }).join(", ")
        : "The API server is up; no model is loaded for inference";
      health.className = "health " + (loaded.length ? "busy" : "up");
    }).catch(function () {
      health.textContent = "API unreachable";
      health.title = "The API server cannot be reached. Is it running? (run_api_server.bat)";
      health.className = "health down";
    });
  }
  refreshHealth();

  // --- the API key: one name in this browser, shared by every page ------------------------
  var KEY = "gep-api-key";
  window.gepNav = {
    pages: function () { return PAGES.slice(); },
    refreshHealth: refreshHealth,
    recallKey: function () { try { return localStorage.getItem(KEY) || ""; } catch (_) { return ""; } },
    rememberKey: function (value) { try { localStorage.setItem(KEY, value); } catch (_) { /* private mode */ } },
  };
})();
