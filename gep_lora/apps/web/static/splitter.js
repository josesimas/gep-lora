// A draggable edge between a page's left column and the rest, remembered per
// page in this browser.
//
// A page asks for it on its two-column grid, and loads this script after it
// (or anywhere: it waits for the document):
//
//     <div class="halves" data-left-column="guide">       <- the name it is stored under
//     ...
//     <script src="/splitter.js"></script>
//
// and writes its first column as var(--split, <its own default>):
//
//     .halves { grid-template-columns: var(--split, minmax(360px, 480px)) minmax(0, 1fr); }
//
// so a page nobody has resized keeps the width it always had, and a narrow
// screen's single-column rule still wins over the variable. Only the width the
// user dragged to is stored -- never the default -- and a double click on the
// edge forgets it. The edge hides itself whenever the grid is one column.
(function () {
  "use strict";

  var PREFIX = "gep-split-";
  var MIN = 260;           // the left column never narrower than this
  var ROOM = 360;          // nor so wide the right one gets less than this
  var STEP = 16;           // an arrow key's worth

  var STYLE = [
    "[data-left-column] { position: relative; }",
    ".split-edge { position: absolute; top: 0; bottom: 0; width: 9px; margin-left: -5px; z-index: 5;",
    "            cursor: col-resize; touch-action: none; background: transparent; }",
    ".split-edge::after { content: ''; position: absolute; top: 0; bottom: 0; left: 3px; width: 3px;",
    "                   background: transparent; transition: background .12s; }",
    ".split-edge:hover::after, .split-edge:focus-visible::after, .split-edge.dragging::after { background: var(--accent); }",
    ".split-edge:focus-visible { outline: none; }",
    "body.splitting, body.splitting * { cursor: col-resize !important; user-select: none !important; }",
  ].join("\n");

  function recall(name) {
    try { var value = parseInt(localStorage.getItem(PREFIX + name), 10); return value > 0 ? value : null; }
    catch (_) { return null; }
  }
  function remember(name, value) {
    try {
      if (value == null) localStorage.removeItem(PREFIX + name);
      else localStorage.setItem(PREFIX + name, String(Math.round(value)));
    } catch (_) { /* private mode: it still resizes, it just forgets */ }
  }

  function attach(grid) {
    var name = grid.getAttribute("data-left-column");
    var handle = document.createElement("div");
    handle.className = "split-edge";
    handle.tabIndex = 0;
    handle.setAttribute("role", "separator");
    handle.setAttribute("aria-orientation", "vertical");
    handle.setAttribute("aria-label", "Resize the left column");
    handle.title = "Drag to resize · double-click for the default width";
    grid.appendChild(handle);

    var wanted = recall(name);   // what the user chose; applied clamped to the room there is

    function twoColumns() {
      return getComputedStyle(grid).gridTemplateColumns.trim().split(/\s+/).length > 1;
    }
    function clamp(width) {
      return Math.max(MIN, Math.min(width, grid.clientWidth - ROOM));
    }
    // The first column's right edge: whichever of its children is showing.
    function edge() {
      var box = grid.getBoundingClientRect();
      for (var i = 0; i < grid.children.length; i++) {
        var child = grid.children[i];
        if (child === handle || child.hidden) continue;
        var rect = child.getBoundingClientRect();
        if (rect.width) return rect.right - box.left;
      }
      return 0;
    }
    function place() {
      handle.hidden = !twoColumns();
      if (!handle.hidden) handle.style.left = edge() + "px";
      handle.setAttribute("aria-valuenow", String(Math.round(edge())));
    }
    function apply() {
      if (wanted == null) grid.style.removeProperty("--split");
      else grid.style.setProperty("--split", clamp(wanted) + "px");
      place();
    }
    function set(width) {
      wanted = clamp(width);
      apply();
    }

    handle.addEventListener("pointerdown", function (event) {
      if (event.button !== 0) return;
      event.preventDefault();
      handle.setPointerCapture(event.pointerId);
      handle.classList.add("dragging");
      document.body.classList.add("splitting");
      var left = grid.getBoundingClientRect().left;
      function move(e) { set(e.clientX - left); }
      function stop() {
        handle.removeEventListener("pointermove", move);
        handle.removeEventListener("pointerup", stop);
        handle.removeEventListener("pointercancel", stop);
        handle.classList.remove("dragging");
        document.body.classList.remove("splitting");
        remember(name, wanted);
      }
      handle.addEventListener("pointermove", move);
      handle.addEventListener("pointerup", stop);
      handle.addEventListener("pointercancel", stop);
    });
    handle.addEventListener("dblclick", function () {
      wanted = null;
      remember(name, null);
      apply();
    });
    handle.addEventListener("keydown", function (event) {
      var by = event.key === "ArrowLeft" ? -STEP : event.key === "ArrowRight" ? STEP : 0;
      if (!by || handle.hidden) return;
      event.preventDefault();
      set(edge() + by);
      remember(name, wanted);
    });

    // The window, a narrow-screen rule or a page swapping which column shows
    // (the console's two sidebars) all move the edge.
    if (window.ResizeObserver) {
      var watch = new ResizeObserver(function () { place(); });
      watch.observe(grid);
      for (var i = 0; i < grid.children.length; i++) if (grid.children[i] !== handle) watch.observe(grid.children[i]);
    }
    window.addEventListener("resize", apply);
    apply();
  }

  function start() {
    var style = document.createElement("style");
    style.id = "gep-split-style";
    style.textContent = STYLE;
    document.head.appendChild(style);
    var grids = document.querySelectorAll("[data-left-column]");
    for (var i = 0; i < grids.length; i++) attach(grids[i]);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
