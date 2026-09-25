// The page's theme: which one, and a picker to change it.
//
// Loaded from <head>, after the page's own <style>, so it runs before the first
// paint: it reads ?theme=NAME or the one remembered in this browser, sets
// <html data-theme>, and adds that theme's stylesheet. The page's own colours
// are the "default" theme, so a theme is only ever an override on top of them,
// and a page with no theme is exactly the page as it was.
(function () {
  "use strict";
  // swatch: paper, surface, ink, accent, second accent -- for pickers that show one.
  var THEMES = [
    { id: "default", name: "Default", about: "Clean and neutral, with an indigo accent.",
      swatch: ["#f3f4f7", "#ffffff", "#161a23", "#4c5fd5", "#0e7490"] },
    { id: "sumi", name: "Sumi (ink & vermilion)", about: "Ink on warm paper, one vermilion sun, a blossom branch.",
      swatch: ["#f4f1ea", "#fbf9f4", "#1d1e21", "#e0472e", "#8c6a3e"] },
    { id: "bauhaus", name: "Bauhaus (poster)", about: "Cream stock, condensed capitals, mustard and pink blocks.",
      swatch: ["#ebe3d2", "#f3ecdd", "#1c1b19", "#e39a98", "#e6b43c"] },
    { id: "acid", name: "Acid (halftone print)", about: "Grey stock, black ink, acid chartreuse, halftone and print marks.",
      swatch: ["#e7e6e1", "#efeeea", "#161616", "#c9d22b", "#62625e"] },
    { id: "bloom", name: "Bloom (mid-century)", about: "Cream, deep teal and coral, rounded type, a flat flower.",
      swatch: ["#f7f3e8", "#fcf9f1", "#14323a", "#ef6a45", "#f6b01e"] },
  ];
  var KEY = "gep-theme";

  function known(id) {
    for (var i = 0; i < THEMES.length; i++) if (THEMES[i].id === id) return true;
    return false;
  }
  function recall() { try { return localStorage.getItem(KEY) || ""; } catch (_) { return ""; } }
  function remember(id) { try { localStorage.setItem(KEY, id); } catch (_) { /* private mode */ } }

  var asked = new URLSearchParams(location.search).get("theme");
  var current = known(asked) ? asked : (known(recall()) ? recall() : "default");
  if (asked && known(asked)) remember(asked);

  var link = null;
  function apply(id) {
    current = id;
    document.documentElement.setAttribute("data-theme", id);
    if (id === "default") {
      if (link) { link.remove(); link = null; }
      return;
    }
    if (!link) {
      link = document.createElement("link");
      link.rel = "stylesheet";
      document.head.appendChild(link);
    }
    link.href = "/themes/" + id + ".css";
  }
  apply(current);

  // For other pickers (the defaults page): the list, which is on, and a way to
  // change it. A change is announced as a "gep-theme" event on document, so
  // every picker on the page follows it.
  function choose(id) {
    if (!known(id)) return;
    apply(id);
    remember(id);
    document.dispatchEvent(new CustomEvent("gep-theme", { detail: { id: id } }));
  }
  window.gepThemes = {
    list: function () { return THEMES.slice(); },
    current: function () { return current; },
    set: choose,
  };

  // The picker: a small select at the end of the top bar.
  function picker() {
    var bar = document.querySelector(".topbar");
    if (!bar || bar.querySelector(".theme-pick")) return;
    var select = document.createElement("select");
    select.className = "theme-pick";
    select.title = "Theme";
    select.setAttribute("aria-label", "Theme");
    select.style.cssText = "width:auto;flex:none;min-height:28px;padding:0 8px;font:inherit;font-size:12px;"
                         + "color:var(--ink);background:var(--panel);border:1px solid var(--line-2);border-radius:8px";
    THEMES.forEach(function (theme) {
      var option = document.createElement("option");
      option.value = theme.id;
      option.textContent = theme.name;
      select.appendChild(option);
    });
    select.value = current;
    select.addEventListener("change", function () { choose(select.value); });
    document.addEventListener("gep-theme", function (event) { select.value = event.detail.id; });
    bar.appendChild(select);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", picker);
  else picker();
})();
