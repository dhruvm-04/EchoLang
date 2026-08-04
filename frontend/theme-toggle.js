/* Theme toggle for the Dhruv editorial/bento UI system.
   Drop this script anywhere on the page (defer or before </body>).
   Requires a button in the nav with id="theme-toggle" — see
   references/components.md for the markup pattern.
   Works by setting document.documentElement.dataset.theme and
   persisting the choice to localStorage under 'theme-preference'. */
(function () {
  var STORAGE_KEY = 'theme-preference';
  var root = document.documentElement;

  function systemPref() {
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
      ? 'dark'
      : 'light';
  }

  function getStored() {
    try { return localStorage.getItem(STORAGE_KEY); } catch (e) { return null; }
  }

  function setStored(value) {
    try { localStorage.setItem(STORAGE_KEY, value); } catch (e) { /* ignore */ }
  }

  function iconMarkup(theme) {
    if (theme === 'dark') {
      return '<svg class="theme-icon" aria-hidden="true" viewBox="0 0 24 24" focusable="false"><circle cx="12" cy="12" r="4" fill="none" stroke="currentColor" stroke-width="1.8"/><g stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M12 2.5v3"/><path d="M12 18.5v3"/><path d="M2.5 12h3"/><path d="M18.5 12h3"/><path d="M4.7 4.7l2.1 2.1"/><path d="M17.2 17.2l2.1 2.1"/><path d="M19.3 4.7l-2.1 2.1"/><path d="M6.8 17.2l-2.1 2.1"/></g></svg>';
    }
    return '<svg class="theme-icon" aria-hidden="true" viewBox="0 0 24 24" focusable="false"><path d="M21 12.8A8.5 8.5 0 0 1 11.2 3a9 9 0 1 0 9.8 9.8Z" fill="currentColor"/></svg>';
  }

  function applyTheme(theme) {
    root.setAttribute('data-theme', theme);
    var btn = document.getElementById('theme-toggle');
    if (btn) {
      btn.innerHTML = iconMarkup(theme);
      btn.setAttribute('aria-label', theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode');
    }
  }

  // Initial theme: stored preference wins, otherwise OS preference.
  var initial = getStored() || systemPref();
  applyTheme(initial);

  // Toggle button wiring.
  document.addEventListener('DOMContentLoaded', function () {
    var btn = document.getElementById('theme-toggle');
    if (!btn) return;
    btn.addEventListener('click', function () {
      var current = root.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
      var next = current === 'dark' ? 'light' : 'dark';
      applyTheme(next);
      setStored(next);
    });
  });

  // Keep in sync if the user changes their OS theme and hasn't explicitly chosen one.
  if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function (e) {
      if (!getStored()) applyTheme(e.matches ? 'dark' : 'light');
    });
  }
})();
