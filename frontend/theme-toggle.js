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

  function applyTheme(theme) {
    root.setAttribute('data-theme', theme);
    var btn = document.getElementById('theme-toggle');
    if (btn) btn.textContent = theme === 'dark' ? 'LIGHT' : 'DARK';
  }

  // Initial theme: stored preference wins, otherwise OS preference.
  var initial = getStored() || systemPref();
  applyTheme(initial);

  // Toggle button wiring.
  document.addEventListener('DOMContentLoaded', function () {
    var btn = document.getElementById('theme-toggle');
    if (!btn) return;
    btn.textContent = initial === 'dark' ? 'LIGHT' : 'DARK';
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
