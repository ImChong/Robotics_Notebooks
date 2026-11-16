(function () {
  var key = 'robotics-notebooks-theme';
  var saved = null;
  try {
    saved = localStorage.getItem(key);
  } catch (_e) { /* storage disabled in private mode */ }
  var dark = saved ? saved === 'dark' : true;
  document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light');
})();
