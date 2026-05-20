(function () {
  var key = 'robotics-notebooks-theme';
  var saved = localStorage.getItem(key);
  var dark = saved ? saved === 'dark' : true;
  document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light');
})();
