(function () {
  var themeKey = 'robotics-notebooks-theme';
  var uiKey = 'robotics-notebooks-ui';
  var savedTheme = null;
  var savedUi = null;
  try {
    savedTheme = localStorage.getItem(themeKey);
    savedUi = localStorage.getItem(uiKey);
  } catch (_e) { /* storage disabled in private mode */ }
  var dark = savedTheme ? savedTheme === 'dark' : true;
  var theme = dark ? 'dark' : 'light';
  var ui = savedUi === 'classic' ? 'classic' : 'tech';
  var root = document.documentElement;
  root.setAttribute('data-theme', theme);
  root.setAttribute('data-ui', ui);

  var themeColorMap = {
    tech: { dark: '#0a0b0d', light: '#f5f4f0' },
    classic: { dark: '#191919', light: '#ffffff' }
  };
  var themeMeta = document.querySelector('meta[name="theme-color"]');
  if (themeMeta) {
    themeMeta.setAttribute('content', themeColorMap[ui][theme]);
  }
})();
