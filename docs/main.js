(function () {
  const root = document.documentElement;
  const themeToggle = document.getElementById('themeToggle');
  const key = 'robotics-notebooks-theme';
  const saved = localStorage.getItem(key);
  const preferDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const dark = saved ? saved === 'dark' : preferDark;
  root.setAttribute('data-theme', dark ? 'dark' : 'light');

  if (themeToggle) {
    themeToggle.addEventListener('click', function () {
      const isDark = root.getAttribute('data-theme') === 'dark';
      root.setAttribute('data-theme', isDark ? 'light' : 'dark');
      localStorage.setItem(key, isDark ? 'light' : 'dark');
    });
  }

  const links = document.querySelectorAll('.main-nav a');
  const sections = Array.from(links)
    .map(link => document.querySelector(link.getAttribute('href')))
    .filter(Boolean);

  function updateActive() {
    const scrollPos = window.scrollY + 100;
    let currentId = sections.length ? '#' + sections[0].id : '';
    sections.forEach(section => {
      if (section.offsetTop <= scrollPos) currentId = '#' + section.id;
    });
    links.forEach(link => {
      link.classList.toggle('active', link.getAttribute('href') === currentId);
    });
  }

  if (sections.length) {
    window.addEventListener('scroll', updateActive);
    updateActive();
  }
})();
