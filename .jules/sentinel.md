## 2025-05-15 - [DOM-based XSS in Template Literals]
**Vulnerability:** Unsanitized variables (`typeLabel` and `communityId`) were interpolated directly into HTML template strings returned by `tooltipHtml` and pushed to `renderLegend`'s `rows` array in `docs/graph.html`.
**Learning:** Even when surrounding variables (like `label`, `summary`) use custom `escapeHtml()`, seemingly innocuous string properties mapped from backend values (like `d.type` or `communityId`) can be overlooked and become vectors for DOM-based XSS if the data source contains special characters.
**Prevention:** Always consistently apply sanitization functions (like `escapeHtml()`) to ALL variables interpolated into strings that will be assigned to `.innerHTML` or HTML attributes, regardless of their perceived "safe" origin.

## 2026-06-02 - [DOM-based XSS in Tooltips and Search Cards]
**Vulnerability:** Dynamic properties mapped from node data (`detailUrl` and `graphUrl`) were concatenated directly into the `href` and `data-result-url` attributes of HTML template strings in `docs/mini-graph.js`, `docs/graph.html`, and `docs/main.js` without escaping.
**Learning:** While `encodeURIComponent` mitigates some injection vectors, it does not escape single quotes, leaving the attribute vulnerable if the injected string is wrapped in single quotes, or if it breaks out in other ways. Additionally, standard codebase security practice demands explicit escaping for all interpolated variables destined for HTML insertion to ensure robust defense in depth.
**Prevention:** Consistently apply `escapeHtml()` to all URL variables before interpolating them into HTML attributes like `href="..."`, even when they appear to just consist of IDs or paths.
