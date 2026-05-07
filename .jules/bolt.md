## 2025-05-05 - RequestAnimationFrame for Scroll Listeners
**Learning:** `docs/main.js` heavily relies on scroll tracking (`updateActive`, `updateActiveTocLink`) for navigation highlighting. Without throttling, this causes layout thrashing via repeated `.offsetTop` and `getBoundingClientRect()` calls, stalling the main thread.
**Action:** Always wrap scroll-bound DOM reads/updates in `window.requestAnimationFrame()` to throttle them to screen refresh rates, drastically improving UI responsiveness. Ensure you do not accidentally include other auto-generated `.json` assets that may have shifted in a commit.

## 2026-05-06 - Tokenization Performance Bottleneck
**Learning:** In Python, calling millions of small functions (`normalize_token`, `_expand_cjk_segment`) and regular expressions (`re.fullmatch`) inside a hot loop like text tokenization (`tokenize_text`) causes severe performance overhead. Checking membership and extending lists with `.extend()` is much faster than repeatedly allocating empty lists (e.g., using `dict.get(key, [])`).
**Action:** When writing or optimizing indexing pipelines, inline simple helper functions to avoid call overhead. Use fast character range checks (e.g., `'\u4e00' <= char <= '\u9fff'`) instead of regex for basic unicode classification if the text is already pre-split.
