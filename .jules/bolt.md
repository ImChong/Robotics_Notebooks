## 2025-05-05 - RequestAnimationFrame for Scroll Listeners
**Learning:** `docs/main.js` heavily relies on scroll tracking (`updateActive`, `updateActiveTocLink`) for navigation highlighting. Without throttling, this causes layout thrashing via repeated `.offsetTop` and `getBoundingClientRect()` calls, stalling the main thread.
**Action:** Always wrap scroll-bound DOM reads/updates in `window.requestAnimationFrame()` to throttle them to screen refresh rates, drastically improving UI responsiveness. Ensure you do not accidentally include other auto-generated `.json` assets that may have shifted in a commit.

## 2026-05-06 - Tokenization Performance Bottleneck
**Learning:** In Python, calling millions of small functions (`normalize_token`, `_expand_cjk_segment`) and regular expressions (`re.fullmatch`) inside a hot loop like text tokenization (`tokenize_text`) causes severe performance overhead. Checking membership and extending lists with `.extend()` is much faster than repeatedly allocating empty lists (e.g., using `dict.get(key, [])`).
**Action:** When writing or optimizing indexing pipelines, inline simple helper functions to avoid call overhead. Use fast character range checks (e.g., `'\u4e00' <= char <= '\u9fff'`) instead of regex for basic unicode classification if the text is already pre-split.

## 2026-05-07 - Frontend Search Array Iteration
**Learning:** In `docs/main.js`, chaining array methods like `.filter(...).map(...)` on large arrays (e.g., search indexes with hundreds or thousands of documents) creates unnecessary intermediate arrays and can cause expensive functions (like `substringScore`) to be evaluated redundantly (once in `filter`, once in `map`).
**Action:** For performance-critical data processing in the frontend, replace chained array methods with a single-pass `for` loop (or `.reduce()`) to minimize array allocations and prevent redundant calculations.

## 2026-05-08 - Garbage Collection and Hot Loop Optimization
**Learning:** In the client-side search indexing loop (`docs/main.js`), generating arrays continuously with methods like `.map()` and `Object.keys()` on every single search token for every single document created massive garbage collection pressure and CPU overhead, slowing search by ~40x.
**Action:** When working on critical O(N*M) hot loops in JavaScript, avoid memory allocations where possible. Initialize strings and keys lazily, and use traditional `for` loops instead of `.forEach()` or `.some()`. Pre-calculate constants (like BM25 length normalization) outside inner loops.
