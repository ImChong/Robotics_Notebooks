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

## 2026-05-10 - Native String Search vs Loop-based Property Matching
**Learning:** In the client-side search ranking (`substringScore`), checking for partial token matches by repeatedly evaluating `Object.keys()` and nesting a `for` loop with `.indexOf()` over each key is extremely slow in JavaScript. Constructing a single string representation with newline boundaries (e.g., `'\n' + Object.keys(docTokens).join('\n') + '\n'`) and executing a single `.indexOf()` delegates the search to highly optimized native C++ string methods, performing ~30-50% faster in hot O(N*M) ranking loops.
**Action:** When performing substring or partial matching across arrays or object keys in a critical loop, stringify the keys once with boundary delimiters (to prevent false cross-word matches) and rely on a single native `.indexOf()` call instead of nested JS iterations.
## 2026-05-11 - Search Scoring Inner Loop Optimization
**Learning:** In string-matching and BM25 hot loops, redundant allocations (like lowercase strings or  operations) and constant math computations executed per query token compound to create CPU overhead. Caching derived strings on static  instances (e.g. ) and hoisting document-level mathematical invariant calculations outside of the query loops drastically reduces unnecessary floating-point ops and GC thrashing.
**Action:** When optimizing performance-critical loops (, ), always identify and pull out calculations that don't depend on the current iteration variable, and lazily cache derived object properties that are re-evaluated frequently.

## 2026-05-11 - Search Scoring Inner Loop Optimization
**Learning:** In string-matching and BM25 hot loops, redundant allocations (like lowercase strings or `.join()` operations) and constant math computations executed per query token compound to create CPU overhead. Caching derived strings on static `doc` instances (e.g. `doc._title_l`) and hoisting document-level mathematical invariant calculations outside of the query loops drastically reduces unnecessary floating-point ops and GC thrashing.
**Action:** When optimizing performance-critical loops (`substringScore`, `compute_score`), always identify and pull out calculations that don't depend on the current iteration variable, and lazily cache derived object properties that are re-evaluated frequently.

## 2026-05-12 - Tokenization Caching Optimization
**Learning:** In the python search ranking code (`scripts/search_wiki_core.py`), computing the document token counts by repeatedly reading files and tokenizing them inside the scoring loop, as well as for the average document length (`compute_avgdl`), results in double the tokenization work.
**Action:** Always avoid redundant work inside data pipelines. When we already know the set of items that will be operated on multiple times, cache expensive operations (such as tokenizing raw content) either in local data structures or add it to existing pre-calculated dictionaries before iteration.

## 2026-05-14 - Python String Search and List Allocation Overhead
**Learning:** In Python, using `any()` with a generator expression inside a hot loop (like `any(word in line.lower() for word in lowered_words)`) creates massive generator overhead. Additionally, repeated string manipulation (`line.lower()`) and list extension/copying during synonym expansion inside loops adds significant garbage collection pressure.
**Action:** When performing substring scanning on a large volume of strings, compute invariant transformations (like `line.lower()`) once per target string. Use explicit `for` loops with `break` instead of `any()` to avoid generator allocation. Always strive to perform dictionary lookups and list extensions in-place directly on the target structure rather than creating multiple intermediate lists.
## 2025-05-15 - [Avoid N+1 File I/O in Hot Search Loop]
**Learning:** In backend CLI scripts that read multiple files (like `search_wiki_core.py`), it is a common anti-pattern to re-read files from disk inside a loop (`path.read_text()`) when their content is already available in memory (e.g., `doc["body"]` from a prior collection step). Doing this creates a massive N+1 file I/O bottleneck which severely degrades performance as the document collection scales.
**Action:** Always verify if needed data is already in memory before initiating file I/O operations inside iterative blocks. Replacing the `read_text` call with `doc["body"]` dramatically reduced CPU wait time. Additionally, properties derived from large strings inside loops (like total token count `dl`) should be computed once and cached on the document dictionary to avoid redundant sum/iteration overheads.
