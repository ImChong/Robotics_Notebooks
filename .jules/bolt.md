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

## 2026-05-15 - Hoisting Invariant Properties Out of Hot Loops
**Learning:** In the frontend JavaScript search loop (`docs/main.js`), repeatedly accessing deep object properties (like `indexData.meta.k1`, `indexData.meta.b`) and recalculating invariant values (like `k1 + 1`) inside a hot loop (evaluating `bm25Score` for every matched document) causes measurable performance degradation due to redundant property lookups and constant math operations across hundreds or thousands of iterations.
**Action:** When optimizing performance-critical loops that evaluate every document in a large array, always hoist invariant object properties and pre-computable constant math (like `k1 + 1` or extracting `idfMap`) to local variables outside the loop. This minimizes redundant CPU work and property lookup overhead per iteration.

## 2026-05-20 - String Operations Optimization
**Learning:** In string sanitization (like HTML escaping), chaining multiple `.replace()` calls with global regular expressions (`.replace(/&/g, '&amp;').replace...`) involves repeatedly parsing the entire string and allocating multiple intermediate string objects.
**Action:** When a function executing basic character escaping is called extremely frequently, write a manual character iteration loop using `charCodeAt()` and build the resulting string by slicing (`substring`). While more verbose, this runs >3x faster.

## 2026-05-20 - Regex and Set Initialization Overhead
**Learning:** Instantiating `new Set(...)` and `new RegExp(...)` (or literal `/.../g`) inside functions that are called repeatedly (like line-by-line syntax highlighters) forces the JavaScript engine to reallocate and recompile these objects on every single function call.
**Action:** In JavaScript hot paths, always hoist static structures (like sets of keywords or fixed regular expressions) to the outer scope to instantiate them exactly once.
## 2026-05-25 - Prevent N+1 Tokenization and File Read Bottlenecks in Search Loop
**Learning:** In backend search logic, calling a document retrieval function (`iter_wiki_documents()`) that hits the file system inside the main `search` function causes large bottlenecks if search is called multiple times. Furthermore, expensive tokenization (via `tokenize_text`) should only be calculated once per document, rather than re-computing it on each new search request.
**Action:** When a loop computes properties over a collection, verify whether the collection retrieval causes file system hits and memoize it. When computing computationally heavy structures (like token counts or derived metrics) on documents, mutate the document dictionary to cache it and skip recalculation on subsequent operations.

## 2026-05-30 - pathlib.Path.relative_to Performance Bottleneck
**Learning:** In Python, calling `pathlib.Path.relative_to()` inside a hot loop (like traversing thousands of files) adds massive overhead because it instantiates a new `Path` object and runs internal path validation and string manipulation logic on every call.
**Action:** When extracting sub-paths from known, validated absolute paths inside a hot loop, calculate the base path's part length once (`REPO_PARTS_LEN = len(REPO_ROOT.parts)`) and use direct tuple slicing `path.parts[REPO_PARTS_LEN:]`. This performs the same logical operation nearly 5x faster by bypassing `pathlib` instantiation entirely.
