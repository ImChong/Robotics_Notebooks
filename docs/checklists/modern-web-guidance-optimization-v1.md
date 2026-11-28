# Modern Web Guidance 前端优化清单 v1

关联 Issue：[#4311](https://github.com/ImChong/Robotics_Notebooks/issues/4311)  
审计基准：`origin/main`（`7a09396f6`，2026-11-28）  
参考标准：[GoogleChrome/modern-web-guidance](https://github.com/GoogleChrome/modern-web-guidance)（全量 206 篇现代 Web 指南 + `npx modern-web-guidance@latest legacy-scan docs --json` 静态扫描 41 处确定性命中）

---

## 已落地确认项（无需重复投入）

- `star-history.html`：已移出阻塞 `<head>` 的 `vendor/plotly.min.js`，改为 `<link rel="preload" as="script">` + `ensurePlotly()` 按需加载，`wiki-activity.json` 延后至节点视图获取。
- `sponsor.js?v=2`：全站 11 个含顶栏页面已启用 `defer`。
- `style.css`：已应用 `color-scheme`、`color-mix()`、同文档 `document.startViewTransition()`、`scrollbar-gutter: stable`、`scrollbar-color` / `scrollbar-width`、`text-wrap: balance/pretty`、`:where(...) :focus-visible`、`prefers-reduced-motion`、`100dvh`、4 处 `content-visibility: auto`（`.home-hub-row` 窄屏、`.updates-day-list`、`.roadmap-major-section:not([open])`、`.co-node-card`）。
- `manifest.json` & `theme-init.js`：图标全部自托管至 `docs/assets/icons/`；`theme-init.js` 与 `main.js` 按暗/亮/UI 风格动态同步 `<meta name="theme-color">`。
- `graph.html`：搜索框 `#graph-search` 已处理 `compositionstart` / `compositionend` / `ev.isComposing`。
- `kb-chat.js`：保持 0 处 `innerHTML`，全部 DOM 节点采用安全构建。

---

## 按 ROI 排序总览

| 排名 | 梯队 | 优化项 | 对应指南 ID | 投入 | 收益 | 状态 |
| :-: | :-: | :--- | :--- | :-: | :-: | :-: |
| 1 | **S** | 修复中文输入法（IME）按回车误提交 / 误跳转 | `ime-safe-enter-submit` | 极低 | 极高 | 已完成 |
| 2 | **S** | 底部外链脚本非阻塞化（`defer`）与请求优先级调度（`fetchpriority` / `priority`） | `optimize-script-priority` / `deprioritize-background-fetches` | 低 | 极高 | 已完成 |
| 3 | **S** | MPA 跨页悬停预取（Speculation Rules API） | `improve-next-page-load-performance` | 极低 | 极高 | 已完成 |
| 4 | **S** | 补全其余屏下重区块的 `content-visibility: auto` | `defer-rendering-heavy-content` | 极低 | 高 | 已完成 |
| 5 | **S** | 补齐外部 CDN 资源 SRI（`integrity`）+ `crossorigin` + 全站 `Referrer-Policy` | `secure-external-subresources` | 极低 | 高 | 已完成 |
| 6 | **S** | AI 问答输入框启用 `field-sizing: content` 自动增高 | `form-fields-automatically-fit-contents` | 极低 | 高 | 已完成 |
| 7 | **S** | 启用跨文档视图过渡（`@view-transition`） | `cross-document-transitions` | 极低 | 高 | 已完成 |
| 8 | **A** | 首页折叠路线/公司改 `hidden="until-found"` 支持页内检索直达 | `search-hidden-content` | 低 | 高 | 待推进 |
| 9 | **A** | 移动端 `enterkeyhint` + OS 字体缩放（`<meta name="text-scale">`）+ PWA `maskable` 图标 | `respect-os-text-scale` / `form-factor-adaptation` | 低 | 中高 | 待推进 |
| 10 | **A** | 模态弹窗迁移原生 `<dialog>` + `showModal()` + `overscroll-behavior` | `dialog` / `light-dismiss-a-dialog` / `prevent-page-scroll-under-modal-dialog` | 中低 | 高 | 待推进 |
| 11 | **A** | 可访问性补强：伪元素静音（`content: "..." / ""`）+ `forced-colors` 高对比度模式 | `accessible-generated-content` / `color` | 低 | 中高 | 待推进 |
| 12 | **A** | 阅读进度环迁移至 CSS Scroll-Driven Animations | `scroll-progress-indicator` | 低 | 中高 | 待推进 |
| 13 | **A** | 清理 `legacy-scan` 检出的 40 处现代 JS 遗留写法 | `js` | 低 | 中 | 待推进 |
| 14 | **B** | 长循环引入 `scheduler.yield()` 优化交互响应（INP） | `break-up-long-tasks` | 中 | 中高 | 待推进 |
| 15 | **B** | Markdown 宽表格键盘可滚（`tabindex="0"`）+ 粘性表头 + 纯 CSS 滚动提示阴影 | `responsive-table` / `scrollability-affordance-hints` | 中低 | 中高 | 待推进 |
| 16 | **B** | 图谱筛选手风琴原生 `<details name>` + `::details-content` | `accordion` / `animate-to-intrinsic-sizes` | 中 | 中 | 待推进 |
| 17 | **B** | 3D 图谱后台标签页与离屏自动暂停渲染 | `efficient-background-processing` | 低中 | 中 | 待推进 |
| 18 | **B** | 渐进收紧 CSP（外置内联脚本去 `'unsafe-inline'`）与 HTML 消毒层 | `security` / `sanitize-untrusted-html` / `trusted-types` | 高 | 中高 | 待推进 |
| 19 | **C** | 新增浮窗采用 Popover + Anchor Positioning（既有跨 Canvas/SVG 浮窗维持现状） | `hovercard` / `viewport-aware-dropdowns` | 高 | 低中 | 暂缓 |
| 20 | **C** | 目录 Scrollspy 迁移 CSS `scroll-target-group` | `scrollspy` | 中 | 低中 | 暂缓 |
| 21 | **C** | 探索端侧 Built-in AI 与只读 WebMCP 工具注册 | `language-model` / `summarizer` / `webmcp` | 中高 | 探索 | 探索 |

---

## Tier S — 极高 ROI（2026-11-28 已落地）

- [x] **#1 修复 IME 组字回车误提交 / 误跳转**（`ime-safe-enter-submit`）：`docs/kb-chat.js`（`#kbChatInput`）与 `docs/main.js`（`#wikiSearchInput`）在 `Enter` `keydown` 分支补充 `isComposing || keyCode === 229` 守卫（覆盖 Safari WebKit bug 165004）。
- [x] **#2 脚本非阻塞化与资源优先级调度**（`optimize-script-priority` / `deprioritize-background-fetches`）：
  - 11 个含 `#sponsor-js` 页面将 `sponsor.js?v=2` 调整为 `async fetchpriority="low"`，避免在 `<head>` 占据 `defer` 队头阻塞后续 `defer` 脚本；
  - `index.html`、`detail.html`、`roadmap.html`、`hubs.html`、`company.html`、`lab.html`、`change-log.html`、`discovery.html`、`404.html` 底部外部脚本统一启用 `defer`（保留 `org-route-order.js` 在 `main.js` 之前，`main.js` 标注 `fetchpriority="high"`；`index.html` 位于 `main.js` 之后的尾随小图谱脚本 `vendor/d3.min.js`、`graph-perspective.js`、`mini-graph.js` 标注 `defer fetchpriority="low"`；`detail.html` 与 `graph.html` 避免在关键脚本之前产生低优先级队头阻塞）；
  - `docs/sw.js` 后台 SWR 刷新与 `docs/main.js` 静默预取（`prefetchWikiSearchIndex()`、`exports/link-graph-summaries.json`）补充 `priority: 'low'`，保持 `CACHE_NAME` 与主动搜索默认优先级不变。
- [x] **#3 接入 Speculation Rules API**（`improve-next-page-load-performance`）：`docs/main.js` 新增幂等 `installSpeculationRules()`，在 `HTMLScriptElement.supports('speculationrules')` 时注入 `prefetch`（`eagerness: "moderate"`），覆盖站内内容页并排除 `graph.html` 与 `star-history.html`。
- [x] **#4 补全其余屏下重区块 `content-visibility: auto`**（`defer-rendering-heavy-content`）：`docs/style.css` 对首页 `#home-hub-section` 及详情页 `#detail-related`、`#detail-recent-ingest-section`、`#detail-recommended`、`#detail-sources`、`#detail-recent` 启用 `content-visibility: auto` 与 `contain-intrinsic-size`，避免影响首屏 `#mini-graph-section`。
- [x] **#5 补齐外部 CDN SRI 与全站 `Referrer-Policy`**（`secure-external-subresources`）：
  - `docs/main.js` `ensureMermaidLoaded()` 为 `mermaid@12.3.0` 补充 `sha384-` SRI 校验，并在 `loadExternalAsset()` 对外域资源默认设置 `referrerPolicy = 'no-referrer'`；
  - `docs/star-history.html` 为 Plotly `2.35.2` CDN 回退补充 `integrity`、`crossOrigin="anonymous"` 与 `referrerPolicy="no-referrer"`；
  - 全站 12 个 HTML `<head>` 统一加入 `<meta name="referrer" content="strict-origin-when-cross-origin" />`。
- [x] **#6 AI 问答输入框启用 `field-sizing: content`**（`form-fields-automatically-fit-contents`）：`docs/style.css` `.kb-chat-input` 补充 `field-sizing: content; height: auto;`（保留 `min-height: 38px; max-height: 84px` 与 `flex: 1 1 auto`）。
- [x] **#7 启用跨文档视图过渡 `@view-transition`**（`cross-document-transitions`）：`docs/style.css` 在 `@media (prefers-reduced-motion: no-preference)` 下启用 `@view-transition { navigation: auto; }`，并用 `html:not(.is-theme-switching)` 与同文档主题圆形扩散动画隔离。

---

## Tier A — 高 ROI

- [ ] **#8 首页折叠路线/公司改用 `hidden="until-found"` + `beforematch`**（`search-hidden-content`）：替换 `docs/index.html` 与 `docs/main.js` 的布尔 `hidden`，让页内 `Ctrl/Cmd+F` 可直接定位并自动展开匹配路线。
- [ ] **#9 移动端 `enterkeyhint`、`<meta name="text-scale" content="scale">` 与 PWA `maskable` 图标**（`respect-os-text-scale` / `form-factor-adaptation`）。
- [ ] **#10 模态弹窗迁移原生 `<dialog>` + `showModal()`**（`dialog` / `light-dismiss-a-dialog` / `prevent-page-scroll-under-modal-dialog` / `animate-to-from-top-layer`）：覆盖 `#sponsorDialog`、`#mermaidLightbox`、`#roadmap-ltree-map`。
- [ ] **#11 可访问性补强：生成内容静音（`content: "..." / ""`）与 `@media (forced-colors: active)`**（`accessible-generated-content` / `color`）。
- [ ] **#12 阅读进度环迁移至 CSS Scroll-Driven Animations**（`scroll-progress-indicator`）：以 `@property --btt-progress` + `animation-timeline: scroll(root block)` 替代滚动时 JS 写变量。
- [ ] **#13 清理 `legacy-scan` 检出的 40 处现代 JS 遗留写法**（`js`）：`replaceChildren()`（20 处）、`AbortSignal.timeout()`（4 处）、`toSorted()`（16 处）。

---

## Tier B — 中高 ROI

- [ ] **#14 长循环引入 `scheduler.yield()` 优化交互响应（INP）**（`break-up-long-tasks`）。
- [ ] **#15 Markdown 表格键盘可滚（`tabindex="0"` + `role="region"`）+ 粘性表头 + 纯 CSS 滚动提示阴影**（`responsive-table` / `scrollability-affordance-hints`）。
- [ ] **#16 图谱筛选手风琴迁移原生 `<details name>` + `::details-content`**（`accordion` / `animate-to-intrinsic-sizes`）。
- [ ] **#17 `graph-3d.js` 结合 `visibilitychange` 后台暂停渲染**（`efficient-background-processing`）。
- [ ] **#18 渐进收紧 CSP（外置内联脚本去 `'unsafe-inline'`）与 Markdown 输出消毒层**（`security` / `sanitize-untrusted-html` / `trusted-types`）。

---

## Tier C — 低 ROI / 暂缓或探索型

- [ ] **#19 新组件采用 Popover + Anchor Positioning（既有跨 Canvas/SVG 的 `graph-tooltip.js` 维持现状）**（`hovercard` / `viewport-aware-dropdowns`）。
- [ ] **#20 目录 Scrollspy 评估迁移 CSS `scroll-target-group`**（`scrollspy`）。
- [ ] **#21 探索端侧 Built-in AI（`Summarizer` / `LanguageModel`）与只读 WebMCP 工具注册**（`language-model` / `summarizer` / `webmcp`）。
