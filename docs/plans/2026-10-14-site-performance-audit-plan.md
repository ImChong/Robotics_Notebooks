# 网站全局优化计划（2026-10-14）

> 状态：**阶段 A（P0 第 1–5 条）已实施，见同名 PR；阶段 B / C 待做**
> 目的：把网站现在最拖慢读者的地方找出来，按「改动小、收益大」排序，先定计划再动手。
> 每条都标了依据：**实测** = 本次用无头 Chromium 量出来的数字；**推测** = 根据实测数据的推断，需要改完再验证。

---

## 一、怎么测的

- 仓库根目录 `make export graph` 生成站点数据，`cd docs && python3 -m http.server 8765` 起本地服务。
- 无头 Chromium（Playwright 1.56）打开各页面，记录：请求数、字节数、DOMContentLoaded、内容就绪时间（`data-loading` 消失）、主线程长任务、JS 堆内存、JS/CSS 实际用到的比例。
- 桌面 1440×900 无节流；另跑一组「手机视口 + CPU 慢 4 倍」模拟弱设备。
- 本地服务不压缩，所以「下载量」另外用 `gzip -6` 算，并用 `curl -I` 核对线上 GitHub Pages 的真实 `content-length`（线上开了 gzip，缓存 `max-age=600`）。
- 本机 CPU 偏弱，**毫秒数偏悲观，看比例不看绝对值**。

---

## 二、现状一览（实测）

| 页面 | 请求数 | 原始下载 | 线上 gzip 后大约 | 内容就绪 | 主线程长任务 | 手机+CPU×4 就绪 |
|---|---|---|---|---|---|---|
| 首页 `index.html` | 15 | 18.7 MB | ≈ 3.5 MB | 1.1–2.5 s | 0.35–0.43 s | 3.9 s |
| 详情页 `detail.html` | 22 | 22.1 MB | ≈ 4.3 MB | 1.2–2.7 s | 1.3–1.7 s | 9.0 s |
| 路线页 `roadmap.html?id=roadmap-motion-control` | 23 | 19.4 MB | ≈ 3.8 MB | 1.5–7.3 s | **6.5 s** | **34.6 s** |
| 图谱页 `graph.html` | 14 | 18.8 MB | ≈ 3.6 MB | 10–15 s | 11–18 s | 未测 |
| 枢纽榜 `hubs.html` | 8 | 3.2 MB | ≈ 0.75 MB | 0.3 s | 0.05 s | 未测 |
| 公司页 `company.html` | 9 | 0.8 MB | ≈ 0.3 MB | 0.3 s | 0.1 s | 未测 |
| 更新记录 `change-log.html` | 8 | 9.9 MB | ≈ 1.8 MB | 0.45 s | 0 | 未测 |

结论一句话：**页面壳子（HTML/CSS/JS）不大，拖慢的是几份巨大的 JSON 数据和路线页的 33 张 Mermaid 图。**

### 数据文件体积（实测，原始 / gzip）

| 文件 | 原始 | gzip | 谁在用 |
|---|---|---|---|
| `search-index.json` | **106 MB** | **26.3 MB** | 首页搜索框聚焦时加载 |
| `exports/link-graph.json` | 9.3 MB | 1.4 MB | 首页、详情页、路线页、图谱页、增长趋势页 |
| `exports/site-catalog-v1.json` | 9.1 MB | 1.9 MB | 详情页、路线页、图谱页 |
| `exports/wiki-activity.json` | 9.7 MB | 1.5 MB | 更新记录页、增长趋势页 |
| `exports/hub-rankings.json` | 2.7 MB | 0.43 MB | 详情页、枢纽榜 |
| `exports/index-v1.json` | 45.7 MB | 12.3 MB | **前端从不请求** |
| `exports/site-data-v1.json` | 45.6 MB | 12.3 MB | **前端从不请求** |
| `main.js` | 332 KB | 85 KB | 每一页 |
| `style.css` | 118 KB | 29 KB | 每一页 |
| `sponsor/wechat-pay.png` | 206 KB | — | 每一页（隐藏的赞助弹窗里） |

---

## 三、问题清单（按「值不值得先做」排序）

### P0：改动小、见效快

**1. 路线页 33 张 Mermaid 图在打开时一次全画**
- 证据（实测）：`roadmap/motion-control.md` 有 33 个 mermaid 块，页面打开时单个长任务 4.5 s（桌面）/ 26 s（手机+CPU×4）。把 Mermaid 请求屏蔽后再测，长任务从合计 6.5 s 降到 0.7 s。
- 怎么改：图滚到视口附近再渲染（IntersectionObserver），或分批渲染让出主线程。`main.js` 的 `renderDetailMermaid` 一处改动。
- 预期效果：首屏不再卡死；主线程阻塞从 6.5 s 降到 1 s 以内（**推测**，渲染成本会分散到滚动时）。手机上从「30 秒点不动」变成正常可滚。

**2. 首页同一次加载请求了两遍 `link-graph.json`**
- 证据（实测）：首页请求列表里 `exports/link-graph.json` 出现 2 次，各 9 MB。原因是 `mini-graph.js` 自己 fetch 一份，`main.js` 里搜索区的社区下拉（`ensureCommunityByPath`）在页面加载时又 fetch 一份，两者并行发出，缓存来不及合并。
- 怎么改：两处共用一个 Promise（`main.js` 已有 `ensureLinkGraphData`，让 `mini-graph.js` 复用；或社区下拉改读 `home-stats.json` 里已有的社区列表）。
- 预期效果：首页少下载 1.4 MB（gzip）、少解析一次 9 MB JSON；首页总下载量约从 3.5 MB 降到 2.1 MB（**实测体积推算**）。

**3. 赞助二维码 206 KB 每页都下载**
- 证据（实测）：每个页面的请求列表都有 `sponsor/wechat-pay.png`，图片在隐藏弹窗里，读者没点也会下载。它比 `main.js + style.css` 的 gzip 总和（115 KB）还大。
- 怎么改：`sponsor.js` 里点击「赞助」时再给 `<img>` 赋 `src`，或加 `loading="lazy"` 并在弹窗打开时插入。
- 预期效果：每页少 206 KB，弱网首屏更快（**推测**：按 4G 约省 0.3–0.5 s）。

**4. 详情页为了两个小字段多下载 2.7 MB `hub-rankings.json`**
- 证据（实测 + 读代码）：`main.js` 的 `ensureDetailCommunityIndex` 拉 `hub-rankings.json` 只为了建 `has_repo` 和 `node_type` 两张映射表；这两项详情页已经加载的 `site-catalog-v1.json` 里都有。
- 怎么改：映射表改从目录数据建，详情页不再请求 `hub-rankings.json`（枢纽榜页不受影响）。
- 预期效果：每个详情页少 0.43 MB（gzip）、少解析 2.7 MB JSON。

**5. 部署产物里有 91 MB 前端从不请求的 JSON**
- 证据（实测）：`index-v1.json` 与 `site-data-v1.json` 合计 91 MB，被镜像到 `docs/exports/` 随站点一起上传，但 `docs/` 下所有 HTML/JS 都没有请求它们。
- 怎么改：部署步骤不再把这两份镜像进 `docs/exports/`（本地 `exports/` 仍保留，脚本和截图流程还用）。需要同步改 `CLAUDE.md`、`cloud-agent-pr-workflow.md` 里「在 `docs/exports/site-data-v1.json` 查页面 id」的说法。
- 预期效果：Pages 产物从 357 MB 降到 266 MB，部署更快、更不容易撞上 Pages 体积上限；读者无感知。

### P1：数据格式瘦身（要改 Python 导出脚本 + 对应 JS，中等工作量）

**6. 详情页每次都要先下载 9 MB 全站目录**
- 证据（实测）：`site-catalog-v1.json` 8.4 MB 是 5988 个页面的元数据，其中 `source_links` 3.3 MB、`related` 1.4 MB。详情页只需要当前页的元数据 + 关联页的标题。
- 怎么改：当前页的元数据和关联项标题并进已有的「每页正文 JSON」（`exports/page-content/<hash>.json`）；全站目录只留 `id → [标题, 类型]` 加别名表。
- 预期效果：目录从 1.9 MB 降到 0.18 MB（gzip，**实测**按新格式算出），解析从 9 MB 降到 0.55 MB；详情页总下载量约减半，长任务明显减少（**推测**）。

**7. `link-graph.json` 的边用完整路径字符串**
- 证据（实测）：5.2 万条边占 6.2 MB，每条边重复写两个长路径；节点里的 `summary` 又占 0.67 MB。
- 怎么改：边改成节点下标 `[i, j]`，`summary` 移到单独文件只给图谱侧栏用。要同步改 `graph.html`、`mini-graph.js`、`main.js`、`star-history.html` 的读取代码。
- 预期效果：9.3 MB / 1.4 MB → 2.6 MB / 0.46 MB（原始 / gzip，**实测**按新格式算出）。首页、详情页、路线页、图谱页都受益。

**8. `wiki-activity.json` 每条记录重复带标题、路径、社区名**
- 证据（实测）：3.4 万条节点引用占 9.2 MB，标题/类型/社区名都能从目录查回。
- 怎么改：只保留 `[detail_id, action]`，渲染时查目录。
- 预期效果：9.7 MB / 1.5 MB → 2.1 MB / 0.47 MB（**实测**按新格式算出）；更新记录页与增长趋势页各省 1 MB 下载。

**9. 搜索索引 106 MB（gzip 26 MB），点一下搜索框就开始下**
- 证据（实测）：`tokens` 字段 64 MB + `idf` 37.5 MB，词表 71.7 万个 token，里面大量是 `------`、`..`、`genesis-sim.md` 这类无意义碎片。线上 `content-length` 为 26.3 MB。
- 怎么改：分两层。第一层只含标题/摘要/标签/别名（2.5 MB / 0.86 MB gzip，**实测**），聚焦搜索框就加载，立刻能搜；第二层正文全文索引只在第一层结果不够或用户切「全文」时再加载，同时清理词表、只保留出现在 ≥2 篇文档的词。
- 预期效果：首次搜索下载 26 MB → 0.86 MB，搜索框从「等几十秒」变成「秒出」（**推测**）；全文搜索仍可用，只是变成按需。
- 备注：试算过「剪枝后的倒排索引」仍有 15 MB gzip，不够小，所以推荐分层而不是单纯换格式。

### P2：工程层面（收益稳定，但不急）

**10. JS/CSS 没有压缩**
- 证据（实测，esbuild `--minify` 试算）：`main.js` 85 KB → 48 KB、`style.css` 29 KB → 16 KB、`graph-3d.js` 22 KB → 8 KB（均为 gzip 后）。
- 怎么改：`pages.yml` 部署时加一步 minify 到 `docs/` 产物，源码不动，ESLint 和测试照常跑源文件。
- 预期效果：每页少约 50 KB 下载；对数据体积大的页面是小头，对公司页/枢纽榜这类轻页面是主要收益。

**11. `main.js` 每页只用到 5%–34%**
- 证据（实测字节覆盖率）：首页 12%、详情页 34%、路线页 30%、图谱页 5%、枢纽榜 10%。
- 怎么改：按页面拆成几个文件（首页 / 详情+路线 / 其他），公共部分抽出。这是 8000 行单文件的重构，风险高。
- 预期效果：每页 JS 下载再降 60%–90%（**推测**）。建议先做第 10 条，这条放最后。

**12. `style.css` 每页只用到 13%–38% 的规则**
- 证据（实测）。
- 预期效果：拆分后每页省 10–15 KB gzip（**推测**）。收益小，可和第 11 条一起做或不做。

**13. `graph.html` 263 KB，其中约 4600 行是内联脚本**
- 证据（实测）：内联脚本随 HTML 一起下载，不能单独缓存。
- 怎么改：抽到 `graph.js`。图谱页本身的渲染性能（就绪 10–15 s、堆 125 MB）已有专项（`docs/checklists/graph-perf-*.md`），本计划不重复。

**14. 增长趋势页在 `<head>` 里同步加载 1.2 MB 的 Plotly**
- 证据（实测）：`plotly.min.js` 1204 KB 从 CDN 同步加载，页面要等它下完才开始渲染。
- 怎么改：加 `defer`，或等用户切到图表再加载。
- 预期效果：首屏提前约 1 s（**推测**，取决于 CDN 速度）。

**15. 其他小项**
- 没有 `robots.txt` 和 `404.html`。
- `manifest.json` 的图标指向外站 icons8，PWA 安装依赖第三方；改成本地图标。
- 5958 个详情页都是 `detail.html?id=...` 由 JS 渲染，标题/描述由 JS 写入；搜索引擎能否正确收录是**推测**，若在意 SEO 可考虑部署时为每页预渲染静态 HTML（约 +60 MB 产物），这是单独决策。
- 仓库里有一个 88 MB 的 mp4（git 包 172 MB，它占一半），每次克隆都要拉。可移到 GitHub Release 附件；历史里的那份要重写历史才能删，建议只讨论、不轻动。

---

## 四、分阶段执行计划

每一步都写了「怎么验证」，照本文第一节的测法再量一次即可对比。

```
阶段 A（P0，预计 1–2 个 PR，改动都在几十行内）
  1. 路线页 Mermaid 懒渲染          → 验证：roadmap 页长任务合计 < 1 s，33 张图滚动到位都能正常显示；现有 tests/*.test.cjs 全过
  2. 首页 link-graph 只请求一次      → 验证：首页请求列表只出现 1 次 link-graph.json；mini-graph 与社区下拉功能不变
  3. 赞助二维码点击再加载            → 验证：每页请求列表无 wechat-pay.png；点「赞助」弹窗能看到二维码
  4. 详情页不再拉 hub-rankings.json  → 验证：详情页请求数 -1；has_repo 徽标、节点类型标签显示不变
  5. 部署不镜像 index-v1/site-data   → 验证：make ci-preflight 通过；Pages 产物体积下降；相关文档已更新

阶段 B（P1，每项一个 PR，改导出脚本 + 前端读取 + 对应测试）
  6. 目录瘦身 + 每页 JSON 带元数据  → 验证：详情页 gzip 下载 < 2 MB；关联项/相关推荐/来源链接显示不变
  7. link-graph 边改下标            → 验证：首页/详情/路线/图谱/趋势页全部正常；文件 gzip < 0.5 MB
  8. wiki-activity 瘦身             → 验证：更新记录页热力图与时间线不变
  9. 搜索索引分层                   → 验证：聚焦搜索框首次下载 < 1 MB；scripts/eval_search_quality.py 的搜索质量回归不下降

阶段 C（P2，视精力）
  10. 部署时 minify                 → 验证：线上 main.js content-length 约 48 KB
  13/14/15. graph.html 外提、Plotly defer、robots/404/本地图标
  11/12. main.js / style.css 按页拆分（最后做，先看阶段 A/B 之后还剩多少问题）
```

### 预期总效果（阶段 A + B 做完，按实测体积推算，时间为推测）

| 页面 | 现在 gzip 下载 | 之后 gzip 下载 | 说明 |
|---|---|---|---|
| 首页 | ≈ 3.5 MB | ≈ 1.1 MB | 去重复 link-graph、瘦身 link-graph、去二维码 |
| 详情页 | ≈ 4.3 MB | ≈ 1.2 MB | 目录瘦身、link-graph 瘦身、去 hub-rankings、去二维码 |
| 路线页 | ≈ 3.8 MB | ≈ 1.1 MB | 同上，且不再被 Mermaid 卡住 |
| 首次搜索 | 26 MB | 0.86 MB | 分层索引 |
| 更新记录 | ≈ 1.8 MB | ≈ 0.7 MB | wiki-activity 瘦身 |

---

## 五、已经做得好的、不建议动

- 3D 图谱库（1.3 MB）只在切到 3D 时加载；KaTeX、Mermaid 按正文内容按需加载；搜索索引只在有搜索意图时加载。这些都有测试守着（`tests/graph-3d-lazy.test.cjs`、`on-demand-assets.test.cjs`、`search-prefetch.test.cjs`）。
- 用系统字体，没有网络字体开销；主题初始化脚本很小且放在头部防闪白。
- Service Worker 只预缓存外壳，大数据按访问缓存，思路是对的。
- 正文已按页拆成单独 JSON（`page-content/<hash>.json`，中位数约 9 KB），这是第 6 条瘦身能做的基础。

---

## 六、复现本次测量

```bash
make export graph                       # 浅克隆时 graph_exports_sync 会拒绝同步，手动 cp exports/*.json docs/exports/
cd docs && python3 -m http.server 8765 &
# 测量脚本本次放在会话 scratchpad，未入库；若要固化，可参考 scripts/measure_graph_perf.cjs 的写法新增 scripts/measure_site_load.cjs
```
