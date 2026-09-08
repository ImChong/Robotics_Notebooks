# 图谱动态渲染性能基线（G0）

对应 [工程计划 · 图谱动态渲染性能专项](../../plan.md#2026-09-08-图谱动态渲染性能专项) 的 **G0 / #35**。
本页只记录**测量方法与实测数字**，不含优化实现；G1–G6 的前后对比必须复用本页的数据与脚本。

复现命令（仓库根目录）：

```bash
make export graph                      # 生成 exports/link-graph.json（浅克隆下 graph_exports_sync 会拒绝同步 docs/exports，手动 cp 即可）
cp exports/link-graph.json exports/hub-rankings.json exports/wiki-activity.json docs/exports/
cd docs && python3 -m http.server 8765 &
node scripts/measure_graph_perf.cjs    # 结果写入 .cursor-artifacts/graph-perf-baseline.json
```

## 1. 固定数据与环境

| 项 | 值 |
| --- | --- |
| `exports/link-graph.json` SHA-256 前 16 位 | `224d420972e9b4c2` |
| 节点 / 边 | 3,867 / 34,409（22 社区） |
| 测量时间 | 2026-09-08 |
| 浏览器 | headless Chromium 141.0.7390.37（puppeteer-core） |
| CPU | Intel Xeon @ 2.10GHz，4 逻辑核 |
| 视口 / DPR | 1440×900 / 1 |
| CPU 节流 | 无 |
| 缓存 | 冷缓存（`setCacheEnabled(false)`） |
| 重复次数 | tick 类 20 次 / 刷新布局 3 轮 / 筛选 5 次 |

**未测（不得据本页外推）**：真机独显/集显 GPU 帧率、手机、Safari、CPU 节流档位、Service Worker 热缓存二次访问。
headless 走软件光栅（SwiftShader），**3D 帧率与 GPU 占用一律记为未测**；本页 3D 只取与硬件无关的 draw calls / 三角形数。
本机 CPU 弱于常见开发笔记本，**绝对毫秒偏悲观**；用于决策的是各段之间的**比例**。

## 2. 结构计数（精确，与硬件无关）

| 指标 | 实测 |
| --- | --- |
| 力模拟节点数 `simulation.nodes()` | 3,867 |
| 力模拟边数 `forceLink.links()` | 34,409 |
| DOM `.edges line` | 34,409 |
| DOM `.nodes g.node-g` | 3,867 |
| DOM 社区标签 `g.community-label` | 21 |
| `#graph-canvas` 元素总数 | 49,944 |
| **每 tick 坐标属性写入** | **141,503**（`4 × 34,409 + 3,867`） |

计划中按旧快照（3,861/34,329）估的 141,177 得到验证；口径与 `syncGraphDomFromSimulation` 一致。

## 3. 主线程耗时分段（每次 tick，单位 ms）

先 `simulation.stop()` 再逐段计时，避免内部 tick 干扰。

| 分段 | p50 | p95 | max | 说明 |
| --- | --- | --- | --- | --- |
| 力计算 `simulation.tick()` | **45.6** | 48.0 | 48.0 | 纯 JS 数值计算，与 GPU 无关 |
| DOM 同步 `syncGraphDomFromSimulation()` | **89.3** | 111.6 | 111.6 | 含社区标签 |
| └ 社区标签 `updateCommunityLabels()` | 1.2 | 1.4 | 1.4 | 已有 bbox 缓存生效 |
| └ 派生：边/节点坐标写入 | **≈88.1** | — | — | DOM 同步 − 社区标签 |
| `applyFilters()` 单次 | 132.2 | 203.6 | 203.6 | 非每帧，但每次筛选都阻塞主线程 |

**合计每 tick ≈ 135 ms**（45.6 + 89.3），即本机满速也只有约 7.4 tick/s，远低于 16.7 ms 帧预算。
多次重跑的 p50 在 tick 43–48 ms、DOM 同步 80–107 ms 区间波动，**DOM 侧始终约为力计算的 1.9–2.2 倍**，该比例是本页最稳定的结论。

> 测量边界：`performance.now()` 只包住 JS 与属性写入，浏览器的样式/布局/绘制发生在其后的帧内，**未计入**。
> 因此 88.1 ms 是「属性写入成本」的下界，真实 SVG 渲染开销只会更高。

## 4. 「刷新布局」按钮：长任务与帧耗时

`restartForceSimulation()` 走同步 warmup（`runStabilizedLayout2D({ reseed: true })`，未传 `async`），点击后采样 4 s：

| 轮次 | >50ms 长任务数 | 长任务总时长 | 最长单个长任务 | 帧耗时 p50 | 采样帧数 |
| --- | --- | --- | --- | --- | --- |
| 1 | 14 | 5,619 ms | 1,328 ms | 330.4 ms | 13 |
| 2 | 16 | 5,353 ms | 1,228 ms | 267.8 ms | 14 |
| 3 | 16 | 5,260 ms | 1,177 ms | 258.0 ms | 15 |

4 s 窗口内**几乎全部时间都在长任务中**，单个长任务最长约 1.3 s；采样期内**没有任何一帧低于 33.3 ms**（≈3–4 FPS）。
这同时印证计划里两点：同步 warmup 造成整块卡顿；现有 `async` 分支的固定 10 tick 分片也不构成时间预算（单 tick 已 ≈135 ms，远超一帧）。

## 5. 筛选是否形成真实子图

拖动「显示节点数」到 Top 300 后读取模拟与 DOM：

| | 筛选前 | Top 300 后 |
| --- | --- | --- |
| `simulation.nodes()` | 3,867 | **3,867** |
| `forceLink.links()` | 34,409 | **34,409** |
| DOM `.edges line` | 34,409 | **34,409** |
| DOM `.nodes g.node-g` | 3,867 | **3,867** |

**计划中「普通筛选仍计算全图」得到实测确认**：Top N 只改 opacity / pointer-events，力计算与 DOM 规模完全不变。G3 的必要性成立。

## 6. 3D 视图（draw calls / 三角形数）

全图进入 3D 后连续 5 帧读取 `renderer.info`，数值完全稳定：

| 指标 | 实测 |
| --- | --- |
| **每帧 draw calls** | **38,272** |
| **每帧三角形数** | **1,660,608** |
| `lines` / `points` | 0 / 0 |
| 几何体数 `memory.geometries` | 2 |
| 材质程序数 | 1 |
| 像素比 | 1 |
| canvas | 1425×796 |

场景内对象数为 `3,867 节点 + 34,409 边 = 38,276`，实测 38,272 次调用（差 4 为视锥剔除），
即**每个节点 Mesh、每条边圆柱各占一次 draw call，无任何合批**。
几何体只有 2 个（球 + 圆柱），说明节点/边几何确实共享（「已有优化」属实），但**共享几何不减少 draw call**。
平均每个对象约 43 个三角形；按边占对象总数 90% 估算，三角形绝大部分来自边圆柱（此项为估算，非分类型实测）。

> 帧率与 GPU 时间在软件光栅下不可外推，故未记录；draw calls 与三角形数由场景图决定，与 GPU 无关，可作为 G2/G6 的前后对比基线。

## 7. 由 G0 得到的决策结论

1. **2D 主要瓶颈是 DOM 写入，不是力计算**：89.3 ms vs 45.6 ms，约 1.9 倍（多轮重跑 1.9–2.2 倍）。
   → **G4（Canvas）优先级应高于 G5（Worker）**。只做 Worker 会把 45.6 ms 移出主线程，主线程仍留 ≈89 ms/tick，帧预算依旧不达标。计划中「G5 可先于 G4」的条件（力计算为主要瓶颈）**在当前数据规模下不成立**。
2. **G3（真实子图）性价比最高**：它同时缩减力计算与 DOM 两侧，且第 5 节证明当前筛选完全没有减负。建议在 G1 之后立即做 G3。
3. **G1 的分片必须按时间预算**：单 tick ≈135 ms 已超一帧，固定 tick 数分片无法消除长任务；需要「每片跑到时间预算就让出」并支持取消。
4. **G2/G6 对 3D 成立且量级明确**：38,272 draw calls 是硬指标，边改细线（`LineSegments` 合批）可望把边的 34,409 次调用降到个位数量级，是 3D 侧收益最大的单项。
5. `applyFilters()` 单次 132.2 ms（p95 203.6 ms）此前未单独立项，建议并入 G3 一起验收。

社区标签（1.2 ms）**已经不是瓶颈**，G1 中「社区标签按变化更新」的收益很小，不应作为主要卖点。

## 8. 环境侧已知问题（非本次改动引入）

在本测量环境（4 核 Xeon 2.1GHz、headless、软件光栅）下，两个既有验证脚本无法跑完，
已用 `git stash` 对**未改动的代码**复跑确认结果一致，**与 G0 的探针改动无关**：

| 脚本 | 现象 | 原因 |
| --- | --- | --- |
| `verify_graph_loading_2d3d.cjs` | `Input.dispatchMouseEvent timed out`（puppeteer 默认 protocolTimeout 180 s） | 点击时主线程被长任务占满，CDP 无法在 180 s 内完成一次鼠标事件派发 |
| `verify_graph_3d_link_thickness.cjs` | `ffmpeg: not found` | 该脚本录屏依赖 ffmpeg，本机未安装 |

第一条本身就是第 4 节结论的旁证：一次点击可以把主线程堵到连输入事件派发都超时。
在性能更强的机器上该脚本此前可以通过，因此**不作为 G1 的回归门禁**，但 G1 完成后应复跑，若能在本机跑通即为额外正向信号。

以下脚本在本次改动后全部通过：`verify_graph_refresh_layout`、`verify_graph_force_damping`、
`verify_graph_community_labels`、`verify_graph_community_labels_3d`，以及 `npm run lint:js` 与 `npm run test:frontend`（33 项）。

---

# G1 前后对比（2026-09-08）

同一份数据（SHA `224d420972e9b4c2`）、同一环境、同一脚本 `scripts/measure_graph_perf.cjs`。
「前」= 本页 G0 基线所在提交（仅含测量探针），「后」= G1 实现提交。

## G1 实施范围

| 计划子项 | 状态 | 说明 |
| --- | --- | --- |
| warmup 按时间预算分片并可取消旧任务 | **已实施** | `FORCE_WARMUP_SLICE_MS = 16`，`do…while` 每片至少 1 tick；`warmupToken` 使新一轮布局作废上一轮未跑完的分片。「刷新布局」改走分片路径 |
| 刷新种子扩大到与图规模相适应的范围 | **已实施** | `randomizeNodePositions` 的散布范围由固定 80px 改为 `max(80, √N × 8)`；本图 N=3,867 → 497px |
| 滑块更新按帧合并 | **已实施** | 新增 `coalesceByFrame()`，用于连接数 Top N、更新时间 Top N、排斥力三个滑块 |
| 社区标签按变化更新 | **不实施** | G0 实测该段仅 1.2 ms/tick，已有 bbox 缓存生效，不是瓶颈；继续做属于无收益改动 |
| 仅初始化当前视图 | **推迟** | 与 #9「按需加载 3D」耦合，且涉及 2D 降级路径，应作为独立变更评估，不并入本次 |

附带修正：分片 warmup 期间浏览器能出帧，若沿用原先「重铺后立刻同步 DOM」，用户会先看到节点挤成一团再跳到展开态。
故仅同步路径（首屏）保留预先落 DOM，分片路径保留上一帧画面，等 `startVisiblePhase()` 再切换。

## 「刷新布局」点击后 4 s（各 3 轮）

| 指标 | 前 | 后 | 变化 |
| --- | --- | --- | --- |
| 最长单个长任务 | 1,141 / 1,164 / 1,307 ms | 289 / 305 / 284 ms | **约 4 倍改善** |
| 长任务总时长 | 5,387 / 5,274 / 5,352 ms | 3,710 / 4,033 / 3,642 ms | 约 −30% |
| 帧耗时 p50 | 284.3 / 252.6 / 258.6 ms | 52.5 / 58.4 / 54.5 ms | **约 4.8 倍改善** |
| 4 s 内出帧数 | 14 / 16 / 14 | 29 / 28 / 30 | **约 2 倍** |
| >50 ms 长任务个数 | 15 / 17 / 15 | 22 / 26 / 18 | 变多（预期） |

长任务**个数变多是分片的预期结果**：一个秒级任务被拆成若干个几十毫秒的任务。
判据应看最长单个任务、总阻塞时长与出帧数，三项均改善。
帧耗时仍在 52–58 ms、未达 16.7 ms 预算，剩余成本主要是可见阶段每 tick ≈88 ms 的 SVG 坐标写入 —— 那是 G4 的范围，G1 不声称解决。

## 滑块连续输入（12 个排队的 input 事件一次性回放）

| 指标 | 前 | 后 |
| --- | --- | --- |
| 事件派发占用主线程 | **1,130.4 ms** | **0.3 ms** |
| 最长单个长任务 | 1,131 ms | 718 ms |
| 长任务总时长 | 4,261 ms | 3,002 ms |
| >50 ms 长任务个数 | 13 | 8 |

主线程被堵住时浏览器会把 input 事件排队，解除阻塞后逐个回放；
合并前每个事件各跑一次 `applyFilters()`（p50 132 ms），12 个事件连成一个 1.13 s 的长任务，合并后降到 0.3 ms。

## 力学行为回归

`verify_graph_refresh_layout.cjs` 判据 `maxV > 15 && flips <= 1 && lateV < 6`，前后均 `ok: true`：

| | 前 | 后 |
| --- | --- | --- |
| `firstV` / `maxV` | 16.54 | 69.35–81.92 |
| `flips`（半径回弹次数） | 1 | 0 |
| `lateV`（末段速度） | 1.44 | 2.31–3.71 |

两点需要如实说明：

1. **`maxV` 的含义变了。** 该脚本点击后等 50 ms 就开始采样，原先同步 warmup 已跑完，采到的是可见阶段速度；
   改分片后首个样本落在 warmup 期间，采到的是内部力学速度（DOM 此时未更新，用户看不到）。
   因此 `maxV > 15` 这条判据**在分片路径下已近乎恒真，不再有效守护「可见展开幅度」**。
   本次未改动该脚本阈值；建议后续单独补一条针对可见阶段的判据，不要把当前的 `ok: true` 当作展开效果未退步的充分证据。
2. **`flips` 由 1 变 0。** 种子散布放大后，布局单调展开到位，不再出现「先冲过头再收回」。
   脚本判据允许 `flips <= 1`，但这确实是刷新动画观感的变化；若认为过冲是想保留的效果，应回退种子缩放或改用更小的放大系数。
   `firstV` 由约 880 降到约 70–82（约 11 倍），说明原先的高速确实来自把 3,867 个节点挤进 80px 方框的斥力爆炸。

## 其他回归

`verify_graph_refresh_layout`、`verify_graph_force_damping`、`verify_graph_community_labels`、
`verify_graph_community_labels_3d`、`verify_graph_topn_collapse`、`verify_graph_recency_topn`、
`verify_graph_opensource_filter`、`verify_graph_timeline_exit_settle` 全部通过；
`npm run lint:js`、`npm run test:frontend`（33 项）、`make ci-preflight`（13/13）通过；页面无 JS 报错。
第 8 节记录的两个环境侧既有问题状态不变。
