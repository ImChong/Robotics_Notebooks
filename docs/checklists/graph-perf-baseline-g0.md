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
