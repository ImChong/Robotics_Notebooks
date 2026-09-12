# 图谱 2D 边层改 Canvas：前后对比（G4）

对应 [工程计划 · 图谱动态渲染性能专项](../../plan.md#2026-09-08-图谱动态渲染性能专项) 的 **G4 / #6**。
口径与脚本沿用 [G0 基线](graph-perf-baseline-g0.md)：`scripts/measure_graph_perf.cjs`，同一份 `link-graph.json`、同一台机器上前后各跑一次。

复现命令（仓库根目录）：

```bash
make export graph                      # 浅克隆下 graph_exports_sync 会拒绝同步，手动 cp 即可
cp exports/link-graph.json exports/hub-rankings.json exports/wiki-activity.json docs/exports/
cd docs && python3 -m http.server 8765 &
node scripts/measure_graph_perf.cjs http://127.0.0.1:8765/graph.html .cursor-artifacts/graph-perf-g4-after.json
node scripts/verify_graph_edge_canvas.cjs http://127.0.0.1:8765/graph.html   # 功能门禁，16 项
```

## 1. 改了什么

2D 的**边**从每条一个 `<line>` 改为**画布下层一张 canvas 批绘制**：

- 一帧内按「高亮批 / 暗淡批」各攒一条路径，每批只 `stroke()` 一次；两个坐标缓冲（`Float32Array`）常驻复用，不每帧新建。
- 原先散落在四处的 `link.attr(...)`（`applyFilters` / 悬停 / 侧栏聚焦 / 时序动画）收敛成一个上色模式状态机，判据在绘制时现算。
- 缩放/平移把 d3-zoom 的 transform 同步给 `ctx.setTransform`，`lineWidth` 仍在世界坐标系内，粗细随缩放的观感与 SVG 一致。
- 新增**视口裁剪**（两端都在可视矩形同一侧外就跳过）与**像素比上限**（常规封顶 2×；`navigator.hardwareConcurrency ≤ 4` 时按 1× 绘制）。

**节点、标签、社区标签与全部交互仍在 SVG**，拾取 / 拖拽 / 浮窗 / 键盘导航逻辑一行未动。

## 2. 固定数据与环境

| 项 | 值 |
| --- | --- |
| `exports/link-graph.json` SHA-256 前 16 位 | `47e04a996f7c4354` |
| 节点 / 边 | 3,968 / 35,490（22 社区） |
| 测量时间 | 2026-09-12 |
| 浏览器 | headless Chromium 141.0.7390.37（puppeteer-core） |
| CPU | Intel Xeon @ 2.80GHz，4 逻辑核 |
| 视口 / DPR | 1440×900 / 1 |
| 缓存 | 冷缓存（`setCacheEnabled=false`） |

> **本页数字不可与 [G0 基线](graph-perf-baseline-g0.md) 横向比较**：机器不同、数据规模也已从 3,867/34,409 涨到 3,968/35,490。
> 本页的 before 是同机、同数据、同脚本对**改动前页面副本**的实测，只在本页内部前后对比。
>
> **未测**：真机独显/集显 GPU、手机、Safari、CPU 节流档位、Service Worker 热缓存二次访问。
> **像素比上限逻辑未在高 DPR 屏上实测**——测量机 `devicePixelRatio=1`，`≤4 核 → 1×` 与「本来就是 1×」在本机无法区分。

## 3. 结构计数（精确，与硬件无关）

| 指标 | 改前 | 改后 |
| --- | --- | --- |
| DOM `.edges line` | 35,490 | **0** |
| DOM `.nodes g.node-g` | 3,968 | 3,968 |
| `#graph-canvas` 元素总数 | 51,429 | **15,938（−69%）** |
| **每 tick 坐标属性写入** | **145,928** | **3,968（−97%）** |
| 边层 canvas 后备缓冲 | — | 1425×796 @ 1× |
| 默认视口实际绘制的边 | — | 23,577 / 35,490（视口裁剪掉约 34%） |

## 4. 每 tick 主线程耗时分段（全图态，p50 / ms，20 次采样）

| 分段 | 改前 | 改后 |
| --- | --- | --- |
| 力计算 `simulation.tick()` | 73.4 | 76.7 |
| 渲染同步 `syncGraphDomFromSimulation()` | **143.6** | **19.7（约 7.3×）** |
| └ 边层 `drawEdges()` | — | 6.1 |
| └ 社区标签 `updateCommunityLabels()` | 1.6 | 2.2 |
| **每 tick 合计** | **≈ 217** | **≈ 96（−56%）** |
| `applyFilters()` 单次 | 115.3 | **56.1（−51%）** |

筛选态（Top 300）：

| 分段 | 改前 | 改后 |
| --- | --- | --- |
| 力计算 | 4.2 | 4.0 |
| 渲染同步 | 13.8 | 11.4 |
| └ 边层 `drawEdges()` | — | 8.9 |
| `applyFilters()` 单次 | 141.8 | **58.1（−59%）** |

> 筛选态的 `drawEdges()`（8.9 ms）反而高于全图态（6.1 ms）：边层始终绘制全量边（含被筛掉的暗边），
> 而筛选后相机被 `fitToScreen` 拉远，视口裁剪命中率下降。裁剪收益随相机位置变化，不是恒定折扣。

## 5. 首屏就绪

| 口径 | 改前 | 改后 |
| --- | --- | --- |
| `measure_graph_perf.cjs` 单次 | 4,212 ms | 3,930 ms |
| 独立探针 3 次中位数（多线程静态服务器） | 5,214 ms | **4,749 ms（−9%）** |

建树省掉 35,490 个 SVG 元素只换来约 9% 的首屏，因为**首屏仍由 `link-graph.json`（5.84 MB）的下载 + 解析 + 首次力布局主导**。
资源计时里 `site-catalog-v1.json` 的 `duration` 长达 3.5 s，但它是 fire-and-forget、不参与首屏判据，
该时长主要反映主线程被建图占满时响应体的滞留，**不能读作纯网络时间**。进一步压缩首屏属 [阶段 B](../../plan.md) 的数据瘦身范畴，不在 G4 内。

## 6. 「刷新布局」长任务与帧耗时（点击后采样 4 s，各 3 轮）

| 指标 | 改前 | 改后 |
| --- | --- | --- |
| 最长单个长任务 | 431 / 573 / 431 ms | **384 / 290 / 350 ms** |
| >50 ms 长任务总时长 | 4,316 / 4,410 / 4,109 ms | 4,390 / 3,995 / 4,206 ms |
| 帧耗时 p50 | 83.5 / 82.1 / 78.6 ms | 81.1 / 87.5 / 81.6 ms |
| 4 s 内出帧数 | 25 / 24 / 24 | 26 / 27 / 27 |

最长阻塞下降约 25–33%，出帧数略升，**帧耗时 p50 基本持平**：warmup 分片按 16 ms 时间预算切，边层变快后同一片里能多跑几个 tick，
省下的时间被力计算吃掉，表现为「同样的帧耗时、更多的布局进度」。这段主线程此时已由力计算主导，边层不再是瓶颈。

## 7. 功能门禁

`node scripts/verify_graph_edge_canvas.cjs` 16 项全通过：边无 SVG 元素 / 节点仍在 SVG / canvas 覆盖画布与像素比上限 /
默认态、悬停、侧栏聚焦、时序动画四种上色模式都画得出 / 筛选后力模拟缩成真实子图且边层仍绘制暗边 /
主题切换重绘 / 放大后裁剪生效 / 2D→3D 边层隐藏、回到 2D 重绘 / 无边层相关 JS 异常。

既有脚本回归通过：`verify_graph_community_labels`、`verify_graph_recency_topn`、`verify_graph_force_damping`、
`verify_graph_timeline_exit_settle`；`npm run test:frontend` 58/58（`graph-fallback` 增加了「边层与 SVG 同进退」断言）；`make ci-preflight` 通过。

**唯一的观感变化**：批绘制下高亮边统一压在暗边之上。原先同一 `<g>` 内按数据序绘制，排在后面的暗边可能盖住高亮边。

**改动前既有问题（本次未处理）**：`applyGraphFitTransform` 里 `t.name(...)` 抛 `t.name is not a function`
（d3 的 transition 没有 `.name()`，应为 `svg.transition(name)`），改动前后同样复现，与边层无关。

## 8. 由 G4 得到的决策结论

1. **G0 第 7.1 条「2D 主要瓶颈是 DOM 写入」在 G4 之后不再成立**：全图态每 tick 96 ms 里力计算占 76.7 ms（约 80%），渲染同步只剩 19.7 ms。
2. 因此 **G5（力模拟移入 Worker）的决策门槛现在成立**——G0 当初否定「G5 可先于 G4」的前提（力计算只占三分之一）已被 G4 消除。
3. 全图态每 tick 96 ms 仍远超 16.7 ms 帧预算，**剩余差距几乎全在力计算**：下一步只能从力计算本身入手（Worker，或把力模拟搬到 GPU）。
   筛选态（Top 300）每 tick 已降到约 15 ms，**首次整体落在帧预算内**。
4. G6（3D 合批）不受本次影响：3D 每帧仍是 39,458 draw calls / 1,708,848 三角形。
