# 图谱 2D / 3D 渲染性能与 Bug 复审（2026-10-19）

对应 [工程计划 · 图谱动态渲染性能专项](../../plan.md#2026-09-08-图谱动态渲染性能专项)。
目标：`docs/graph.html` + `docs/graph-3d.js` 在**低配电脑浏览器（集显、2–4 核）与手机**上流畅渲染 2D / 3D 图谱。
本页是**审查结论与分批方案**，不含实现；G0–G4 已完成的项（见 [G0 基线](graph-perf-baseline-g0.md)、[G2/G3](graph-perf-g2-g3.md)、[G4](graph-perf-g4.md)）不重复，G2 实测否定的「细线 `linkWidth→0`」不再提。

状态：**v1** —— 五个审查视角（2D 力模拟、2D 渲染、3D 场景、3D 库 API、加载）与实测已完成；交互事件、移动端 CSS、Bug 专项三个视角及逐条对抗验证仍在进行，结论若有变化在本页追加「v2 修订」小节。

证据等级：
- **[实测]** 本次在 headless Chromium 141（4 核 Xeon 2.8 GHz，软件光栅 SwiftShader）测得；CPU / 主线程数字可信，GPU 帧率不可外推。
- **[代码确认]** 逐行核对过对应源码 / vendor 压缩包并引用行号。
- **[agent 实测]** 单个审查 agent 用 scratch 脚本在同一环境测得，尚未经第二方复核。
- **[推测]** 未实测、按代码推断或经验估计，实现时须验证。

---

## 0. 结论先行

1. **数据规模已比 9 月 G4 测量时大 57%**：今日 `link-graph.json` 为 6,216 节点 / 53,740 边（G4：3,968 / 35,490）。G0–G4 的毫秒数全部过时 [实测]。
2. **全图在低端设备上靠调参救不回来**。桌面全图 1 tick ≈ 160–180 ms；手机模拟（4× CPU 节流）1 tick ≈ 760 ms，首屏 33 s，其中一个 **23.7 s** 的长任务（页面完全无响应）[实测]。必须三管齐下：**移动端默认子图 + 导出期预计算布局**（P1）、**力模拟进 Worker**（G5）、**3D 合批**（G6）。
3. **先修的 Bug 本身就是性能问题**：3D 连线不透明度 NaN（开连线后一团白）、每次力模拟结束都 `fitToScreen` 吃掉用户缩放、时序动画 `t.name` 抛错、3D hover 触发整库重建 53,740 个连线对象（每次悬停 3.3–3.7 s）[实测 + 代码确认]。
4. 有一批**一行到几十行**的改动，合计可把桌面全图每 tick 成本砍约 40%、把缩放/平移的主线程开销砍一个数量级、把 3D 空闲时的 CPU 降到接近 0（§3 P0）。

---

## 1. 本次实测基线

数据：`docs/exports/link-graph.json`（SHA-256 前 16 位 `625943f4d9656f03`），6,216 / 53,740，21 社区。
桌面 = 1440×900 DPR 1 不节流；手机 = 390×844 DPR 3、`isMobile + hasTouch`、Android UA、`(hover:none) / (pointer:coarse)`、**CPU 4× 节流**（CDP `Emulation.setCPUThrottlingRate`）。
桌面列同时跑了仓库脚本 `scripts/measure_graph_perf.cjs` 与一份加了手机 profile 的探针脚本（原始 JSON 为 headless 产物，未入库；§4 要求把手机 profile 并入仓库脚本）。

| 指标 | 桌面（不节流） | 手机模拟（4× 节流） | 9 月 G4（3,968 节点） |
|---|---|---|---|
| 首屏就绪（loading 遮罩消失） | 5.7–7.8 s | **33.1 s** | 3.9–4.7 s |
| 加载期最长长任务 | 4.3 s | **23.7 s** | — |
| `JSON.parse`（二次取回计时） | 19 ms | 65 ms | — |
| 力 tick p50 | 142–147 ms | **681 ms** | 76.7 ms |
| 渲染同步 p50（关连线 / 开连线） | 13.5 / 21.1 ms | 81 ms / 未测 | 19.7 ms |
| `drawEdges` p50（开连线，裁剪后画 32,133 条） | 8.3 ms（JS 录制时间；栅格化见 §3） | 未测 | 6.1 ms |
| `applyFilters` p50 | 69–124 ms | 438 ms | 56 ms |
| 刷新布局：帧 p50 / 最长长任务 | 126–144 ms / 455 ms（关连线）；**2.4–2.7 s**（开连线） | 2.6 s / 2.6 s（4 s 只出 2 帧） | 81–87 ms / 290–384 ms |
| 边层 canvas 后备尺寸 | 1440×796 @1× | hc=4 → 390×740 @1×；hc=8 → 780×1480 @2× | — |
| 3D draw calls / 三角形（开连线） | **59,956 / 2.63 M** | 未测（页面把 CDP 卡死） | 39,458 / 1.71 M |
| 3D draw calls（默认关连线） | 6,216 / 1.34 M | 未测 | — |
| 3D 点击 → 画面可见 | 58.7 s（软件 GL，不代表真机） | 未测 | — |
| 3D hover 同步部分 / `syncLabels` p50 | 99–121 / 51–60 ms | 未测 | 18–22 / 13–15 ms |
| 3D hover **延迟 digest**（kapsule 1 ms 定时器之后） | **3.3–3.7 s**（开连线）/ 8–48 ms（关连线） | 未测 | 从未计过 |

要点：
- 力计算占全图 tick 的 ~90%，G4 的「渲染不再是瓶颈」结论成立且更强 [实测]。
- `JSON.parse` 不是瓶颈（19 / 65 ms）；主线程被建图 / warmup 占满才是。线上 GitHub Pages 发 gzip（link-graph 约 1.07 MB gz），不支持 brotli，所以只能靠减原始体积 [实测 + agent 核查线上响应头]。
- 未能测：手机 3D（页面长任务把 CDP 协议堵死）、真机 GPU、Safari。

---

## 2. 已确认 Bug 清单（按严重度）

| # | 现象 | 位置 | 证据 | 修法（工作量） |
|---|---|---|---|---|
| B1 | **3D 连线不透明度 = NaN**：`linkOpacity` 在 3d-force-graph 1.80.0 是**标量**，代码传了函数 → 库内 `p = e.linkOpacity * mf(c)` 得 NaN → `transparent:false, opacity:NaN`，开「显示连线」后 53,740 条线全部不透明白线盖住整图；`linkOpacityFor` 的淡化 / 高亮值从未生效 | `graph-3d.js:1020, 1433`；vendor `p=e.linkOpacity*mf(c)` | [代码确认] + [agent 实测] 53,740/53,740 NaN、截图 | 改标量 `linkOpacity(1)` 并把 alpha 放进 `linkColor` 返回的 rgba 串（库按颜色串缓存共享材质）；或用 `linkMaterial` 访问器返回 2–3 个共享材质。改后连线进入透明队列需每帧 z 排序，宜与 G6a 一起做（S） |
| B2 | **每次力模拟 `end` 都 `fitToScreen`**，不看 `userZoomed2D`：拖一个节点、碰滑块、从 3D 回来，几秒到一分钟后视图跳回全图 | `graph.html:4579-4582 → 4496-4500 → 5894` | [代码确认] + [agent 实测]（放大到 k=1.447 拖拽后 43.7 s 视图跳回 k=0.18） | 首屏 / 刷新布局设一次性 `pendingAutoFit`，`end` 时仅在标志置位且用户未缩放时 fit；去掉 4589 的 2.2 s 兜底定时器（全图 alpha 2.2 s 时仍 ≈0.44，从不触发）（S） |
| B3 | **时序动画 `t.name is not a function`**：d3 v7 transition 无 `.name()`，`fitToTimelineNodes` 传 `name:'tlfit'` 时每次跟随相机都抛错，播放期间视口不跟随 | `graph.html:5871`（调用方 5542） | [代码确认]（G4 §7 曾记「既有问题」但未定位） | 改为 `svg.transition(transitionOpts.name)`（S） |
| B4 | **3D `refreshAppearance` 每次重设 5 个访问器**，kapsule setter 不比较值、一律标脏；其中 `linkWidth` 变更触发 `linkDataMapper.clear()` → 拆掉并重建全部 53,740 个连线 Mesh（`Object3D.remove` 在 6 万子节点的 Group 上是 O(E²)）。hover on/off、筛选、侧栏、换主题各付一次 | `graph-3d.js:1006-1024`；vendor `n(["linkThreeObject","linkThreeObjectExtend","linkWidth"]))&&e.linkDataMapper.clear()` | [代码确认] + [实测] 3.3–3.7 s/次（开连线） | 5 个访问器只在 `bindGraphInstance` 设一次（它们本就是读闭包状态的函数）；hover / 侧栏改为直接改 `link.__lineObj` 的材质与缩放（与 `updateNodeMeshes` 走 `__threeObj` 同法）；只有可见连线**集合**变化时才碰 setter，且永不再碰 `linkWidth`（S–M） |
| B5 | **3D 首次进入 `graphData` 设了 3 次**（`show` → `installCustomNodeMeshes` → `firstShowKick`），每次重建 6 万对象并把 alpha 重置为 1；`nodeThreeObject` setter 本身就会重建节点 | `graph-3d.js:1521, 1048, 1498` | [代码确认] | 删后两处 `graphData`，连线数组建一次复用（让 mapper 只做 diff）（S） |
| B6 | **3D `cooldownTime` 沿用库默认 15 s**，且库每帧只跑 1 tick：慢设备 15 s 只够几十 tick，alpha ≈ 0.3 时引擎停机，相机 fit 到未收敛布局；注释（1409-1410）以为已处理 | `graph-3d.js:1417-1420`；vendor `cooldownTime:{default:15e3}` | [代码确认] + [agent 实测]（6 tick 后停机） | `cooldownTime(Infinity)`；初次 fit 改由 `onEngineStop` + tick 计数驱动，9 s 定时器只做 60 s 兜底（S）；根治见 G5-3D |
| B7 | `graph.d3Alpha` / `graph.d3AlphaTarget` 在 1.80.0 **不存在**（库只透出 refresh / getGraphBbox / d3Force / d3ReheatSimulation / emitParticle），四处 `typeof === 'function'` 守卫全是死代码；`maybeMarkLabelsLayoutReady` 因此第一 tick 就标「就绪」 | `graph-3d.js:570, 1090, 1136, 1660` | [代码确认] | 删死守卫；就绪改用 tick 计数或 `onEngineStop`（S） |
| B8 | `updateForces` 磁吸模式 `getTargetY` 兜底写成 `W/2` 应为 `H/2` | `graph.html:4123` | [代码确认] | 一字修（S） |
| B9 | 侧栏邻居排序：`sort` 比较函数里两次 `nodes.find`（O(N)）+ 每次 `localeCompare('zh-CN')`；点枢纽节点（度 >1,300）同步卡住主线程 | `graph.html:6291-6296` | [代码确认] | 建一次 `nodeById` Map；先装饰再排序；复用一个 `Intl.Collator`（S） |
| B10 | 加载进度条把**解码字节数**除以 **gzip Content-Length**：线上 1.07 MB gz / 4.14 MB 原始 → 下到 26% 时进度已到 78% 封顶然后停住 | `graph.html:3200-3210, 3232` | [代码确认]（ReadableStream 产出解码字节是规范行为；线上响应头 agent 核过） | 部署期把原始字节数 stamp 进页面（pages.yml 已有 stamp 步骤）作分母；无法得知时改不确定态进度条（S） |
| B11 | 刷新布局分片 warmup 期间，滑块 / 拖拽 / 筛选会打断：`updateForces` 的 `alpha(0.25).restart()` 把挤成一团的种子态画出来并重置 alpha | `graph.html:4504-4556, 4131, 4148, 5330` | [代码确认]，中置信度 | 加 `warmupInFlight`：期间 `updateForces` 只换力不 restart；拖拽开始先取消 warmup 并 `startVisiblePhase`（S） |
| B12 | DPR 变化（拖到另一块屏）清空边层 canvas，但 `runGraphViewportResize` 因 W/H 未变提前返回，不重绘 | `graph.html:3685-3694, 5929-5934` | [代码确认]，低频 | `resizeEdgeCanvas` 真改了尺寸就 `scheduleEdgeDraw()`；加 `matchMedia('(resolution)')` 监听（S） |
| B13 | 无 `webglcontextlost` 处理：iOS 在 6 万对象 + MSAA 下丢上下文后 3D 黑屏且不回退 2D | `graph-3d.js` 全文无 webglcontext | [代码确认]（发生概率为推测） | 监听 lost → 暂停渲染环并走既有「恢复 2D」提示；restored → 恢复（S） |
| B14 | `<meta name="description">` 仍写「75 个概念节点、429 条关联边」 | `graph.html:6` | [代码确认] | 部署期 stamp（S，非性能） |

另：3D 时序暂停时 `alphaTarget` 钉在 0.3，暂停期间力模拟永不停（电量 / 发热），`graph.html:5602-5607` [代码确认]，归入 P0。

---

## 3. 性能方案（按性价比分批）

### P0 小改动（每项独立 PR，合计 1–2 天）

**2D**

| 项 | 位置 | 预期收益 | 证据 |
|---|---|---|---|
| 首屏 warmup 走已有的 `async` 分片路径（现在只有「刷新布局」走） | `graph.html:4586` | 消掉首屏最大长任务（桌面 20×147 ≈ 2.9 s，手机 20×681 ≈ 13.6 s） | [代码确认] + [实测] |
| warmup 结束立刻 `fitToScreen(0)`；`end` 的 fit 改一次性（B2） | `4520-4525, 4579` | 现在首屏在 `end` 之前完全没有 fit：叶序种子半径 ≈ 790 px，手机上看到的是被裁掉的一小块，持续几十秒 | [代码确认] + [agent 实测]（fit 出现在 43–48 s） |
| 隐藏标签用 `display:none` 而非 `opacity:0` | `4183-4191, 2990-2998` | 6,216 个 `<text>` 在任何**缩放**（滚轮 / 捏合 / fit 过渡的每一帧）都要重排：230–540 ms/步 → 7 ms | [agent 实测] |
| 平移手势期间给 `g.nodes` 加 `pointer-events:none` | `3975-3986` | 每次 mousemove 对 24,927 个 SVG 元素做命中测试：34–62 ms/步 | [agent 实测] trace |
| 点击 / 轻触节点不再 reheat（drag `start` 只记 fx/fy，首个 `drag` 事件才 `alphaTarget(0.2).restart()`；`clickDistance(4)`） | `4145-4159` | 手机上每次点节点 = 55 个全图 tick（≈22 s 墙钟） | [agent 实测] |
| `alphaMin(0.005)`（≤4 核或触屏 0.01） | 力模拟初始化 `4071` | 全图冷却 170 tick 中最后 ~57 tick 位移 <10 px；砍掉的是亚像素运动，不是「提前冻结」 | [agent 实测] Node 基准；须用 `verify_graph_refresh_layout`（lateV）与 `verify_graph_force_damping` 回归 |
| `forceManyBody().theta(1.2)`（低端 1.5） | `4074` | 斥力占 tick 61%；theta 1.2 省 ~20 ms、1.5 省 ~33 ms（98 ms 基准） | [agent 实测] Node 基准；须截图对比布局观感 |
| `forceCollide` 门控：alpha>0.3 时跳过，低端隔 tick 跑 | `4078` | 碰撞占 34%；爆炸期每 tick 位移 ~34 px，单 tick 解重叠是白做 | [agent 实测]，中置信度 |
| 3D→2D 不重新加热（`resumeSimulation2D` 只在暂停时正在冷却才 restart；`updateForces` 加 `{reheat:false}`） | `2928-2938, 4090, 4131` | 每次切回 2D 白跑 ~135 tick（≈13 s CPU）再触发 B2 | [代码确认] |
| hover leave 不走 `applyFilters`；预建邻接 Map 替代遍历 53,740 边；用缓存的 `__nodeParts` 替代 `node.select()`；`.node-circle` 的 CSS transition 只作用于被悬停节点 | `4821-4877, 4772, 846` | 扫过密集区每个节点 ≈110 ms JS + 3–4 帧全量 style / paint | [agent 实测] |
| 去掉 `<g>` 组级 `opacity`（0.08 / 0.4 / 0.05），淡化全交给已写的 `fill-opacity` | `5271, 6154-6157` | 组级半透明让 ~5,900 组各成一层，每步平移多 ~20% 栅格 | [agent 实测] trace，中置信度 |
| 逐 tick 跳过亚像素位移的 `setAttribute`；无节点变化时跳过 `drawEdges` / 社区标签；坐标保留 1–2 位小数 | `4484-4494` | 冷却末段只剩力计算成本 | [推测]，收益随 alphaMin 调整而减少 |
| 边层 DPR 上限改按设备类别（`pointer:coarse` / `maxTouchPoints` → 1×），不只看核数 | `3671` | iOS 不暴露或报 6 核、安卓中端报 8 核 → 现在手机全走 2×（4 倍像素） | [代码确认]（iOS 具体取值为 agent 记忆，中置信度） |
| 平移 / 捏合手势进行中不重绘边层，改给 canvas 加 CSS transform，手势结束再 `drawEdges` 一次 | `3977-3981, 3889` | `drawEdges` 的 8 ms 只是录制；栅格化在 commit：开连线平移每步 Commit 768 ms（软件光栅，真机比例待测） | [agent 实测] trace，中置信度 |
| 时序暂停 800 ms 后 `alphaTarget(0)` | `5602-5607` | 暂停期间不再每帧全图 tick | [代码确认] |

**3D**

| 项 | 位置 | 预期收益 | 证据 |
|---|---|---|---|
| B1 + B4 + B5 + B6 + B7（§2） | — | hover 3.5 s → 几十 ms；首次进入少 2 次全量重建；慢设备布局能收敛 | [实测] + [代码确认] |
| `setPixelRatio(min(DPR, 低端 1 : 1.5))`、低端 `antialias:false`、`alpha:false`（背景本就是不透明色） | `graph-3d.js:1471-1486`；vendor `setPixelRatio(Math.min(2,window.devicePixelRatio))` + `antialias:!0,alpha:!0` | DPR 3 手机现在按 2× 渲染（4 倍像素）+ MSAA + 半浮点 composer 目标；填充率是移动 GPU 最弱项 | [代码确认]；GPU 收益量级 [推测] |
| 渲染环空闲暂停：`onEngineStop` 后 ≥500 ms 无 controls `change` / 指针 / 滚轮 / 触摸 → `pauseAnimation()`，任何输入 / reheat / 相机飞行恢复 | `1087-1116`；vendor `_animationCycle` 无条件 rAF | 空闲时每帧仍 6k–60k draw calls + 60k 矩阵更新（20.8 ms/帧）+ 每 50 ms 全场景 raycast | [实测]（CPU 部分） |
| 连线排除拾取：连线对象 `layers.set(1)`（库 Raycaster 只查 layer 0）；无指针活动 300 ms 后 `enablePointerInteraction(false)`，触屏只在 touchstart–touchend 间开启 | vendor `pointerRaycasterThrottleMs:50` 未透出；`intersectObjects(e.objects,!0)` 递归全场景 | 开连线时每 50 ms 17–39 ms CPU，指针不动也跑；且圆柱挡在节点前会抢走 hover | [代码确认] + [agent 实测] |
| 材质按（颜色，量化不透明度）缓存，替代 6,216 个独立 `MeshLambertMaterial`；颜色串只解析一次 | `948-986` | 每个 draw call 都刷新 uniform；hover 时 6,216 次 CSS 颜色解析 | [代码确认] |
| `linkResolution(3)`、球体 `SphereGeometry(1,8,6)`（低端） | `943, 1424`；vendor 圆柱 `openEnded=false`（24 tri） | 2.63 M → ≈1.1 M 三角形，节点在屏上只有 2–8 px | [代码确认] |

**加载**

| 项 | 位置 | 预期收益 | 证据 |
|---|---|---|---|
| `<link rel="preload" as="fetch" href="exports/link-graph.json" crossorigin>` | `graph.html` head | 现在数据请求在 ~750 KB 同步 JS 下载 + 执行之后才发出 | [代码确认]；延迟量 [推测] 0.5–1 s（4G） |
| SW 注册从 `load` 推迟到 `finishLoading()` 之后 | `1849-1853`；`sw.js:18-52` 预缓存 17 项 | 首访时 ~0.5 MB gz 与 1.07 MB 数据抢带宽 | [代码确认] |
| `site-catalog-v1.json`（3.9 MB 原始 / 1.2 MB gz）推迟到首次打开侧栏；按 path 建 Map 替代 `.find` | `3246-3252, 6280, 6218` | 现在在建图期并行下载并主线程解析，只为侧栏 tags / related | [代码确认] |
| 搜索 datalist 的 6,216 个 `<option>` 延后到首次聚焦搜索框 | `3642-3652` | 首绘路径里少一次 ~300 KB HTML 解析 | [代码确认] |
| 用 `new Response(body.pipeThrough(计数流)).json()` 替代手工拼 Uint8Array + TextDecoder；删掉 `edges.map(e=>({...e}))` 多余拷贝 | `3206-3217, 3271` | 少 3 份 4 MB 拷贝 + 53,740 对象 | [代码确认]；收益 [推测] 手机 100–200 ms |

### P1 低端 / 移动端策略（2–4 天）

1. **设备分级 + 「性能模式」开关**（localStorage 持久化，UI 放现有物理面板）：
   - 信号：`(pointer:coarse)`、`hardwareConcurrency ≤ 4`、`deviceMemory ≤ 4`，以及**首帧实测**（第一个可见 tick 的 rAF 间隔 >50 ms 自动降档）。
   - 各档改什么：边层 / 3D DPR、标签与光晕、warmup tick 数、碰撞力频率、theta、3D 球体精度、`antialias`、默认 Top N。
2. **移动端默认子图**：`(pointer:coarse)` 上默认 Top N（建议 500，面板可改回全部）。G3 已让子图成本按比例缩减：Top 300 时力 tick 3.8 ms、渲染同步 2.4 ms [实测]。设默认值的位置：`topDegreeLimit` 初始化（`graph.html:3317` 附近）+ `initDegreeTopSlider`。
3. **导出期预计算布局**（新脚本，Node 跑 vendor d3 到 alphaMin，坐标取整后写入 `layout` 字段，约 +29 KB gz [agent 估算]）：首屏直接用预计算坐标，不跑 warmup，只以 `alpha(0.08)` 做轻量落稳；「刷新布局」/ 拖拽才跑完整力模拟。这是对 G1「只分片」之外的新角度，也是手机首屏最大的单项。约束：力参数须与 `graph.html:3999-4083` 共用一个小模块，避免漂移。
4. **数据瘦身**：`summary` 占 gzip 的 55%、`detail_id` 可由 `id` 推导、`recency` 全表同值（导出时 mtime = 部署时间，死字段）。拆成拓扑文件（≈435 KB gz，−58%）+ 懒加载的 `link-graph-summaries.json`；tooltip / 侧栏 / 3D 的 `_info.summary` 改读共享 Map。需同步改 `scripts/generate_link_graph.py:1627-1634, 1886-1911`、`main.js` `expandLinkGraph`、`mini-graph.js`、`tests/test_generate_link_graph_compact_formats.py`。

### P2 结构性（G5 / G6，各 3–6 天）

**G5 2D 力模拟进 Worker**（2D 侧触碰 x/y/vx/vy/fx/fy 的写点：`graph.html:4017-4029, 4040-4051, 4085-4093, 4100-4131, 4147-4159, 4504-4556, 4573-4582, 5326-5332, 5480-5498, 5512-5516, 5520-5529, 5586-5607`）：
- Worker 持有 d3-force 与 `{degree[], radius[], edges Int32Array, W, H, charge, theta, alphaMin, 磁吸中心}`；主线程保留节点对象，只把 `Float32Array` 快照（可转移）镜像进 `n.x / n.y`。
- 消息：init / setMembership(子图索引, token) / setPositions / pin·unpin / alpha·alphaTarget·stop·restart / setForces；Worker → 主线程每 rAF 最多一帧快照（ack 节流）+ end{token}。
- 它不减总 CPU，只把 ~150 ms/tick 移出主线程；所以 P0 的 alphaMin / theta / collide 仍然要做。

**G6a 3D 连线合批（先做）**：`linkVisibility` 只放行高亮边（库的圆柱 + 现有 `linkScaleUpdate` 加粗逻辑原样留给这几十条），其余 53,740 条用**一个** `LineSegments`（position 6·E、RGBA 顶点色 8·E，`frustumCulled=false`、`raycast=noop`），在 `onEngineTick` 重写 position、只在 hover / 筛选 / 主题时重写颜色。一次消掉 53,740 个 draw call、库每帧 122–136 ms 的逐连线定位循环、以及 raycast 成本 [代码确认 + agent 实测]。

**G6b 3D 节点合批**：目标 6,216 draw call → 1。**决策点**：vendor 包未导出 `InstancedMesh` / `Points` 类（`setMatrixAt` 0 次出现，渲染器侧 `isInstancedMesh` 检查 19 处）[代码确认]。可选：
- (a) 换 ESM 构建自带 three（可 tree-shake，长期干净；需验证库版本 API）；
- (b) 不动 vendor：从现有对象原型链反抓 `BufferGeometry` / `BufferAttribute` / `Line` / `LineBasicMaterial`（G6a 够用），节点用「Mesh + `isInstancedMesh=true` + 伪装 `isInstancedBufferAttribute`」或合并几何体；拾取改为屏幕空间最近点（6k 节点很便宜）；3D 节点拖拽需自实现（库的 DragControls 依赖逐节点对象）。

建议：G6a 用 (b) 先落地验证，G6b 再决定是否走 (a)。

**G5-3D**：库把 tick 绑在帧上（B6 根因）。路线：移除库内力（`d3Force('charge'|'link'|'center', null)`）、`cooldownTicks(1)`，Worker 回传坐标后写 `node.x/y/z` 并 `d3ReheatSimulation()` 让库只做一步廉价的位置传播。拖拽回弹保留。

**2D 节点层改 canvas**（G4 剩余部分）：标签 `display:none` 后 SVG 节点层每 tick 仍 ≈43 ms 主线程 style / layout / paint + 栅格 [agent 实测]；改 canvas + `d3.quadtree` 拾取可降到个位数 ms，但要重做拖拽 / 浮窗 / 键盘导航，放最后。

---

## 4. 验收与度量

- 目标：桌面接近 60 fps、手机稳定 30 fps（帧预算 16.7 / 33.3 ms）；移动端 4G 首屏 <5 s；任意交互长任务 <200 ms；3D 空闲 CPU ≈0。
- `scripts/measure_graph_perf.cjs` 需扩展：(1) 手机 profile（CPU 4×、DPR 3、touch）；(2) 3D hover 要等 kapsule 的 1 ms digest 跑完再计时（现在 334-336 行只计同步部分）；(3) 用 trace 的 Commit / RasterTask 评估边层，而不是 `performance.now()`；(4) 记录 3D 实际 pixel ratio、三角形数。
- 回归门禁：`verify_graph_refresh_layout`（alphaMin / theta 改动后 `lateV<6`、`flips≤1` 仍需成立）、`verify_graph_force_damping`、`verify_graph_timeline_*`（补 B3 的相机跟随断言）、`verify_graph_3d_link_thickness`（补「材质 opacity 为有限数且 <1」断言）、`verify_graph_edge_canvas`、`tests/cache-isolation.test.cjs`（改 SW 预缓存 / 目录策略时）。
- 真机：至少 1 台 iPhone（Safari）、1 台中端 Android、1 台集显笔记本；记录 GPU 帧率，这是 headless 永远测不到的。

## 5. 未覆盖 / 存疑

- 交互事件、移动端 CSS（`backdrop-filter` 等 10 处）、Bug 专项三个审查视角与逐条对抗验证仍在进行；本页 [agent 实测] 条目是单 agent 结论，§2 B1–B10 与 §3 标 [代码确认] 的项已逐条核对源码。
- 2D 平移 / 缩放的栅格化数字来自软件光栅，真机 GPU 下的绝对值会小很多，但「每手势全量重绘 + DPR 2×」的结构不变。
