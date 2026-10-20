# 图谱 2D / 3D 渲染性能与 Bug 复审（2026-10-19）

对应 [工程计划 · 图谱动态渲染性能专项](../../plan.md#2026-09-08-图谱动态渲染性能专项)。
目标：`docs/graph.html` + `docs/graph-3d.js` 在**低配电脑浏览器（集显、2–4 核）与手机**上流畅渲染 2D / 3D 图谱。
本页是**审查结论与分批方案**，不含实现；G0–G4 已完成的项（见 [G0 基线](graph-perf-baseline-g0.md)、[G2/G3](graph-perf-g2-g3.md)、[G4](graph-perf-g4.md)）不重复，G2 实测否定的「细线 `linkWidth→0`」不再提。

状态：**v2**（2026-10-20，见 §6）—— v1 五个审查视角（2D 力模拟、2D 渲染、3D 场景、3D 库 API、加载）与实测已完成；v2 补齐交互事件、移动端 CSS、Bug 专项三个视角，并对 F01–F10 做了 3 路对抗验证（§6）。

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

---

## 6. v2 修订（2026-10-20）：补跑视角与对抗验证

补跑了交互事件、移动端 CSS、Bug 专项三个视角；全部发现去重后共 **64 条**（22 Bug / 42 perf）。对抗验证（每条 3 个独立反驳者：代码追踪 / 收益 / 回归风险）只完成了 F01–F10 这 10 条，**全部成立**，修正意见已并入下文；其余 54 条未经第二方验证，其中标 [代码确认] 的为本人逐行核对，其余为单 agent 结论。

### 6.1 对 v1 的修正（来自对抗验证）

| v1 项 | 修正 |
|---|---|
| B2（`end` 必 fit） | 一次性标志还须盖住 4589 的 2.2 s 兜底定时器；3D→2D 路径（2934-2936）直接同步 `fitToScreen()` 且**不重置** `userZoomed2D`，需显式处理。 |
| B6（`cooldownTime` 15 s） | **单独设 `cooldownTime(Infinity)` 不改善流畅度**：只是把 15 s 未收敛冻结换成 ~270 帧「每帧 1 tick、2–4 fps」（慢设备上 60–110 s 抖动）。必须配合按设备缩放的 `d3AlphaDecay`（plan.md 允许「更快冷却作为测量后的补充调参」）或 G5-3D 解耦 tick 与帧。 |
| B1（NaN 不透明度） | 补充：因 `NaN !== NaN`，库每次刷新连线都走 `material.dispose()` + 重新赋材质，现状额外付费；改 rgba 方案后 `getNodeColor` 的高亮色（hex、alpha 1）需显式带 alpha。 |
| P0「点击不 reheat」 | 懒 reheat 本身是 S 且无风险；agent 顺带提的「拖拽时只模拟 1 跳子图」是独立的较大改动，不并入。 |
| B3 | `svg.interrupt('tlfit')` 实际在 5578 / 5669；修好后每 650 ms 一次 500 ms 缩放过渡会在弱设备上每帧触发边层重绘 + 标签同步，与 F24 联动。 |

### 6.2 新增 Bug（v1 未列）

| # | 现象 | 位置 | 证据 | 修法 |
|---|---|---|---|---|
| B15 | **3D 渲染环可在 2D 视图下被唤醒**：`show()` 排队的 `afterGraphDataApplied`（最多 240 帧轮询）/ `scheduleCustomMeshInstall`（90 帧）回调最终调 `resumeRenderLoop()`，它只查 `graph` 与 `renderer()`，不查 `container.hidden`。慢设备上首次进 3D 的几秒内点回 2D，之后隐藏的 3D 继续跑 ~270 tick 力模拟 + 每帧渲染 6 万对象 | `graph-3d.js:1094-1115, 1134-1139, 1380-1399, 1510-1548` | [代码确认] + 3 路验证成立 | `resumeRenderLoop` / `reheatAfterGraphData` 开头加 `if (!graph \|\| container.hidden) return;`；show/hide 维护代数计数，回调带代数、过期即丢 |
| B16 | **「刷新布局」后切 3D**：分片 warmup 的 rAF 循环只认 `warmupToken`，`pauseSimulation2D` 不 bump，余下分片跑完后 `startVisiblePhase()` 在 3D 下 `simulation.restart()`，隐藏的全图 2D 模拟与 3D 初始化抢 CPU 10–30 s，冷却后 `end` 还会动 3D 相机（与 B11 同根） | `graph.html:4085-4088, 4537-4551, 4520-4525, 4579` | [代码确认] + 3 路验证成立 | `pauseSimulation2D` 里 `warmupToken++`；`startVisiblePhase` 在 `is3DView()` 时不 restart；`end` 处理器在 3D 下直接返回 |
| B17 | **视图切换竞态**：3D 库异步加载期间点「2D」被 `mode === spatialViewMode` 早退忽略，加载完成后的 `then()` 仍切到 3D；已保存 3D 偏好的自动切换还带 `force:true`，直接覆盖用户刚点的 2D | `graph.html:2863-2894, 4328-4339, 6068` | [代码确认] + 3 路验证成立 | 请求序号：`requestSpatialViewMode` 每次自增，`then()` 里序号不符即放弃并恢复按钮；2D 点击也 bump 序号 |
| B18 | **`?focus=<id>` 飞行被打断**（桌面）：`flyToFocus` 起 700 ms 缩放过渡后立刻 `openSidebar` → 220 ms 后 `recenterAfterViewportChange` 调 `zoom.transform`，d3 内部 `interrupt()` 掐断飞行；焦点节点停在屏外直到力模拟结束（实测 9.5 s 后仍在 y=−421）。另：`end.focus` 与 1400 ms 定时器各触发一次，飞行每次都跑两遍 | `graph.html:6449-6467, 6098-6112, 5905-5914` | [agent 实测] + 3 路验证成立 | 先开侧栏，在 `scheduleSidebarViewportSync` 的 finish 回调里再起飞行（用侧栏后的 W 算目标）；`end.focus` 跑过后清掉定时器 |
| B19 | **「清除全部」不清图例聚焦与搜索**：`focusedCommunityId/Type/Opensource/Health` 与 `searchQuery` 不重置，而 `hasActiveGraphFilter()` 包含它们 → 图仍淡化、子图仍生效、灰节点 `pointer-events:none`；徽标计数也不含这几项 | `graph.html:5133-5145, 5148-5160, 4783-4788` | [agent 实测] + 3 路验证成立 | 复用 `setColorMode`（4461-4471）里已有的清零块；`updateFilterCount` 改由 `hasActiveGraphFilter()` 派生 |
| B20 | **3D 侧栏高亮跨视图泄漏**：`applyNodeNeighborHighlight / clearNodeNeighborHighlight` 只在 `is3DView()` 时同步给 3D，但 graph-3d 自持 `sidebarNodeId`；3D 选中 → 切 2D 关侧栏 → 切回 3D：全图仍淡化、hover 被 `if (!sidebarNodeId)` 挡死 | `graph.html:6161-6163, 6172`；`graph-3d.js:817-821, 1455-1458` | [代码确认] | 去掉 `is3DView()` 门，只要 `graph3dView` 存在就同步；进 3D 时按当前侧栏状态重放高亮 |
| B21 | 磁吸模式无 `health` 分支：按健康度上色时仍按类型聚类（2D/3D 同） | `graph.html:2715-2724` | [代码确认] | 补 `health` 的 key 与 categories |
| B22 | **移动端 2D 轻触节点「先显后隐」**：触屏派发的兼容鼠标事件先触发 `mouseenter → showTooltip`，`isMobile` 分支在那里就把 `pinnedNode = d`，随后 `click` 看到 `pinnedNode === d` 即关闭；第二次点才生效。3D→2D 的 `hideTooltip()` 也不清 `pinnedNode` | `graph.html:4821-4855, 4640-4652, 6380-6393, 2931` | [代码确认]，中置信度（依赖浏览器兼容事件行为，未真机验证） | `showTooltip` 不再设 `pinnedNode`，只在 click 路径设；`isMobile` 时 mouseenter/mousemove/mouseleave 直接返回；切 2D 时清 `pinnedNode` |
| B23 | **3D 触屏手势后出现「幽灵 hover」**：库在 pointerup 后不清 `pointerPos`，每 50 ms 的 raycast 以抬指位置命中节点 → `onNodeHover` 不分指针类型就 `refreshAppearance()`，全图淡化直到下次点击；每次轻触还付两次全量刷新 | `graph-3d.js:1453-1460, 1369-1375`；`graph.html:2795-2808` | [代码确认] vendor 指针逻辑，中置信度 | 把 `isMobile` / `pointerType` 传进 `RNGraph3D.create`，触屏下 `onNodeHover` 不改 hover 态；可扩展 `verify_graph_3d_touch_highlight.cjs` 断言手势后 `node_dimmed === 0` |
| B24 | 触屏上任何**高度**变化（键盘弹出等）都走 `fitToScreen()` 而非 recenter，丢掉用户缩放 | `graph.html:5934-5949, 5957-5967` | [代码确认]，低置信度（键盘行为因浏览器而异） | 移动端只在宽度变化 / `orientationchange` 时 refit；搜索框聚焦期间忽略；viewport meta 可加 `interactive-widget=resizes-visual` |
| B25 | iOS PWA（manifest `standalone`）下无 `viewport-fit=cover` 与 `env(safe-area-inset-*)`，底部 FAB / 提示 / 固定浮窗落在 Home 指示条手势区 | `graph.html:5, 549-556, 670-673, 830-836, 4647-4650` | [代码确认]，影响范围推测 | 加 `viewport-fit=cover`，底部偏移改 `calc(16px + env(safe-area-inset-bottom))` |

### 6.3 新增性能项

| 项 | 位置 | 要点 | 证据 |
|---|---|---|---|
| **3D 筛选只改外观**：Top N / 社区筛选在 3D 仍模拟 6,216 节点 / 53,740 边、保留 6 万场景对象；只有时序模式有真子集路径 | `graph-3d.js:836-840, 1565-1567, 1612` | **P1「移动端默认子图」在 3D 侧收益为零，除非 `syncFilters` 改成按子集重建 `graphData`**（复用时序子集机制；库按对象身份保留留下的节点，位置不丢） | [代码确认] |
| 2D 边层在 Top-N 态仍每 tick 遍历并描全部 53,740 条边，而被冻结端点的边坐标不变 | `graph.html:3800-3875, 4490-4492` | 拆 `activeEdges / frozenEdges`，冻结暗批画到离屏 canvas，仅筛选 / 主题 / 缩放变化时重绘，每 tick 只描活跃边 | [代码确认] + G4 §4（Top 300 的 drawEdges 8.9 ms > 全图 6.1 ms） |
| 全局 CSS transition：`.node-circle { transition: r .15s, fill-opacity .15s }`、`.node-label { transition: opacity .15s }` 作用于全部节点，任何批量写属性（滑块、hover、轻触、缩放）都重启 ~7,900 个 CSS 动画 | `graph.html:846-847` | 删全局规则，只给被交互节点加过渡 | [代码确认] |
| 7 处 `backdrop-filter`（工具栏、图例 FAB 常驻、图例抽屉关闭态只是平移出屏仍参与合成、物理 / 筛选面板、侧栏） | `graph.html:53, 242, 568, 601, 1145, 1427, 1758` | 背景 alpha ≥ 0.92 的直接删（无视觉差）；其余只在 `(hover:hover) and (pointer:fine)` 或高档位启用；抽屉关闭后 `visibility:hidden` | [代码确认]；GPU 开销量级 [推测] |
| 2D 时序播放：14 ms `setInterval` 每步 `nodes.filter` + `edges.filter`（53,740）+ `simulation.nodes()` 全量力重初始化，`revealTimelineNode` 还用 `node.filter` O(N) 找元素（`d.__el` 已存在） | `graph.html:5431-5433, 5506-5518, 5546-5574, 5588-5600` | 增量维护成员数组（用已有 `timelineIncident`），`d3.select(d.__el)`，改 rAF + 时间预算驱动 | [代码确认]；headless 4 s 只显现 72/6216 |
| 每个节点一个 `.node-glow` 圆（6,216 个，只有 hover/pinned/hub 时非零） + 同步建 6,216 个 `<option>` | `graph.html:4162-4167, 3642-3651` | 单个共享 glow 元素随 hover 移动；datalist 懒建 | [代码确认] |
| SVG `filter: drop-shadow` 在每 tick 重绘的层里：社区标签胶囊（每 tick 21 次离屏模糊）、`.node-depth-hub` | `graph.html:852-854, 288-292` | 用几何体（第二个低 alpha 矩形 / 描边）替代 filter | [代码确认] |
| `pointerdown` / `visualViewport.scroll` 每次都强制回流 + 重写 svg 尺寸 + 重建 center/x/y 力，再判断尺寸是否变化 | `graph.html:5929-5949, 3941-3955` | 先只读算出 W/H，不变即返回；fit 去抖 150 ms | [代码确认] |
| 3D tooltip：`mousemove` 与 `pointermove` 都注册 → 每次移动两次 `innerHTML` 重建 + KaTeX 扫描 + 强制布局 | `graph.html:2810-2819, 4637-4665`；`graph-3d.js:1369-1375` | 同一节点只 `moveTooltip`；只注册 `pointermove`；缓存 offsetWidth/Height | [代码确认] |
| Service Worker：缓存名按部署日轮换、`activate` 清掉包括不变的 vendor 在内的全部旧缓存；`link-graph.json` 走 stale-while-revalidate，每次访问后台整份重下并写缓存，与首屏建图抢带宽 | `sw.js:2, 54-64, 92-113` | vendor 单独缓存桶不随版本清；大导出按哈希 / 时间条件再验证，且推迟到首屏完成后 | [代码确认] |
| `main.js`（345 KB / 90 KB gz）整份进图谱页只为 2 个小 helper | `graph.html:2126-2128`；`main.js:4316-4330` | 拆 `rn-core.js`（与 v1「加载」表的 defer 项合并） | [代码确认] |
| 3D 视图在 2D 期间全量常驻（WebGL 上下文 + 6 万 mesh + 标签 DOM），`destroy()` 从未被调用；主题切换在 3D 隐藏时也全量 `refreshColors()` | `graph-3d.js:1544-1548, 1811-1832`；`graph.html:2739-2742, 5980-5987` | 触屏 / `deviceMemory ≤ 4` 时 hide 即丢连线对象（`graphData({links:[]})`），show 时按缓存的 link 记录重建（依赖 B5 的「link 数组只建一次」）；`refreshColors` 加 hidden 守卫 | [代码确认]；内存量级 [推测] |
| 3D 节点标签 div 各带 `will-change: transform`（每个标签一层）+ opacity 过渡；目前开关不在 DOM 里，为潜在问题 | `graph.html:411-423` | `will-change` 移到容器；重新启用时低档位限制渲染数量 | [代码确认] |
| `#graph-canvas` 无 `touch-action` 规则（只有 `#graph-canvas-3d` 有），页面原生双击缩放 / 捏合可能与 d3-zoom 并行 | `graph.html` CSS 397 附近 | 加一行 `touch-action: none` | [代码确认]，真机行为 [推测] |

### 6.4 本页结论的修订

- **P1「移动端默认子图」必须同时做 3D 的真子集 `syncFilters`**，否则手机进 3D 后收益归零。
- **B6 的修法升级**：不是只把 `cooldownTime` 放开，而是「按设备缩放 `d3AlphaDecay` + 解耦 tick 与帧（G5-3D）」。
- 对抗验证只覆盖了 F01–F10；§3 中标 [agent 实测] 的缩放 / 平移 / 栅格化数字（标签 `display:none` 230–540 ms、命中测试 34–62 ms、Commit 768 ms）仍是单 agent 结论，实施前先用 `measure_graph_perf.cjs` 的扩展 profile 复测。

---

## 7. 实施进度（2026-10-20，同一 PR 内逐项提交）

每项一个 commit，均在本机 headless Chromium（4 核 → 低端档）复测；绝对毫秒偏悲观，看相对变化。

| 项 | 状态 | 实测（改前 → 改后） |
|---|---|---|
| B3 时序 `t.name` 抛错 | 已修 | pageerror 5 → 0，播放中相机跟随（k 跟到 2.2） |
| B8 / B21 磁吸 Y 兜底、健康度分支 | 已修 | 按健康度聚类生效 |
| B2 / B16 `end` 一次性 fit、warmup 后先 fit、3D 下不 restart | 已修 | 放大拖拽后视口保持 k=1.447（改前跳回 0.18） |
| 首屏 warmup 分片 | 已做 | 加载期最长长任务 4,306 → 1,421 ms |
| 点击不加热、首次拖动才加热 | 已做 | 点击后 alpha 不变；拖拽 alpha 0.056、节点随手移动 |
| alphaMin / theta / 碰撞力门控 + LOW_END_DEVICE | 已做 | 力 tick p50 147 → 111 ms，首屏落稳 53 → 28 s，`verify_graph_refresh_layout` 首次 ok |
| B1 / B4 3D 连线 NaN、hover 不再整图 digest | 已修 | 材质 NaN 53,740 → 0；hover 延迟 digest 3.3–3.7 s → 无；同步 99–121 → 117–176 ms |
| B5 首次进 3D 1 次 graphData、连线记录复用 | 已做 | 点击 → 53k 连线 38 → 25–27 s（软件 GL） |
| B6 / B7 / B15 冷却分档、tick 计数取景、死守卫、hidden 守卫 | 已修 | 切回 2D 后隐藏 3D 帧计数不再增长 |
| 3D 像素比 / MSAA 分档 | 已做 | DPR 2：低端 1× 无 MSAA；高端 1.5× |
| 3D 渲染环空闲暂停 | 已做 | 引擎停机后帧计数 96 → 96，滚轮唤醒，阻尼结束再停 |
| 3D 连线排除 raycast、指针拾取按输入开关 | 已做 | 53,740 条全在 layer 1，draw calls 不变；悬停命中 97 → 20 s（软件 GL） |
| 2D 标签 `display:none` | 已做 | 12 步滚轮 >100 ms 帧 12 → 2，最长帧 879 → 118 ms |
| 平移时节点层 `pointer-events:none` | **已放弃** | trace：HitTest 592 → 480 ms，但 UpdateLayoutTree 64 → 280 ms（继承属性引发 25k 元素重算），净负 |
| hover 离开不整跑 applyFilters、邻接表 | 已做 | mouseleave 110 → 31 ms（筛选态 90 → 26） |
| B9 侧栏 id 索引、装饰排序 | 已修 | 枢纽开侧栏 3,953 → 57 ms |
| 目录按需加载、datalist 懒建 | 已做 | 首屏不再请求 3.9 MB 目录；开侧栏 0.9 s 内补齐 |
| preload 数据、SW 延后、流式解析 | 已做 | 数据请求 234 → 29 ms 起步；首屏就绪 5.7–8.7 → 4.4 s |
| B17 / B19 / B20 / 时序暂停冷却 | 已修 | 清除全部回到 6,216；暂停 0.8 s 后 alphaTarget 0；3D 高亮跨视图同步 |
| B22 / B23 触屏轻触、3D 幽灵 hover | 已修 | 轻触一次钉住，空白处轻触关闭；旋转后 node_dimmed 0 |

未做（待后续）：全局 CSS transition / `backdrop-filter` / 组级 opacity（F40 / F41 / F50）、3D 材质缓存与几何精简（F52 / F53）、进度条分母（B10）、DPR 变化重绘（B12）、WebGL 上下文丢失（B13）、移动端视口（B24 / B25）；P1 的移动端默认子图（含 3D 真子集）、导出期预计算布局、数据瘦身；P2 的 G5 / G6。
