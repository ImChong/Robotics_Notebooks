# 前端体验优化清单 v1 (Frontend Optimization Checklist)

状态：**进行中** (已完成结构重构)
目标：**极简化首页 (Minimalist Homepage)** —— 消除视觉噪音，突出目标入口、学习路线、搜索与知识图谱。

---

## 一、 视觉降噪 (Visual De-cluttering)

- [x] **重构英雄区 (Hero Section)**：
    - [x] 压缩文字：将目前的 4 段文字（Kicker, Title, Subtitle, Desc）精简为 1 行核心标语 + 1 行副标题。
    - [x] 移除右侧侧边栏 (Hero Panel)：将“你现在更适合从哪进？”的逻辑融入目标导航卡片。
    - [x] 整合 Stats Bar：将 3 个统计卡片改为一行半透明的微型数据条。
- [x] **合并冗余区块**：
    - [x] 将“两条成长路线”、“当前技术栈主干”、“关系页”、“论文导航”这四个大区块（约 16 个卡片）合并为一个 **“探索入口 (The Gateway)”** 网格，仅保留 4-6 个最具代表性的入口。
    - [x] 移除“这个项目是做什么的”区块：其内容已在 Hero 区和 README 中体现，首页不再重复平铺。

## 二、 交互增强 (Interaction First)

- [x] **搜索框置顶 (Search-Centric Design)**：
    - [x] 保留搜索框的核心地位，并在目标入口与学习路线之后提供明确的知识检索入口。
    - [x] 增加搜索框的视觉权重（更大、带有毛玻璃效果、更明显的聚焦反馈）。
- [x] **图谱深度集成 (Graph Integration)**：
    - [x] 将 `mini-graph` 移至英雄区背景或作为右侧核心视觉元素。
    - [x] 实现图谱节点与搜索框的联动：在搜索时，背景图谱自动高亮相关节点。`mini-graph.js` 暴露 `window.RNMiniGraph.highlight(ids, query)/clear()`；首页 `main.js` 搜索命中后按结果 id + 标签词命中高亮相关节点、其余淡出（`.mini-node-dim`/`.mini-node-hit`），命中为空则不淡出避免整图变暗；清空/Esc 复原。验证脚本 `scripts/screenshot_home_search_graph_link.cjs`（`slam` → 命中 2 / 淡出 48 / 共 50）。
- [x] **全局导航精简**：
    - [x] Header 仅保留：Logo、搜索图标（移动端）、图谱入口、GitHub。
    - [x] **约束**：保留当前 `site-header` 的基础视觉设计（毛玻璃效果与布局），仅精简文字。

## 三、 风格对齐 (Aesthetic Refinement)

- [ ] **引入 Technical Blueprint 风格**：
    - [ ] 背景：采用深色调 (#0d1117) 配合细密网格线 (Dot Grid)。
    - [ ] 字体：标题切换为工程感更强的 `DM Mono`，正文优化衬线比例。
- [x] **响应式体验优化**：
    - [x] 确保极简后的首页在手机端首屏就能看到搜索框和图谱入口。
    - [x] 图谱页 `graph.html`：「按社区着色」图例全端（PC + 移动）改为左下角 FAB，默认折叠；点击后自左侧滑出抽屉，点击空白/遮罩或 Esc 收起，避免遮挡画布。PC 端展开尺寸约为视口宽 1/4、高 1/3。
    - [x] 图谱页 `graph.html`：移动端 Chrome 下滑隐藏底栏后，画布壳层按 `visualViewport` / `dvh` 同步高度，避免 graph view 下方留白。
    - [x] 录制验证视频 `graph-mobile-viewport-fix.mp4`；同步刷新 README `media/graph-demo.gif`（1776 节点）。
    - [x] README 演示 GIF 升级为全站演示 `media/site-demo.gif`（替换并删除旧 `media/graph-demo.gif`）：首页入口卡 → 全库即时搜索（MPC）→ 图谱预览 → 完整图谱悬停 / 缩放 / 节点详情侧栏 / 3D 切换，带分步字幕；由 `scripts/record_readme_demo.cjs` 生成（1799 节点）。

---

## 执行节奏 (Implementation Phases)

### Phase 1: 结构手术 (Structure) - [x] *已完成*
- [x] 修改 `index.html`：删除冗余 Section，保留搜索和核心 Entry。
- [x] 将搜索框上移，Stats Bar 缩小。

### Phase 2: 视觉焕新 (Aesthetics) - [x] *已完成*
- [x] 更新 `style.css`：引入网格背景、新字体和组件阴影优化。
- [x] 重构 Hero 区布局，引入 D3 动态背景。

### Phase 3: 体验打磨 (UX)
- [ ] 搜索结果列表优化：支持卡片式快速预览。
- [ ] 详情页浮窗联动。
- [x] 图谱页移动端动态视口：Chrome 底栏显隐时 `#graph-wrap` 用 `--app-vh`（`visualViewport.height`）撑满，消除下方空白条。
- [x] 详情页 Markdown 渲染修复：正文独立 `---` 行渲染为分隔线，并按整行分隔符剥离 YAML frontmatter。
- [x] 首页搜索索引修复：将 `tech-map/`、roadmap 与 references 纳入 `search-index.json`，支持搜索 `tech-map`。
- [x] 技术路线页 `roadmap.html`：基于导出阶段数据渲染可折叠阶段树（垂直 `<details>` 列表）；已移除折叠 Mermaid 线框总图与路线页内 Mermaid CDN。
- [x] 技术路线页 `roadmap.html`：favicon、顶栏统一为 🚀（与首页「技术路线指南」CTA 一致）；主题切换仍为 ☀️/🌙。
- [x] 技术路线页侧栏 TOC（桌面 ≥1632px）：卡片封顶 `min(76vh, 680px)`，列表区内滚动；移动端抽屉保持原样全高滚动。
- [x] 详情页「关联知识图谱」迷你图：1-hop 邻居按全图度数（节点大小）优先；≤16 全显示、超出取 Top-16；弹簧距离/强度随邻居大小变化（近=重要）；完整邻居仍走 `graph.html?focus=`。
- [x] 首页知识图谱预览：Top-N 诱导子图剔除孤立点并按度数回填可连通候选，避免「全局度数高但邻居不在预览集」的孤儿节点漂浮。
- [x] 详情页 Mermaid：正文 14px / 移动端 12px，增大节点 `padding` 与 `wrappingWidth`，减轻大字贴边与单行裁切；灯箱仍按 1.75× 离屏高清重绘。
- [x] 详情页 Mermaid 标签裁切：`htmlLabels` 下 `foreignObject` 比 `nodeLabel` 略窄时 post-render 扩框 + CSS `overflow: visible`，修复如 world-models-15 技术地图等长标签贴边裁切。
- [x] 纵深路线「路线一览」Mermaid：节点标签 `**Stage N**` 改为 `<b>Stage N</b>`（与已有 `<br/>`/`<em>` 对齐）；`main.js` 渲染前将残留 Markdown `**…**` 规范为 `<b>`，避免星号原样显示。
- [x] Markdown 嵌套列表：`renderMarkdownContent` 保留缩进层级，路线页「其它纵深路径 / 关联知识页」等不再被拍平为同级 bullet。
- [x] 首页「更多路线」六按钮排版：`home-entry-route-links` 最小列宽 132px → 118px，861–940px 窗口宽度下不再出现 5+1 孤行；≤860px 移动端保持 2×3 网格。
- [x] 首页「更多路线」按钮按方向起点里程碑的历史顺序排列：传统控制（ZMP 1972）→ 安全控制（CLF 1983）→ 接触操作（阻抗控制 1985）→ 模仿学习（行为克隆 1988）→ 强化学习（Q-learning 1989）→ 感知越障（2020s）。
- [x] 首页「更多路线」扩为十按钮并保持历史序：新增导航（概率 SLAM 1986）、移动操作（移动操作臂协调 1994）、BFM（DeepMimic 2018）；原 VLA·BFM 合并按钮拆为独立 BFM 与 VLA（RT-2 2023）两个入口。
- [x] 首页「更多路线」扩为十三按钮并保持历史序：新增 WAM（World Action Models 综述形式化，2026）；默认展示最新 4 条（感知越障 / 动作生成 / VLA / WAM），BFM 并入折叠区；展开文案同步为 13 条。
- [x] 首页「更多路线」扩为十八按钮并保持历史序：新增人形群控展演（央视春晚 540 台 Alpha 1S 群舞，2016；插在动作重定向与 Sim2Real 之间）；展开文案与 `main.js` 同步为 18 条。
- [x] 修复合并冲突冲掉的「人形群控展演」首页按钮：`docs/index.html` 补回入口，默认折叠态文案「17 → 18」。
- [x] 修复移动端折叠态 WAM 通栏：末行居中规则改为仅 `.is-expanded` 时生效，避免 hidden 节点被 `:nth-child` 计入导致第 13 项误判孤行。
- [x] 更新记录页 `change-log.html`：超量日先预览 10 条；提供「再展开 10 项 / 展开全部 N 项 / 收起至前 10 项」，视觉沿用箭头+文案样式（PR#1245）。
  - [x] 详情页 Mermaid 灯箱：加载态文案 + `stage-pending` 在 `fit` 完成前隐藏，消除「空面板 / 先大后缩」闪烁。
  - [x] 图谱页 `graph.html`：参数面板新增「显示节点名称」开关（默认关闭）；2D SVG 标签与 3D HTML 叠加层共享同一状态，开启后两端同时显示。
  - [x] 图谱页 `graph.html`：参数浮窗去掉「显示节点名称」勾选项；`showNodeLabels` / `setShowNodeLabels` / 2D·3D 标签同步逻辑源码保留（默认仍关闭，无 UI 入口）。
  - [x] 图谱页 `graph.html`：参数面板新增「更新明度渐变」开关（默认关闭）；节点时间口径对齐「更新记录」页（`link-graph.json` 新增节点级 `activity` 字段，取 log.md 最近活跃日），开启后最新节点全亮、越旧越暗（下限 0.2 保持可见），无更新日期的节点按最暗显示；2D SVG 与 3D 视图共享同一开关。
  - [x] 图谱页 `graph.html`：「更新明度渐变」明暗分级由线性改为指数曲线 `f(t)=(e^{k t}-1)/(e^k-1)`（`k=3`）：归一化时间 t∈[0,1]（0=最旧，1=最新），越新越亮、近期对比更陡；2D / 3D 仍共享同一开关与下限 0.2。
  - [x] 图谱页 `graph.html`：参数面板「显示社区标签」勾选框（**默认开启**，用户可在参数浮窗自行关闭）；开启时社区名称以**胶囊卡片**（社区色圆角底 + 亮度自适应文字）漂浮显示在各聚类中心点（2D SVG 图层 + 3D HTML overlay 双端实现，3D 按 3D 质心投影），并随筛选/聚焦只统计可见节点；仅「按社区」筛选/着色时可勾选，按类型/健康度筛选时置灰不可选。
  - [x] 图谱页 `graph.html`：修复 3D 社区标签在力引擎收敛后冻结的问题——标签重投影原仅由 `onEngineTick` 驱动（alpha 收敛即停发），现同时挂 `controls` 的 `change` 事件，相机旋转/缩放/飞行动画期间持续跟随（3D 节点名称标签同因并修）；2D 标签在 `gRoot` 组变换内天然跟随，实证无此问题。
  - [x] 图谱页 `graph-3d.js`：修复 3D 社区标签旋转/平移卡顿——相机 `change` 路径改为 rAF 合并 + 缓存世界坐标质心后只做屏幕重投影（避免每帧 O(N) 重算）；定位由 `left/top` 改为 `translate3d`（`will-change: transform`），减少 layout thrash。
  - [x] 图谱页社区漂浮标签：字号随社区节点数缩放（√n 插值）；**2D / 3D 分开**——2D **8–28px**，3D **8–16px**（3D 另有相机 scale，字号收窄）；胶囊内边距等比跟随。
  - [x] 图谱页社区漂浮标签：兜底桶 `community-other`（「其他（Other） 社区」）不漂浮显示；图例与筛选仍保留；字号区间亦排除该桶。
  - [x] 图谱社区上限：`PRIMARY_COMMUNITY_CAP` / `MAX_COMMUNITIES` **21**（命名席位；含「其他」时图例总数目标约 **20**）。
  - [x] 图谱页 3D 社区漂浮标签：随相机距离（滚轮缩放）与节点一起放大缩小（`translate3d` + `scale`，相对首次适配基线距离；zoom 钳制随画布短边约 **0.4–1.75** / 窄屏 **0.55–1.25**）；2D 标签在 `gRoot` 变换内天然跟随。
  - [x] 图谱页 3D 社区漂浮标签：按画布短边相对 ~800px 参考值 **连续缩放**字号（√ 比例，钳制约 **0.78–1.28**，绝对字号约 **7–22px**），取代旧的「移动端字号×0.55 + zoom×0.55」二元收紧，避免手机/平板有效字号落到 ~3–5px、大屏却完全不放大；验证脚本 `scripts/verify_graph_community_labels_3d_responsive.cjs`。
  - [x] 图谱页 3D 社区漂浮标签：点「适配屏幕」飞行动画期间锁定胶囊 scale 为落稳尺寸，并提前写入目标距离基线，避免中途 `|cam−lookAt|` 偏离导致标签大小闪一下。
  - [x] 图谱页 `graph.html`：筛选浮窗保留顶部「按类型 / 按社区 / 按健康度」按钮三选一；当前维度的勾选项改为与「路线视图 / 研究机构」一致的可折叠 `<details>`，折叠标题在有选中时显示数量（如 `3 个社区`）。
- [x] 首页「互链枢纽 · Top 10」底部入口改为「查看完整榜单 →」，新增 `docs/hubs.html` 全量互链榜单页（数据源 `exports/hub-rankings.json`，全站 / 论文双 tab）。
- [x] 首页 Hero 盘点条新增「主路线」（数字 1，位于「互链关系」与「纵深路线」之间）；四项数字可点：知识节点 / 互链关系 → `graph.html`，主路线 → 锚到「从零开始」卡并顺时针描边一圈，纵深路线 → 锚到「更多路线」卡描边一圈（不展开）后短时高亮「展开全部…」文案。
- [x] 首页入口卡边框描边多分辨率对齐：SVG 改为按 border-box 像素定位（计入 border 宽度与亚像素 `getBoundingClientRect`），去掉 padding-box `inset`/`%` 尺寸冲突；圆角跟随 computed `border-radius`；动画期间 ResizeObserver 跟随卡片尺寸。
- [x] 首页 Hero 顶栏下方留白收紧：桌面 `.hero` padding `68px 0 44px` → `32px 0 28px`，移动端 `38px 0 30px` → `24px 0 22px`，减少「持续更新的机器人技术栈地图」徽章上方空档，入口卡更易落在首屏。
---

### Phase 4: 信息架构优化 (Information Architecture) - [x] *已完成*
- [x] 首页调整为：目标入口 → 搜索 → 最新内容 → 知识图谱 → 互链榜单。
- [x] 新增目标入口：从零开始、项目查询两张卡，强化学习 / 传统控制并入「更多路线」通栏卡。
- [x] 移除 Hero 区与目标入口、知识图谱预览重复的两个 CTA 按钮。
- [x] 详情页移除“通用详情页”“正文同步内容”“消费 JSON”等面向实现的措辞，统一为读者语言。
- [x] 去重收敛：删除独立「学习路线」区块，纵深路线（强化学习 / 传统控制 / 模仿学习 / 安全控制 / 接触操作）统一并入目标入口区的「更多路线」卡。
- [x] 「项目查询」入口卡点击后直接聚焦搜索输入框（`data-focus-search`），减少锚点跳转后的二次寻找。
- [x] 「项目查询 / 知识图谱」入口卡：滚到对应区块（`#wiki-search` / `#mini-graph-section`）**窗口顶端**（`scrollIntoView({ block: 'start' })`，与旧锚点一致），再对模块（`#wiki-search-panel` / `#mini-graph-wrap`）顺时针描边一圈；项目查询仍额外聚焦搜索框；深链同效。Hero「主路线 / 纵深路线」仍居中。
- [x] 「项目查询」点击卡顿优化：点击后立刻 `focus`（不再等 scrollend）；滚动接近落点即开描边（rAF 轮询，fallback 420ms）；空查询 focus 静默预取搜索索引（不写「加载中…」）；`pointerdown`/`mouseenter`/idle 预取 `search-index.json`；描边 SVG 推迟到下一帧挂载；去掉描边 `drop-shadow` 滤镜以免动画期掉帧。
- [x] 删除搜索区副标题（与输入框 placeholder 重复），Hero 副标题改为面向读者的引导语。
- [x] 删除目标入口区标题「选择你的入口」与副标题「按当前目标进入最短路径…」：与 Hero「先选一个入口」重复，入口卡本身已自解释。
- [x] 同步刷新 README `media/site-demo.gif`（`scripts/record_readme_demo.cjs`，1933 节点；首页帧已无入口区标题/副标题）。
- [x] 再次同步刷新 README `media/site-demo.gif`（`scripts/record_readme_demo.cjs`，2031 节点 / 17987 边；70 frames / 3.21 MB）。
- [x] 再次同步刷新 README `media/site-demo.gif`（`scripts/record_readme_demo.cjs`，2058 节点 / 18475 边；70 frames / 3.12 MB）。
- [x] 再次同步刷新 README `media/site-demo.gif`：录制脚本真实点击「项目查询 / 知识图谱」入口卡并纳入顺时针描框特效（`scripts/record_readme_demo.cjs`，2060 节点 / 18500 边；88 frames / 3.48 MB）。

---

## 验收标准 (DoD)
- [x] 首页滚动条长度缩减 50% 以上。
- [x] 首页首屏 (Above the fold) 必须包含：核心定位、规模统计，并紧接按目标选择的入口。
- [x] Lighthouse 性能/可访问性评分保持在 90+。
