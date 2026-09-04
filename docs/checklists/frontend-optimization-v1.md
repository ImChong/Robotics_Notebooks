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
    - [x] 背景：暗色主题底色改为 `#0d1117`（与 `graph-3d.js` 画布同色，2D/3D 切换无色差），配套调深 `--bg-alt`/`--surface-strong` `#161b22`、`--border` `#30363d`、`--border-strong` `#6e7681`、`--text` `#e6edf3`、`--text-muted` `#9ba4b0`，`theme-color` / `manifest.theme_color` 同步 `#0d1117`；全站叠加细密点阵（`--grid-size: 22px` + `--grid-dot`，`body` 背景图与 `.hero-backdrop::before` 共用同一尺寸、同为文档原点且随内容滚动，叠加不产生摩尔纹；亮色主题 `--grid-dot: transparent` 保持纯净底不变）。对比度实测（暗色）：正文 16.0:1 / 次要文字 7.5:1 / 链接 6.9:1，均高于改动前（6.2:1）。
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
- [x] 搜索结果列表优化：支持卡片式快速预览。首页搜索结果卡片摘要默认按 2 行 `-webkit-line-clamp` 收起（仅摘要长度 > 120 字符时），卡片内提供「预览全文 / 收起」就地展开开关（`.result-preview-toggle`，`aria-expanded` 同步），无需跳转详情页即可快速预览完整摘要；短摘要卡片不显示开关。验证脚本 `scripts/verify_search_preview.cjs`（`slam` → 10 张卡片 / 4 个预览开关；收起态 clientH 45 < scrollH 89、展开后全显 89）。
- [x] 详情页浮窗联动：正文里指向站内知识页的内链（`detail.html?id=` / `roadmap.html?id=`）标记为 `.detail-inline-link`，悬停弹出与图谱同款 hover 卡片（类型徽标 + 标题 + 摘要 + 「打开详情页 →」）；徽标优先取 `link-graph.json` 的细类型（concept / task / paper…）与社区色，与迷你图浮窗同口径。双向联动：悬停正文内链 → 「关联知识图谱」迷你图同一节点点亮（`.mini-node-linked`）；悬停迷你图节点 → 正文中指向它的内链点亮（`.detail-inline-link-linked`）。触屏（`hover: none`）不绑定，点击仍直接跳转。验证脚本 `scripts/verify_detail_inline_link_preview.cjs`（`wiki-concepts-sim2real` → 内链 Locomotion 浮窗可见 / 迷你图点亮 1 个节点；反向悬停迷你图节点 → 正文点亮 1 条内链）。
- [x] 技术路线页 `roadmap.html` 本库超链接同样悬停浮窗：正文、知识地图叶子、阶段速览相关项中的 `detail.html?id=` / `roadmap.html?id=` 复用同一套 hover 卡片；无迷你图时类型徽标回退 `roadmapKmapNodeType`（path 前缀）。桌面端绑浮窗时去掉原生 `title`，避免与浏览器 tooltip 重叠。验证脚本 `scripts/verify_roadmap_inline_link_preview.cjs`。
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
- [x] 站点活动改由 git 驱动：首页「最新知识节点」/ 更新记录热力图 / 图谱 `activity` 取 `wiki/` `roadmap/` 提交（`A`=新增）；`log.md` 降为叙事层。浅克隆回退日志。
- [x] 修复首页「最新知识节点」回填方向：`wiki-activity.days` 为升序，回填须自新到旧；并为回填项补齐 `recency`（避免空白日期）；社区短标签与 `community_short_label` 对齐（去掉英文括注）。
- [x] 更新记录页热力图计数与筛选同步：默认按 `added_count` 着色/提示「个新增节点」；点击「显示维护节点」后按 `count`（新增+维护总量）重绘色阶与 tip，再切回仅新增。
- [x] 更新记录页说明文案精简：副标题 / 筛选提示 / 时间线 intro 缩短；筛选提示加 `word-break: keep-all`，避免中途突然断行。
- [x] 修复更新记录页移动端大段空白：`.updates-filter-hint` 的 `flex: 1 1 12rem` 在 column 布局下被当成高度；改为仅 ≥641px 横排时启用，移动端 `flex: 0 1 auto`。
- [x] 详情页标题：有 `sources/repos/` 源码关联的节点在 `#detailTitle` 末尾显示 ⭐️（与首页「最新知识节点」/更新记录/互链榜同口径；`site-data` 写出 `has_repo`，前端 `detailPageHasRepo` 另有正文回退检测）。
- [x] 详情页 Mermaid 灯箱：加载态文案 + `stage-pending` 在 `fit` 完成前隐藏，消除「空面板 / 先大后缩」闪烁。
  - [x] 图谱页 `graph.html`：参数面板新增「显示节点名称」开关（默认关闭）；2D SVG 标签与 3D HTML 叠加层共享同一状态，开启后两端同时显示。
  - [x] 图谱页 `graph.html`：参数浮窗去掉「显示节点名称」勾选项；`showNodeLabels` / `setShowNodeLabels` / 2D·3D 标签同步逻辑源码保留（默认仍关闭，无 UI 入口）。
  - [x] 图谱页 `graph.html`：参数面板新增「更新明度渐变」开关（默认关闭）；节点时间口径对齐「更新记录」页（`link-graph.json` 节点级 `activity` 字段，取 git 最近触达日；浅克隆回退 log.md），开启后最新节点全亮、越旧越暗（下限 0.2 保持可见），无更新日期的节点按最暗显示；2D SVG 与 3D 视图共享同一开关。
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
  - [x] 图谱页 `graph.html`：筛选浮窗保留顶部「按类型 / 按社区 / 按开源 / 按健康度」按钮四选一；当前维度的勾选项改为与「路线视图 / 研究机构」一致的可折叠 `<details>`，折叠标题在有选中时显示数量（如 `3 个社区`）。
  - [x] 图谱页 `graph.html`：筛选浮窗在「按健康度」前新增「按开源」着色/筛选——已开源节点 `#2dd4bf`（teal，关联 `sources/repos/`，与详情页 ⭐️ / `has_repo` 同口径），未开源节点 `#fb7185`（rose）；图例可点选聚焦，筛选勾选项带数量；2D / 3D 共享着色；验证脚本 `scripts/verify_graph_opensource_filter.cjs`。
  - [x] 图谱页 `graph.html`：筛选浮窗内「按社区（当前维度）/ 路线视图 / 研究机构」三区改为手风琴——同时仅展开一个，默认展开「按社区」；展开区吃满中间剩余高度，收起项靠在上方或下方；验证脚本 `scripts/verify_graph_filter_accordion.cjs`。
  - [x] 图谱页 `graph.html`：路线视图 chip 保持紧凑换行，不随筛选浮窗手风琴吃满剩余高度；「按社区 / 研究机构」仍吃满；收起的「研究机构」仍钉在容器底。
  - [x] 图谱页 `graph.html`：筛选浮窗在「连接数 Top N」下方新增「更新时间 Top N」滑块——左=最新子集、右=全部（默认全部）；排序口径对齐 `activity` / 更新明度；与连接数 Top N 同时生效时取交集；清除筛选一并复位；验证脚本 `scripts/verify_graph_recency_topn.cjs`。
  - [x] 图谱页 `graph.html`：筛选浮窗「连接数 Top N / 更新时间 Top N」改为与下方手风琴同款可折叠 `<details>`（默认收起；折叠标题显示当前值如「全部」/`N`；两区互斥二选一，可全部收起；不进三区互斥）；验证脚本 `scripts/verify_graph_topn_collapse.cjs`。
  - [x] 图谱页 `graph.html`：更新时间 Top N 增加「按节点 / 按日期」切换——默认仍按最新 N 个节点；按日期则整日保留（滑块改为最近 N 个活跃日，同一天节点一并留下）；从节点子集首次切到日期时按该子集覆盖的日期展开；两种口径各自记住滑块；清除筛选复位为按节点+全部；验证脚本 `scripts/verify_graph_recency_date_mode.cjs`。
  - [x] 图谱页 `graph.html`：更新时间 Top N 默认仅计 **新增**（git 首次加入日 `added`）；「显示维护更新」打开后改用最近触达日（`activity`，含维护）；与按节点/按日期正交；清除筛选复位为仅新增；`link-graph.json` 节点写出 `added` 字段。
  - [x] 图谱页 `graph.html`：筛选浮窗「当前按…着色与筛选」文案上移到标题栏正下方（模式 tab 之上）。
  - [x] 图谱页 2D 力模拟：略欠阻尼、**最多一次回弹**（验收以「刷新布局」为准，不用时序动画）——`velocityDecay=0.54`、极短 warmup 20 tick 后 `alpha=0.9` 开场、度数加权弹簧；3D 力参数保持对话前原状。
  - [x] 图谱页 2D 时序动画：退出时清零 `alpha`/`alphaTarget` 并停模拟（不再走 pause 路径把加热目标钉在 0.3）；`closeSidebar` 仅在侧栏确实打开时才 viewport sync / relayout restart，避免「退出后点空白 → 力模拟自振抖动」；验证脚本 `scripts/verify_graph_timeline_exit_settle.cjs`。
- [x] 首页「互链枢纽 · Top 10」底部入口改为「查看完整榜单 →」，新增 `docs/hubs.html` 全量互链榜单页（数据源 `exports/hub-rankings.json`，全站 / 论文双 tab）。
- [x] 首页 Hero 盘点条新增「主路线」（数字 1，位于「互链关系」与「纵深路线」之间）；四项数字可点：知识节点 / 互链关系 → `graph.html`，主路线 → 锚到「从零开始」卡并顺时针描边一圈，纵深路线 → 锚到「更多路线」卡描边一圈（不展开）后短时高亮「展开全部…」文案。
- [x] 首页入口卡边框描边多分辨率对齐：SVG 改为按 border-box 像素定位（计入 border 宽度与亚像素 `getBoundingClientRect`），去掉 padding-box `inset`/`%` 尺寸冲突；圆角跟随 computed `border-radius`；动画期间 ResizeObserver 跟随卡片尺寸。
- [x] 首页 Hero 顶栏下方留白收紧：桌面 `.hero` padding `68px 0 44px` → `32px 0 28px`，移动端 `38px 0 30px` → `24px 0 22px`，减少「持续更新的机器人技术栈地图」徽章上方空档，入口卡更易落在首屏。
- [x] 首页 Hero 规模数字（知识节点 / 互链关系 / 主路线 / 纵深路线）每次进入/刷新 count-up 翻滚；内联脚本首屏前归零避免终值闪跳；立刻用 `data-fallback` 开播（不等 fetch、不重播）；翻滚中保持终值文字色（不变强调色）；去掉位移动画；重 DOM 延后以减轻卡顿；`prefers-reduced-motion` 时直接显示终值。验证脚本 `scripts/verify_hero_stats_countup.cjs`。
- [x] 首页 Hero 徽章「持续更新…」前蓝点改为慢呼吸灯（`hero-badge-dot-breathe`，3.6s 正弦感曲线，透明度 0.68↔1 + 峰谷短暂停留 + 外环微强弱，避免闪烁）；`prefers-reduced-motion` 下保持静态圆点。
- [x] 详情/模块/路线「未找到」与长正文切换抖动：根因是经典滚动条有无导致居中主栏横移（宽仍同为 912px），以及空态 `.section` 在 JS 隐藏前吃掉 88px 顶底留白；修复为 `html { scrollbar-gutter: stable }`、空态 section 默认 `hidden` + `padding: 0`，并把空态卡片放进与正文相同的 `detail-content-layout` / `detail-content-main`。验证脚本 `scripts/verify_detail_empty_column_stable.cjs`。
- [x] 详情页英雄区简介：`#detailSummary` 走与正文相同的 `renderMathBlocks` + `renderDetailMath`，避免 `$O(n)$` 泄漏为 `\(...\)`；`.detail-summary .math-inline` 沿用正文公式高亮。
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
- [x] 再次同步刷新 README `media/site-demo.gif`（`scripts/record_readme_demo.cjs`，3000 节点 / 26062 边；78 frames / 2.86 MB）。
- [x] 全站 footer 作者名「Chong Liu」可点击跳转个人主页 `https://imchong.github.io/index.html`（新标签打开）；`.footer a` 继承次要文字色、悬停强调色。
- [x] 全站 sticky footer：`body:has(> .footer)` 用 flex 列布局 + `main { flex: 1 }`，短内容页把 `.footer` 推到视口最底部；无 footer 的页面（`graph.html` / `references.html`）不启用。

---

## 验收标准 (DoD)
- [x] 首页滚动条长度缩减 50% 以上。
- [x] 首页首屏 (Above the fold) 必须包含：核心定位、规模统计，并紧接按目标选择的入口。
- [x] Lighthouse 性能/可访问性评分保持在 90+。
