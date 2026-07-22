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
    - [ ] 实现图谱节点与搜索框的联动：在搜索时，背景图谱自动高亮相关节点。
- [x] **全局导航精简**：
    - [x] Header 仅保留：Logo、搜索图标（移动端）、图谱入口、GitHub。
    - [x] **约束**：保留当前 `site-header` 的基础视觉设计（毛玻璃效果与布局），仅精简文字。

## 三、 风格对齐 (Aesthetic Refinement)

- [ ] **引入 Technical Blueprint 风格**：
    - [ ] 背景：采用深色调 (#0d1117) 配合细密网格线 (Dot Grid)。
    - [ ] 字体：标题切换为工程感更强的 `DM Mono`，正文优化衬线比例。
- [x] **响应式体验优化**：
    - [x] 确保极简后的首页在手机端首屏就能看到搜索框和图谱入口。
    - [x] 图谱页 `graph.html`：「按社区着色」图例全端（PC + 移动）改为左下角 FAB，默认折叠；点击后自左侧滑出抽屉，点击空白/遮罩或 Esc 收起，避免遮挡画布。PC 端展开尺寸约为视口宽/高各 1/3。

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
- [x] 详情页 Markdown 渲染修复：正文独立 `---` 行渲染为分隔线，并按整行分隔符剥离 YAML frontmatter。
- [x] 首页搜索索引修复：将 `tech-map/`、roadmap 与 references 纳入 `search-index.json`，支持搜索 `tech-map`。
- [x] 技术路线页 `roadmap.html`：基于导出阶段数据渲染可折叠阶段树（垂直 `<details>` 列表）；已移除折叠 Mermaid 线框总图与路线页内 Mermaid CDN。
- [x] 技术路线页 `roadmap.html`：favicon、顶栏统一为 🚀（与首页「技术路线指南」CTA 一致）；主题切换仍为 ☀️/🌙。
- [x] 技术路线页侧栏 TOC（桌面 ≥1632px）：卡片封顶 `min(76vh, 680px)`，列表区内滚动；移动端抽屉保持原样全高滚动。
- [x] 详情页 Mermaid：正文 14px / 移动端 12px，增大节点 `padding` 与 `wrappingWidth`，减轻大字贴边与单行裁切；灯箱仍按 1.75× 离屏高清重绘。
- [x] 详情页 Mermaid 标签裁切：`htmlLabels` 下 `foreignObject` 比 `nodeLabel` 略窄时 post-render 扩框 + CSS `overflow: visible`，修复如 world-models-15 技术地图等长标签贴边裁切。
- [x] 首页「更多路线」六按钮排版：`home-entry-route-links` 最小列宽 132px → 118px，861–940px 窗口宽度下不再出现 5+1 孤行；≤860px 移动端保持 2×3 网格。
- [x] 首页「更多路线」按钮按方向起点里程碑的历史顺序排列：传统控制（ZMP 1972）→ 安全控制（CLF 1983）→ 接触操作（阻抗控制 1985）→ 模仿学习（行为克隆 1988）→ 强化学习（Q-learning 1989）→ 感知越障（2020s）。
- [x] 首页「更多路线」扩为十按钮并保持历史序：新增导航（概率 SLAM 1986）、移动操作（移动操作臂协调 1994）、BFM（DeepMimic 2018）；原 VLA·BFM 合并按钮拆为独立 BFM 与 VLA（RT-2 2023）两个入口。
- [x] 首页「更多路线」扩为十三按钮并保持历史序：新增 WAM（World Action Models 综述形式化，2026）；默认展示最新 4 条（感知越障 / 动作生成 / VLA / WAM），BFM 并入折叠区；展开文案同步为 13 条。
- [x] 修复移动端折叠态 WAM 通栏：末行居中规则改为仅 `.is-expanded` 时生效，避免 hidden 节点被 `:nth-child` 计入导致第 13 项误判孤行。
  - [x] 详情页 Mermaid 灯箱：加载态文案 + `stage-pending` 在 `fit` 完成前隐藏，消除「空面板 / 先大后缩」闪烁。
  - [x] 首页「互链枢纽 · Top 10」底部入口改为「查看完整榜单 →」，新增 `docs/hubs.html` 全量互链榜单页（数据源 `exports/hub-rankings.json`，全站 / 论文双 tab）。
---

### Phase 4: 信息架构优化 (Information Architecture) - [x] *已完成*
- [x] 首页调整为：目标入口 → 搜索 → 最新内容 → 知识图谱 → 互链榜单。
- [x] 新增目标入口：从零开始、项目查询两张卡，强化学习 / 传统控制并入「更多路线」通栏卡。
- [x] 移除 Hero 区与目标入口、知识图谱预览重复的两个 CTA 按钮。
- [x] 详情页移除“通用详情页”“正文同步内容”“消费 JSON”等面向实现的措辞，统一为读者语言。
- [x] 去重收敛：删除独立「学习路线」区块，纵深路线（强化学习 / 传统控制 / 模仿学习 / 安全控制 / 接触操作）统一并入目标入口区的「更多路线」卡。
- [x] 「项目查询」入口卡点击后直接聚焦搜索输入框（`data-focus-search`），减少锚点跳转后的二次寻找。
- [x] 删除搜索区副标题（与输入框 placeholder 重复），Hero 副标题改为面向读者的引导语。

---

## 验收标准 (DoD)
- [x] 首页滚动条长度缩减 50% 以上。
- [x] 首页首屏 (Above the fold) 必须包含：核心定位、规模统计，并紧接按目标选择的入口。
- [x] Lighthouse 性能/可访问性评分保持在 90+。
