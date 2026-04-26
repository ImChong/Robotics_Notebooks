# 前端体验优化清单 v1 (Frontend Optimization Checklist)

状态：**进行中** (已完成结构重构)
目标：**极简化首页 (Minimalist Homepage)** —— 消除视觉噪音，突出搜索、图谱与核心入口。

---

## 一、 视觉降噪 (Visual De-cluttering)

- [x] **重构英雄区 (Hero Section)**：
    - [x] 压缩文字：将目前的 4 段文字（Kicker, Title, Subtitle, Desc）精简为 1 行核心标语 + 1 行副标题。
    - [x] 移除右侧侧边栏 (Hero Panel)：将“你现在更适合从哪进？”的逻辑融入核心 CTA 或导航卡片。
    - [x] 整合 Stats Bar：将 3 个统计卡片改为一行半透明的微型数据条。
- [x] **合并冗余区块**：
    - [x] 将“两条成长路线”、“当前技术栈主干”、“关系页”、“论文导航”这四个大区块（约 16 个卡片）合并为一个 **“探索入口 (The Gateway)”** 网格，仅保留 4-6 个最具代表性的入口。
    - [x] 移除“这个项目是做什么的”区块：其内容已在 Hero 区和 README 中体现，首页不再重复平铺。

## 二、 交互增强 (Interaction First)

- [x] **搜索框置顶 (Search-Centric Design)**：
    - [x] 将 `wiki-search` 区块提升至英雄区下方或直接整合进英雄区。
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

---

## 验收标准 (DoD)
- [x] 首页滚动条长度缩减 50% 以上。
- [x] 首页首屏 (Above the fold) 必须包含：搜索框、图谱预览、Top 3 统计、2 个核心 CTA。
- [x] Lighthouse 性能/可访问性评分保持在 90+。
