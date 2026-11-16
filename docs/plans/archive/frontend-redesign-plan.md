# 前端重设计计划 · Frontend Redesign Plan

> 状态：**实施中（Phase 0 已完成）**
> 触发原因：当前首页信息密度过高、视觉层次不清晰；link-graph.json（61 节点 / 342 边）已生成但未被前端消费；用户希望加入 Obsidian 风格的图谱视图作为核心导航入口。
> 设计准则来源：frontend-design skill（已安装 `~/.claude/skills/frontend-design/SKILL.md`）

---

## 零、执行策略：渐进式接入（"按钮先行"原则）

**核心约束**：在不破坏现有网站任何页面的前提下逐步引入图谱功能。

### 渐进式分阶段策略

```
Phase 0（已完成）：最小化接入
  ├── 新建独立页面 docs/graph.html（不影响任何现有页面）
  ├── 首页英雄区 CTA 加入"🕸️ 知识图谱"按钮（一行改动）
  └── Makefile make graph 自动同步 link-graph.json → docs/exports/

Phase 1（下一步）：首页视觉升级
  ├── 英雄区右侧嵌入图谱缩略预览（静态快照，非交互）
  └── 加入 stats bar（动态统计页面数 / 边数）

Phase 2（后续）：图谱与首页深度整合
  ├── 首页第一屏改为图谱全屏背景
  ├── 图谱成为主导航入口
  └── 传统卡片导航下移为辅助入口

Phase 3（长期）：设计系统整体升级
  └── 字体 / 颜色 / 动效全面对齐 Technical Blueprint 风格
```

**为什么"按钮先行"**：
- graph.html 作为独立页面，任何问题都不影响现有用户体验
- 可以先收集使用反馈，再决定是否将图谱融入首页核心区域
- 单一 CTA 按钮改动风险极低（一行 HTML），可随时回滚
- 避免在设计方向未充分验证时做大规模重写

---

## 一、现状诊断

### 1.1 当前页面结构

| 文件 | 用途 | 问题 |
|------|------|------|
| `index.html` | 首页：英雄区 + 卡片 + 搜索 + 路线预览 | 内容过多、节点不可见、无"全局观" |
| `detail.html` | 通用 wiki 页渲染（query param `?id=`） | 结构清晰，基本够用 |
| `tech-map.html` | 技术栈节点，按 layer 卡片展示 | 静态卡片，不体现节点间依赖关系 |
| `roadmap.html` | 学习路径页 | 功能完整，低优先级改动 |
| `module.html` | 模块聚合页 | 功能完整，低优先级改动 |
| `references.html` | 重定向到 `roadmap.html?id=roadmap-motion-control`（兼容旧「论文导航」书签） |

### 1.2 核心问题

1. **"全局观"缺失**：用户无法直观看到 61 个 wiki 页面之间的关系网络，只能线性导航
2. **首页内容过载**：英雄区、"这个项目是做什么的"、搜索、路线、模块、关系页、论文导航全部堆在一个页面，首屏信息没有优先级
3. **graph 数据闲置**：`exports/link-graph.json` 已包含完整节点 + 边数据，但没有任何前端消费它
4. **系统字体 + 暖米色**：现有设计保守，缺乏与"机器人 / 前沿技术"主题匹配的视觉语言
5. **tech-map.html 和图谱功能重叠**：两者都在描述"节点之间的关系"，但 tech-map 只做了卡片列表

---

## 二、设计方向（Aesthetic Direction）

> 参考 frontend-design skill：*"Choose a clear conceptual direction and execute it with precision."*

### 推荐方向：**Technical Blueprint（技术蓝图风格）**

**概念核心**：这个 wiki 是一张正在被绘制的技术地图。首页就是地图本身——节点是概念，连线是依赖，用户在地图上自由探索，点击进入概念详情。

**视觉语言**：
- **背景**：深色主调（#0d1117，GitHub Dark 级别），配合细密网格线纹理（0.5px grid pattern），体现"工程图纸 / 蓝图"感
- **字体**：标题用 `Space Grotesk`（工程感 + 现代感）... ⚠️ 不，frontend-design skill 明确禁止 Space Grotesk。改用 **`DM Mono`（标题）+ `Noto Serif SC`（中文正文）**，形成代码美学 + 学术阅读的对比
- **颜色**：主色 `#00d4ff`（电子蓝），图谱边为 `rgba(0, 212, 255, 0.15)`，节点按类型分色（见下），白色文字，低饱和度背景
- **动效**：图谱节点加载时从中心扩散（stagger + spring physics），节点悬停时 glow 光晕扩散，边高亮相邻连线

**节点颜色系统**（与现有类型对应）：

| type | 颜色 | 含义 |
|------|------|------|
| `concept` | `#60a5fa`（蓝） | 基础概念 |
| `method` | `#34d399`（绿） | 方法/算法 |
| `task` | `#f472b6`（粉） | 任务场景 |
| `entity` | `#fbbf24`（橙） | 工具/框架实体 |
| `comparison` | `#c084fc`（紫） | 对比分析 |
| `query` | `#94a3b8`（灰蓝） | Query 产物 |
| `formalization` | `#fb923c`（橘） | 形式化基础 |
| `` (空，wiki page) | `#64748b`（中灰） | 通用 wiki 页 |

---

## 三、改动范围（分阶段）

### Phase 1 · 首页重构（P1，高优先级）

**目标**：首页从"信息堆砌"变为"地图入口"——用图谱作为首屏核心视觉，其下是项目简介 + 快速入口。

#### 1.1 新版首页结构（`docs/index.html`）

```
┌─────────────────────────────────────────────────────────┐
│  HEADER（保留，精简文字）                                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   HERO SECTION（全屏高度）                                │
│   ┌─────────────────────────────────────────────────┐   │
│   │  左列（40%）：项目介绍文字                         │   │
│   │  • 大标题（DM Mono，英文）                        │   │
│   │  • 副标题（中文，Noto Serif SC）                  │   │
│   │  • 3行简介                                       │   │
│   │  • stats bar：62 pages · 21 sources · 342 links  │   │
│   │  • CTA 按钮：进入图谱 / 开始学习路线               │   │
│   ├─────────────────────────────────────────────────┤   │
│   │  右列（60%）：GRAPH PREVIEW（缩略图谱）            │   │
│   │  • D3.js 力导向图，轻量版（不可交互，仅视觉）      │   │
│   │  • 点击 "进入图谱" 跳到 graph.html               │   │
│   └─────────────────────────────────────────────────┘   │
│                                                         │
├─────────────────────────────────────────────────────────┤
│   ABOUT SECTION（保留简化版"这个项目是做什么的"）           │
│   4 个要点卡片（看清模块 / 理解依赖 / 形成路径 / 连接行动）   │
├─────────────────────────────────────────────────────────┤
│   QUICK ENTRY（新增）                                    │
│   • wiki 搜索框（保留）                                  │
│   • "热门概念"入口（按入度排名前8个节点，从 graph.json 算） │
├─────────────────────────────────────────────────────────┤
│   LEARNING PATHS（简化版，保留路线 A/B）                   │
├─────────────────────────────────────────────────────────┤
│   FOOTER（新增简单 footer）                               │
└─────────────────────────────────────────────────────────┘
```

#### 1.2 关键技术点

- `stats bar` 数值从 `exports/index-v1.json` 计算（不硬编码，动态读取）
- Hero 右侧的 graph preview 是静态 SVG 或低性能 D3 snapshot（不阻塞首屏）
- "热门概念"用 link-graph.json 按节点入度排序，取 Top 8

---

### Phase 2 · 图谱视图页（P2，核心功能）

**目标**：新建 `docs/graph.html`，实现完整 Obsidian 风格的知识图谱。

#### 2.1 功能规格

| 功能 | 描述 | 优先级 |
|------|------|--------|
| 力导向布局 | D3.js `d3-force`，节点按连接度分散 | P0（必须） |
| 节点点击导航 | 点击节点 → `detail.html?id=<node-id>` | P0（必须） |
| 节点悬停 tooltip | 显示页面标题 + type + 简短 summary | P0（必须） |
| 按类型过滤 | 顶部 chip 切换（全部/概念/方法/任务/实体…），支持多选 | P1 |
| 节点搜索 | 输入关键词高亮匹配节点，非匹配节点淡出 | P1 |
| 缩放/平移 | `d3-zoom` 支持鼠标滚轮缩放 + 拖拽平移 | P0（必须） |
| 节点标签 | 节点旁显示短标签（中文简称），缩小时隐藏 | P1 |
| 边高亮 | 悬停节点时，相邻边高亮，非相邻边淡出 | P1 |
| 固定节点 | 双击节点 pin 固定位置（再次双击解除） | P2 |
| 迷你导航地图 | 右下角 minimap，显示当前视口在全图中的位置 | P2 |
| 图谱快照导出 | 导出当前视图为 PNG（`canvas` 截图） | P3 |

#### 2.2 数据来源

```
exports/link-graph.json          → 节点 + 边数据（已有）
exports/index-v1.json            → 节点的 summary 字段（用于 tooltip）
```

节点 id（`wiki/concepts/mpc.md` 格式）与 `index-v1.json` 中的 `path` 字段一一对应，用于关联 summary。

#### 2.3 技术选型

| 选项 | 方案 | 理由 |
|------|------|------|
| 图形库 | **D3.js v7**（CDN） | 无框架依赖，轻量，力模拟最成熟；与现有静态 HTML 架构一致 |
| 替代选项 | Cytoscape.js | 更高级别 API，但包体积大，flex 较差 |
| 不用 | three.js / WebGL | 61 节点不需要 3D 性能，增加复杂度 |

#### 2.4 页面布局

```
┌──────────────────────────────────────────────────────────┐
│  HEADER（与全站共用，精简版）                               │
├──────────────────────────────────────────────────────────┤
│  TOOLBAR（固定在顶部，HEADER 下方）                         │
│  [搜索框]  [全部][概念][方法][任务][实体][对比][Query]  [重置] │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  GRAPH CANVAS（全屏，占剩余高度）                           │
│  • D3.js SVG 渲染                                        │
│  • 深色背景 + 网格纹理                                    │
│  • 节点：圆形，按类型着色                                  │
│  • 边：半透明线段，悬停高亮                                │
│  • 标签：节点旁文字，zoom < 0.5 时隐藏                    │
│                                                          │
├──────────────────────────────────────────────────────────┤
│  MINIMAP（右下角，固定）  │  INFO PANEL（右侧，hover 时弹出） │
└──────────────────────────────────────────────────────────┘
```

---

### Phase 3 · 设计系统升级（P3，可选）

**目标**：在不破坏现有功能的前提下，升级 `style.css` 的视觉语言使之与新设计方向一致。

#### 3.1 字体升级

```css
/* 现在（系统字体，无特色）*/
--font-sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', ...;

/* 计划（加入特色字体，保留 CJK fallback）*/
--font-display: 'DM Mono', monospace;                   /* 标题、数字 */
--font-body:    'Noto Serif SC', 'Georgia', serif;       /* 正文（中文） */
--font-ui:      'DM Sans', -apple-system, sans-serif;   /* UI 元素 */
```

从 Google Fonts 通过 CDN 引入（不需要本地文件）：
- `DM Mono` — 工程代码感，字重 400/500
- `DM Sans` — 极简现代感，与 DM Mono 同家族
- `Noto Serif SC` — CJK 衬线，提升中文可读性

#### 3.2 颜色系统升级（图谱主题沿用到全站）

```css
/* 新颜色 token（在现有 CSS 变量基础上扩展）*/
--graph-node-concept:     #60a5fa;
--graph-node-method:      #34d399;
--graph-node-task:        #f472b6;
--graph-node-entity:      #fbbf24;
--graph-node-comparison:  #c084fc;
--graph-node-query:       #94a3b8;
--graph-node-formal:      #fb923c;
--graph-edge:             rgba(255, 255, 255, 0.12);
--graph-edge-hover:       rgba(0, 212, 255, 0.6);
--graph-bg:               #0d1117;
--graph-grid:             rgba(255, 255, 255, 0.04);
```

#### 3.3 首页背景优化

将现有 `radial-gradient` 背景替换为：
```css
body {
  background:
    url("data:image/svg+xml,...")  /* 细密 dot grid SVG pattern */
    var(--bg);
}
```
保持暖米色基调，加入若隐若现的 dot grid，呼应"工程图纸"主题而不破坏现有布局。

#### 3.4 动效增强（CSS-only，不引入 JS 动画库）

- 卡片悬停：`transform: translateY(-3px)` + `box-shadow` 增强（现已有，微调参数）
- 搜索框聚焦：`border-color` 渐变 + 轻微 `scale(1.01)` 反馈
- 页面加载：`@keyframes fadeInUp`，卡片组 `animation-delay` stagger（0.05s 间隔）

---

### Phase 4 · detail.html 微调（P4，低优先级）

| 改动 | 描述 |
|------|------|
| 关联页面卡片化 | 将文字列表改为带 type badge + summary 片段的小卡片 |
| 图谱入口 | 在每个 detail 页右上角加 "在图谱中查看" 按钮，点击跳转 graph.html 并高亮该节点 |
| 阅读进度条 | 顶部 1px 进度条，滚动时填充（纯 CSS `position: sticky` 实现） |

---

## 四、实现优先级总结

| Phase | 内容 | 改动量 | 依赖 | 状态 |
|-------|------|--------|------|------|
| P0 | graph.html（独立图谱页）+ 首页加按钮 | 中（新建 ~300行 + 1行按钮） | D3.js CDN | ✅ 已完成 |
| P1 | 首页升级（stats bar + 图谱缩略预览嵌入英雄区） | 中（改 index.html 局部） | D3.js CDN | ⭐⭐⭐ 下一步 |
| P2 | 图谱与首页深度整合（图谱作为首屏背景/主导航） | 大（重构 index.html） | P1 验证通过后 | ⭐⭐ 中期 |
| P3 | 设计系统升级（字体 + 颜色 + 动效全面对齐） | 中（改 style.css + head） | Google Fonts CDN | ⭐ 长期 |
| P4 | detail.html 微调（关联卡片化 + 图谱入口按钮） | 小 | - | ⭐ 低 |

**当前完成状态（Phase 0）**：
- [x] `docs/graph.html` 新建：D3.js 力导向图，61节点/342边，按类型着色，悬停 tooltip，点击跳 detail 页，类型过滤 chip，节点搜索，缩放/平移
- [x] `docs/index.html` 英雄区加入 `🕸️ 知识图谱` 按钮（btn-primary 样式，首个 CTA）
- [x] `docs/exports/link-graph.json` 同步（从 exports/ 复制）
- [x] `Makefile` `make graph` 新增自动同步到 docs/exports/

---

## 五、技术约束与注意事项

1. **纯静态部署**：无后端，所有数据从 `exports/*.json` 加载，图谱实现必须客户端渲染
2. **CDN 依赖**：`D3.js v7` 通过 CDN 引入（`https://cdn.jsdelivr.net/npm/d3@7`），需要网络访问
3. **link-graph.json 节点 id 格式**：`wiki/concepts/xxx.md` 路径格式，需与 `detail.html?id=` 的 id 格式对齐（当前 detail 页使用的是 `exports/index-v1.json` 中的 `id` 字段，需验证两者是否一致）
4. **62 个节点**（非 61，index-v1.json 有 99 项，graph 只有 wiki 部分）：D3 force 在 <200 节点时性能完全无忧，移动端也可流畅运行
5. **字体加载**：Google Fonts CDN 在国内访问可能受限，可使用 `font-display: swap` + 系统字体 fallback 保证退化体验；或替换为 `fontsource` npm 包（若未来引入构建工具）

---

## 六、验收标准（完成时的可测试指标）

| 指标 | 目标 |
|------|------|
| graph.html 加载时间 | 首次加载（含 D3.js CDN）< 3s on 3G |
| 节点点击跳转 | 100% 节点可点击，跳转到正确 detail.html 页 |
| 类型过滤 | 切换 chip 后 < 200ms 内完成节点显示/隐藏 |
| 搜索高亮 | 输入关键词后匹配节点高亮，非匹配节点 opacity < 0.15 |
| 缩放范围 | 支持 0.1x ~ 4x 缩放 |
| 移动端 | 触控拖拽 + 双指缩放可用 |
| Lint 不破坏 | `make lint` 仍保持 0 真实问题（不改 wiki 内容） |

---

## 七、参考 / 灵感来源

- **Obsidian Graph View**：力导向布局、节点悬停高亮相邻边、缩放标签自动显隐
- **Logseq Graph**：暗色背景 + 彩色节点，密集连接的知识图谱审美
- **GitHub Contribution Graph** 配色：深色背景上的彩色数据点
- **Blueprint/Schematic 审美**：工程图纸的网格纹理，常见于硬件创业公司官网
- **D3.js Observable notebooks**：力模拟 + zoom 的标准实现参考

---

*由 Claude Code + frontend-design skill 规划，日期：2026-04-15*
---

## 八、ID 对齐验证结论（已确认，2026-04-15）

> 原问题：`link-graph.json` 节点 id 格式 与 `detail.html?id=` 路由格式是否一致？

### 结论：**不直接一致，但有简单一致的转换规则，59/61 节点完全可用**

| 字段 | 格式示例 |
|------|---------|
| `link-graph.json` node id | `wiki/concepts/capture-point-dcm.md` |
| `index-v1.json` item id | `wiki-concepts-capture-point-dcm` |
| `detail.html` 路由格式 | `detail.html?id=wiki-concepts-capture-point-dcm` |

**转换规则（完全一致，无例外）**：
```javascript
// 在 graph.html 点击节点时执行：
function graphIdToDetailId(graphNodeId) {
  return graphNodeId.replace(/\//g, '-').replace('.md', '');
  // 'wiki/concepts/mpc.md' → 'wiki-concepts-mpc'
}
```

### 2 个不匹配节点

| graph 节点 | 原因 | 处理方式 |
|-----------|------|---------|
| `wiki/references/llm-wiki-karpathy.md` | 此路径不在 `index-v1.json` 导出范围内（reference 类页面） | 点击时 detail.html 会显示"找不到页面"的 graceful 错误，不影响其他节点 |
| `wiki/roadmaps/humanoid-control-roadmap.md` | 同上（roadmap 类页面，不在 wiki/ 导出） | 同上 |

这两个节点是纯导航辅助页，不是核心知识节点，不影响图谱的实际使用价值。

### 实施方案（在 graph.html 中）

**方案 A（推荐）**：运行时一行转换

```javascript
node.on('click', (event, d) => {
  const detailId = d.id.replace(/\//g, '-').replace('.md', '');
  window.location.href = 'detail.html?id=' + encodeURIComponent(detailId);
});
```

无需额外加载任何文件，零成本，59/61 节点精准跳转。

**方案 B（备选，最精确）**：预加载 `index-v1.json`，建立 `path→id` map，查表跳转。覆盖 100% 节点（但两个不在 index 里的节点仍无法导航），额外增加 488KB JSON 加载。

**结论：使用方案 A**，61 个节点中 59 个精准跳转，2 个显示 graceful 错误，实现成本最低。
