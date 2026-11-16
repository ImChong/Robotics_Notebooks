# Robotics_Notebooks Tech-map Data-driven Plan

> **For Hermes:** 在已有 detail page 的基础上，把 `tech-map.html` 从静态说明页推进成真正消费 `tech_map_page` 的数据驱动页面；优先验证结构闭环，不追求复杂可视化。

**Goal:** 让 `docs/tech-map.html` 直接消费 `docs/exports/site-data-v1.json` 的 `tech_map_page`，展示 graph meta、layer 分布、节点卡片，并把节点统一接到 `detail.html?id=...`。

**Architecture:** 保持原生 HTML / CSS / JS。`tech-map.html` 提供数据挂载点，`docs/main.js` 增加 `renderTechMapPage()`，直接读取 `tech_map_page.nodes`。第一阶段用「分层统计 + 节点卡片」替代复杂图可视化。

**Tech Stack:** 原生 HTML / CSS / JS、Python 导出脚本（已可用）、`unittest`

---

## 1. 目标拆解

### 本次必须完成
1. `tech-map.html` 变成数据驱动页面，而不是纯静态文案页
2. 页面至少展示：
   - overview / dependency graph meta
   - layer 分布
   - 节点卡片列表
3. 节点要能跳到 `detail.html?id=...`
4. 页面要能处理空数据 / 读取失败

### 本次暂不做
- 力导图 / 关系图动画
- 节点拖拽
- 搜索框
- 多层筛选器

---

## 2. 具体执行步骤

### Task 1: 写失败测试
文件：`tests/test_tech_map_page.py`

验证：
- `docs/tech-map.html` 含有新的挂载点：
  - `techMapHeroSummary`
  - `techMapGraphMeta`
  - `techMapLayerList`
  - `techMapNodeGrid`
- `docs/main.js` 含有：
  - `function renderTechMapPage`
  - `document.getElementById('techMapNodeGrid')`
  - `detail.html?id=`

### Task 2: 改造 `docs/tech-map.html`
目标：
- 保留页面标题和导航
- 主体内容改成数据面板 + layer chips + node cards
- 加 loading skeleton

### Task 3: 在 `docs/main.js` 增加 tech-map 专用渲染器
目标：
- 从 `site-data-v1.json.pages.tech_map_page` 取数据
- 渲染：
  - hero summary
  - graph meta
  - layer 分布
  - 节点卡片
- 节点标题/按钮跳 `detail.html?id=...`

### Task 4: 复用现有样式，少量补 CSS
目标：
- 优先复用 `.card` / `.chip-list` / `.data-card`
- 只补 tech-map 页面需要的少量布局样式

### Task 5: 验证
命令：
- `python -m unittest tests/test_tech_map_page.py`
- 浏览器打开 `http://127.0.0.1:8766/tech-map.html`

验收：
- 页面能显示 layer 分布
- 节点卡片能展示 title / summary / layer / node_kind
- 节点可跳 `detail.html?id=...`

---

## 3. 设计约束

### 为什么先做“列表化 tech-map”
因为当前目标不是做最终产品，而是验证：
- `tech_map_page` 结构够不够用
- `id / summary / layer / node_kind / related` 是否足以支撑真实页面

### 为什么要接到 detail page
因为现在项目已经有了统一 detail route：
- tech-map 不该再成为孤立页面
- 节点卡片跳 detail page，才能形成真正的网站结构闭环

---

## 4. 后续最自然的下一步

如果这一步完成，下一步优先级建议：
1. 给 tech-map 增加 layer filter
2. 再决定要不要做 markdown 正文同步
3. 然后更新 `docs/tech-stack-next-phase-checklist-v2.md`，让项目阶段描述跟上现实
