# Robotics_Notebooks Roadmap Page Rollout Plan

> **For Hermes:** 不再维持手写静态 `roadmap.html`，直接把它升级成统一的 data-driven roadmap route；优先补齐页面结构，而不是继续扩展示例文案。

**Goal:** 把 `roadmap_pages` 从 preview 升级成真实页面，让 `roadmap.html?id=...` 可以直接渲染任一路线页。

**Architecture:** 复用现有 `docs/roadmap.html` 作为统一路线页入口，直接消费 `docs/exports/site-data-v1.json` 的 `roadmap_pages`。先展示标题、摘要、阶段列表、相关项、来源链接，不额外引入路由器。

**Tech Stack:** 原生 HTML / CSS / JS、现有导出结构、`unittest`

---

## 执行步骤

### Task 1: 写失败测试
文件：`tests/test_roadmap_page.py`

验证：
- `docs/roadmap.html` 含有挂载点：
  - `roadmapTitle`
  - `roadmapSummary`
  - `roadmapMeta`
  - `roadmapStageList`
  - `roadmapRelatedList`
  - `roadmapSourceList`
- `docs/main.js` 包含：
  - `function renderRoadmapPage`
  - `document.getElementById('roadmapStageList')`
  - `roadmap.html?id=`

### Task 2: 重写 `docs/roadmap.html`
目标：
- 支持 URL：`roadmap.html?id=roadmap-route-a-motion-control`
- 支持空态
- 和 detail / module / roadmap 之间统一跳转

### Task 3: 在 `docs/main.js` 中新增路线页渲染器
目标：
- 读取 `pages.roadmap_pages`
- 渲染阶段列表
- 相关项跳 `detail.html?id=...`
- 来源链接跳外部地址

### Task 4: 更新站内入口
目标：
- 首页 CTA“开始看路线”默认指向 `roadmap.html?id=roadmap-route-a-motion-control`
- 相关页面里尽量使用统一 roadmap route

### Task 5: 验证与推送
命令：
- `python -m unittest tests/test_roadmap_page.py`
- 浏览器打开 `http://127.0.0.1:8766/roadmap.html?id=roadmap-route-a-motion-control`
- 完成后提交并推送 GitHub，提交信息用中文
