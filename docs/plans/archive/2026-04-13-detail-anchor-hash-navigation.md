# Robotics_Notebooks Detail Page Anchor Hash Navigation Plan

> **For Hermes:** 在 detail page 已有 TOC、标题锚点复制和正文内链路由之后，继续补齐“带 `#heading` 的深链真正落到正文目标位置”这一段闭环；仍然保持原生 HTML / JS，不引入额外前端框架。

**Goal:** 让 `detail.html?id=...#some-heading` 在正文异步渲染完成后仍能稳定滚动到目标标题，并给目标标题一个短暂高亮反馈。

**Architecture:** 继续使用当前 detail page 的浏览器端 markdown 渲染链路。在 `renderDetailPage()` 完成 `contentEl.innerHTML`、KaTeX 渲染、标题增强后，读取 `window.location.hash`，在正文容器内查找对应锚点标题并调用 `scrollIntoView()`。同时监听 `hashchange`，保证 TOC 点击、复制锚点后的再次打开、正文内跨页带 hash 内链都可复用同一逻辑。样式层增加目标标题的短暂高亮态。

**Tech Stack:** 原生 HTML / CSS / JS、`unittest`

---

## 执行步骤

### Task 1: 先写失败测试
文件：`tests/test_content_sync.py`

验证：
- `docs/main.js` 中存在 hash 深链恢复函数
- detail 渲染链路里会在正文渲染后调用该函数
- 存在 `hashchange` 监听
- `docs/style.css` 中存在目标标题高亮样式

### Task 2: 在 detail page 中补 hash 深链恢复
目标：
- 如果 URL 带 `#heading`，正文渲染后自动滚动到该标题
- 若目标存在，给对应标题打一个短暂高亮态
- 若目标不存在，不报错、不白屏

### Task 3: 保持与当前路由体系兼容
目标：
- 不破坏当前 TOC active、高亮逻辑
- 不破坏正文内站内 markdown 内链到 `detail.html?id=...#...` 的跳转
- 不引入新的构建依赖

### Task 4: 验证
命令：
- `python -m unittest tests/test_content_sync.py`
- `python -m unittest`
- `python scripts/export_minimal.py`
- 本地打开 `detail.html?id=wiki-concepts-centroidal-dynamics#key-equations`（若存在对应标题）或任意长文带锚点链接验证

验收：
- 带锚点的 detail URL 在正文渲染后能落到正确标题
- 目标标题有短暂视觉反馈
- 无 hash / 错误 hash 不影响页面正常加载
