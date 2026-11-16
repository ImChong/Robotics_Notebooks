# Robotics_Notebooks Module Page Rollout Plan

> **For Hermes:** 不再继续维护三张手写静态模块页，先建立统一的 `module.html?id=...` 数据驱动入口，让 module page 正式进入真实页面体系。

**Goal:** 把当前只存在于 preview 里的 `module_pages` 升级成真实页面，并统一接入 `detail.html?id=...`。

**Architecture:** 新增一个通用 `docs/module.html`，直接消费 `docs/exports/site-data-v1.json` 的 `module_pages`。优先展示模块标题、摘要、tag、entry items、references、roadmaps、related modules，不依赖额外构建器。

**Tech Stack:** 原生 HTML / CSS / JS、现有导出结构、`unittest`

---

## 执行步骤

### Task 1: 写失败测试
文件：`tests/test_module_page.py`

验证：
- `docs/module.html` 存在
- 页面包含挂载点：
  - `moduleTitle`
  - `moduleSummary`
  - `moduleMeta`
  - `moduleEntryList`
  - `moduleReferenceList`
  - `moduleRoadmapList`
  - `moduleRelatedModules`
- `docs/main.js` 包含 `renderModulePage()` 和 `module.html?id=`

### Task 2: 新增通用 module page
目标：
- 支持 URL：`module.html?id=control`
- 支持空态
- 支持和 detail / roadmap / module 之间跳转

### Task 3: 在 main.js 中新增 module renderer
目标：
- 读取 `pages.module_pages`
- 用 `detail.html?id=` 渲染 entry items / references
- 用 `module.html?id=` 渲染 related modules
- 用 `detail.html?id=` 渲染 roadmap items（先走统一详情路由，后续再决定 roadmap 是否独立）

### Task 4: 更新站内入口
目标：
- 首页中现有模块入口改为指向 `module.html?id=...`
- tech-map / preview 中涉及模块入口的地方尽量统一

### Task 5: 验证
命令：
- `python -m unittest tests/test_module_page.py`
- 浏览器打开 `http://127.0.0.1:8766/module.html?id=control`

验收：
- 页面能展示模块摘要、入口项、references、roadmaps、related modules
- 内部链接可跳转
