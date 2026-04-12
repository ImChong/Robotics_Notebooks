# Robotics_Notebooks Tech-map Filter Rollout Plan

> **For Hermes:** 在已落地的 data-driven tech-map 页面上补最小 filter / 分组导航，再把当前筛选状态同步到 URL，并把节点列表改成按 layer 分组的最小折叠视图；仍然不做复杂状态管理，优先保证页面可用性。

**Goal:** 让 `tech-map.html` 支持按 layer 过滤节点、让筛选状态在刷新和分享链接后仍然保留，并让节点区在默认全量状态下更容易扫读。

**Architecture:** 继续使用原生 HTML / JS。页面保留现有 filter 区块，`main.js` 基于 `tech_map_page.nodes` 生成 layer chips，并通过点击切换当前 layer，实时重渲染节点列表。当前 layer 同步到 `?layer=...`；如果回到 `all`，则自动移除该查询参数。节点列表改为按 layer 分组的 `details` / `summary` 结构，每组默认展开，后续可手动折叠。

**Tech Stack:** 原生 HTML / CSS / JS、`unittest`

---

## 执行步骤
1. 先写失败测试，锁定 filter 挂载点、渲染函数、URL 同步逻辑与分组折叠渲染逻辑
2. 在 `tech-map.html` 保持现有 filter 区块，不再重复造 UI
3. 在 `main.js` 增加 tech-map 过滤状态与 URL 查询参数同步逻辑
4. 在 `main.js` / `style.css` 中把节点列表切成按 layer 分组的可折叠区块
5. 用浏览器验证：默认展示全部、切换后只展示单 layer、URL 会同步变化、分组数量正确
6. 提交并推送 GitHub（中文 commit）
