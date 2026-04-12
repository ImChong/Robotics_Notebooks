# Robotics_Notebooks Tech-map Filter Rollout Plan

> **For Hermes:** 在已落地的 data-driven tech-map 页面上补最小 filter / 分组导航，再把当前筛选状态同步到 URL；仍然不做复杂状态管理，优先保证页面可用性。

**Goal:** 让 `tech-map.html` 支持按 layer 过滤节点，并让筛选状态在刷新和分享链接后仍然保留，降低节点数量上来后的浏览成本。

**Architecture:** 继续使用原生 HTML / JS。页面增加过滤器挂载点，`main.js` 基于 `tech_map_page.nodes` 生成 layer chips，并通过点击切换当前 layer，实时重渲染节点列表。当前 layer 同步到 `?layer=...`；如果回到 `all`，则自动移除该查询参数。

**Tech Stack:** 原生 HTML / CSS / JS、`unittest`

---

## 执行步骤
1. 先写失败测试，锁定 filter 挂载点、渲染函数与 URL 同步逻辑
2. 在 `tech-map.html` 保持现有 filter 区块，不再重复造 UI
3. 在 `main.js` 增加 tech-map 过滤状态与 URL 查询参数同步逻辑
4. 用浏览器验证：默认展示全部、切换后只展示单 layer、URL 会同步变化、回到 all 会清参数
5. 提交并推送 GitHub（中文 commit）
