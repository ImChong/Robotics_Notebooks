# Robotics_Notebooks Tech-map Filter Rollout Plan

> **For Hermes:** 在已落地的 data-driven tech-map 页面上补最小 filter / 分组导航，不做复杂状态管理，优先保证页面可用性。

**Goal:** 让 `tech-map.html` 支持按 layer 过滤节点，降低节点数量上来后的浏览成本。

**Architecture:** 继续使用原生 HTML / JS。页面增加过滤器挂载点，`main.js` 基于 `tech_map_page.nodes` 生成 layer chips，并通过点击切换当前 layer，实时重渲染节点列表。

**Tech Stack:** 原生 HTML / CSS / JS、`unittest`

---

## 执行步骤
1. 先写失败测试，锁定 filter 挂载点和渲染函数
2. 在 `tech-map.html` 增加 filter 区块
3. 在 `main.js` 增加 tech-map 过滤状态与节点重渲染逻辑
4. 用浏览器验证默认展示全部、切换后只展示单 layer
5. 提交并推送 GitHub（中文 commit）
