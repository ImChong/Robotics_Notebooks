# 项目变更日志

本文档是 `Robotics_Notebooks` 的维护变更记录。

旧的重构日志见 `log.md`。

本文档只记录：
- 新增重要页面
- 结构性改动（模板、入口、路线重写）
- 版本升级（v1 → v2）
- 项目阶段变化

**不记录**：单次提交的具体内容（那是 git 的事）。

---

## 2026-04-11 — 页面级聚合导出推进

### 完成内容

**导出层从对象级推进到页面级**
- 扩展 `scripts/export_minimal.py`，让它在生成 `exports/index-v1.json` 之外，同时生成 `exports/site-data-v1.json`
- 新增 `site-data-v1.json` 的 5 类页面聚合结果：`home_page`、`module_pages`、`roadmap_pages`、`tech_map_page`、`detail_pages`
- 当前首页聚合已包含：hero、quick entries、featured chain、featured modules
- 当前模块页聚合已覆盖：`control`、`rl`、`il`、`sim2real`、`locomotion`、`tooling`
- 当前路线页、tech-map 页和 detail pages 已可直接被前端按页面类型消费

### 项目阶段变化

- 导出层从“对象 schema + 对象池导出”推进到“页面消费层聚合导出”
- 下一步可以直接进入最小网页渲染验证，而不是回到继续补散页

---

## 2026-04-12 — 页面级导出最小网页验证

### 完成内容

**开始直接消费 `site-data-v1.json` 做前端验证**
- 新增 `docs/site-data-preview.html`，作为页面级聚合导出的最小验证页
- 扩展 `docs/main.js`，直接读取 `exports/site-data-v1.json` 并渲染：导出摘要、首页数据、模块页数据、路线页数据
- 扩展 `docs/style.css`，补充预览页所需的 KPI 卡片、chip 列表和数据面板样式
- 更新 `docs/index.html` 首页 CTA，加入“页面级导出预览”入口

### 项目阶段变化

- 项目从“已经具备页面级导出”推进到“已经开始用页面级导出做最小网页消费验证”
- 下一步重点可以转向：补 detail page / tech-map page 的直接渲染，而不是回到手写静态首页内容

---

## 2026-04-12 — 补齐 detail page / tech-map 预览验证

### 完成内容

**把页面级导出验证从首页 / 模块 / 路线继续推进到详情页与技术栈图层**
- 扩展 `docs/site-data-preview.html`，新增 detail page 和 tech-map 两个预览分区
- 扩展 `docs/main.js`，从 `detail_pages` 中选取代表页面做通用详情页渲染验证
- 扩展 `docs/main.js`，直接消费 `tech_map_page`，渲染 graph meta、layer 分布与节点卡片
- 扩展 `docs/style.css`，补充 detail / tech-map 预览所需样式

### 项目阶段变化

- 现在已经不只是验证“页面级导出能不能驱动首页”，而是开始验证“能不能支撑通用详情页和技术栈节点页”
- 下一步可以继续把预览页中的 detail / tech-map 渲染，推进为真正独立的数据驱动页面

---

## 2026-04-11 — V2 阶段第二次推进

### 完成内容

**入口与路线升级**
- 重写 `README.md`，从项目介绍升级为真正入口指南（适合谁 / 怎么用 / 从哪开始 / 项目结构 / 和其他项目边界）
- 重写 `index.md`，从目录列表升级为导航总入口（快速入口表 / 四模块分工 / 推荐阅读顺序 / 主线知识链）

**Roadmap 执行性强化**
- 重写 `roadmap/route-a-motion-control.md`，从空章节升级为完整执行路线（L0 数学基础 → L6 综合实战，每阶段有前置知识 / 核心问题 / 推荐做什么 / 推荐读什么 / 学完输出）
- 重写 `roadmap/learning-paths/if-goal-locomotion-rl.md`，从空列表升级为 6 Stage 完整路径
- 重写 `roadmap/learning-paths/if-goal-imitation-learning.md`，从空列表升级为 6 Stage 完整路径

**Tech-map 结构联动**
- 重写 `tech-map/overview.md`，从简单列表升级为模块总览 + 知识入口 + 当前主攻主线
- 重写 `tech-map/dependency-graph.md`，从箭头列表升级为完整依赖图（分层关系 / 横向桥接 / 推荐阅读顺序）

**Wiki 主干补全**
- 新增 `wiki/concepts/lip-zmp.md`
- 新增 `wiki/concepts/centroidal-dynamics.md`
- 新增 `wiki/concepts/tsid.md`
- 新增 `wiki/concepts/state-estimation.md`
- 新增 `wiki/concepts/system-identification.md`
- 新增 `wiki/methods/trajectory-optimization.md`

**执行清单升级**
- 建立 `docs/tech-stack-next-phase-checklist-v2.md`，从 v1 待办表升级为阶段判断 + 优先级重排 + 当前状态 + 执行看板
- README 固定入口指向 v2

### 项目阶段变化

- V1（补主干 wiki 页面）→ **V2（结构联动、入口优化、路线执行化、维护规范化）**

---

## 2026-04-11 — V1 阶段完成

### 完成内容

**Wiki 主干基本成型**
- `Sim2Real`
- `Whole-Body Control`
- `Domain Randomization`
- `Optimal Control (OCP)`
- `Model Predictive Control (MPC)`
- `Reinforcement Learning`
- `Imitation Learning`
- `Locomotion`
- `Manipulation`
- `WBC vs RL`
- `Robot Learning Overview`
- `Humanoid Control Roadmap`

**项目结构确立**
- `wiki/` / `roadmap/` / `tech-map/` / `sources/` / `references/` / `docs/` 目录体系
- 基础 schema（`page-types.md`、`linking.md`、`naming.md`）
- 执行清单 v1

### 项目阶段

V1 阶段：搭建知识骨架，建立基本目录结构和部分内容。

---

## 项目定位回顾

| 项目 | 职责 |
|------|------|
| [`Humanoid_Robot_Learning_Paper_Notebooks`](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks) | 单篇论文深读笔记 |
| [`Robotics_Notebooks`](https://github.com/ImChong/Robotics_Notebooks)（本项目）| 跨模块知识组织、成长路线、技术栈地图 |
| [`ImChong.github.io`](https://github.com/ImChong/ImChong.github.io) | 个人简历与对外展示 |
