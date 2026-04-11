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
