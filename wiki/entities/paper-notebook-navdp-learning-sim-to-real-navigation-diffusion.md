---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2505.08712"
related:
  - ../overview/paper-notebook-category-08-navigation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_navdp.md
summary: "在动态复杂开放世界中导航是自主机器人的关键且困难的能力。已有方法多依赖级联模块化框架（需大量调参）或有限真实演示学习。NavDP（Navigation Diffusion Policy）是一个端到端网络，仅在仿真训练就能实现零样本 sim-to-real，跨多样环境与机器人本体迁移。它用统一的 Transformer 架构同时做轨迹生成与评估：以局部 RGB-D 观测为条件，为对比轨迹样本预测评论值（critic values），并借助特权仿真信息提升空间理解。训练数据大规模——跨 3000 个场景、累计超百万米导航。结果：在仿真与真机评测中均显著超越此前 SOTA。"
---

# NavDP

**NavDP: Learning Sim-to-Real Navigation Diffusion Policy with Privileged Information Guidance** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：08_Navigation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

在动态复杂开放世界中导航是自主机器人的关键且困难的能力。已有方法多依赖级联模块化框架（需大量调参）或有限真实演示学习。NavDP（Navigation Diffusion Policy）是一个端到端网络，仅在仿真训练就能实现零样本 sim-to-real，跨多样环境与机器人本体迁移。它用统一的 Transformer 架构同时做轨迹生成与评估：以局部 RGB-D 观测为条件，为对比轨迹样本预测评论值（critic values），并借助特权仿真信息提升空间理解。训练数据大规模——跨 3000 个场景、累计超百万米导航。结果：在仿真与真机评测中均显著超越此前 SOTA。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| NavDP | Navigation Diffusion Policy |
| Diffusion Policy | 扩散策略，用扩散生成动作/轨迹 |
| Privileged Info | 特权信息，训练时可用的真值/全局信息 |
| Critic Value | 评论值，对候选轨迹打分 |
| RGB-D | 彩色 + 深度观测 |
| Cross-Embodiment | 跨本体，迁移到不同机器人 |

## 为什么重要

- **"生成 + 评估"一体的 Transformer**是导航策略的优雅设计，避免级联调参；
- **特权信息引导**是 sim-to-real 的常用强力手段（与 VIRAL、Opening-Door 同思路）；
- **跨本体**意味着人形可直接复用，导航能力与本体解耦；
- 大规模仿真数据是零样本泛化的底座。

## 解决什么问题

开放世界导航难点： - 级联**模块化**框架需**大量调参**； - 从**有限真实演示**学习数据少； - 想**纯仿真训练、零样本上真机**且**跨本体**。

NavDP 要：一个**端到端、可大规模仿真训练、零样本迁移**的导航策略。

## 核心机制

1. **端到端导航扩散策略**：纯仿真训练、零样本 sim-to-real、跨本体；
2. **统一 Transformer 生成 + 评估**：扩散生轨迹 + 评论值打分；
3. **特权信息引导**：训练期提升空间理解，部署期无需特权；
4. **大规模数据 + SOTA**：3000 场景/百万米，仿真与真机均领先。

方法拆解（深读笔记小节）：统一 Transformer：生成 + 评估；局部 RGB-D 条件 + 特权信息引导；大规模仿真数据；结果；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 08_Navigation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/NavDP__Learning_Sim-to-Real_Navigation_Diffusion_Policy/NavDP__Learning_Sim-to-Real_Navigation_Diffusion_Policy.html> |
| arXiv | <https://arxiv.org/abs/2505.08712> |
| 作者 | Wenzhe Cai、Jiaqi Peng、Yuqiang Yang、Yujian Zhang、Meng Wei、Hanqing Wang、Yilun Chen、Tai Wang、Jiangmiao Pang（上海 AI Lab 等） |
| 发表 | 2025 年 5 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-08-navigation](../overview/paper-notebook-category-08-navigation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_navdp.md](../../sources/papers/humanoid_pnb_navdp.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/NavDP__Learning_Sim-to-Real_Navigation_Diffusion_Policy/NavDP__Learning_Sim-to-Real_Navigation_Diffusion_Policy.html>
- 论文：<https://arxiv.org/abs/2505.08712>

## 推荐继续阅读

- [机器人论文阅读笔记：NavDP](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/NavDP__Learning_Sim-to-Real_Navigation_Diffusion_Policy/NavDP__Learning_Sim-to-Real_Navigation_Diffusion_Policy.html)
