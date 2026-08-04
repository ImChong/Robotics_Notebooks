---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2509.19301"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_residual-off-policy-rl-for-finetuning-behavior-c.md
summary: "行为克隆（BC）能学到不错的视觉运动策略，但受限于人类演示质量、采集人力、离线数据的边际收益递减。强化学习（RL）靠自主交互、潜力大，但直接在真机训 RL 难——样本低效、安全、稀疏奖励长时程，对高自由度（DoF）系统尤甚。本文给出一个把 BC 与 RL 优点结合的残差学习配方：把 BC 策略当黑盒基座，用样本高效的离策略 RL 学每步的轻量残差修正。方法只需稀疏二值奖励，即可在高自由度系统（仿真与真机）上改进操作策略。尤其，作者据其所知首次在带灵巧手的人形真机上成功进行 RL 训练，在多项视觉任务上取得 SOTA，指向把 RL 真正部署到现实的可行路径。"
---

# Residual Off-Policy RL for Finetuning Behavior Cloning Policies

**Residual Off-Policy RL for Finetuning Behavior Cloning Policies** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

行为克隆（BC）能学到不错的视觉运动策略，但受限于人类演示质量、采集人力、离线数据的边际收益递减。强化学习（RL）靠自主交互、潜力大，但直接在真机训 RL 难——样本低效、安全、稀疏奖励长时程，对高自由度（DoF）系统尤甚。本文给出一个把 BC 与 RL 优点结合的残差学习配方：把 BC 策略当黑盒基座，用样本高效的离策略 RL 学每步的轻量残差修正。方法只需稀疏二值奖励，即可在高自由度系统（仿真与真机）上改进操作策略。尤其，作者据其所知首次在带灵巧手的人形真机上成功进行 RL 训练，在多项视觉任务上取得 SOTA，指向把 RL 真正部署到现实的可行路径。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| BC | Behavior Cloning，行为克隆 |
| Off-Policy RL | 离策略强化学习（样本高效） |
| Residual | 残差，对基座策略的逐步修正 |
| Sparse Binary Reward | 稀疏二值奖励（成功/失败） |
| High-DoF | 高自由度系统 |
| Dexterous Hands | 灵巧手 |

## 为什么重要

- **"冻结基座 + 学残差"是真机 RL 的安全样本高效范式**，呼应 ResMimic、SteadyTray 的残差思路；
- **稀疏二值奖励**降低奖励工程门槛，利于真机；
- **首次灵巧手人形真机 RL**是里程碑，证明真机 RL 可行；
- 对高 DoF 系统（人形）尤其有价值。

## 解决什么问题

BC 与 RL 各有短板： - **BC**：受演示质量限制、边际收益递减； - **真机 RL**：样本低效、安全难、稀疏奖励长时程难，高 DoF 更甚。

论文要：把二者结合，**安全样本高效**地在**真机高 DoF**（含灵巧手人形）上改进策略。

## 核心机制

1. **残差 BC+RL 配方**：BC 基座 + 离策略 RL 轻量残差，安全样本高效；
2. **仅需稀疏二值奖励**：免稠密奖励工程；
3. **首次灵巧手人形真机 RL**：据作者所知；
4. **视觉任务 SOTA**：指向真机 RL 的可行路径。

方法拆解（深读笔记小节）：残差学习：BC 基座 + RL 修正；样本高效离策略 RL + 稀疏二值奖励；真机灵巧手人形 RL（首次）；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Residual_Off-Policy_RL_for_Finetuning_Behavior_Cloning_Policies/Residual_Off-Policy_RL_for_Finetuning_Behavior_Cloning_Policies.html> |
| arXiv | <https://arxiv.org/abs/2509.19301> |
| 作者 | Lars Ankile、Zhenyu Jiang、Rocky Duan、Guanya Shi、Pieter Abbeel、Anusha Nagabandi |
| 发表 | 2025 年 9 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 结论

**这篇工作的赌注是「不要重训基座」：把 BC 策略冻结成黑盒，只用离策略 RL 学每步的轻量残差，从而在高自由度真机上同时压下 RL 的样本代价与安全风险。**

- 起作用的是 **残差结构 + 离策略 RL** 的组合，缺一不可：前者让探索始终停留在 BC 行为附近（安全），后者让昂贵的真机交互数据可被反复复用（样本高效）。
- **只需稀疏二值奖励** 是关键工程红利——真机任务最难写的恰恰是稠密奖励，这一步被绕开了。
- 适用边界写在前提里：必须先有一个能用的 BC 基座。BC 受演示质量与采集人力限制、离线数据边际收益递减，残差修正的是「最后一段」，不是从零学会任务。
- 里程碑意义在于据作者所知 **首次在带灵巧手的人形真机上跑通 RL**，多项视觉任务取得 SOTA；但本页仅为索引级摘要，量化指标以深读笔记与论文 PDF 为准。
- 同簇对照：与 ResMimic、SteadyTray 共享「冻结基座 + 学残差」思路，差别在于本文把它推到了真机高 DoF 视觉操作这一最难的场景。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_residual-off-policy-rl-for-finetuning-behavior-c.md](../../sources/papers/humanoid_pnb_residual-off-policy-rl-for-finetuning-behavior-c.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Residual_Off-Policy_RL_for_Finetuning_Behavior_Cloning_Policies/Residual_Off-Policy_RL_for_Finetuning_Behavior_Cloning_Policies.html>
- 论文：<https://arxiv.org/abs/2509.19301>

## 推荐继续阅读

- [机器人论文阅读笔记：Residual Off-Policy RL for Finetuning Behavior Cloning Policies](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Residual_Off-Policy_RL_for_Finetuning_Behavior_Cloning_Policies/Residual_Off-Policy_RL_for_Finetuning_Behavior_Cloning_Policies.html)
