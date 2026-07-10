---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2511.10021"
related:
  - ../overview/paper-notebook-category-12-hardware-design.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_decart-leg-novel-humanoid-robot-leg-with-decoupl.md
summary: "人形腿长期卡在「好看 vs 敏捷」的二选一里：要拟人外观（前向膝、串联结构）就牺牲速度，要敏捷（Cassie/Digit 的解耦结构）就长得像鸟腿。DecARt Leg 用 准伸缩（pantograph + 齿轮膝）机构把「腿长」和「腿俯仰」解耦、把所有电机都放到膝盖以上以压低摆动惯量、再用多连杆把 2 自由度踝的力矩从近端远程传过来，做到一条前向膝、拟人外观的腿也能拿到 Cassie 级的摆腿速度。"
---

# DecARt Leg

**DecARt Leg: Design and Evaluation of a Novel Humanoid Robot Leg with Decoupled Actuation for Agile Locomotion** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：12_Hardware_Design），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

人形腿长期卡在「好看 vs 敏捷」的二选一里：要拟人外观（前向膝、串联结构）就牺牲速度，要敏捷（Cassie/Digit 的解耦结构）就长得像鸟腿。DecARt Leg 用 准伸缩（pantograph + 齿轮膝）机构把「腿长」和「腿俯仰」解耦、把所有电机都放到膝盖以上以压低摆动惯量、再用多连杆把 2 自由度踝的力矩从近端远程传过来，做到一条前向膝、拟人外观的腿也能拿到 Cassie 级的摆腿速度。

## 英文缩写速查

| 缩写 | 全称 | 解释 |
|---|---|---|
| DoF | Degree of Freedom | 自由度 |
| FAST | Fastest Achievable Swing Time | 最快可达摆动时间（本文提出的敏捷度指标） |
| TSID | Task Space Inverse Dynamics | 任务空间逆动力学（求 FAST 时所用的全动力学控制器） |
| WBC | Whole-Body Control | 全身控制 |
| QP | Quadratic Programming | 二次规划（QP-WBC 求解器） |
| IK | Inverse Kinematics | 逆运动学 |
| RL | Reinforcement Learning | 强化学习 |

## 为什么重要

- **腿部机构设计**：给出「拟人外观 + 解耦敏捷」可兼得的一个具体机构样本，打破二选一惯性
- **敏捷度评测**：FAST 提供一个脱离整机尺度/上半身重量的腿级横向对比标尺，且开源可复现
- **控制范式无关**：IK / QP-WBC / RL 都能驱动，说明机构收益不绑定特定控制栈
- **硬件-控制协同**：把「低摆动惯量」做进机构，等于在硬件层先替控制器减负

## 解决什么问题

人形机器人的腿设计存在一条长期的张力：

- **拟人 / 串联（serial）结构**：膝盖朝前、外观像人，但电机沿腿串联布置 → 大腿小腿都挂着电机、**摆动惯量大**，限制了摆腿速度和敏捷度； - **解耦 / 并联结构（Cassie、Digit）**：把电机集中到髋部、用连杆/弹簧远程传力，**摆动惯量小、跑得快**，但外观像鸟腿、反屈膝，不拟人。

## 核心机制

1. **新型解耦腿机构**：用「准伸缩（pantograph + 齿轮膝）+ 无源齿轮/4 杆平行 + 多连杆踝传动」，把腿长、腿俯仰、2-DoF 踝**全部用膝上电机解耦驱动**，在**保留前向膝拟人外观**的同时压低摆动惯量；
2. **FAST 敏捷度指标**：提出一个**只评测腿本身、脱离上半身重量、可跨机器人比较**的标准化敏捷度量，并**开源**了计算代码；
3. **多控制器验证**：在仿真与真机上分别用 **IK / QP-WBC / RL** 三套控制方法跑通免缆行走，证明该腿结构对不同控制范式都适配；
4. **量化收益**：FAST 0.17 s 优于其串联消融版（0.25 s）、Cassie（0.24 s）与 GR1T2（0.24 s），并通过仿真与真机实验交叉验证。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 12_Hardware_Design |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/12_Hardware_Design/DecARt_Leg_Novel_Humanoid_Robot_Leg_with_Decoupled_Actuation/DecARt_Leg_Novel_Humanoid_Robot_Leg_with_Decoupled_Actuation.html> |
| arXiv | <https://arxiv.org/abs/2511.10021> |
| 发表 | 2025-11-13 (arXiv) |
| 笔记阅读日期 | 2026-06-26 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-12-hardware-design](../overview/paper-notebook-category-12-hardware-design.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_decart-leg-novel-humanoid-robot-leg-with-decoupl.md](../../sources/papers/humanoid_pnb_decart-leg-novel-humanoid-robot-leg-with-decoupl.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/12_Hardware_Design/DecARt_Leg_Novel_Humanoid_Robot_Leg_with_Decoupled_Actuation/DecARt_Leg_Novel_Humanoid_Robot_Leg_with_Decoupled_Actuation.html>
- 论文：<https://arxiv.org/abs/2511.10021>

## 推荐继续阅读

- [机器人论文阅读笔记：DecARt Leg](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/12_Hardware_Design/DecARt_Leg_Novel_Humanoid_Robot_Leg_with_Decoupled_Actuation/DecARt_Leg_Novel_Humanoid_Robot_Leg_with_Decoupled_Actuation.html)
