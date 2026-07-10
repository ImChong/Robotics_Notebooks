---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2408.11805"
related:
  - ../overview/paper-notebook-category-07-teleoperation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_ace.md
summary: "从演示学习对机器人操作很有效，跨平台高效遥操作系统因此愈发关键。但当前缺少面向不同末端（拟人手、夹爪）且能跨平台的低成本、易用遥操作系统。ACE 是一套跨平台的视觉-外骨骼系统，做低成本灵巧遥操作：用一个面向手的相机捕捉 3D 手部姿态，配一个便携底座上的外骨骼，实时精确捕捉手指与手腕姿态。相比以往常需按机器人定制硬件的系统，ACE 的单一系统即可泛化到拟人手、臂-手、臂-夹爪、四足-夹爪等多种构型，实现高精度遥操作，从而支撑多平台上复杂操作任务的模仿学习。"
---

# ACE

**ACE: A Cross-Platform Visual-Exoskeletons System for Low-Cost Dexterous Teleoperation** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：07_Teleoperation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

从演示学习对机器人操作很有效，跨平台高效遥操作系统因此愈发关键。但当前缺少面向不同末端（拟人手、夹爪）且能跨平台的低成本、易用遥操作系统。ACE 是一套跨平台的视觉-外骨骼系统，做低成本灵巧遥操作：用一个面向手的相机捕捉 3D 手部姿态，配一个便携底座上的外骨骼，实时精确捕捉手指与手腕姿态。相比以往常需按机器人定制硬件的系统，ACE 的单一系统即可泛化到拟人手、臂-手、臂-夹爪、四足-夹爪等多种构型，实现高精度遥操作，从而支撑多平台上复杂操作任务的模仿学习。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Visual-Exoskeleton | 视觉 + 外骨骼混合捕捉 |
| Cross-Platform | 跨平台，泛化到多种末端 |
| Hand-facing Camera | 面向手的相机，捕 3D 手姿 |
| Finger/Wrist Pose | 手指 / 手腕姿态 |
| End-Effector | 末端执行器（手/夹爪） |
| Imitation Learning | 模仿学习 |

## 为什么重要

- **跨平台单一系统**大幅降低多机型数据采集成本；
- **视觉 + 外骨骼互补**是兼顾精度与成本的好折中；
- **手指 + 手腕同时捕**对灵巧操作至关重要；
- 与 Bunny-VisionPro、NuExo 等灵巧/外骨骼遥操作共同推进低成本数据采集。

## 解决什么问题

跨平台遥操作缺**通用、低成本**方案： - 不同末端（拟人手 vs 夹爪）通常**各自定制硬件**； - 缺**单一系统跨多平台**的高精度遥操作。

ACE 要：一套**低成本、跨平台**、能同时捕**手指 + 手腕**的灵巧遥操作系统。

## 核心机制

1. **跨平台视觉-外骨骼系统**：单一系统泛化到多种末端构型；
2. **视觉 + 外骨骼混合捕捉**：相机捕手指、外骨骼捕手腕，实时高精度；
3. **低成本、便携**：免按机器人定制硬件；
4. **支撑模仿学习**：高质量演示用于复杂操作任务。

方法拆解（深读笔记小节）：视觉 + 外骨骼混合捕捉；跨平台泛化（单一系统）；支撑模仿学习；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 07_Teleoperation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/ACE__A_Cross-Platform_Visual-Exoskeletons_System_for_Low-Cost_Dexterous_Teleoperation/ACE__A_Cross-Platform_Visual-Exoskeletons_System_for_Low-Cost_Dexterous_Teleoperation.html> |
| arXiv | <https://arxiv.org/abs/2408.11805> |
| 作者 | Shiqi Yang、Minghuan Liu、Yuzhe Qin、Runyu Ding、Jialong Li、Xuxin Cheng、Ruihan Yang、Sha Yi、Xiaolong Wang（UCSD 等） |
| 发表 | 2024 年 8 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-07-teleoperation](../overview/paper-notebook-category-07-teleoperation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_ace.md](../../sources/papers/humanoid_pnb_ace.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/ACE__A_Cross-Platform_Visual-Exoskeletons_System_for_Low-Cost_Dexterous_Teleoperation/ACE__A_Cross-Platform_Visual-Exoskeletons_System_for_Low-Cost_Dexterous_Teleoperation.html>
- 论文：<https://arxiv.org/abs/2408.11805>

## 推荐继续阅读

- [机器人论文阅读笔记：ACE](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/ACE__A_Cross-Platform_Visual-Exoskeletons_System_for_Low-Cost_Dexterous_Teleoperation/ACE__A_Cross-Platform_Visual-Exoskeletons_System_for_Low-Cost_Dexterous_Teleoperation.html)
