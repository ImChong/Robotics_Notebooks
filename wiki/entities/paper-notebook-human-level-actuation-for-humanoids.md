---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2511.06796"
related:
  - ../overview/paper-notebook-category-12-hardware-design.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_human-level-actuation-for-humanoids.md
summary: "厂商常说自家人形「达到人级驱动」，但没有统一、可量化的标准——峰值扭矩这类单点指标既能被姿态/温度「刷分」，也掩盖了扭矩、速度、带宽、效率、热持续之间的耦合取舍。本文提出三件套：① DoF Atlas（用 ISB 生物力学规范统一人/机关节坐标与功能活动范围）、② 人等效包络 HEE（要求驱动器在人真正做正功的「同一个姿态+转速」点上同时满足扭矩与功率）、③ 人级驱动评分 HLAS（把工作空间覆盖、HEE 覆盖、扭矩带宽、效率、热持续等六项加权成一个标量，人在自己任务上得 1.0），从而把「人级」这件事做成一把可复现、防刷分的尺子。"
---

# Human-Level Actuation for Humanoids

**Human-Level Actuation for Humanoids** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：12_Hardware_Design），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

厂商常说自家人形「达到人级驱动」，但没有统一、可量化的标准——峰值扭矩这类单点指标既能被姿态/温度「刷分」，也掩盖了扭矩、速度、带宽、效率、热持续之间的耦合取舍。本文提出三件套：① DoF Atlas（用 ISB 生物力学规范统一人/机关节坐标与功能活动范围）、② 人等效包络 HEE（要求驱动器在人真正做正功的「同一个姿态+转速」点上同时满足扭矩与功率）、③ 人级驱动评分 HLAS（把工作空间覆盖、HEE 覆盖、扭矩带宽、效率、热持续等六项加权成一个标量，人在自己任务上得 1.0），从而把「人级」这件事做成一把可复现、防刷分的尺子。

## 英文缩写速查

| 缩写 | 全称 | 解释 |
|---|---|---|
| DoF | Degree of Freedom | 自由度 |
| ISB | International Society of Biomechanics | 国际生物力学学会（关节坐标系规范） |
| ROM | Range of Motion | 关节活动范围（分主动/被动/功能三类） |
| HEE | Human-Equivalence Envelope | 人等效包络（同姿态+转速下同时达标的扭矩-功率区） |
| HLAS | Human-Level Actuation Score | 人级驱动评分（六项加权标量） |
| SEA | Series Elastic Actuator | 串联弹性驱动器 |
| QDD | Quasi-Direct Drive | 准直驱 |

## 为什么重要

- **驱动器设计规格**：提供以人类生物力学为锚、可量化的目标，而非模糊的「人级」口号
- **横向评测**：HLAS 让不同尺度/结构的人形关节在同一把尺子上比较
- **反宣传水分**：把「峰值扭矩」这类易刷分指标换成连续安全性能图，倒逼厂商公开更真实的数据
- **硬件-任务对齐**：以真实任务（行走/举重/够取）的做功包络为准绳，避免为跑分而过度设计

## 解决什么问题

「人级驱动（human-level actuation）」是人形圈的高频宣传词，但缺少可核验的定义。现有做法的两个通病：

- **单点峰值指标误导**：一个标称「峰值 100 N·m」的关节，可能只在某个特定姿态、瞬时、冷态下达到；热浸泡后掉到 30 N·m，或在需要的转速点上根本给不出功率——传统规格把这些缺陷全藏起来了。 - **无法跨平台公平比较**：不同机器人尺度、关节定义、测试条件各异，「谁更接近人」没有共同标尺。

## 核心机制

1. **DoF Atlas**：用 ISB 规范统一人/机关节定义与功能活动范围，给出全身 110 DoF 的可比坐标账目；
2. **人等效包络 HEE**：把评测限制在「人真正做正功」的姿态×转速点上，并要求扭矩与功率**同时**达标——从机制上封堵单点刷分；
3. **HLAS 评分**：将 ROM/DoF/HEE/带宽/效率/热六项加权为一个可跨平台比较的标量，人在自己任务上得 1.0；
4. **完整测量协议**：给出关节级 + 任务级的可复现测试流程，主张用「连续安全性能图」取代「孤立峰值规格」。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 12_Hardware_Design |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/12_Hardware_Design/Human-Level_Actuation_for_Humanoids/Human-Level_Actuation_for_Humanoids.html> |
| arXiv | <https://arxiv.org/abs/2511.06796> |
| 发表 | 2025-11-10 (arXiv) |
| 源码 | 论文未公开代码或项目页（纯方法/基准框架） |
| 笔记阅读日期 | 2026-07-07 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-12-hardware-design](../overview/paper-notebook-category-12-hardware-design.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_human-level-actuation-for-humanoids.md](../../sources/papers/humanoid_pnb_human-level-actuation-for-humanoids.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/12_Hardware_Design/Human-Level_Actuation_for_Humanoids/Human-Level_Actuation_for_Humanoids.html>
- 论文：<https://arxiv.org/abs/2511.06796>

## 推荐继续阅读

- [机器人论文阅读笔记：Human-Level Actuation for Humanoids](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/12_Hardware_Design/Human-Level_Actuation_for_Humanoids/Human-Level_Actuation_for_Humanoids.html)
