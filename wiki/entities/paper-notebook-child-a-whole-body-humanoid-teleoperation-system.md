---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2508.00162"
related:
  - ../overview/paper-notebook-category-07-teleoperation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_child.md
summary: "遥操作已能让机器人做复杂操作，但很少支持人形的「全身关节级」遥操作，限制了任务多样性。CHILD（Controller for Humanoid Imitation and Live Demonstration）是一套紧凑、可重构的遥操作系统，实现对人形的关节级控制。它能塞进标准婴儿背带（baby carrier），让操作者同时控制四肢，并支持直接关节映射的全身控制与移动操作。系统内置自适应力反馈，提升操作体验并防止不安全关节运动。作者在一台人形与多款双臂系统上验证了移动操作与全身控制，并开源硬件设计以促进可及性与可复现性。"
---

# CHILD

**CHILD: a Whole-Body Humanoid Teleoperation System** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：07_Teleoperation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

遥操作已能让机器人做复杂操作，但很少支持人形的「全身关节级」遥操作，限制了任务多样性。CHILD（Controller for Humanoid Imitation and Live Demonstration）是一套紧凑、可重构的遥操作系统，实现对人形的关节级控制。它能塞进标准婴儿背带（baby carrier），让操作者同时控制四肢，并支持直接关节映射的全身控制与移动操作。系统内置自适应力反馈，提升操作体验并防止不安全关节运动。作者在一台人形与多款双臂系统上验证了移动操作与全身控制，并开源硬件设计以促进可及性与可复现性。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| CHILD | Controller for Humanoid Imitation and Live Demonstration |
| Joint-level | 关节级，直接控制各关节 |
| Reconfigurable | 可重构，适配不同机器人 |
| Direct Joint Mapping | 直接关节映射，人关节→机器人关节 |
| Adaptive Force Feedback | 自适应力反馈，防不安全运动 |
| Loco-manipulation | 移动操作 |

## 为什么重要

- **关节级全身接口拓展任务多样性**：比末端控制能表达更丰富的全身行为；
- **可穿戴 + 低门槛硬件**利于规模化数据采集，呼应 TWIST2、ACE 的低成本理念；
- **力反馈做安全护栏**是遥操作的实用安全设计；
- 开源硬件降低社区复现门槛。

## 解决什么问题

现有遥操作**很少支持人形全身关节级控制**： - 多为末端/上身控制，丢失全身关节自由度； - 缺乏**可穿戴、可重构**且**安全**的关节级接口。

CHILD 要：一套**便携可穿戴、关节级、带安全力反馈**的全身人形遥操作装置。

## 核心机制

1. **关节级全身人形遥操作**：填补"很少支持全身关节级"的空白；
2. **可穿戴可重构形态**：婴儿背带式，便携、同时控四肢、适配多机型；
3. **自适应力反馈**：提升体验并防止不安全关节运动；
4. **开源硬件**：促进可及性与可复现。

方法拆解（深读笔记小节）：紧凑可重构、可穿戴（婴儿背带形态）；直接关节映射（全身 + 移动操作）；自适应力反馈（安全）；验证 + 开源；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 07_Teleoperation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/CHILD__a_Whole-Body_Humanoid_Teleoperation_System/CHILD__a_Whole-Body_Humanoid_Teleoperation_System.html> |
| arXiv | <https://arxiv.org/abs/2508.00162> |
| 作者 | Noboru Myers、Obin Kwon、Sankalp Yamsani、Joohyung Kim（UIUC） |
| 发表 | 2025 年 8 月 |
| 源码 | 开源硬件设计 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-07-teleoperation](../overview/paper-notebook-category-07-teleoperation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_child.md](../../sources/papers/humanoid_pnb_child.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/CHILD__a_Whole-Body_Humanoid_Teleoperation_System/CHILD__a_Whole-Body_Humanoid_Teleoperation_System.html>
- 论文：<https://arxiv.org/abs/2508.00162>

## 推荐继续阅读

- [机器人论文阅读笔记：CHILD](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/CHILD__a_Whole-Body_Humanoid_Teleoperation_System/CHILD__a_Whole-Body_Humanoid_Teleoperation_System.html)
