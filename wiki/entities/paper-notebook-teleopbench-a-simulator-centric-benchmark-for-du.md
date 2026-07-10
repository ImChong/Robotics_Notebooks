---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2505.12748"
related:
  - ../overview/paper-notebook-category-07-teleoperation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_teleopbench.md
summary: "遥操作是具身机器人学习的基石，双臂灵巧遥操作尤其能提供自主系统难得的丰富演示。TeleOpBench 是一个统一的、以仿真为中心的基准：包含 30 个高保真任务环境，涵盖取放、工具使用、协作操作。它实现了四种遥操作模态——动捕、VR 设备、外骨骼、单目视觉跟踪——并提供统一协议与指标；在一个物理双臂平台上跨验证。通过 10 个留出（held-out）真实任务做仿真↔硬件交叉验证，发现仿真表现与真机行为强相关，确认了该基准作为可靠评测平台的外部效度。"
---

# TeleOpBench

**TeleOpBench: A Simulator-Centric Benchmark for Dual-Arm Dexterous Teleoperation** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：07_Teleoperation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

遥操作是具身机器人学习的基石，双臂灵巧遥操作尤其能提供自主系统难得的丰富演示。TeleOpBench 是一个统一的、以仿真为中心的基准：包含 30 个高保真任务环境，涵盖取放、工具使用、协作操作。它实现了四种遥操作模态——动捕、VR 设备、外骨骼、单目视觉跟踪——并提供统一协议与指标；在一个物理双臂平台上跨验证。通过 10 个留出（held-out）真实任务做仿真↔硬件交叉验证，发现仿真表现与真机行为强相关，确认了该基准作为可靠评测平台的外部效度。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Simulator-Centric | 以仿真为中心的基准 |
| Modality | 遥操作模态（动捕/VR/外骨骼/视觉） |
| Held-out Task | 留出任务，用于验证泛化 |
| External Validity | 外部效度，仿真结论对真机的可推广性 |
| Dual-Arm Dexterous | 双臂灵巧（操作） |
| Protocol/Metric | 统一协议与评测指标 |

## 为什么重要

- **遥操作需要统一基准**：否则不同模态/系统难公平比较，TeleOpBench 填补此空白；
- **"仿真↔真机强相关"是关键卖点**：让低成本仿真评测可信，加速迭代；
- **多模态统一**便于研究"哪种接口更适合哪类任务"；
- 对人形双臂灵巧数据采集与策略评测有直接价值，呼应本仓 11 仿真基准板块。

## 解决什么问题

遥操作研究缺**统一、可复现**的评测： - 不同**模态**（动捕/VR/外骨骼/视觉）难公平比较； - 真机评测**贵、难复现**； - 不清楚**仿真结论能否推广到真机**。

TeleOpBench 要：一个**仿真为中心**、**多模态**、**统一指标**且**经真机验证外部效度**的遥操作基准。

## 核心机制

1. **统一仿真为中心的遥操作基准**：30 任务 + 统一协议/指标；
2. **四模态实现**：动捕/VR/外骨骼/单目视觉，公平横比；
3. **外部效度验证**：10 真实留出任务交叉验证，仿真↔真机强相关；
4. **可靠评测平台**：为遥操作研究提供可复现基准。

方法拆解（深读笔记小节）：30 个高保真仿真任务；四种遥操作模态 + 统一协议；仿真↔真机交叉验证（外部效度）；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 07_Teleoperation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/TeleOpBench__A_Simulator-Centric_Benchmark_for_Dual-Arm_Dexterous_Teleoperation/TeleOpBench__A_Simulator-Centric_Benchmark_for_Dual-Arm_Dexterous_Teleoperation.html> |
| arXiv | <https://arxiv.org/abs/2505.12748> |
| 作者 | Hangyu Li、Qin Zhao、Haoran Xu、Xinyu Jiang、Qingwei Ben、Feiyu Jia 等（上海 AI Lab 等） |
| 发表 | 2025 年 5 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-07-teleoperation](../overview/paper-notebook-category-07-teleoperation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_teleopbench.md](../../sources/papers/humanoid_pnb_teleopbench.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/TeleOpBench__A_Simulator-Centric_Benchmark_for_Dual-Arm_Dexterous_Teleoperation/TeleOpBench__A_Simulator-Centric_Benchmark_for_Dual-Arm_Dexterous_Teleoperation.html>
- 论文：<https://arxiv.org/abs/2505.12748>

## 推荐继续阅读

- [机器人论文阅读笔记：TeleOpBench](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/TeleOpBench__A_Simulator-Centric_Benchmark_for_Dual-Arm_Dexterous_Teleoperation/TeleOpBench__A_Simulator-Centric_Benchmark_for_Dual-Arm_Dexterous_Teleoperation.html)
