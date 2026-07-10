---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2409.04639"
related:
  - ../overview/paper-notebook-category-07-teleoperation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_high-speed-and-impact-resilient-teleoperation-of.md
summary: "人形遥操作长期是难题，需要软硬件协同进步才能实现无缝直观的控制。本文提出一个集成方案，由几大要素构成：① 免标定（calibration-free）的动作捕捉与重定向——仅用 7 个 IMU 即可生成全身机器人参考；② 低延迟的快速全身运动学流式工具箱，降低端到端延迟；③ 高带宽摆线执行器（cycloidal actuators），使机器人能高速且抗冲击。在人形机器人 Nadia 上验证，通过感知、控制与驱动的协同进步，展示了遥操作的前所未有的性能。"
---

# High-Speed and Impact Resilient Teleoperation of Humanoid Robots

**High-Speed and Impact Resilient Teleoperation of Humanoid Robots** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：07_Teleoperation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

人形遥操作长期是难题，需要软硬件协同进步才能实现无缝直观的控制。本文提出一个集成方案，由几大要素构成：① 免标定（calibration-free）的动作捕捉与重定向——仅用 7 个 IMU 即可生成全身机器人参考；② 低延迟的快速全身运动学流式工具箱，降低端到端延迟；③ 高带宽摆线执行器（cycloidal actuators），使机器人能高速且抗冲击。在人形机器人 Nadia 上验证，通过感知、控制与驱动的协同进步，展示了遥操作的前所未有的性能。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Calibration-free | 免标定，无需繁琐标定流程 |
| IMU | 惯性测量单元（本文仅用 7 个） |
| Kinematics Streaming | 运动学流式传输，低延迟下发参考 |
| Cycloidal Actuator | 摆线执行器，高带宽、抗冲击 |
| Impact Resilient | 抗冲击，承受碰撞不损坏 |
| Retargeting | 动作重定向 |

## 为什么重要

- **遥操作性能受限于"最慢的一环"**：动捕、延迟、执行器需同步优化；
- **硬件（摆线执行器）是高速/抗冲击的物理前提**，提醒算法之外的本体重要性；
- **免标定 + 少传感器**降低使用门槛，利于现场部署；
- 高速抗冲击能力为采集"动态/接触丰富"演示提供条件。

## 解决什么问题

人形遥操作要做到**高速、抗冲击、低延迟**，需软硬件同步突破： - **动捕标定**繁琐、传感器多； - **延迟**高导致控制不跟手； - **执行器**带宽不足，难高速/抗冲击。

论文要：一套**感知 + 控制 + 驱动**协同的集成方案。

## 核心机制

1. **软硬件一体的高速抗冲击遥操作**：感知 + 控制 + 驱动协同；
2. **免标定动捕（7 IMU）**：简化穿戴/标定即得全身参考；
3. **低延迟运动学流式**：提升跟手性；
4. **高带宽摆线执行器**：使高速与抗冲击成为可能（Nadia 验证）。

方法拆解（深读笔记小节）：免标定动捕 + 重定向（仅 7 IMU）；低延迟全身运动学流式工具箱；高带宽摆线执行器（高速 + 抗冲击）；验证；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 07_Teleoperation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/High-Speed_and_Impact_Resilient_Teleoperation_of_Humanoid_Robots/High-Speed_and_Impact_Resilient_Teleoperation_of_Humanoid_Robots.html> |
| arXiv | <https://arxiv.org/abs/2409.04639> |
| 作者 | Sylvain Bertrand、Luigi Penco、Dexton Anderson、Duncan Calvert、Jerry Pratt、Robert Griffin 等（IHMC / Boardwalk Robotics） |
| 发表 | 2024 年 9 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-07-teleoperation](../overview/paper-notebook-category-07-teleoperation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_high-speed-and-impact-resilient-teleoperation-of.md](../../sources/papers/humanoid_pnb_high-speed-and-impact-resilient-teleoperation-of.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/High-Speed_and_Impact_Resilient_Teleoperation_of_Humanoid_Robots/High-Speed_and_Impact_Resilient_Teleoperation_of_Humanoid_Robots.html>
- 论文：<https://arxiv.org/abs/2409.04639>

## 推荐继续阅读

- [机器人论文阅读笔记：High-Speed and Impact Resilient Teleoperation of Humanoid Robots](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/High-Speed_and_Impact_Resilient_Teleoperation_of_Humanoid_Robots/High-Speed_and_Impact_Resilient_Teleoperation_of_Humanoid_Robots.html)
