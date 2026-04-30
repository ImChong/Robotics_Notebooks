---
type: concept
title: Data Flywheel (具身数据飞轮)
tags: [data-cycle, robot-learning, scaling, automation]
summary: "数据飞轮通过“采集-清洗-训练-部署”的自动化闭环，利用 Scaling Law 实现机器人策略性能与场景覆盖的持续自我强化。"
updated: 2026-05-01
---

# Data Flywheel (具身数据飞轮)

**具身数据飞轮 (Data Flywheel)** 指的是机器人学习中通过**自动化闭环**实现数据规模化与性能持续提升的机制。它的核心逻辑是：更强的模型吸引更多场景使用 → 产生更多样化的数据 → 自动化的数据清洗与标注 → 进一步强化模型。

## 为什么重要？

具身智能的最终落地依赖于 [[embodied-scaling-laws]]。数据飞轮是实现规模效应的核心手段：
- **突破“人力”瓶颈**：传统的遥操作（Teleoperation）数据采集昂贵且低效，飞轮效应通过仿真（[[robotwin]]）或自监督学习减少对人的依赖。
- **长尾场景覆盖**：通过策略在真机或仿真中失败的案例，自动触发针对性的数据补全（[[generative-data-augmentation]]），从而攻克边缘情况（Edge Cases）。

## 核心闭环

1. **采集 (Collection)**：利用 [[lerobot]] 等框架在仿真或实物中生成初始轨迹。
2. **清洗与标注 (Cleaning & Labeling)**：利用 [[auto-labeling-pipelines]] 自动剔除低质数据并添加语义标签。
3. **训练 (Training)**：在海量异构数据上进行大规模预训练。
4. **验证与反馈 (Eval & Feedback)**：模型在实测中发现弱点，反馈给采集端进行针对性补全。

## 与其他系统的关系

- **实战路径**：[[xbotics-embodied-guide]] 将数据飞轮视为从 0 到 1 落地具身智能项目的核心目标。
- **基础设施**：飞轮的转动需要强大的仿真底座（如 [[isaac-gym-isaac-lab]]、[[genesis-sim]]）和自动化标注工具支撑。

## 参考来源
- [Xbotics-Embodied-Guide](../../sources/repos/xbotics-embodied-guide.md)
- [Embodied Scaling Laws](../concepts/embodied-scaling-laws.md)
