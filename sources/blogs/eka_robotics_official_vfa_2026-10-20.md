# Eka Robotics 官方文章：Vision-Force-Action 与实验室操作吞吐

> 原始资料归档

- **标题：** Eka Robotics Raises $150M to Build Superhuman Intelligence for the Physical World
- **类型：** blog（公司官方文章）
- **来源：** Eka Robotics
- **链接：** https://ekarobotics.com/blog/eka-robotics-raises-150m-to-build-superhuman-intelligence-for-the-physical-world
- **发布日期：** 2026-10-20
- **入库日期：** 2026-10-21
- **一句话说明：** Eka 介绍其 Vision-Force-Action（VFA）路线：将视觉与接触力信息结合，通过仿真强化学习训练操作策略，并报告实验室中每小时约 1,500 次操作。

## 官方介绍摘要

Eka 将 VFA 描述为结合视觉、力与动作的操作模型。视觉用于感知物体及其位置，接触产生的力信息帮助机器人判断抓取、滑移、插入阻力等交互状态，并据此调整动作。公司主张力是机器人理解物理交互的重要信号。

Eka 表示，其策略主要通过物理仿真中的大规模强化学习训练，仿真可覆盖不同物体、环境与物理条件，也能产生失败经验并学习恢复；之后将策略迁移至真实机器人。文章称可在数分钟内生成数千小时的仿真机器人经验。训练规模和迁移效果为公司自述，文章未公开实现细节、训练配置或独立复现材料。

## 性能数据口径

Eka 报告机器人在训练分布外的多种物体上达到 **每小时 1,500 次操作**，并以约 **每小时 600 次**作为人工基准；公司还列举拧入灯泡、插入导线等操作。该数据来自公司文章，未附标准化测试协议或第三方评测，不能直接视作生产线长期节拍或跨任务基准。

## 开源核查

截至 2026-10-21，本文未提供 VFA 论文、代码仓库、模型权重或可下载数据集链接。此结论仅针对本文及 Eka 官网公开链接的核查结果。

## 对 wiki 的映射

- 现有实体页：[Eka Robotics — Vision-Force-Action（VFA）](../../wiki/entities/eka-robotics-vfa.md)
- 可与 [VLA 方法页](../../wiki/methods/vla.md)、[Sim2Real 概念页](../../wiki/concepts/sim2real.md) 对照阅读。
