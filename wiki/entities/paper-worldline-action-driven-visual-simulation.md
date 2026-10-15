---
type: entity
project_id: worldline-visual-simulator
tags: [paper, world-models, manipulation, video-generation, simulation, hkust, pku, cuhk]
topic: [world-models, manipulation]
status: complete
updated: 2026-10-15
arxiv: "2609.38059"
project: https://zhengsh123.github.io/WorldLine/
code: https://github.com/Zhengsh123/WorldLine
summary: "WorldLine 将跨本体操纵动态预训练与图像空间动作接地解耦，生成动作条件视频用于策略评估和候选筛选；演示与研究资料公开，训练/推理代码尚未发布。"
related:
  - ../methods/generative-world-models.md
  - ../concepts/functional-taxonomy-world-models.md
  - ../concepts/world-action-models.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/worldline_arxiv_2609_38059.md
  - ../../sources/sites/worldline-project.md
  - ../../sources/repos/worldline.md
---

# WorldLine：动作驱动的机器人操纵视觉仿真

**WorldLine**（*Action-Driven Visual Simulation for Robotic Manipulation*，arXiv:2609.38059）是一种动作条件视频模拟器：先从大规模无动作标注机器人视频学习可迁移的机器人–物体动态，再将不同机器人的控制映射到共享图像空间，用预测 rollout 评估策略与筛选候选动作。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 预测动作或状态变化后的环境未来 |
| DiT | Diffusion Transformer | WorldLine 所用视频生成骨干的一类架构 |
| Rot6D | 6D Rotation Representation | 用六维表示旋转姿态，便于投影到动作图 |
| IoU | Intersection over Union | 衡量预测与真值机器人分割掩码重合度 |
| OOD | Out-of-Distribution | 训练中未见的环境、任务或本体评估设置 |

## 为什么重要

动作条件视频模型的瓶颈有两面：大规模被动视频不告诉模型某条机器人控制会导致什么结果；动作标注轨迹又少且受本体、坐标系和控制接口限制。WorldLine 将两类数据拆开使用，使无动作视频扩展动态先验，再通过少量得多的控制轨迹学习动作接地。

它的定位是可学习的**视觉模拟器 / rollout 评估器**，接收动作并预测其后果；它不是直接输出机器人动作的策略。因此可与 [World Action Models（WAM）](../concepts/world-action-models.md) 对照：WAM 讨论世界预测与动作生成的耦合，而 WorldLine 可作为策略外部的候选结果模拟与排序�