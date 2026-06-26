---
type: overview
tags: [agibot, data-pipeline, imitation-learning, category-hub, survey]
status: complete
updated: 2026-06-26
summary: "智元 2026-06 发布 · 01 数据入口 — AGIBOT WORLD 2026 如何用真实多模态采集支撑模仿学习？"
related:
  - ./agibot-june-2026-release-technology-map.md
  - ./agibot-release-category-02-sim-training-eval.md
  - ../entities/agibot-world-2026.md
  - ../entities/ewmbench.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_agibot_june_2026_release.md
---

# 智元发布分类 01：数据入口

> **图谱分类节点**：对应 [具身智能研究室 · 智元 2026-06 发布解读](https://mp.weixin.qq.com/s/QWj7F2vhhRrRpX41SaNyaA) 的 **01 数据入口**；总地图见 [智元 2026-06 发布技术地图](./agibot-june-2026-release-technology-map.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| IL | Imitation Learning | 从示范轨迹学习策略 |
| RGB-D | RGB-Depth | 彩色图与深度图组合传感 |
| IMU | Inertial Measurement Unit | 惯性测量单元 |
| LiDAR | Light Detection and Ranging | 激光雷达距离传感 |

## 核心问题

**机器人学习用的数据，离真实世界部署有多近？** 须覆盖杂乱场景、遮挡、动态干扰，并记录 **失败与修正** 而不仅是干净成功轨迹。

## 本组项目（1 个）

| # | 项目 | Wiki 实体 |
|---|------|-----------|
| 01 | AGIBOT WORLD 2026 | [agibot-world-2026.md](../entities/agibot-world-2026.md) |

## 与 EWMBench / Agibot-World 生态

| 维度 | 本数据集 | [EWMBench](../entities/ewmbench.md) |
|------|----------|-------------------------------------|
| 角色 | **新一轮真实 IL 数据入口** | 基于 Agibot-World 的 **世界模型生成评测** |
| 采集 | 精灵 G2 + 多模态传感 + 力控/遥操作 | 评测子集与协议 |

## 关联页面

- [智元 2026-06 发布技术地图](./agibot-june-2026-release-technology-map.md)
- [仿真训练与评测](./agibot-release-category-02-sim-training-eval.md)

## 参考来源

- [wechat_embodied_ai_lab_agibot_june_2026_release.md](../../sources/blogs/wechat_embodied_ai_lab_agibot_june_2026_release.md)

## 推荐继续阅读

- [AGIBOT WORLD 2026 官网](https://agibot-world.com)
