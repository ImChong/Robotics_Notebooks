---
type: entity
tags:
  - dataset
  - 3d-scene
  - synthetic-scene
  - navigation
status: draft
updated: 2026-10-20
summary: "211 个高质量合成室内场景，含 18,656 个真实物体模型；论文用实验说明“少而精”优于“多而粗”。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../entities/habitat-sim.md
  - ../tasks/vision-language-navigation.md
  - ../entities/matterport3d-simulator.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# HSSD / HSSD-200

## 一句话定义

**HSSD**：按真实住宅布局搭建的高质量合成室内场景数据集。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| HSSD | Habitat Synthetic Scenes Dataset | 数据集名称 |
| ProcTHOR | Procedurally generated THOR | 程序化生成场景对照 |
| ObjectNav | Object Goal Navigation | 物体目标导航任务 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | Simon Fraser University、Georgia Tech、Meta AI 等 |
| 时间 | 2023-06 |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 规模 | 211 个场景；18,656 个物体模型 |
| 实验 | 122 个场景训练，在真实扫描环境零样本优于 1 万场景训练 |
| 获取 | Hugging Face hssd/hssd-hab |

## 为什么重要

- 论文表明场景视觉逼真度与真实相关性比数量更重要：122 个场景训练的智能体零样本表现超过 ProcTHOR-10K 训练的智能体。

## 核心原理

按真实住宅布局人工搭建场景并放置真实物体模型，用于物体目标导航实验。

## 工程实践

| 项 | 说明 |
|----|------|
| 代码 | 3dlg-hcvc/hssd 仓库（文章给出链接） |

## 局限与风险

- 场景数量较少，用于大规模训练时需与程序化场景混合（推测）。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [habitat-sim](../entities/habitat-sim.md)
- [vision-language-navigation](../tasks/vision-language-navigation.md)
- [matterport3d-simulator](../entities/matterport3d-simulator.md)

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://3dlg-hcvc.github.io/hssd/)
