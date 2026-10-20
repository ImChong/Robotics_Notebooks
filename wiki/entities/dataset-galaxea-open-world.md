---
type: entity
tags:
  - dataset
  - real-robot
  - mobile-manipulation
status: draft
updated: 2026-10-20
summary: "单一 Galaxea R1 Lite 双臂移动机器人在 11 处真实场地采集的 500 小时开放世界数据。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../tasks/loco-manipulation.md
  - ../tasks/teleoperation.md
  - ../methods/vla.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# Galaxea Open-World Dataset

## 一句话定义

**Galaxea Open-World Dataset**：同一本体（R1 Lite）在真实住宅、餐饮、零售、办公等场景采集的双臂移动操作数据集。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| G0 | 文中模型名（双系统） | G0-VLM 规划 + G0-VLA 执行 |
| VLA | Vision-Language-Action | 视觉-语言-动作模型 |
| VLM | Vision-Language Model | 视觉语言模型 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | 星海图（Galaxea） |
| 时间 | 2025-08 |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 规模 | 约 500 小时，150 余项任务，50 个场景 |
| 模态 | 真机遥操作轨迹（视觉观测 + 双臂/移动底盘动作）+ 子任务级语言标注；传感器配置以官方页为准 |
| 本体 | Galaxea R1 Lite 双臂移动机器人 |
| 标注 | 精确到子任务的语言标注 |

## 为什么重要

- 单本体、长时长、子任务级语言标注，适合研究单本体预训练的作用。

## 核心原理

训练分为跨本体预训练、单本体预训练与任务后训练三阶段；论文认为单本体预训练对整体性能影响大。

## 工程实践

| 项 | 说明 |
|----|------|
| 获取 | Hugging Face 数据页（文章给出链接） |
| 代码 | GalaxeaVLA 仓库 |

## 局限与风险

- 数据集中仅一种本体，跨本体泛化需依赖其他数据集（文章论述）。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [loco-manipulation](../tasks/loco-manipulation.md)
- [teleoperation](../tasks/teleoperation.md)
- [vla](../methods/vla.md)

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset)
