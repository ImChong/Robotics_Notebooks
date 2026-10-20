---
type: entity
tags:
  - dataset
  - hand-object
  - articulated
  - bimanual
status: draft
updated: 2026-10-20
summary: "双手灵巧操作铰接物体的视频数据：210 万帧，提供精确三维手部与物体网格真值。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../tasks/manipulation.md
  - ../methods/diffusion-policy.md
  - ../methods/vla.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# ARCTIC

## 一句话定义

**ARCTIC**：记录双手开合剪刀、笔记本等铰接物体的三维手-物交互数据集。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MANO | Hand Model with Articulated and Non-rigid Deformations | 手部模型 |
| SMPL-X | SMPL eXpressive | 全身参数化人体模型 |
| CVPR | Conference on Computer Vision and Pattern Recognition | 会议 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | ETH Zurich、Max Planck Institute for Intelligent Systems |
| 时间 | 2022-04（CVPR 2023） |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 规模 | 210 万帧视频 |
| 标注 | 三维手部与物体网格；动态接触信息 |
| 模型 | MANO 与 SMPL-X 标注 |

## 为什么重要

- 同时提供物理一致的手与铰接物体运动及动态接触信息，填补此前数据空白。

## 核心原理

同步视频与三维网格，提出单目视频重建与稠密手-物交互场估计两项任务。

## 工程实践

| 项 | 说明 |
|----|------|
| 获取 | arctic.is.tue.mpg.de 下载页（文章给出链接） |

## 局限与风险

- 物体类别以剪刀、笔记本等为主（文章举例），泛化范围需看原论文（推测）。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [manipulation](../tasks/manipulation.md)
- [diffusion-policy](../methods/diffusion-policy.md)
- [vla](../methods/vla.md)

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://arctic.is.tue.mpg.de/)
