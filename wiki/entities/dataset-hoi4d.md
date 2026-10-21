---
type: entity
tags:
  - dataset
  - hand-object
  - 4d
  - egocentric
status: draft
updated: 2026-10-20
summary: "四维第一视角手-物交互数据：240 万帧 RGB-D、4,000 段序列、800 个物体实例。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../tasks/manipulation.md
  - ../methods/diffusion-policy.md
  - ../methods/vla.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# HOI4D

## 一句话定义

**HOI4D**：在 610 个室内房间中与 16 类物体交互采集的大规模四维第一视角数据集。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| HOI | Human-Object Interaction | 人-物交互 |
| 4D | 三维空间加时间 | 动态四维数据 |
| RGB-D | RGB + Depth | 彩色与深度 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | 清华大学、北京大学、上海期智研究院 |
| 时间 | 2022-03（CVPR 2022） |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 规模 | 240 万帧 RGB-D；4,000 段序列；800 个物体实例；16 类 |
| 场景 | 610 个不同室内房间 |
| 标注 | 全景分割、运动分割、三维手部姿态、类别级位姿、手部动作 |

## 为什么重要

- 逐帧标注全景分割、运动分割、手部姿态与类别级物体位姿，适合类别级交互研究。

## 核心原理

第一视角 RGB-D 视频配合逐帧标注，设三类基准（四维点云分割、位姿跟踪、动作分割）。

## 工程实践

| 项 | 说明 |
|----|------|
| 代码 | leolyliu/HOI4D-Instructions 仓库（文章给出链接） |

## 局限与风险

- 类别数量（16）有限，开放类别泛化需额外验证（推测）。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [manipulation](../tasks/manipulation.md)
- [diffusion-policy](../methods/diffusion-policy.md)
- [vla](../methods/vla.md)

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://hoi4d.github.io/)
