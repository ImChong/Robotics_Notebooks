---
type: entity
tags:
  - dataset
  - articulated-object
  - simulation-asset
status: draft
updated: 2026-10-20
summary: "2,346 个可交互铰接三维模型、覆盖 46 类，标注部件运动类型、关节轴与运动范围。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../tasks/manipulation.md
  - ../methods/diffusion-policy.md
  - ../methods/vla.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# PartNet-Mobility（SAPIEN）

## 一句话定义

**PartNet-Mobility**：随 SAPIEN 模拟器发布的可动关节物体模型集合，可直接用于物理仿真。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SAPIEN | Simulation of Articulated Parts in Environments | 模拟器名称（文章未展开，缩写为推测） |
| DoF | Degrees of Freedom | 关节自由度 |
| CVPR | Conference on Computer Vision and Pattern Recognition | 会议 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | UC San Diego 等 |
| 时间 | 2020-03（CVPR 2020） |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 规模 | 2,346 个模型；46 个类别 |
| 标注 | 部件运动类型、关节轴、运动范围 |
| 获取 | Hugging Face sapien-sim/PartNetMobility |

## 为什么重要

- 铰接物体操作研究的常用资产库，被 ManiSkill 等仿真基准广泛采用。

## 核心原理

部件级标注配合物理仿真，可用于部件检测与运动属性识别。

## 工程实践

| 项 | 说明 |
|----|------|
| 代码 | haosulab/SAPIEN 仓库 |

## 局限与风险

- 物体类别偏家居，且为仿真资产，真实物理参数需另行核对（推测）。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [manipulation](../tasks/manipulation.md)
- [diffusion-policy](../methods/diffusion-policy.md)
- [vla](../methods/vla.md)

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://huggingface.co/datasets/sapien-sim/PartNetMobility)
