---
type: entity
tags:
  - dataset
  - tactile
  - audio
  - multisensory
  - simulation
status: draft
updated: 2026-10-20
summary: "1,000 个虚拟家居物体的多感官神经表示：可渲染视觉、撞击声与触觉读数。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../concepts/tactile-sensing.md
  - ../concepts/visuo-tactile-fusion.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# ObjectFolder 2.0（仿真数据）

## 一句话定义

**ObjectFolder 2.0**：以隐式神经表示描述物体视觉、声音与触觉的大规模多感官仿真数据集。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| INR | Implicit Neural Representation | 隐式神经表示 |
| Sim2Real | Simulation to Real | 仿真到真实迁移 |
| ObjectFolder | 物体文件（数据集名） | 每个物体一个多模态文件 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | Stanford University |
| 时间 | 2022-04（CVPR 2022） |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 规模 | 1,000 个物体（1.0 版为 100 个） |
| 模态 | 视觉、声音、触觉 |
| 任务 | 尺寸估计、接触定位、形状重建（论文迁移实验） |

## 为什么重要

- 每个物体为一个“物体文件”，可由内在属性与外部条件渲染多模态信号，并论文称可迁移到真实物体。

## 核心原理

神经网络编码物体的多感官响应，渲染速度较 1.0 版显著提升。

## 工程实践

| 项 | 说明 |
|----|------|
| 获取 | objectfolder.stanford.edu 下载页（文章给出链接） |

## 局限与风险

- 仿真数据迁移到真实物体的效果依赖论文实验（文章转述）。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [tactile-sensing](../concepts/tactile-sensing.md)
- [visuo-tactile-fusion](../concepts/visuo-tactile-fusion.md)
- [manipulation](../tasks/manipulation.md)

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://objectfolder.stanford.edu/objectfolder2-0-download)
