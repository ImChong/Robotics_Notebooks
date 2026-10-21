---
type: entity
tags:
  - dataset
  - hand-object
  - pose
  - grasp
status: draft
updated: 2026-10-20
summary: "人手抓取 YCB 物体的多视角 RGB-D 数据：1,000 段序列、58.2 万帧，附手部与物体 6D 位姿。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../tasks/manipulation.md
  - ../methods/diffusion-policy.md
  - ../methods/vla.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# DexYCB

## 一句话定义

**DexYCB**：记录人手从桌面抓取 20 种 YCB 物体过程的手-物交互数据集。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| YCB | Yale-CMU-Berkeley Object Set | 标准物体集 |
| 6D | Six-Degree-of-Freedom | 六自由度位姿 |
| RGB-D | RGB + Depth | 彩色与深度 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | NVIDIA、华盛顿大学 |
| 时间 | 2021-04（CVPR 2021） |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 规模 | 10 名受试者；20 种物体；1,000 段序列；58.2 万帧 |
| 采集 | 8 台标定同步的 RGB-D 相机，30fps，640×480 |
| 许可 | CC BY-NC |

## 为什么重要

- 同时支持手部姿态、物体位姿估计，并提出面向机器人的人机交接抓取任务。

## 核心原理

多台标定相机同步记录，并用于 2D/6D/3D 三类估计任务。

## 工程实践

| 项 | 说明 |
|----|------|
| 代码 | NVlabs/dex-ycb-toolkit（文章给出链接） |

## 局限与风险

- CC BY-NC 限制商用（文章给出许可）。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [manipulation](../tasks/manipulation.md)
- [diffusion-policy](../methods/diffusion-policy.md)
- [vla](../methods/vla.md)

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://dex-ycb.github.io/)
