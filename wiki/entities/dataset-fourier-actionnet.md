---
type: entity
tags:
  - dataset
  - real-robot
  - humanoid
  - dexterous-hand
status: draft
updated: 2026-10-20
summary: "傅利叶人形机器人灵巧手双臂 VR 遥操作数据，约 3 万条轨迹、140 小时。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../tasks/loco-manipulation.md
  - ../tasks/teleoperation.md
  - ../methods/vla.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# Fourier ActionNet Dataset

## 一句话定义

**Fourier ActionNet**：面向人形灵巧双臂操作的第一视角遥操作数据集，覆盖 GR1-T1、GR1-T2、GR2 三款人形机器人。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DoF | Degrees of Freedom | 自由度 |
| VR | Virtual Reality | 虚拟现实遥操作 |
| NC | Non-Commercial | 非商业许可条款 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | 傅利叶 + 上海交通大学 |
| 时间 | 文章未注明 |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 规模 | 3 万余条遥操作轨迹，约 140 小时 |
| 模态 | 第一视角视觉（OAK-D W 97 广角相机）+ 双臂与灵巧手遥操作动作 |
| 本体 | GR1-T1 / GR1-T2 / GR2；6-DoF 与 12-DoF 灵巧手 |
| 许可 | 数据 CC BY-NC-SA 4.0；工具 Apache 2.0 |

## 为什么重要

- 同时覆盖 6 自由度与 12 自由度两类灵巧手，便于研究手型差异下的策略迁移。

## 核心原理

操作者佩戴 Vision Pro，从机器人第一人称视角进行 VR 遥操作，广角 OAK-D W 97 相机记录视觉。

## 工程实践

| 项 | 说明 |
|----|------|
| 工具 | fourier-lerobot 仓库（文章给出链接） |

## 局限与风险

- CC BY-NC-SA 许可限制商用，选用前需确认授权。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [loco-manipulation](../tasks/loco-manipulation.md)
- [teleoperation](../tasks/teleoperation.md)
- [vla](../methods/vla.md)

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://huggingface.co/datasets/FourierIntelligence/ActionNet)
