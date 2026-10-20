---
type: entity
tags:
  - dataset
  - hand-object
  - pose
  - real-world
status: draft
updated: 2026-10-20
summary: "真实场景手与物体联合三维位姿数据：77,558 帧、68 段序列，由自动标注方法 HOnnotate 生成真值。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../tasks/manipulation.md
  - ../methods/diffusion-policy.md
  - ../methods/vla.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# HO-3D（含 H2O-3D、HO-3D v3）

## 一句话定义

**HO-3D**：真实手-物遮挡场景下的三维手部姿态与物体 6D 位姿联合标注数据集。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| HOnnotate | 文中自动标注方法名 | 无标记点联合优化 |
| YCB | Yale-CMU-Berkeley Object Set | 物体集 |
| 6D | Six-Degree-of-Freedom | 六自由度位姿 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | Graz University of Technology、LIGM |
| 时间 | 2019-07（CVPR 2020） |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 规模 | 77,558 帧；68 段序列；10 名参与者；10 个 YCB 物体 |
| 后续 | HO-3D v3；双手版本 H2O-3D |
| 真值 | 标注真值已于 2024 年 11 月公开（文章） |

## 为什么重要

- 无需标记点即可在严重遮挡下联合优化手与物体位姿，是真实场景标注的代表方法。

## 核心原理

利用单台或多台 RGB-D 相机联合优化手与物体位姿。

## 工程实践

| 项 | 说明 |
|----|------|
| 代码 | shreyashampali/ho3d 仓库（文章给出链接） |

## 局限与风险

- 自动标注精度依赖多视角与优化收敛（推测）。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [manipulation](../tasks/manipulation.md)
- [diffusion-policy](../methods/diffusion-policy.md)
- [vla](../methods/vla.md)

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://github.com/shreyashampali/ho3d)
