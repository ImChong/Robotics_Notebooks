---
type: entity
tags:
  - dataset
  - whole-body-grasp
  - mocap
  - hand-object
status: draft
updated: 2026-10-20
summary: "全身抓取动作数据：10 名受试者与 51 种日常物体交互，含全身、面部与双手网格。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../tasks/manipulation.md
  - ../methods/diffusion-policy.md
  - ../methods/vla.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# GRAB

## 一句话定义

**GRAB**：提出“全身抓取”概念并采集身体整体参与抓取的三维形状与姿态序列。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GRAB | GRasping Actions with Bodies | 数据集名称 |
| MoCap | Motion Capture | 动作捕捉 |
| ECCV | European Conference on Computer Vision | 会议 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | Max Planck Institute for Intelligent Systems |
| 时间 | 2020-08（ECCV 2020） |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 规模 | 10 名受试者；51 种形状各异的物体 |
| 标注 | 完整人体形状与姿态（含面部与双手）、物体三维位姿、身体-物体接触 |
| 衍生 | GrabNet 手部抓取生成模型 |

## 为什么重要

- 说明抓取不仅是手的动作，并据此计算身体与物体间的接触。

## 核心原理

动捕标记点拟合全身参数化网格，并据此计算接触。

## 工程实践

| 项 | 说明 |
|----|------|
| 获取 | grab.is.tue.mpg.de 下载页（文章给出链接） |
| 代码 | otaheri/GRAB 仓库 |

## 局限与风险

- 受试者人数少（10 人），人群多样性有限（推测）。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [manipulation](../tasks/manipulation.md)
- [diffusion-policy](../methods/diffusion-policy.md)
- [vla](../methods/vla.md)

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://grab.is.tue.mpg.de/)
