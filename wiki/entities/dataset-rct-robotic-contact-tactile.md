---
type: entity
tags:
  - dataset
  - tactile
  - language
  - material
  - evaluation
status: draft
updated: 2026-10-20
summary: "机器人用三个 DIGIT 传感器按压 122 种工业材料：29,279 帧触觉图像、1,832 段接触序列。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../concepts/tactile-sensing.md
  - ../concepts/visuo-tactile-fusion.md
  - ../tasks/manipulation.md
  - ../queries/embodied-eval-benchmark-selection-loop.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# RCT（Robotic Contact Tactile）

## 一句话定义

**RCT**：机器人采集的触觉、视觉与语言数据集，支持按材料、类别、传感器做留出评测。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DIGIT | DIGIT tactile sensor | 视觉触觉传感器 |
| Recall@1 | Top-1 检索召回率 | 检索评测指标 |
| RCT | Robotic Contact Tactile | 数据集名称 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | TU Dresden（德累斯顿工业大学） |
| 时间 | 2026-06 |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 规模 | 122 种材料（7 大类）；29,279 帧；1,832 段接触序列 |
| 附加 | 逐帧接触力、材料照片、触感描述 |
| 泄漏分析 | 去除序列重叠后触觉到文本 Recall@1 下降 17.7 个百分点；留出材料时为 25.1% |

## 为什么重要

- 论文指出同一次按压的帧高度相关，随机按帧划分会造成训练测试泄漏，并给出按序列划分的评测口径。

## 核心原理

机器人在多个接触位置按压材料，按接触序列组织数据。

## 工程实践

| 项 | 说明 |
|----|------|
| 代码 | faerber-lab/RCT 仓库（文章给出链接） |

## 局限与风险

- 论文的泄漏分析是本数据的主要贡献，结论适用于同类按压数据（推测）。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [tactile-sensing](../concepts/tactile-sensing.md)
- [visuo-tactile-fusion](../concepts/visuo-tactile-fusion.md)
- [manipulation](../tasks/manipulation.md)
- [具身大模型评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md) — 评测基准在选型闭环中的定位与取舍

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://github.com/faerber-lab/RCT)
