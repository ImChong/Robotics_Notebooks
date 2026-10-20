---
type: entity
tags:
  - dataset
  - tactile
  - whole-hand
  - egocentric
status: draft
updated: 2026-10-20
summary: "首个野外第一视角全手触觉数据：5.1 小时视频、全手触觉与手部姿态同步。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../concepts/tactile-sensing.md
  - ../concepts/visuo-tactile-fusion.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# OpenTouch

## 一句话定义

**OpenTouch**：同步第一视角视频、全手触觉与手部姿态轨迹的野外数据集。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| OT | OpenTouch（数据集名） | 本数据集 |
| FPV | First-Person View | 第一视角 |
| MIT | Massachusetts Institute of Technology | 麻省理工学院 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | MIT 等 |
| 时间 | 2025-12 |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 规模 | 5.1 小时；约 3 小时接触密集的精细标注片段；2,900 个带文字标注片段 |
| 模态 | 第一视角视频、可感知力度的全手触觉、手部姿态 |
| 基准 | 检索与分类 |

## 为什么重要

- 补足“整只手触觉 + 第一视角视频”的真实环境数据空白，论文称触觉是理解抓取的紧凑线索。

## 核心原理

同步采集视频与全手触觉，附精细标注片段用于检索与分类基准。

## 工程实践

| 项 | 说明 |
|----|------|
| 获取 | Hugging Face rayxsong/opentouch（文章给出链接） |
| 代码 | OpenTouch-MIT/opentouch 仓库 |

## 局限与风险

- 总时长仅 5.1 小时，规模小（文章数字）。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [tactile-sensing](../concepts/tactile-sensing.md)
- [visuo-tactile-fusion](../concepts/visuo-tactile-fusion.md)
- [manipulation](../tasks/manipulation.md)

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://opentouch-tactile.github.io/)
