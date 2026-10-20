---
type: entity
tags:
  - dataset
  - real-robot
  - humanoid
  - whole-body
status: draft
updated: 2026-10-20
summary: "宇树 G1 人形全身操作开源数据合集：55 个子数据集，约 1.26 万片段、132 小时（截至 2026-09）。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../tasks/loco-manipulation.md
  - ../tasks/teleoperation.md
  - ../methods/vla.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# UnifoLM-WBT Dataset

## 一句话定义

**UnifoLM-WBT Dataset**：面向人形机器人全身移动操作的开源数据合集，由 Hugging Face 上的多个子数据集组成。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| G1 | Unitree G1 humanoid | 宇树人形机器人型号 |
| HF | Hugging Face | 数据托管平台 |
| WBT | Whole-Body Teleoperation（推测） | 名称中的缩写，文章未展开 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | 宇树科技（Unitree） |
| 时间 | 2026-03-05 |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 规模 | 55 个子数据集；约 1.26 万片段、约 132 小时（按官方元数据合计） |
| 本体 | G1 + 强脑 / 因时 / Dex1 三类手部 |
| 模态 | 头部双目与双腕共 4 路视频；下肢、腰部、双臂、手部、躯干位姿与底盘速度指令 |

## 为什么重要

- 同时包含行走与双手操作的协同动作，是人形全身控制数据的典型公开样例。

## 核心原理

按任务场景（收纳衣物、整理床铺、装洗碗机、超市理货等）采集，统一记录视觉与关节/底盘指令。

## 工程实践

| 项 | 说明 |
|----|------|
| 合集入口 | Hugging Face collection（文章给出链接） |

## 局限与风险

- 子数据集许可与格式不完全一致，需逐个核查（推测，待官方页核实）。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [loco-manipulation](../tasks/loco-manipulation.md)
- [teleoperation](../tasks/teleoperation.md)
- [vla](../methods/vla.md)

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://huggingface.co/collections/unitreerobotics/unifolm-wbt-dataset-69a8fddf623895884e663d0f)
