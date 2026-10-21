---
type: entity
tags:
  - dataset
  - tactile
  - handheld-gripper
  - in-the-wild
status: draft
updated: 2026-10-20
summary: "集成触觉传感器的手持夹爪采集的野外视觉触觉数据：2,700 余条示范、43 项任务。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../concepts/tactile-sensing.md
  - ../concepts/visuo-tactile-fusion.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# Touch in the Wild

## 一句话定义

**Touch in the Wild**：在多种室内外环境中用便携触觉夹爪同步采集的视觉与触觉示范数据集。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RSS | Robotics: Science and Systems | 会议 |
| NeurIPS | Neural Information Processing Systems | 会议 |
| VT | Visuo-Tactile（视觉触觉） | 视觉与触觉联合 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | Columbia University |
| 时间 | 2025-07（NeurIPS 2025） |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 规模 | 2,700 余条示范；43 项任务；12 个室内外环境 |
| 样本 | 260 万对以上视觉触觉样本 |
| 获奖 | RSS 2025 机器人硬件感知智能研讨会最佳演示奖（文章） |

## 为什么重要

- 补足手持夹爪普遍缺少触觉的问题，并在试管插入、移液等精细操作上验证效果。

## 核心原理

手持夹爪同步录制视觉与触觉，并用保留各自特性的跨模态表征做预训练。

## 工程实践

| 项 | 说明 |
|----|------|
| 获取 | Hugging Face binghaohuang-robot/touch_in_the_wild-dataset（文章给出链接） |
| 代码 | XinyueZhuXY/touch_in_the_wild 仓库 |

## 局限与风险

- 手持采集的夹爪构型与工业夹爪不同，迁移到产线需重新标定（推测）。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [tactile-sensing](../concepts/tactile-sensing.md)
- [visuo-tactile-fusion](../concepts/visuo-tactile-fusion.md)
- [manipulation](../tasks/manipulation.md)

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://binghao-huang.github.io/touch_in_the_wild/)
