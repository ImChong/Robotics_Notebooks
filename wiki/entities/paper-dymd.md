---
type: entity
tags: [paper, robotics, robot-learning]
status: complete
updated: 2026-10-07
project_id: dymd
arxiv: "2609.31349"
related:
  - ../overview/humanoid-motion-intelligence-day5-world-models-decision.md
  - ../methods/generative-world-models.md
sources:
  - ../../sources/blogs/humanoid_motion_intelligence_day5_world_models_decision_2026_10_06.md
summary: "DyMD压缩视频世界模型时着重保留抓取、搬运中的交互变化。"
---

# DyMD: Preserving Interaction Dynamics through Distribution Matching Distillation in Few-Step Video World Models

DyMD压缩视频世界模型时着重保留抓取、搬运中的交互变化。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| VLA | Vision-Language-Action | 视觉与指令条件化的机器人动作策略。 |
| WM | World Model | 预测行动后的环境变化。 |
| WAM | World Action Model | 联合建模未来视觉和动作。 |

## 为什么重要

DyMD压缩视频世界模型时着重保留抓取、搬运中的交互变化。 评读时应将问题、观测条件与执行接口一并记录，避免从单个演示或指标外推能力。

## 方法栈

用分布匹配蒸馏将14B视频扩散教师压缩为1.3B少步生成器，并提高困难交互样本权重。

## 实验与评测

文章报告R-Bench遵循分数34.6升至44.2，另两套视频基准的领域分数提高；没有实体机器人试验。

## 与其他工作对比

它优化预测模型的成本与动态保真，不增加运行时控制模块；生成视频分数不是实机策略成功率。

## 工程实践

当前归档仅核实到arXiv入口，未核实可运行官方代码。

## 源码运行时序图

**不适用**：当前没有核实到该论文的可运行官方训练、推理或部署入口。

## 结论

**一句话总判：** DyMD压缩视频世界模型时着重保留抓取、搬运中的交互变化。

1. 核对方法实际改变的是数据、表征、规划、控制还是评测。
2. 只把论文覆盖的任务、平台与扰动范围当作证据。
3. 复现前确认代码、权重、数据许可和硬件依赖状态。

## 局限与风险

它优化预测模型的成本与动态保真，不增加运行时控制模块；生成视频分数不是实机策略成功率。 当前导读以用户提供的文章和预印本题录为索引；详细配置与结果应核对论文全文。

## 关联页面

- [Day 5 导读](../overview/humanoid-motion-intelligence-day5-world-models-decision.md)
- [生成式世界模型](../methods/generative-world-models.md)

## 参考来源

- [Day 5 原文归档](../../sources/blogs/humanoid_motion_intelligence_day5_world_models_decision_2026_10_06.md)
- [arXiv:2609.31349](https://arxiv.org/abs/2609.31349)

## 推荐继续阅读

- [Day 5 导读](../overview/humanoid-motion-intelligence-day5-world-models-decision.md)
