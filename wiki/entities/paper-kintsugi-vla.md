---
type: entity
tags: [paper, robotics, robot-learning]
status: complete
updated: 2026-10-07
project_id: kintsugi-vla
arxiv: "2609.31048"
related:
  - ../overview/humanoid-motion-intelligence-day5-world-models-decision.md
  - ../methods/generative-world-models.md
sources:
  - ../../sources/blogs/humanoid_motion_intelligence_day5_world_models_decision_2026_10_06.md
summary: "Kintsugi-VLA从仿真失败轨迹中筛选适合学习恢复行为的起始状态。"
---

# Kintsugi-VLA: Turning Failed Robot Rollouts into Recovery Data through Interventional Recoverability

Kintsugi-VLA从仿真失败轨迹中筛选适合学习恢复行为的起始状态。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| VLA | Vision-Language-Action | 视觉与指令条件化的机器人动作策略。 |
| WM | World Model | 预测行动后的环境变化。 |
| WAM | World Action Model | 联合建模未来视觉和动作。 |

## 为什么重要

Kintsugi-VLA从仿真失败轨迹中筛选适合学习恢复行为的起始状态。 评读时应将问题、观测条件与执行接口一并记录，避免从单个演示或指标外推能力。

## 流程总览

```mermaid
flowchart TD
    A["仿真中的失败 rollout<br/>(PPO 教师 33 + 基础 VLA 167，共 200 条)"] --> B["保存 simulator 快照<br/>(各 checkpoint 状态)"]
    B --> C["恢复状态并分支<br/>固定特权专家 πE 重复续作"]
    C --> D["自适应预算 5→15→30 次<br/>估计 recoverability + Wilson 区间"]
    D --> E["找出末端低可恢复前沿 tf<br/>(Ui &lt; τ 的最大末端后缀)"]
    E --> F["恢复状态选择<br/>深度 0.15 / 0.50 / 0.85"]
    F --> G["特权专家从所选状态续作<br/>成功续作导出为恢复示范"]
    G --> H["难度匹配 / 帧预算匹配"]
    H --> I["与名义示范一起训练 SmolVLA"]
    I --> J["评测 Protocol A / B / G"]
```

图依据：论文 §III / Fig.2

## 方法栈

在候选状态恢复仿真并由特权教师多次续作，以成功比例估计可恢复性，再围绕低可恢复前沿采集恢复示范。

## 实验与评测

原文报告定向采样恢复成功率34.6%，高于随机25.9%与窗口均匀28.8%；受扰执行由59.5%升至63.4%，干净任务略降。

## 与其他工作对比

这是恢复数据选择方法，不是部署时在线故障控制器；应单独报告干净任务与受扰任务。

## 工程实践

当前归档仅核实到arXiv入口，未核实可运行官方代码。

## 源码运行时序图

**不适用**：当前没有核实到该论文的可运行官方训练、推理或部署入口。

## 结论

**一句话总判：** Kintsugi-VLA从仿真失败轨迹中筛选适合学习恢复行为的起始状态。

1. 核对方法实际改变的是数据、表征、规划、控制还是评测。
2. 只把论文覆盖的任务、平台与扰动范围当作证据。
3. 复现前确认代码、权重、数据许可和硬件依赖状态。

## 局限与风险

这是恢复数据选择方法，不是部署时在线故障控制器；应单独报告干净任务与受扰任务。 当前导读以用户提供的文章和预印本题录为索引；详细配置与结果应核对论文全文。

## 关联页面

- [Day 5 导读](../overview/humanoid-motion-intelligence-day5-world-models-decision.md)
- [生成式世界模型](../methods/generative-world-models.md)

## 参考来源

- [Day 5 原文归档](../../sources/blogs/humanoid_motion_intelligence_day5_world_models_decision_2026_10_06.md)
- [arXiv:2609.31048](https://arxiv.org/abs/2609.31048)
- [arXiv PDF](https://arxiv.org/pdf/2609.31048)：已通读方法部分（实验与附录以略读为准）：§III Method、§IV 实验设置（§IV-A 至 §IV-C）

## 推荐继续阅读

- [Day 5 导读](../overview/humanoid-motion-intelligence-day5-world-models-decision.md)
