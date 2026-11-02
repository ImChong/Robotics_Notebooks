---
type: entity
tags: [paper, robotics, robot-learning]
status: complete
updated: 2026-10-07
project_id: interaction-centric-gripper
arxiv: "2609.31207"
related:
  - ../overview/humanoid-motion-intelligence-day5-world-models-decision.md
  - ../methods/generative-world-models.md
sources:
  - ../../sources/blogs/humanoid_motion_intelligence_day5_world_models_decision_2026_10_06.md
summary: "交互中心表征用于缓解两指夹爪跨相机视角与夹爪域迁移的问题。"
---

# Enabling a Unified Cross-Domain Representation for Two-Finger Gripper Manipulation via Interaction-Centric Modeling

交互中心表征用于缓解两指夹爪跨相机视角与夹爪域迁移的问题。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| VLA | Vision-Language-Action | 视觉与指令条件化的机器人动作策略。 |
| WM | World Model | 预测行动后的环境变化。 |
| WAM | World Action Model | 联合建模未来视觉和动作。 |

## 为什么重要

交互中心表征用于缓解两指夹爪跨相机视角与夹爪域迁移的问题。 评读时应将问题、观测条件与执行接口一并记录，避免从单个演示或指标外推能力。

## 流程总览

```mermaid
flowchart TD
    A["语言指令 L + 当前 RGB-D 观测"] --> B["VLM (GLM-4.6V)<br/>解析任务并定位交互三元组<br/>(每 δ=5 步刷新)"]
    B --> C["SAM 2.1 分割 / 跟踪<br/>得到交互三元组 mask"]
    C --> D["RGB-D 投影为点云<br/>变换到夹爪 ego 坐标系 Gi"]
    D --> E["全局：APF 表征<br/>碰撞场 + 目标场，8 关键点 → (16,4)"]
    D --> F["局部：目标 / 被持物点云<br/>FPS + 加权采样至 256 点"]
    E --> G["每夹爪 zone 的 4 个 token<br/>PointNet / APF Transformer / 位姿 MLP"]
    F --> G
    B -->|"子任务文本 Ct"| H["冻结文本编码器 + MLP adapter"]
    G --> I["Flow-Matching Transformer 去噪器<br/>(8 层因果 Transformer)"]
    H --> I
    I --> J["horizon-H 的 7-DoF 动作块"]
```

图依据：论文 §IV-A 至 §IV-C / Alg.1、Fig.3

## 方法栈

融合语言指定物体关系、分割点云和接触几何，再生成机械臂动作，使策略聚焦物体交互而非视角坐标。

## 实验与评测

文章报告多任务真机迁移结果，但复杂任务成功次数仍有限，迁移受场景覆盖约束。

## 与其他工作对比

与全身控制或灵巧手策略范围不同；需要检查接触几何和对象覆盖。

## 工程实践

当前归档仅核实到arXiv入口，未核实可运行官方代码。

## 源码运行时序图

**不适用**：当前没有核实到该论文的可运行官方训练、推理或部署入口。

## 结论

**一句话总判：** 交互中心表征用于缓解两指夹爪跨相机视角与夹爪域迁移的问题。

1. 核对方法实际改变的是数据、表征、规划、控制还是评测。
2. 只把论文覆盖的任务、平台与扰动范围当作证据。
3. 复现前确认代码、权重、数据许可和硬件依赖状态。

## 局限与风险

与全身控制或灵巧手策略范围不同；需要检查接触几何和对象覆盖。 当前导读以用户提供的文章和预印本题录为索引；详细配置与结果应核对论文全文。

## 关联页面

- [Day 5 导读](../overview/humanoid-motion-intelligence-day5-world-models-decision.md)
- [生成式世界模型](../methods/generative-world-models.md)

## 参考来源

- [Day 5 原文归档](../../sources/blogs/humanoid_motion_intelligence_day5_world_models_decision_2026_10_06.md)
- [arXiv:2609.31207](https://arxiv.org/abs/2609.31207)
- [arXiv PDF](https://arxiv.org/pdf/2609.31207)：已通读方法部分（实验与附录以略读为准）：§IV Method（A 感知、B 表征、C 动作、D 部署效率）

## 推荐继续阅读

- [Day 5 导读](../overview/humanoid-motion-intelligence-day5-world-models-decision.md)
