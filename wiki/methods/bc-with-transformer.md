---
type: method
tags: [imitation-learning, transformers, behavior-cloning, machine-learning]
status: complete
updated: 2026-04-21
related:
  - ./behavior-cloning.md
  - ./action-chunking.md
  - ./vla.md
sources:
  - ../../sources/papers/imitation_learning.md
summary: "基于 Transformer 的行为克隆（BC with Transformer）利用自注意力机制捕获长时序观测中的多峰分布与因果关系，是目前 ACT、RT 系列算法的核心骨干架构。"
---

# Behavior Cloning with Transformer

在模仿学习（IL）中，传统的基于 MLP 或 CNN 的行为克隆往往难以处理**多模态动作**（例如专家有时左绕，有时右绕）和**长时间依赖**。将 **Transformer** 引入 BC 架构，利用其自注意力机制，已成为具身智能的主流趋势。

## 主要技术路线

1. **时序观测序列化**：将过去 $k$ 帧的视觉特征和状态特征作为 Token 序列输入。
2. **动作分词**：通常需要将动作空间离散化，详见 [Action Tokenization](../formalizations/vla-tokenization.md)。
3. **因果掩码 (Causal Mask)**：确保模型只能看到过去，不能预见未来。
3. **动作序列输出**：通常配合 [Action Chunking](./action-chunking.md)，一次性输出未来多个步长的动作，提高执行的连贯性。

## 代表算法

- **ACT (Action Chunking with Transformers)**：使用 CVAE 结构配合 Transformer 处理双臂协作任务。
- **RT-1 (Robotics Transformer)**：将机器人操作建模为 Token 流预测任务。

## 关联页面
- [Behavior Cloning (行为克隆)](./behavior-cloning.md)
- [Action Chunking](./action-chunking.md)
- [VLA (Vision-Language-Action Models)](./vla.md)

## 参考来源
- Zhao, T. Z., et al. (2023). *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*.
- Brohan, A., et al. (2022). *RT-1: Robotics Transformer*.
