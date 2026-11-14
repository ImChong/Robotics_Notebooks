---
type: method
tags: [imitation-learning, transformers, behavior-cloning, machine-learning]
status: complete
updated: 2026-11-14
related:
  - ./behavior-cloning.md
  - ./action-chunking.md
  - ./humanoid-transformer-touch-dreaming.md
  - ./vla.md
  - ../concepts/foundation-policy.md
sources:
  - ../../sources/papers/imitation_learning.md
  - ../../sources/papers/humanoid_touch_dream.md
  - ../../sources/papers/act_arxiv_2304_13705.md
  - ../../sources/papers/aloha_act_reading_arxiv_2304_13705_reading_teleop.md
  - ../../sources/papers/rt_1_arxiv_2212_06817.md
original:
  - ../../sources/papers/act_arxiv_2304_13705.md
  - ../../sources/papers/aloha_act_reading_arxiv_2304_13705_reading_teleop.md
  - ../../sources/papers/rt_1_arxiv_2212_06817.md
summary: "基于 Transformer 的行为克隆（BC with Transformer）利用自注意力机制捕获长时序观测中的多峰分布与因果关系，是目前 ACT、RT 系列算法的核心骨干架构。"
---

# Behavior Cloning with Transformer

在模仿学习（IL）中，传统的基于 MLP 或 CNN 的行为克隆往往难以处理**多模态动作**（例如专家有时左绕，有时右绕）和**长时间依赖**。将 **Transformer** 引入 BC 架构，利用其自注意力机制，已成为具身智能的主流趋势。

## 一句话定义

把多帧图像、本体状态与自然语言条件映射为 Transformer token 序列，以注意力机制直接回归连续动作块（如 ACT）或预测离散化动作 token（如 RT-1）的行为克隆策略族。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BC | Behavior Cloning | 将状态映射到动作的监督式模仿，易受分布偏移影响 |
| ACT | Action Chunking Transformer | 预测动作块的序列模型架构，常与 ALOHA 配套 |
| IL | Imitation Learning | 从专家演示学习策略，奖励难定义时的主路线 |
| MLP | Multi-Layer Perceptron | 多层感知机，处理本体向量等低维输入 |
| CNN | Convolutional Neural Network | 卷积神经网络，处理图像/深度感知 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态基础策略方向 |
| Manipulation | Robot Manipulation | 抓取、移动、操作物体的任务总称 |

## 为什么重要

- **解决时间关联与遮挡记忆**：单帧 MLP 无法判断状态历史方向，引入多帧历史窗口（如 RT-1 的 6 帧历史、ACT 的短时序关节状态）显著缓解部分可观测与停滞歧义。
- **桥接连续高频控制与大模型序列接口**：同一套注意力结构既能配 CVAE + L1 回归做双臂精细装配（ACT / ALOHA），也能配 256-bin 交叉熵做千任务级规模化控制（RT-1），并构成 [Foundation Policy](../concepts/foundation-policy.md) 与 VLA 的共同底座。

## 核心原理

- **多模态 Token 化与压缩**：视觉主干（如 ResNet / EfficientNet + FiLM）与压缩器（如 TokenLearner 将每帧 81 token 压至 8 token）把图像、本体状态和指令压入可控序列长度。
- **双分支序列解码范式**：
  - **CVAE + DETR 式 Action Chunking 解码（ACT）**：训练期引入编码器抽取隐风格变量 $z$ 对齐动作多峰性，测试期置 $z=0$ 并一次性输出未来 $k$ 步目标关节位置配合 Temporal Ensembling 平滑误差累积。
  - **因果序列自回归预测（RT-1）**：把 11 维离散化动作写为 256-bin token 序列并用因果解码器逐 token 预测，直接复用语言模型扩展配方。

## 主要技术路线

1. **时序观测序列化**：将过去 $k$ 帧的视觉特征和状态特征作为 Token 序列输入。
2. **动作分词**：通常需要将动作空间离散化，详见 [Action Tokenization](../formalizations/vla-tokenization.md)。
3. **因果掩码 (Causal Mask)**：确保模型只能看到过去，不能预见未来。
3. **动作序列输出**：通常配合 [Action Chunking](./action-chunking.md)，一次性输出未来多个步长的动作，提高执行的连贯性。

## 代表算法

- **ACT (Action Chunking with Transformers)**：使用 CVAE 结构配合 Transformer 处理双臂协作任务。
- **RT-1 (Robotics Transformer)**：将机器人操作建模为 Token 流预测任务。
- **HTD (Humanoid Transformer with Touch Dreaming)**：在 action chunk 行为克隆上加入未来手部力和触觉 latent 预测，用辅助目标提高接触丰富型人形操作的表示质量。

## 工程实践

- **选择 chunk size 与 ensembling 权衡**：ALOHA/ACT 中过大的 chunk 降低突发扰动响应，过小则重现复合误差；实际部署需按相机曝光与控制周期联调 `chunk_size` 与时间集成系数 $m$。
- **离散 bin vs 连续动作头**：离散化（RT-1）适合规模化预训练与语言混合，但会损失毫米级/力控精度；高精度接触操作通常转向连续动作块、残差预测或流匹配头。
- **数据采集一致性**：遥操作演示的延迟抖动和静止停顿段会直接污染动作预测，采集管线须记录固定控制周期帧并清理无效静止片段。

## 局限与风险

- **复利误差（Compounding Error）并未根除**：Action Chunking 只在块内缓解分布偏移，一旦块间偏离专家轨迹流形，仍需要 DAgger 纠偏或 RL 后训练接管。
- **算力与历史窗口受限**：高分辨率多视角图像直接拼长序列会推高注意力复杂度，端侧高频闭环仍依赖 TokenLearner 或轻量视觉编码压缩。

## 关联页面
- [Behavior Cloning (行为克隆)](./behavior-cloning.md)
- [Action Chunking](./action-chunking.md)
- [Foundation Policy](../concepts/foundation-policy.md)
- [Humanoid Transformer with Touch Dreaming](./humanoid-transformer-touch-dreaming.md)
- [VLA (Vision-Language-Action Models)](./vla.md)

## 参考来源
- Zhao, T. Z., et al. (2023). *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*.
- Brohan, A., et al. (2022). *RT-1: Robotics Transformer*.
- [sources/papers/act_arxiv_2304_13705.md](../../sources/papers/act_arxiv_2304_13705.md) — ACT 算法机制、CVAE 多峰建模与 Temporal Ensembling 原文核对
- [sources/papers/aloha_act_reading_arxiv_2304_13705_reading_teleop.md](../../sources/papers/aloha_act_reading_arxiv_2304_13705_reading_teleop.md) — ALOHA 遥操作硬件与数据采集协议核对
- [sources/papers/rt_1_arxiv_2212_06817.md](../../sources/papers/rt_1_arxiv_2212_06817.md) — RT-1 架构、TokenLearner 与 256-bin 接口核对
- [sources/papers/imitation_learning.md](../../sources/papers/imitation_learning.md)
- [sources/papers/humanoid_touch_dream.md](../../sources/papers/humanoid_touch_dream.md)

## 推荐继续阅读

- [Action Chunking](./action-chunking.md)
- [RT-1 论文实体](../entities/paper-rt-1.md)
- [Diffusion Policy](./diffusion-policy.md)
