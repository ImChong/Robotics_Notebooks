---
type: method
tags: [vla, foundation-policy, deepmind, flow-matching, manipulation]
status: complete
updated: 2026-05-15
related:
  - ./vla.md
  - ./pi07-policy.md
  - ./diffusion-policy.md
  - ../formalizations/vla-tokenization.md
  - ../formalizations/cross-modal-attention.md
sources:
  - ../../sources/papers/diffusion_and_gen.md
summary: "π₀ (Pi-zero) 是由 Physical Intelligence 提出的一种通用的 Vision-Language-Action 模型，通过结合流匹配（Flow Matching）与大规模预训练，实现了对复杂机器人操作任务的高效建模。"
---

# π₀ (Pi-zero) 策略模型

**π₀ (Pi-zero)** 是具身智能大模型（VLA）领域的最新突破，由 Physical Intelligence 团队于 2024 年提出。它旨在打破“一个机器人一个模型”的限制，通过单一的大型神经网络同时掌控不同形态（如双臂机械臂、灵巧手、移动底座）的机器人执行多样化的复杂任务。

## 主要技术路线

π₀ 的设计融合了语言模型的大规模预训练优势与生成式动作建模的精确性：

1. **流匹配 (Flow Matching) 骨干**：
   有别于 RT-1 等采用的标量离散化路线，π₀ 在动作输出层使用了**流匹配（Flow Matching）**。这是一种基于 [概率流形式化](../formalizations/probability-flow.md) 的高效生成式建模方法，能够以更少的推理步数生成高质量、连续且多模态的动作分布。
2. **视觉语言对齐**：
   π₀ 借用了预训练多模态大模型（如 VLM）的权重，使其天然具备理解自然语言指令（如“把弄脏的毛巾放进篮子里”）并识别图像中复杂物体的能力。
3. **后训练 (Post-training) 范式**：
   类似于 LLM 的指令微调，π₀ 首先在海量的多源机器人数据集上进行基础预训练（Pre-training），随后通过特定任务的高质量演示数据进行对齐微调。

## 为什么是“π₀”？

“π”在强化学习中代表策略（Policy），而“0”则隐喻“从零开始的通用基础”。它的出现标志着机器人控制正经历从“特征工程”到“**Scaling Law (规模法则)**”的范式转移。

## 性能优势

- **多任务泛化**：能够处理从折衣服到整理餐具等完全不同的任务。
- **跨平台通用**：同一模型可以无缝适配不同厂商的机械臂，证明了动作表征的普适性。
- **鲁棒性**：面对视觉背景的变化和物体的轻微位移，展现出了极强的闭环纠错能力。

## 关联页面
- [VLA (Vision-Language-Action Models)](./vla.md)
- [π₀.7（Pi-zero 0.7）通才 VLA](./pi07-policy.md) — 同一 π 系路线在「多模态提示 + 异质数据对齐」上的后继公开版本（2026）
- [Diffusion Policy](./diffusion-policy.md)
- [Action Tokenization (动作分词)](../formalizations/vla-tokenization.md)
- [Cross-modal Attention (跨模态注意力)](../formalizations/cross-modal-attention.md)
- [LWD（Learning while Deploying）](./lwd.md) — 其 QAM 组件正是为 flow-based 动作头（如 π₀）设计的策略抽取方法

## 参考来源
- Black, K., et al. (2024). *π₀: A Vision-Language-Action Flow Model for General Robot Control*.
- [Physical Intelligence Blog](https://www.physicalintelligence.company/blog/pi0).
- [sources/papers/pi07.md](../../sources/papers/pi07.md) — π₀.₇ 后继工作与多模态提示条件（若只关心 π₀ 本体的历史语境可略读摘录节）
