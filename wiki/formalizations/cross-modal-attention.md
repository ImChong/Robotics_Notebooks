---
type: formalization
tags: [vla, deep-learning, attention, multimodal, machine-learning]
status: complete
updated: 2026-04-21
related:
  - ../methods/vla.md
  - ../formalizations/vla-tokenization.md
  - ../concepts/tactile-sensing.md
sources:
  - ../../sources/papers/perception.md
summary: "跨模态注意力机制（Cross-modal Attention）允许 VLA 模型在统一的 Transformer 架构中动态关联视觉特征、自然语言指令与机器人本体状态，是具身智能语义理解的核心。"
---

# Cross-modal Attention (跨模态注意力)

在具身大模型（VLA）中，**跨模态注意力 (Cross-modal Attention)** 是实现“理解指令并根据视觉反馈执行动作”的核心数学机制。它允许模型在处理 Token 序列时，显式地计算不同感官模态（如 RGB 图像、自然语言命令、力觉反馈）之间的相关性。

## 数学表达

基于标准的 Transformer Scaled Dot-Product Attention，定义来自不同模态的 Query ($Q_i$)、Key ($K_j$) 和 Value ($V_j$)。

### 1. 互注意力 (Cross-Attention)
常用于语言指令引导视觉特征的提取。假设语言 Token 序列为 $S_{lang}$，视觉特征 Token 序列为 $S_{vis}$：

$$ \text{Attention}(Q_{vis}, K_{lang}, V_{lang}) = \text{softmax}\left( \frac{Q_{vis} K_{lang}^T}{\sqrt{d_k}} \right) V_{lang} $$

- **Query ($Q_{vis}$)**：由视觉补丁（Patch）产生，询问“哪些图像区域与指令有关？”。
- **Key ($K_{lang}$)**：由指令词产生，作为匹配的索引。
- **Value ($V_{lang}$)**：携带指令的语义信息，融入到图像特征中。

### 2. 多模态统一自注意力 (Unified Multi-modal Self-Attention)
这是 RT-2 和 π₀ 等主流 VLA 采用的范式。将所有模态的 Token 拼接为一个超长序列 $S_{total} = [S_{vis}, S_{lang}, S_{robot}]$：

$$ S_{out} = \text{SelfAttention}(S_{total}) $$

在这种形式下，模型通过内部权重矩阵 $W_Q, W_K, W_V$ 自动学习跨模态的依赖关系：
- 视觉 Token 关注语言 Token 以获取任务目标。
- 动作 Token 同时关注视觉（以确定空间位姿）和语言（以确定操作逻辑）。

## 具身智能中的关键作用

1. **语义定位 (Grounded Semantic)**：通过注意力图（Attention Map），可以观察到当指令提到“红色杯子”时，模型对红色像素区域的注意力权重显著升高。
2. **多模态对齐 (Alignment)**：将非结构化的物理信号（如力觉）映射到与自然语言相同的语义空间。
3. **因果推理**：在预测未来动作 $a_t$ 时，通过注意力机制回溯历史观测序列 $o_{t-k:t}$。

## 关联页面
- [VLA (Vision-Language-Action Models)](../methods/vla.md)
- [Action Tokenization](./vla-tokenization.md)
- [触觉感知 (Tactile Sensing)](../concepts/tactile-sensing.md)

## 参考来源
- Vaswani, A., et al. (2017). *Attention Is All You Need*.
- Team, G., et al. (2023). *RT-2: Vision-Language-Action Models Transfer Knowledge from Web to Robots*.
