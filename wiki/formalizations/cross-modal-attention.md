---
type: formalization
tags: [vla, deep-learning, attention, multimodal, machine-learning]
status: complete
updated: 2026-11-07
related:
  - ../methods/vla.md
  - ../formalizations/vla-tokenization.md
  - ../concepts/tactile-sensing.md
  - ../concepts/transformer.md
  - ../concepts/multi-head-attention.md
  - ../entities/paper-attention-is-all-you-need.md
  - ../concepts/flow-matching-embodied-policy.md
original:
  - ../../sources/papers/attention_is_all_you_need.md
  - ../../sources/papers/pi0_arxiv_2410_24164.md
  - ../../sources/papers/rt2_vla_reading_arxiv_2307_15818.md
  - ../../sources/papers/gr00t_n1_arxiv_2503_14734.md
  - ../../sources/papers/smolvla_arxiv_2506_01844.md
sources:
  - ../../sources/papers/attention_is_all_you_need.md
  - ../../sources/papers/pi0_arxiv_2410_24164.md
  - ../../sources/papers/rt2_vla_reading_arxiv_2307_15818.md
  - ../../sources/papers/gr00t_n1_arxiv_2503_14734.md
  - ../../sources/papers/smolvla_arxiv_2506_01844.md
  - ../../sources/papers/perception.md
summary: "跨模态注意力机制（Cross-modal Attention）允许 VLA 模型在统一的 Transformer 架构中动态关联视觉特征、自然语言指令与机器人本体状态，是具身智能语义理解的核心。"
---

# Cross-modal Attention (跨模态注意力)

## 一句话定义

**跨模态注意力（Cross-modal Attention）** 让一种模态的 token（如动作、图像块）作为 Query，去读取另一种模态的 token（如语言、视觉特征）作为 Key/Value，从而把视觉、语言和机器人本体状态融合进同一个 Transformer。在 VLA 中它有两种主流实现：把各模态 token 拼进同一序列做（带掩码的）自注意力，或用交叉注意力层让动作头读取 VLM 特征。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作多模态基础策略方向 |
| VLM | Vision-Language Model | 视觉语言模型，VLA 的常见骨干 |
| RGB | Red-Green-Blue | 彩色图像通道，常与深度 (RGB-D) 配合 |
| RT-2 | Robotics Transformer 2 | 将 web 规模 VLM 能力迁移到机器人控制的代表工作 |
| MHA | Multi-Head Attention | 多头注意力，见 [多头注意力](../concepts/multi-head-attention.md) |
| DiT | Diffusion Transformer | 以 adaLN + 交叉注意力注入条件的去噪骨干，见 [DiT](../concepts/diffusion-transformer.md) |

## 为什么重要

- 在具身大模型（VLA）中，它是实现“理解指令并根据视觉反馈执行动作”的核心机制：模型处理 token 序列时，可显式计算不同模态之间的相关性。
- 原版 Transformer 就内置了跨序列注意力：解码器的“编码器-解码器注意力”层中，Query 来自解码器上一层，Key/Value 来自编码器输出（Vaswani et al., 2017, §3.2.3）。VLA 把这一思路推广到图像、语言、状态与动作。
- 选择哪一侧做 Query、用拼接序列还是独立交叉注意力层，直接影响动作头的延迟、可缓存性和效果（见下文 π₀、GR00T N1、SmolVLA 的对比）。

## 核心原理

基于标准的缩放点积注意力 $\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\big(QK^\top/\sqrt{d_k}\big)V$（[Transformer](../concepts/transformer.md)）。

### 1. 交叉注意力 (Cross-Attention)

Query 与 Key/Value 来自不同序列。原论文的写法：Q 来自解码器、K/V 来自编码器输出。以语言指令引导视觉特征为例的**示意**写法（非原文公式，属常见设计）：

$$ \text{Attention}(Q_{vis}, K_{lang}, V_{lang}) = \text{softmax}\left( \frac{Q_{vis} K_{lang}^T}{\sqrt{d_k}} \right) V_{lang} $$

- **Query**：询问方，输出的序列长度与 Query 序列一致。
- **Key**：被检索方的匹配索引；**Value**：被检索方携带的内容。

VLA 动作头里的实际方向正相反——动作 token 作 Query：GR00T N1 的 DiT 动作模块“交替的交叉注意力与自注意力块”，自注意力处理加噪动作 token 与状态嵌入，交叉注意力读取 VLM 输出的视觉-语言 token；SmolVLA 的动作专家在交叉注意力层里以 VLM 的 K/V 为被读取对象，并与对动作 token 的因果自注意力交错。

### 2. 多模态拼接自注意力 (Concatenated Multi-modal Self-Attention)

把各模态 token 排进一个序列，用同一套 $W_Q,W_K,W_V$ 做自注意力，模型自行学习跨模态依赖。示意：$S_{total}=[S_{vis},S_{lang},S_{robot}]$，$S_{out}=\text{SelfAttention}(S_{total})$。几个真实系统并不是简单的全连接拼接：

| 系统 | 做法（原文核实） |
|------|------------------|
| RT-2 | 把动作写成文本 token（256 个均匀 bin 的整数），与网络图文数据一起 co-fine-tune；RT-2-PaLI-X 的图像 token 经投影后与文本一起进入 UL2 风格的编码器-解码器骨干（32B、50 层），RT-2-PaLM-E 是把图像与文本投影到语言 token 空间的仅解码器模型；机器人任务下解码只采样合法动作 token |
| π₀ | 单个 Transformer、两套权重（VLM 专家 + 约 300M 动作专家），权重只通过自注意力交互。**分块因果注意力掩码**共 3 块：[图像, 语言]、[机器人状态 $q_t$]、[噪声动作 $A_t^\tau$]；块内双向，块只能看前面的块。状态单独成块是因为它在各去噪步不变，可缓存 KV。动作 token 作为最后一块能看到全部输入 |
| GR00T N1 | 视觉语言 token 经交叉注意力进入 DiT 块；动作编码器 MLP 同时编码去噪时间步与噪声动作 |
| SmolVLA | 动作专家交错交叉注意力（K/V 来自 VLM 特征）与因果自注意力；消融里交错最优（LIBERO 85.5），纯交叉 79.0，纯自注意力 74.5 |

### 3. 以 π₀ 为例的掩码

```mermaid
flowchart LR
  B1["块1：图像 + 语言\n块内双向，看不到后两块"] --> B2["块2：机器人状态\n看不到动作块"]
  B2 --> B3["块3：噪声动作 chunk\n可看全部输入"]
```

## 工程实践

- **选 Query 一侧：** 想让动作读取多模态上下文，就让动作 token 作 Query（GR00T N1、SmolVLA、π₀ 的动作块）；想让视觉特征被语言调制，可用语言作 K/V（示意写法，需按具体论文核实）。
- **KV 缓存：** π₀ 的块掩码让图像/语言/状态前缀的 K/V 在 10 步去噪中只算一次，每步只重算动作块；论文 4090 上总计 73 ms（图像编码 14 ms + 观测前向 32 ms + 10 步动作前向 27 ms）。
- **注意力掩码是超参，不是定论：** π₀ 的动作专家块内双向；SmolVLA 的消融里动作 token 间双向自注意力（67.5）反而不如因果（74.5）；两者数据与架构不同，不能互相推翻。
- **状态放哪里：** SmolVLA 消融中，把状态作为 VLM 前缀比作为动作专家后缀好（CA 设定 80.3 对 73.3）；π₀ 则把状态单独成块。
- **可视化：** 可通过注意力图检查模型是否关注指令中提到的物体，但这是诊断手段，不保证因果解释（推测，非来源结论）。

## 局限与风险

- **原论文范围：** Vaswani 等只在机器翻译和句法分析上验证注意力机制；把跨模态注意力当作“语义对齐”的结论，需要靠 VLA 论文自己的实验支持。
- **“注意力图显示红色杯子权重升高”“力觉映射到与语言相同的语义空间”**等说法在本页此前未附出处，此处只作示例性设想（推测），不是上述论文的结论。
- 二次复杂度：拼接序列越长（多相机多帧）算力越大；RT-2 55B 只能约 1–3 Hz，需云端推理；VLA 通常靠小动作头与前缀缓存控制延迟。
- 触觉/力觉等模态的 token 化与对齐，本页引用的论文均未覆盖（见 [触觉感知](../concepts/tactile-sensing.md)）。

## 关联页面

- [VLA (Vision-Language-Action Models)](../methods/vla.md)
- [Action Tokenization](./vla-tokenization.md)
- [触觉感知 (Tactile Sensing)](../concepts/tactile-sensing.md)
- [Transformer](../concepts/transformer.md)
- [多头注意力](../concepts/multi-head-attention.md)
- [Flow Matching 与具身策略](../concepts/flow-matching-embodied-policy.md)

## 参考来源

- Vaswani, A., et al. (2017). *Attention Is All You Need*. [归档](../../sources/papers/attention_is_all_you_need.md)
- Brohan, A., et al. (2023). *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*. [归档](../../sources/papers/rt2_vla_reading_arxiv_2307_15818.md)（此前本页误写作者为 Team, G.）
- Black, K., et al. (2024). *π0*. [归档](../../sources/papers/pi0_arxiv_2410_24164.md)
- Bjorck, J., et al. (2025). *GR00T N1*. [归档](../../sources/papers/gr00t_n1_arxiv_2503_14734.md)
- Shukor, M., et al. (2025). *SmolVLA*. [归档](../../sources/papers/smolvla_arxiv_2506_01844.md)

## 推荐继续阅读

- [Attention Is All You Need（论文实体）](../entities/paper-attention-is-all-you-need.md)
- [π0 论文 arXiv:2410.24164](https://arxiv.org/abs/2410.24164)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
