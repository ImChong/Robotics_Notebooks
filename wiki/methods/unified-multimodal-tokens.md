---
type: method
tags: [vla, transformers, tokenization, multi-modal, architecture]
status: complete
updated: 2026-04-21
related:
  - ../formalizations/vla-tokenization.md
  - ../formalizations/cross-modal-attention.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/rl_foundation_models.md
summary: "统一多模态 Token（Unified Multimodal Tokens）是现代 VLA 模型的架构趋势，通过将视觉 Patch、自然语言词向量和量化动作编码进同一个嵌入空间，实现了纯粹的序列建模。"
---

# Unified Multimodal Tokens (统一多模态 Token)

**统一多模态 Token** 是一种先进的具身智能架构设计。它摒弃了为每种感官模态设计专用神经网络分支的传统做法，转而将所有输入（图像、语言、状态、动作）全部转换为格式一致的 Token 序列，并在一个通用的 **Transformer** 中统一处理。

## 架构组成

### 1. 视觉 Token 化 (Visual Patching)
- 将输入的 RGB 图像划分为 $16 \times 16$ 或 $14 \times 14$ 的固定大小补丁（Patches）。
- 每个补丁通过线性投影层（Linear Projection）映射为特征向量。
- **代表**：Vision Transformer (ViT)。

### 2. 状态与动作 Token 化
- **本体感受**：将关节弧度、末端位姿通过感知器层映射为单个或多个 Token。
- **动作**：通过 [Action Tokenization](../formalizations/vla-tokenization.md) 将连续动作转换为离散的 ID。

## 主要技术路线

模型看到的输入是一个混合序列：
`[SEP] [Text_Tokens] [SEP] [Visual_Patches] [SEP] [Robot_State] [SEP]`
模型的目标是自回归地预测随后的 `[Action_Tokens]`。

### 关键技术技巧


1. **模态指示 Embedding (Modality Type Embeddings)**：
   类似于 BERT 的 Segment Embeddings，在每个 Token 上加上一个可学习的向量，告诉模型这个 Token 是“看的”、“读的”还是“动的”。
2. **位置编码同步**：
   - 文本使用一维位置编码。
   - 视觉补丁使用二维位置编码。
   - 时序序列使用绝对或相对时序编码。
3. **因果屏蔽 (Causal Masking)**：
   确保模型在预测 $t$ 时刻的动作时，只能关注 $t$ 时刻之前的观测，防止信息泄漏。

## 带来的优势

- **极强的灵活性**：如果机器人新增了一个触觉传感器，只需增加一种新的 Tokenizer，而无需改动核心 Transformer 架构。
- **跨任务迁移**：模型可以像在不同语种间切换一样，在“拿杯子”和“走路”两个完全不同的任务序列间共享底层特征。
- **硬件友好的扩展性**：所有的 Scaling 工作都集中在增加 Transformer 的层数和宽度上，利用现有的算力优化。

## 关联页面
- [动作分词 (Action Tokenization)](../formalizations/vla-tokenization.md)
- [跨模态注意力 (Cross-modal Attention)](../formalizations/cross-modal-attention.md)
- [VLA (Vision-Language-Action Models)](./vla.md)

## 参考来源
- Black, K., et al. (2024). *π₀: A Vision-Language-Action Flow Model for General Robot Control*.
- Team, G., et al. (2023). *RT-2: Vision-Language-Action Models Transfer Knowledge from Web to Robots*.
