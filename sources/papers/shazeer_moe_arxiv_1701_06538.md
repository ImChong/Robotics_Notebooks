# Outrageously Large Neural Networks：稀疏门控 MoE（arXiv:1701.06538）

> 论文来源归档（ingest）

- **标题：** Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer
- **作者：** Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean
- **类型：** paper / deep-learning / architecture / moe
- **arXiv：** <https://arxiv.org/abs/1701.06538> · PDF：<https://arxiv.org/pdf/1701.06538.pdf>
- **入库日期：** 2026-09-21
- **一句话说明：** 用 **可训练稀疏门控** 在每一步只激活少数专家 MLP，使参数容量可扩到千亿级而逐步算力近似不变。

## 核心摘录（面向 wiki 编译）

### 1) 条件计算：容量与算力解耦

- **要点：** 网络吸收信息的能力受参数量限制；**按样本激活子集**（conditional computation）理论上可大幅扩容而不按比例增加计算。本文把该想法做成可在 GPU 集群上跑通的稀疏 MoE 层。
- **对 wiki 的映射：** [`wiki/concepts/mixture-of-experts.md`](../../wiki/concepts/mixture-of-experts.md)

### 2) 稀疏门控 + 上千专家

- **要点：** 一层 MoE 含多达数千个前馈子网络；门控网络为每个样本选出 **稀疏组合**。作者报告相对稠密模型 **>1000×** 容量提升，计算效率损失很小。
- **对 wiki 的映射：** [`wiki/concepts/mixture-of-experts.md`](../../wiki/concepts/mixture-of-experts.md)、[`wiki/overview/ai-architecture-map.md`](../../wiki/overview/ai-architecture-map.md)

### 3) 夹在 LSTM 之间的卷积式 MoE

- **要点：** 语言模型与机器翻译实验里，MoE 被卷积式地插在堆叠 LSTM 之间，最大约 **1370 亿** 参数；在更低计算代价下超过当时稠密 SOTA。
- **对 wiki 的映射：** [`wiki/concepts/recurrent-neural-network.md`](../../wiki/concepts/recurrent-neural-network.md)

### 4) 对机器人策略的读法

- **要点：** 今日 VLA 动作专家、多技能 locomotion gating 复用的是「**门控选专家**」而不是 137B 翻译模型本身。真机低层高频策略仍常是稠密小 MLP；MoE 出现在需要 **多模态/多技能容量** 的慢层。
- **对 wiki 的映射：** [`wiki/concepts/humanoid-policy-network-architecture.md`](../../wiki/concepts/humanoid-policy-network-architecture.md)、[`wiki/methods/vla.md`](../../wiki/methods/vla.md)

## 开源状态（步骤 2.5）

- 论文本身未维持单一现代官方仓；思想进入 tensor2tensor / Switch Transformer / 现代 LLM MoE。
- **结论：** 算法已开源扩散；本条按「无统一官方仓、生态已开源」处理。

## 当前提炼状态

- [x] 要点摘录与 wiki 映射
