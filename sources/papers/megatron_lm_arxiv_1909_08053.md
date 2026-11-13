# Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism / 使用模型并行训练数十亿参数语言模型

> 原始论文来源归档（paper），基于作者原文提炼，非全文转载。

- **标题：** Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism
- **中文：** Megatron-LM：使用模型并行训练数十亿参数语言模型
- **类型：** paper
- **作者：** Mohammad Shoeybi、Mostofa Patwary、Raul Puri、Patrick LeGresley、Jared Casper、Bryan Catanzaro
- **机构：** 英伟达（NVIDIA）
- **arXiv：** https://arxiv.org/abs/1909.08053
- **HTML 全文：** https://arxiv.org/html/1909.08053
- **PDF：** https://arxiv.org/pdf/1909.08053
- **版本：** arXiv v4，2020-03-13（初稿 2019-09-17）
- **代码：** https://github.com/NVIDIA/Megatron-LM（已开源；当前仓库包含 Megatron Core）
- **官方文档：** https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html
- **入库日期：** 2026-11-13
- **核对范围：** 通读 arXiv HTML v4 的摘要、背景、§3 模型并行、§4 配置、§5 实验、§6 结论与附录 A–E（包含 DP/TP 组、随机数、强弱缩放与测试口径）；额外对照 NVIDIA 主仓 README、官方并行指南与 pretrain_gpt.py 入口。附录 C 为原文生成文本示例，不作为真实性依据。
- **通读：** 已完成
- **一句话说明：** 在 PyTorch Transformer 内将 MLP 与 attention 的 GEMM 切分到多 GPU，减少跨 GPU 同步，同时与数据并行结合以突破单卡显存和训练规模限制。

## 原文机制核对

1. **MLP 张量并行（§3）：** 第一层投影按输出维切列，令各分片先独立执行 GeLU；第二层投影按输入维切行，再跨 GPU 归约。避免在 GeLU 前额外同步。
2. **Self-attention 并行（§3）：** Q/K/V 按注意力头对应的维度切分，头内注意力局部计算；输出投影按行切分后做跨卡归约。
3. **通信优化（§3）：** 经典 Transformer 层中 MLP 与 attention 前向各需一次 all-reduce，反向各一次；此为原论文 2019 结构的结论，并非对当前所有 Megatron Core 配置（如 sequence parallel）的无条件断言。
4. **词表并行（§3）：** Embedding 按 vocabulary 切分，将并行 logits 与 cross-entropy 融合，以避免先 all-gather 全词表 logits 的高通信代价。
5. **DP + TP（附录 B）：** 8 路模型并行 × 64 路数据并行 = 512 GPU；不同 TP 组同位置的 rank 组成 DP 组并同步梯度；对模型并行域内外 dropout 维护不同随机数流。

## 论文实验与边界

| 实验（原论文，非现行框架性能承诺） | 结果 |
|---|---|
| GPT-2 类模型规模 | 至 8.3B 参数；512 × V100 32GB |
| 512 GPU 综合训练计算 | 约 15.1 PFLOP/s；摘要报告 76% 相对强单卡基线效率（§5.1 的具体曲线口径另给约 74%） |
| WikiText-103 zero-shot | 8.3B：困惑度 10.81（Table 3，按论文规范化测试） |
| LAMBADA zero-shot | 8.3B：66.51%（Table 3） |
| BERT 3.9B | 重新安排 LayerNorm/残差，缓解模型扩大带来的退化 |
| 强缩放（附录 D） | 固定 1.2B、batch 8：2 GPU 1.64×，8 GPU 2.98×；通信与计算碎片化限制收益 |

**局限：** 2019 年实验基于 V100/DGX-2H 与特定 GPT/BERT 数据及评估方式，无法直接外推为今天 H100/Blackwell 或机器人 VLA 的吞吐；该论文首先研究层内张量并行，并非最初就实现现在文档中 TP/PP/DP/CP/EP 的全套策略。

## 对 wiki 的映射

- [Megatron-LM 独立论文详情节点](../../wiki/entities/paper-megatron-lm.md)
- [NVIDIA/Megatron-LM 代码归档](../repos/megatron-lm.md)
- [官方 Megatron Core 文档归档](../sites/nvidia-megatron-core.md)
- [LoongForge：基于 Megatron 栈的上层训练框架](../../wiki/entities/cn-os-loongforge.md)

## 相关独立研究（仅作延伸引用，未在本次 ingest 中合并论文身份）

- Narayanan 等（2021），*Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*，https://arxiv.org/abs/2104.04473：进一步研究 TP、PP、DP 的组合和交错流水线调度；是另一篇独立论文。
