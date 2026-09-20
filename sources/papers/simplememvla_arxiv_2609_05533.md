# SimpleMemVLA（arXiv:2609.05533）

> 来源归档（paper）

- **标题：** SimpleMemVLA: A Simple but Effective Native-Video Memory for Vision-Language-Action Models
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.05533>
- **PDF：** <https://arxiv.org/pdf/2609.05533>
- **代码：** <https://github.com/OpenBMB/SimpleMemVLA>
- **模型集合：** <https://huggingface.co/collections/yinchenghust/simplememvla>
- **机构：** 清华大学（Tsinghua University）、面壁智能（ModelBest / OpenBMB）等
- **作者：** Cheng Yin, Wang Xu, Junpeng Yang, Sikyuen Tam, Hanyu Liu, Yuan Yao, Xiangrui Zeng, Junbo Cui, Yequan Wang, Zhouping Yin, Yankai Lin
- **入库日期：** 2026-09-20
- **一句话说明：** 无专用记忆模块的 VLA：把采样历史以带 plaintext 时间戳的原生视频喂给 Qwen3.5-4B，sub-task 文本 span 隐状态作为唯一历史→动作通道，DiT flow-matching 动作头；四套记忆基准 SOTA 且 LIBERO 97.5 无损，官方全栈已开源。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-09-20）：[OpenBMB/SimpleMemVLA](https://github.com/OpenBMB/SimpleMemVLA)（MIT）含五基准统一训练/闭环评测；HF 每套件 checkpoint + LeRobot v3 数据集；ModelScope 镜像。arXiv 摘要仍指向 `wadeKeith/SimpleMemVLA`，以 OpenBMB 组织仓为准。

## 核心摘录

1. **问题：** 长程操作部分可观测；检索库、学习压缩器、循环状态需在 **决策前** 决定保留什么（write-time commitment），可能丢掉未来才需要的帧。
2. **方法：** **SimpleMemVLA** 不建专用记忆模块——保留采样历史，以 **带 `<X.X seconds>` 时间戳的原生视频** 输入 **Qwen3.5-4B**；监督生成当前 **sub-task** 文本（≤64 token），其 **hidden states + token embeddings**（加 1 个 proprio token）为 **DiT flow-matching 动作头**（~0.9B）唯一条件；共享前缀 **prefill** 使流式推理 **1.02 s → 0.68 s**（字节级等价全重算）。
3. **记忆接口消融（同骨干/数据/训练器，仅换历史接口）：** RoboMME 上 native context **88.3%** vs retrieval **31.5%** / token compression **22.6%** / recurrent **20.6%**。
4. **主结果（每套件一模型，官方协议闭环）：** RMBench **94.0**、RoboMME **88.3**、MIKASA-Robo **74.0**、RoboMemArena **63.6 TSR / 72.1 CSR**；LIBERO **97.5**（并列最佳）、LIBERO-Plus **78.4**。
5. **真机：** 双臂 Cover Blocks **35/60 (58.3%)**、Put Back Block **28/40 (70.0%)**（各初始位 10 次自主试验）；180 / 308 条真机 demo 微调。
6. **工程：** 单仓五基准，`simplememvla/benchmarks/` 规格模块；checkpoint 自包含 `config.json`+`stats.json`；Python 3.10 + torch 2.4.1；SAPIEN 基准需 **分 conda 环境**。

**对 wiki 的映射**

- [paper-simplememvla](../../wiki/entities/paper-simplememvla.md)
- [openbmb_simplememvla](../repos/openbmb_simplememvla.md)
