# Causal World Modeling for Robot Control (LingBot-VA)

> 来源归档（ingest · 主论文原文）

- **标题：** Causal World Modeling for Robot Control
- **模型名：** LingBot-VA
- **类型：** paper
- **机构：** 蚂蚁灵波（Robbyant）
- **arXiv：** <https://arxiv.org/abs/2601.21998>（HTML 全文：<https://arxiv.org/html/2601.21998v2>）
- **版本：** arXiv v2（2026-03-22）
- **项目页：** <https://technology.robbyant.com/lingbot-va>
- **代码：** <https://github.com/robbyant/lingbot-va>
- **入库日期：** 2026-11-21
- **核对范围：** 2026-11-21 通读 arXiv `2601.21998v2` HTML 全文（摘要、§1 引言、§2 Flow Matching 预备知识、§3.1–§3.4 自回归视频–动作世界建模/双流 MoT 架构/Noisy History Augmentation/FDM-grounded 异步推理与 Algorithm 1–2、§4.1–§4.5 数据集 16K 小时与 30D 统一动作空间、5.3B 实现细节、真机 6 任务与 RoboTwin 2.0 / LIBERO 评测、消融及记忆/泛化分析、§5–§6 结论与局限，以及 Appendix Table S1）；未在本地 GPU 上复跑预训练或真机实验。
- **通读：** 已完成
- **一句话说明：** 提出以 Wan2.2-5B 视频流（`d_v=3072`）与非对称 350M 动作流（`d_a=768`）构成的因果自回归双流 MoT 世界–动作模型（总计 5.3B 参数），在交错序列中联合建模视觉动态与逆动力学动作解码，并通过真实观测写回 KV cache 与 FDM-grounded 异步流水线实现闭环机器人控制。

## 原文核心要点（arXiv:2601.21998v2）

1. **统一交错序列与双流 MoT（§3.2–§3.3）：** 采用 Wan2.2 因果 VAE（`4×16×16` 压缩 + `2×` patchify，每帧 `N=192` token）与时间稀疏化（`τ=4`，每个潜帧对应 4 个动作 token），通过双流 MoT 共享注意力并各自保留投影参数；动作流按 $\sqrt{d_v/d_a}$ 缩放插值自视频预训练权重。
2. **Noisy History Augmentation 与异步闭环（§3.3–§3.4）：** 训练时以概率 0.5 对历史视频潜变量注入 `s_aug ∈ [0.5, 1.0]` 噪声，推理时视频仅需部分去噪（如整合至 `s=0.6` / 3 Euler 步）即可解码动作（10 步至 `s=1.0`）；异步管线在执行当前动作块 $a_t$ 的同时，先用真实反馈 $z_{t-1}$ 执行一步 FDM 落地预测，再预测下一块 $a_{t+1}$（Naive Async 在 RoboTwin Horizon=3 仅 32.9%，FDM-grounded 达 85.6%）。
3. **数据与统一动作接口（§4.1–§4.2）：** 聚合 Agibot、RoboMind、InternData-A1、OXE、UMI Data（不含 DexUMI）、RoboCOIN 与自采数据共约 16K 小时；统一 30 维双臂动作空间 `(7 EEF + 7 joints + 1 gripper) × 2`；预训练 1.4T tokens。
4. **实验结果（§4.3–§4.5）：** RoboTwin 2.0（50 任务）Easy 92.93% / Hard 91.55%（Horizon=3 达 93.22% / 93.28%）；LIBERO 四套件平均 98.5%（Long 98.5%）；真机 6 任务每任务仅 50 条演示微调即显著超越 π0.5，并在 Wipe Plate（计数 6 次）和 Search Box 等非马尔可夫记忆任务上验证 KV cache 记忆优势。

## 对 wiki 的映射

- [`wiki/entities/paper-sa-2601-21998-lingbot-va-causal-video-action-world-model-for-g.md`](../../wiki/entities/paper-sa-2601-21998-lingbot-va-causal-video-action-world-model-for-g.md)
