# Stabilizing Reinforcement Learning with LLMs: Formulation and Practices｜原文通读归档

- **标题：** Stabilizing Reinforcement Learning with LLMs: Formulation and Practices
- **中文标题：** 稳定大语言模型的强化学习：形式化分析与实践
- **作者：** Chujie Zheng, Kai Dang, Bowen Yu, Mingze Li, Huiqiang Jiang, Junrong Lin, Yuqiong Liu, Hao Lin, Chencan Wu, Feng Hu, An Yang, Jingren Zhou, Junyang Lin
- **机构：** 阿里巴巴（Alibaba）Qwen 团队
- **类型：** paper / LLM-RL / policy-gradient / MoE / training-stability
- **论文：** <https://arxiv.org/abs/2512.01374>；PDF <https://arxiv.org/pdf/2512.01374>（arXiv:2512.01374v3，2025-12-03 版本，13 页）
- **入库日：** 2026-11-08
- **一句话说明：** 用一阶近似解释何时可以使用 token 级代理目标优化完整回答的序列级奖励，并在 30B MoE 的 LLM 强化学习中检验训练—推理偏差、策略陈旧、重要性采样、裁剪和专家路由重放的影响。
- **核对范围：** 已通读 v3 PDF 全部 13 页：Abstract、§1 引言、§2 公式 (1)–(5) 与一阶近似条件、§3 R2/R3 路由重放及 Table 1、§4.1–4.5 MiniRL/实验设置/on-policy/off-policy/cold-start、Figure 1–5、§5 结论、参考文献、Appendix A MiniRL/GRPO/CISPO 目标对照、Appendix B Figure 6–10 分项基准。并核对 arXiv 原始条目。
- **通读：** 已完成

## 研究问题与重要边界

- 这里的 **with LLMs** 表示**对 LLM 本身进行强化学习训练**，并非让 LLM 充当机器人环境反馈校验器、奖励模型或 reward shaping 模块。
- 论文使用数学推理题的**可验证二值奖励**（完整回答正确与否，R∈{0,1}）；未研究稀疏/噪声环境奖励修复或具身机器人训练。
- 关键矛盾是完整响应由 **sequence-level** 奖励评价，但常见 REINFORCE/GRPO 训练使用 **token-level** 策略梯度目标；直接用序列级重要性权重因响应概率跨度大、方差高而难以优化。

## 原文核心摘录（带章节对应）

1. **§2.2–2.3 / Eq. (1)–(4)：** 真正目标是期望完整回答奖励。token 级代理目标把整段概率比乘积替换为逐 token 比率和；设各 token 比率为 1+δ_t，乘积的一阶展开丢弃 δ_iδ_j 等高阶项。因此 **仅在目标策略与 rollout 策略足够接近时**，代理梯度才近似序列级目标梯度。
2. **§2.4 / Eq. (5)：** 每个 token 的概率比可分解为 [训练引擎旧策略 / 推理引擎 rollout 策略] × [当前训练策略 / 训练引擎旧策略]，分别对应 **training–inference discrepancy** 和 **policy staleness**。两者都影响上述近似可靠性。
3. **§3 / Eq. (6)、Table 1：** MoE 的专家路由变化会放大两种偏差。**R2 (Vanilla Routing Replay)** 在更新时复用训练引擎旧策略选中的专家，主要抑制 staleness；**R3 (Rollout Routing Replay)** 复用推理引擎 rollout 时选中的专家，同时缓和训练—推理路由不一致与 staleness，但两者都可能改变原始目标策略并引入偏差。
4. **§4.1 / Eq. (7)：** MiniRL 采用组内均值作为 reward baseline（而非本文给出的 GRPO 标准差归一化），再把 token-level IS 权重按 stop-gradient 使用；clip/mask 通过正负优势与当前策略/训练旧策略的概率比决定是否停止某 token 的梯度。
5. **§4.2：** 主实验为 Qwen3-30B-A3B-Base 衍生 30B MoE，4,096 道数学训练题、AIME24/AIME25/HMMT25 各 30 题、每题 32 次采样的平均正确率；推理 FP8、训练 BF16；每次更新 1,024 responses（B=64, G=16），最大输出 32,768 tokens，ε_high=0.27、ε_low=0.20，Truncated IS 阈值 5，估计每 gradient step 5–6 GPU hours。
6. **§4.3 / Figure 1：** on-policy（global batch = mini-batch = 1,024）下，**保留 IS 校正的基础 MiniRL 最稳定且表现最佳**；去掉 train–infer IS 迅速崩溃；额外的长度归一化表现次优；加入 R3 不带来提升，反而可能产生目标偏置。
7. **§4.4 / Figures 2–4：** off-policy 将 global batch 增为 2,048、4,096、8,192（分别 N=2、4、8 mini-batches）。**Clipping 与 Routing Replay 需要组合使用**；N=2 时 R2 较 R3 好，N=4/8 时 R3 更稳；R2 在高 off-policiness 下也可能崩溃。
8. **§4.5 / Figure 5：** 对早期实验 Qwen3Next 小型 MoE，分别用三个模型的 cold-start 蒸馏数据，在 MiniRL+R2 稳定训练后，AIME24/25 的最终表现接近。这不等于证明所有基础模型/任务/初始化都会收敛到同一点。
9. **Appendix A：** 与 GRPO/CISPO 目标比较指出，原始 GRPO/CISPO 目标没有显式修正 train–infer discrepancy，且含长度归一化；CISPO 不直接 mask 对应 token 梯度。论文将这些差异与其消融结果联系起来。
10. **Appendix B：** Figure 6–10 提供 HMMT25/AIME25/AIME24 以及不同 cold start 的分项曲线，支持正文的趋势性判断；不将图像曲线估读为精确数表。

## 开放与复现状态

- 论文和 arXiv 条目**未列出与本研究对应的官方可运行代码/专属项目主页**；截至本次核对（2026-11-08）未核实到作者官方复现仓库。
- 第三方 MiniRL 复现不等于作者实验源码；因此**不绘制伪造的源码运行时序图**。
- 文中 SGLang、vLLM、Megatron、FSDP 是通用训练/推理引擎示例，并非这篇论文公开的脚本入口。

## 对 wiki 的映射

- [稳定 LLM 强化学习：论文实体](../../wiki/entities/paper-stabilizing-rl-with-llms.md) — 理论近似、MiniRL、R2/R3、实验、局限与流程图
- [策略优化](../../wiki/methods/policy-optimization.md) — 与机器人 RL 中 PPO 更新稳定性对照，明确方法适用边界

## 参考来源

- [arXiv 原文条目](https://arxiv.org/abs/2512.01374)
- [论文 v3 PDF（13 页）](https://arxiv.org/pdf/2512.01374)
