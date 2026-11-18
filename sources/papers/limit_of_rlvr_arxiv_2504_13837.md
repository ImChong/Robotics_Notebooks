# Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? / 强化学习真的能超越基础模型的推理能力边界吗？

> 来源归档（paper · Limit of RLVR）

- **标题：** Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?
- **作者：** Yang Yue（乐洋）、Zhiqi Chen、Rui Lu、Andrew Zhao、Zhaokai Wang、Yang Yue（乐阳）、Shiji Song、Gao Huang
- **机构：** 清华大学（Tsinghua University / LeapLab）、上海交通大学（Shanghai Jiao Tong University）
- **版本：** [arXiv:2504.13837v5](https://arxiv.org/abs/2504.13837v5)，2025-11-24；首版 2025-04-18；NeurIPS 2025
- **原文：** [arXiv HTML 全文](https://arxiv.org/html/2504.13837v5) · [PDF](https://arxiv.org/pdf/2504.13837v5)
- **项目页：** https://limit-of-rlvr.github.io/
- **代码：** https://github.com/LeapLabTHU/limit-of-RLVR （公开数学与代码评测脚本，非完整训练发布）
- **入库日期：** 2026-11-18
- **身份纠错：** 用户提供 arXiv:2608.24053 实际对应《WeMM-Embedding: WeChat Multi-Modal Embedding Technical Report》，与本论文标题不符；本文以 arXiv:2504.13837 作为唯一主论文身份。

## 核心摘录与核对

1. **实验对象：** 数学 Qwen2.5-7B/14B/32B、LLaMA-3.1-8B、Oat-Zero、DAPO 等；编程 CodeR1-Zero 与 DeepCoder；视觉推理 Qwen2.5-VL-7B；另有近前沿 Magistral-Medium 的初步验证（原文 §3、§4.6）。
2. **统一评测协议：** 对基座与 RLVR 版本使用匹配的 zero-shot 提示和生成设置，主实验 temperature=0.6、top-p=0.95，最多 16,384 tokens；以 pass@k 在大 k 时估计有限采样预算内可解问题覆盖率，不等于数学意义的绝对能力上界（§2.2–§3）。
3. **数学对照：** Minerva 上 32B 模型 k=128 时基座 pass@k 比 RLVR 高约 9 个百分点。AIME24 在 k=1024 的共同可解占 63.3%，仅基座可解占 13.3%，仅 RLVR 可解为 0%；MATH500 在 k=128 时对应 92.4%、3.6%、1.0%（§3.1、§4.1 Table 2）。
4. **机制解释：** 以基座模型计算 RLVR 输出序列的 PPL，与基座自身采样的低 PPL 区间重叠；奖励后训练强化既有高回报轨迹的生成概率，而不是在这些实验中稳定出现基座完全没有的推理轨迹（§4.1、Appendix C.4）。
5. **算法消融：** PPO、GRPO、Reinforce++、RLOO、ReMax、DAPO 的差异相对有限。Omni-MATH 测试集：Qwen2.5-7B Base pass@1=10.2、pass@256=69.1；GRPO 25.1/68.3；PPO 26.8/69.2；DAPO 26.5/67.0（百分比，Appendix C.5 Table 3）。
6. **训练轨迹：** Omni-MATH 测试集 GRPO 的 pass@1 随 step 150/300/450 从 25.1→27.1→28.3，而 pass@256 从 68.3→66.6→63.9，提示要同时观察单样本成功率和覆盖率（Appendix C.5 Table 4）。
7. **蒸馏对照：** DeepSeek-R1-Distill-Qwen-7B 相比基础 Qwen2.5-Math-7B 扩大 pass@k 曲线；作者将其归于强教师的 CoT 信息转移，不能把这类收益归因于零额外信息的 RLVR（§4.2）。
8. **范围限制：** 论文主要研究可验证的二元结果奖励；部分 coding/vision 从 instruction-tuned 模型起步，近前沿模型不易全面评测；原文明确不主张否定所有 RL、所有奖励形态或更大规模探索（§5、§7；官方 Q&A）。

## 与机器人知识体系的关系

- **核心是评测方法而不是机器人真机实验。** 对应单次成功率与多次尝试任务覆盖率的不同目标；不可把 LLM 文本 pass@k 的结论直接推广成足式 RL、VLA 或机器人闭环成功率结论。
- 作为 [GRPO](../../wiki/methods/grpo.md) 的 RLVR 评测反例，与 [强化学习](../../wiki/methods/reinforcement-learning.md)、[Reasoning with Sampling](../../wiki/entities/paper-reasoning-with-sampling.md) 的多次采样研究形成交叉链接。
- **唯一论文节点：** [Limit of RLVR](../../wiki/entities/paper-limit-of-rlvr.md)。

## 核对记录

- **核对范围：** 通读 arXiv v5 HTML 正文 §1–7，核对数学/代码/视觉任务、pass@k 估计、perplexity、蒸馏与算法比较；检查附录 A–E 的算法、评测设置、表 2–6、消融、提示和边界说明；对照作者项目页 Q&A 与公开代码入口。静态核读，未实际运行 GPU 实验。
- **通读：** 已完成
