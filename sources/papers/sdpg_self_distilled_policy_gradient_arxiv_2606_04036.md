# sdpg_self_distilled_policy_gradient_arxiv_2606_04036

> 来源归档（ingest）

- **标题：** Self-Distilled Policy Gradient
- **类型：** paper
- **作者：** Yifeng Liu, Shiyuan Zhang, Yifan Zhang, Quanquan Gu（UCLA / Princeton）
- **arXiv：** <https://arxiv.org/abs/2606.04036>
- **项目页：** <https://lauyikfung.github.io/SDPG>
- **代码：** <https://github.com/lauyikfung/SDPG>（**已开源**，基于 verl）
- **入库日期：** 2026-09-06
- **一句话说明：** LLM 推理后训练框架 SDPG：在 GRPO/DAPO 稀疏 verifier 优势上叠加 **全词表 on-policy 自蒸馏**（privileged context 条件教师 vs 无 context 学生 reverse KL）与 reference-policy KL 正则；正优势门控 + β  warmup-decay 提升 RLVR 稳定性。

## 核心论文摘录（MVP）

### 1) 问题与动机（Abstract / §1）

- **链接：** <https://arxiv.org/abs/2606.04036>
- **核心贡献：** RLVR（如 GRPO）依赖稀疏序列级 verifier 奖励，token 级信用分配困难；纯 on-policy 自蒸馏（OPCD/OPSD）可把 privileged context 转为稠密 KL 信号，但无 verifier 时可能强化错误轨迹上的局部合理 token。SDPG **合并** GRPO 二值 outcome 与全词表 privileged OPD，并加 reference KL（UFKL/URKL 等）。
- **对 wiki 的映射：**
  - [SDPG 自蒸馏策略梯度论文实体](../../wiki/entities/paper-sdpg-self-distilled-policy-gradient.md)
  - [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)
  - [Prism-GRPO](../../wiki/entities/paper-prism-grpo.md)（同为 GRPO 族改进，领域不同：VLA vs LLM 数学推理）

### 2) SDPG 目标与稳定器（§2–3）

- **链接：** <https://github.com/lauyikfung/SDPG>
- **核心贡献：**
  - \(\mathcal{L}_{\text{SDPG}} = \mathcal{L}_{\text{out}} + \beta(k)\mathcal{L}^{+}_{\text{OPD}} + \alpha \mathcal{L}_{\mathcal{K}}(\pi_\theta,\pi_{\text{ref}})\)
  - **Full-vocab reverse KL：** \(D_{\mathrm{KL}}(p_t \| \mathrm{SG}[q_t])\)，\(p_t=\pi_\theta(\cdot|x,y_{<t})\)，\(q_t=\pi_\theta(\cdot|c,x,y_{<t})\)
  - **正优势门控** 与 **β warmup-decay** 抑制噪声教师信号
  - 数据格式：`[TEACHER_CONTEXT_TOKEN]` 分隔 actor 问题与教师 privileged context
- **对 wiki 的映射：**
  - [Policy Optimization](../../wiki/methods/policy-optimization.md)

### 3) 实验与复现（README）

- **链接：** <https://github.com/lauyikfung/SDPG>
- **核心贡献：**
  - Qwen3-4B 数学 DAPO 数据；对比 GRPO / OPSD / RLSD
  - 依赖 **verl** + Ray；默认 8× A100/H100；`examples/rpg2_trainer/run_qwen3_4b_sdpg_boxed.sh`
- **对 wiki 的映射：**
  - [SDPG 自蒸馏策略梯度论文实体](../../wiki/entities/paper-sdpg-self-distilled-policy-gradient.md)

## 同名消歧

- 缩写 **SDPG** 亦指 Yale 视觉机器人 RL 论文 [arXiv:2605.26478](https://arxiv.org/abs/2605.26478)（Stochastic Decoupled Policy Gradient）→ [wiki/entities/paper-sdpg-visual-rl-stochastic-decoupled.md](../../wiki/entities/paper-sdpg-visual-rl-stochastic-decoupled.md)

## 当前提炼状态

- [x] 摘要与 GitHub README 方法表对齐
- [x] 项目页核查：**已开源**
- [x] wiki 页面映射确认
