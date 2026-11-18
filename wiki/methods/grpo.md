---
type: method
tags: [reinforcement-learning, llm, policy-optimization, grpo]
status: complete
original:
  - ../../sources/papers/deepseekmath_grpo_2024.md
updated: 2026-11-18
related:
  - ./policy-optimization.md
  - ./ppo.md
  - ./reinforcement-learning.md
  - ../formalizations/mdp.md
  - ../entities/paper-limit-of-rlvr.md
  - ../entities/paper-lightnav-0.md
  - ../entities/paper-prism-grpo.md
  - ../entities/minimind.md
  - ../entities/paper-atomic-wam.md
  - ../entities/paper-aesformer.md
sources:
  - ../../sources/papers/deepseekmath_grpo_2024.md
  - ../../sources/blogs/lightorigins_lightnav_0_2026-09-01.md
summary: "GRPO（Group Relative Policy Optimization）：DeepSeekMath 提出的组内相对优势 RL 变体；LightNav-0 第三阶段在线 RL 采用 GRPO 比较规划执行结果。"
---

# GRPO（Group Relative Policy Optimization）

**GRPO** 是 DeepSeekMath 提出的 RL 算法变体：在同一 prompt/状态下采样一组轨迹，用**组内相对回报**估计 advantage，省去独立 critic 网络。

## 一句话定义

**用一组并行 rollout 的相对排序代替 per-token critic，适合大模型 / VLA 后训练阶段的在线 RL。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GRPO | Group Relative Policy Optimization | 组相对策略优化 |
| RL | Reinforcement Learning | 强化学习 |
| PPO | Proximal Policy Optimization | 近端策略优化；GRPO 常作为其 advantage 估计替代 |
| VLA | Vision-Language-Action | 视觉-语言-动作策略 |

## 为什么重要

- [LightNav-0](../entities/paper-lightnav-0.md) 第三阶段在仿真中用 **GRPO** 比较自主规划执行结果，EVT-Bench 跟踪 SR 从 74.4→82.6。
- [MiniMind](../entities/minimind.md) 在 **文本 LLM** 尺度提供 **从 0 实现的 GRPO/CISPO/PPO** 与 `train_agent.py`，便于对照公式与代码（与 VLA/导航 GRPO 场景不同）。
- [DriveTeach-VLA](../entities/paper-driveteach-vla.md)（ECCV 2026）在 **TGP-guided SFT** 后用 **GRPO** 对齐 BEV 轨迹与驾驶偏好；RL 实现见 [Curious-VLA](https://github.com/Mashiroln/curious_vla)。
- 仓库另有 [Prism-GRPO](../entities/paper-prism-grpo.md)、[Temporal-GRPO](../entities/paper-temporal-grpo.md) 等变体论文页。
- 与 [PPO](./ppo.md) 同属 on-policy 家族，**advantage 来自同题多输出奖励的组内标准化**（并非仅按名次排序），而非独立 value/critic network（见 [MDP](../formalizations/mdp.md) 回报定义）。

## 主要技术路线

```mermaid
flowchart LR
  ctx["同一状态/指令上下文"] --> samp["并行采样 G 条 rollout"]
  samp --> ret["每条轨迹回报 R_i"]
  ret --> rel["组内均值/标准差归一化 → Â_i"]
  rel --> pg["Clipped policy gradient（类 PPO 目标）"]
  pg --> pol["更新 π_θ"]
```

典型部署：**SFT 后在线 RL**（LightNav-0）、LLM RLVR（[SDPG](../entities/paper-sdpg-self-distilled-policy-gradient.md) 等变体）。

## 核心原理

1. 对同一上下文采样多条 rollout（规划/动作序列）。
2. 原始结果监督 GRPO（DeepSeekMath §4.1.2）计算组内标准化优势：对同一 prompt 的奖励向量 \(\mathbf r\)，有 \(\hat A_{i,t}=(r_i-\mathrm{mean}(\mathbf r))/\mathrm{std}(\mathbf r)\)，在该回答的所有 token 上使用同一优势；不是只有名次先后。
3. 用 clipped policy gradient 更新策略（与 PPO 类似的目标，但 advantage 来自组内比较）；在目标中**单独施加**相对参考策略的 KL 正则，不把 KL 惩罚混入组内奖励。原论文另给出逐步奖励的过程监督：当前 token 的优势来自后续各推理步骤的归一化奖励之和（§4.1.3）。

## 工程实践

| 项 | 建议 |
|----|------|
| 适用 | LLM/VLM/VLA 后训练、可并行采样的仿真环境 |
| 与 PPO | 见 [policy-optimization](./policy-optimization.md)；GRPO 主要改 advantage 估计 |
| 导航实例 | LightNav-0 三阶段：SFT 后 GRPO 在线改进跟踪与恢复 |
| 原论文核验 | DeepSeekMath 使用约 **144K** 个 GSM8K/MATH 提示、每题 **64** 个候选、KL 系数 **0.04**；MATH CoT Top1 **46.8% → 51.7%**（Instruct → GRPO-RL） |

## 局限与风险

- 组大小与采样成本 trade-off；仿真吞吐不足时 RL 阶段增益有限。
- 相对回报对 reward 标定敏感；同题多次采样增加吞吐负担；导航任务需与 SFT 阶段接口一致（LightNav 复用 RVQ 动作词表）。
- DeepSeekMath §5.2 指出 RL 提升多数投票 Maj@K，但对 Pass@K 没有明显改善，**不能仅凭 RL 分数上升就断言基础可解问题集合扩大**；其数学结果也不能直接视为机器人任务的结果。

- [Limit of RLVR](../entities/paper-limit-of-rlvr.md)（NeurIPS 2025）进一步以同题 base/RLVR 的大 k 采样揭示：pass@1 增益通常来自高回报轨迹采样概率升高，而有限采样可解问题的覆盖可能收窄。该结论针对语言模型的可验证奖励设置，不能直接推广到机器人控制。

## 关联页面

- [AesFormer：七维美学编辑规划](../entities/paper-aesformer.md) — GRPO-A 在照片编辑计划上用格式、参考意图对齐与创造性奖励做组内优化；这里的 action 是照片编辑指令，并非机器人控制量。

- [policy-optimization](./policy-optimization.md)
- [LightNav-0](../entities/paper-lightnav-0.md)
- [Prism-GRPO](../entities/paper-prism-grpo.md)
- [Atomic-WAM](../entities/paper-atomic-wam.md) — EG-GRPO 用执行奖励后训练「原子动作描述」推理器，固定 WAM 据此出视频与动作块

## 参考来源

- [Limit of RLVR 原文（arXiv:2504.13837）核读](../../sources/papers/limit_of_rlvr_arxiv_2504_13837.md) — pass@k 与 GRPO/RLVR 训练对照。

- [DeepSeekMath 原文通读归档（arXiv:2402.03300v3）](../../sources/papers/deepseekmath_grpo_2024.md)
- [DeepSeekMath 官方全文](https://arxiv.org/html/2402.03300v3) — GRPO 原始定义、目标与实验（§4–§6，Appendix A.1）
- [lightorigins_lightnav_0_2026-09-01.md](../../sources/blogs/lightorigins_lightnav_0_2026-09-01.md)

## 推荐继续阅读

- DeepSeekMath 技术报告（GRPO 原始提出）
- [LightNav-0 Tech Blog](https://www.lightorigins.com/blog/lightnav-0)
