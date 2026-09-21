---
type: entity
tags: [paper, rlhf, alignment, nlp, openai, post-training]
status: complete
updated: 2026-09-21
arxiv: "2203.02155"
venue: "NeurIPS 2022"
related:
  - ./light-o1.md
  - ../methods/vla.md
  - ../concepts/embodied-scaling-laws.md
sources:
  - ../../sources/papers/instructgpt_arxiv_2203_02155.md
  - ../../sources/courses/karpathy_deep_dive_llms_youtube.md
summary: "InstructGPT（arXiv:2203.02155）：SFT + 奖励模型 + PPO 的 RLHF 管线，使 LM 更遵循人类意图——Light-O1 用 RLHF 对齐「先推理再动作」输出格式。"
---

# InstructGPT（RLHF）

**Training language models to follow instructions with human feedback**（Ouyang et al.，[arXiv:2203.02155](https://arxiv.org/abs/2203.02155)，OpenAI / NeurIPS 2022）提出 **InstructGPT**：在基座 LM 上 **SFT → 训练 reward model → PPO 强化学习**，显著改善 **helpfulness / honesty / harmlessness**，奠定 **ChatGPT 对齐栈**。

## 一句话定义

**用人类偏好当奖励信号，把「大 LM 会写」变成「大 LM 听人话」的三段式 post-training 模板。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RLHF | Reinforcement Learning from Human Feedback | 人类反馈强化学习 |
| SFT | Supervised Fine-Tuning | 示范微调 |
| RM | Reward Model | 拟合人类偏好的标量模型 |
| PPO | Proximal Policy Optimization | 常用 RL 优化器 |
| LM | Language Model | 自回归语言模型 |

## 为什么重要

- **Light-O1 post-training：** Tech Blog [17] 明确用 **RLHF** 对齐 **human intent** 与 **两段式输出**（先语言推理 body 需求，再生成 action）——因 **intent 无 held-out loss**。
- **具身对齐范式：** VLA/VLM 的 **preference / RLHF / DPO** 分支均溯源至此。

## 核心管线

```mermaid
flowchart LR
  base["基座 LM"] --> sft["SFT 示范"]
  sft --> rm["训练 Reward Model\n（人类 pairwise 偏好）"]
  rm --> ppo["PPO 优化策略"]
  ppo --> inst["InstructGPT / 对齐模型"]
```

## 实验与评测

- 人类评估：**InstructGPT 1.3B** 可在多维度 **优于未对齐 GPT-3 175B**（论文人类 side-by-side）。
- **Public API 用户** 偏好显著偏向 RLHF 模型。
- **毒性 / 幻觉** 部分改善，但未消除——后续 RLHF/DPO 迭代仍继续。

## 结论

**RLHF 是「预训练压缩知识 → post-training 对齐用法」的标准第二段；Light-O1 把它搬到「对齐推理-动作格式与人类意图」。**

1. **SFT alone 不够** — 需 preference 信号刻画 subtle quality。
2. **RM + PPO** 成为工业默认（后接 DPO/RLAIF 等变体）。
3. **具身复用：** 同一逻辑用于 **language reasoning trace + action** 对齐。
4. **成本：** 人类标注贵；Light-O1 仅称 **scale RLHF**，细节未开源。
5. **边界：** 原文是 **text**；机器人 **物理安全** 需额外 constraint，不能照搬 RM。

## 关联页面

- [Light-O1](./light-o1.md) — RLHF post-training 引用方
- [Embodied Scaling Laws](../concepts/embodied-scaling-laws.md)

## 参考来源

- [instructgpt_arxiv_2203_02155.md](../../sources/papers/instructgpt_arxiv_2203_02155.md)
- 论文：<https://arxiv.org/abs/2203.02155>

## 推荐继续阅读

- [Karpathy Deep Dive LLMs](../../sources/courses/karpathy_deep_dive_llms_youtube.md) — 课程中的 RLHF 章节
