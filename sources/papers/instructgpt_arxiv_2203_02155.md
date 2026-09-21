# Training language models to follow instructions with human feedback

> 来源归档（ingest · Light-O1 ref [17] · RLHF 原典）

- **标题：** Training language models to follow instructions with human feedback
- **作者：** Long Ouyang 等（OpenAI）
- **类型：** paper / rlhf / alignment
- **arXiv：** <https://arxiv.org/abs/2203.02155>
- **入库日期：** 2026-09-21
- **一句话说明：** **InstructGPT / RLHF** 奠基：SFT + 人类偏好奖励模型 + PPO 对齐，使 LM 更 helpful/honest/harmless——Light-O1 post-training 用 RLHF 教「先语言推理再出动作」的两段式输出。

## 核心摘录

- **问题：** 更大 LM 不自动更 **follow user intent**（幻觉、毒性、无用回答）。
- **管线：** 1) 监督微调（SFT）示范；2) 训练 **reward model** 拟合人类偏好；3) **PPO** 优化策略 against RM。
- **Light-O1 用法：** post-training 阶段 **无 held-out loss 可评 intent**，故 scale **RLHF** 对齐人类意图与 **reasoning-then-action** 格式（tech blog §Pretraining and Adaptation）。
- **对 wiki 的映射：** [`wiki/entities/paper-instructgpt-rlhf.md`](../../wiki/entities/paper-instructgpt-rlhf.md)

## 参考来源

- 论文：<https://arxiv.org/abs/2203.02155>
