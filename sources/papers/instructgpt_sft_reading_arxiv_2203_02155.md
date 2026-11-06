# InstructGPT 论文通读（SFT 视角）

> 来源归档（原文通读 · 支撑 `wiki/concepts/supervised-fine-tuning.md`）

- **标题：** Training language models to follow instructions with human feedback
- **作者：** Long Ouyang 等（OpenAI）
- **类型：** paper / sft / rlhf
- **原文：** <https://arxiv.org/abs/2203.02155>（2026-11 通过 arXiv PDF 获取全文）
- **一句话说明：** 把 SFT 明确为「示范数据监督微调 → 奖励模型 → PPO」三步对齐管线的第一步，并给出 SFT 的数据规模、训练超参与相对 GPT-3 / PPO 的评测。
- **核对范围：** 通读正文 §1–§5（含 3.2 数据集、3.5 模型 / SFT 训练、4.1–4.3 结果、5 讨论与局限）及附录 A.3（数据集规模）、C.1 / C.3（SFT 训练细节与 RLHF 初始化）；其余附录（标注员问卷、全部毒性 / 偏见细表）仅浏览，未逐项核对。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **三步管线（§3.1）：** Step 1 收集示范数据，用监督学习微调预训练 GPT-3 得到 SFT 策略；Step 2 收集模型输出的人类排序训练奖励模型；Step 3 用 PPO 针对奖励模型优化 SFT 策略。
- **SFT 数据（§3.2、附录 A.3）：** SFT 数据集约 13k 训练 prompt（API 与标注员撰写）；表 6：训练集标注员 11,295 条、客户 1,430 条。标注员条目偏多，是因为早期让标注员写模板指令加 few-shot 示例，并通过采样不同 few-shot 组合从同一指令合成多条样本。
- **SFT 训练（§3.5、附录 C.1）：** 训练 16 epoch，cosine 学习率衰减到初值 10%，residual dropout 0.2；1.3B / 6B 学习率 9.65e-6、batch 32，175B 学习率 5.03e-6、batch 8。
- **过拟合但仍有益（§3.5）：** SFT 模型在 1 个 epoch 后验证 loss 就开始过拟合，但继续训练能提升奖励模型得分与人类偏好评分，故最终以验证集上的 RM 得分选模型，而非验证 loss。
- **RLHF 初始化（附录 C.3）：** 作为 RL 初始化的 SFT 只训 2 epoch，并混入 10% 预训练数据（对 PPO 训练有帮助）。
- **效果（§4.1）：** 偏好提升呈台阶式：GPT-3 < 精心 few-shot 的 GPT-3 < SFT < PPO；175B InstructGPT 相对 GPT-3 被偏好的比例为 85 ± 3%，相对 few-shot GPT-3 为 71 ± 4%。
- **公开 NLP 数据集不够（§4.2）：** 在 FLAN、T0 数据上微调的 175B GPT-3 优于 GPT-3，与精心 prompt 的 GPT-3 相当，但差于 SFT 基线，说明这些数据多样性不足以覆盖真实 API 提示分布。
- **SFT 的副作用（§4.3）：** SFT 基线毒性最低，但连贯性（continuity）最低、排序中最不受偏好，作者推测其输出可能过短或退化。
- **成本（§5.1）：** 训练 175B SFT 约 4.9 petaflops/s-days，175B PPO-ptx 约 60，GPT-3 预训练约 3,640；对齐成本相对预训练很小。
- **局限（§5.3）：** 行为取决于约 40 名主要说英语的标注员的价值判断；模型仍会产生有害 / 虚构内容，且多数情况下会照做用户指令，即便可能造成现实伤害。

## 对 wiki 的映射

- [`wiki/concepts/supervised-fine-tuning.md`](../../wiki/concepts/supervised-fine-tuning.md)
- [`wiki/entities/paper-instructgpt-rlhf.md`](../../wiki/entities/paper-instructgpt-rlhf.md)

## 参考来源

- 论文：<https://arxiv.org/abs/2203.02155>
