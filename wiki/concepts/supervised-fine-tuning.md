---
type: concept
tags:
  - post-training
  - fine-tuning
  - imitation-learning
  - vla
  - llm
status: complete
updated: 2026-11-06
summary: "监督微调（SFT）：在预训练模型上用带标注的「输入 → 目标输出」样本做最大似然训练，把通用模型适配到目标任务分布；在 LLM 中是 InstructGPT 式「SFT → 奖励模型 → RL」的第一段，在机器人 VLA 中通常就是在示范数据上做行为克隆，之后常接 RL 后训练。"
related:
  - ../entities/paper-instructgpt-rlhf.md
  - ./universal-post-training-robotics.md
  - ../methods/behavior-cloning.md
  - ../methods/dagger.md
  - ../methods/vla.md
  - ../overview/hub-embodied-foundation-model.md
  - ../methods/reinforcement-learning.md
  - ../entities/qwen-vla.md
  - ../methods/hy-motion-1.md
  - ../entities/philia.md
  - ../entities/paper-alloego-vlm.md
  - ../entities/paper-notebook-commanding-humanoid-by-free-form-language.md
original:
  - ../../sources/papers/instructgpt_sft_reading_arxiv_2203_02155.md
sources:
  - ../../sources/papers/instructgpt_sft_reading_arxiv_2203_02155.md
  - ../../sources/papers/instructgpt_arxiv_2203_02155.md
  - ../../sources/courses/karpathy_deep_dive_llms_youtube.md
  - ../../sources/papers/qwenvla_arxiv_2605_30280.md
---

# 监督微调（Supervised Fine-Tuning, SFT）

## 一句话定义

**监督微调（SFT）是在已预训练的模型上，用带标注的「输入 → 目标输出」样本做最大似然训练，把通用能力收窄到目标任务分布的后训练步骤；在机器人里，目标输出通常是示范动作，此时 SFT 与行为克隆是同一件事。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SFT | Supervised Fine-Tuning | 监督微调：在标注样本上做最大似然微调 |
| RLHF | Reinforcement Learning from Human Feedback | 基于人类偏好反馈的强化学习，常接在 SFT 之后 |
| CPT | Continued Pre-Training | 持续预训练：在新领域数据上继续预训练，常在 SFT 之前 |
| BC | Behavior Cloning | 行为克隆：在（观测, 动作）示范对上做监督学习 |
| VLA | Vision-Language-Action | 视觉–语言–动作模型 |
| GRPO | Group Relative Policy Optimization | 组相对策略优化，常见的 SFT 后 RL 算法之一 |

## 为什么重要

- **它是后训练链路的固定第一段。** InstructGPT 把对齐管线定为「SFT 示范 → 训练奖励模型 → PPO 强化学习」三段（[InstructGPT](../entities/paper-instructgpt-rlhf.md)）；机器人基础模型沿用了同一结构，[通用后训练](./universal-post-training-robotics.md) 一页把它类比为 LLM 的「SFT → RLHF → RLVR」收敛路径。
- **VLA 论文的训练配方几乎都含这一步。** 例如 [Qwen-VLA](../entities/qwen-vla.md) 的配方是「动作预训练 → 多模态持续预训练（CPT）→ SFT → RL」；[Philia](../entities/philia.md) 描述「SFT → rollout 干预 → advantage-conditioned 后训练」的部署闭环。
- **读论文时要知道它学到了什么、学不到什么。** SFT 只能逼近示范分布：示范没覆盖的状态，模型不会自己学会恢复——这正是很多工作在 SFT 之后再接 RL 或交互式纠错的原因。

## 核心原理

**目标函数。** 给定预训练参数 \(\theta_0\) 与标注数据集 \(\mathcal D=\{(x_i, y_i)\}\)，SFT 从 \(\theta_0\) 出发最小化负对数似然：

$$
\mathcal L_{\text{SFT}}(\theta) = -\frac{1}{|\mathcal D|}\sum_{(x,y)\in\mathcal D}\log p_\theta(y\mid x)
$$

- \(x\)：输入。LLM 中是提示词；VLA 中是图像观测 + 语言指令（+ 本体状态）。
- \(y\)：目标输出。LLM 中是示范回答；VLA 中是示范动作（离散动作 token，或连续动作块）。
- 对自回归 token 输出，\(\log p_\theta(y\mid x)\) 拆成逐 token 的交叉熵之和（next-token 预测）；对扩散 / 流匹配动作头，似然项换成对应的去噪或速度回归损失（本库归纳：损失形式随动作头而变，"监督"的本质不变）。

**与行为克隆的关系（本库归纳）。** 当 \(y\) 是示范动作时，上式就是 [行为克隆](../methods/behavior-cloning.md) 的目标；两个名字来自不同社区——LLM 社区强调"在预训练模型上微调"，机器人模仿学习社区强调"从示范学策略"。因此 SFT 继承了行为克隆的已知弱点：训练分布与执行分布不一致时误差会累积（见 [DAgger](../methods/dagger.md)）。

**在训练链路中的位置。**

```mermaid
flowchart LR
  PT["预训练<br/>（大规模通用数据）"] --> CPT["持续预训练 CPT<br/>（可选：领域数据）"]
  CPT --> SFT["SFT<br/>（标注 / 示范数据）"]
  SFT --> RL["RL 后训练<br/>（RLHF / GRPO / 真机 RL）"]
  RL --> DEP["部署"]
```

## 工程实践

库内几个 SFT 的具体用法，可以看出"标注数据"在机器人里的不同形态：

| 页面 | SFT 的数据 | SFT 之后 |
|------|-----------|----------|
| [Qwen-VLA](../entities/qwen-vla.md) | 操作、导航与轨迹预测数据，衔接离散 VLM token 与连续控制轨迹 | RL |
| [Philia](../entities/philia.md) | 灵巧 / 长程任务的专家示范 | rollout 干预 + advantage-conditioned 后训练 |
| [HY-Motion 1.0](../methods/hy-motion-1.md) | 「随意用户 prompt → 优化 prompt + 时长」三元组（prompt 改写模型） | GRPO，用更强模型作裁判 |
| [自由语言指挥人形](../entities/paper-notebook-commanding-humanoid-by-free-form-language.md) | 「CoT + 动作 token」的 next-token 预测，使用全部增强后的人类语料 | RL 只用机器人可执行子集 |
| [AlloEgo-VLM](../entities/paper-alloego-vlm.md) | AlloEgo-View 参照系消歧数据 | — |

**读配方时的检查点：**
- 数据从哪来：真机遥操作、仿真、人类视频还是模型生成？生成数据会把生成模型的偏差一起学进去。
- 冻结了什么：全参数微调，还是只训动作头 / LoRA？这决定预训练能力保留多少。
- 和下一阶段怎么分工：最后一行的例子里，SFT 用全部人类语料，RL 只用机器人可执行的子集——SFT 负责覆盖面，RL 负责可执行性。

**InstructGPT 原文中的 SFT 要点**（[通读归档](../../sources/papers/instructgpt_sft_reading_arxiv_2203_02155.md)）：SFT 数据约 13k 条 prompt；训练 16 epoch，但验证 loss 1 epoch 后就过拟合，作者按奖励模型得分选模型，因为多训几轮仍提升人类偏好；作为 RL 初始化的 SFT 只训 2 epoch 并混入 10% 预训练数据。效果上呈台阶：GPT-3 < few-shot GPT-3 < SFT < PPO。

## 局限与风险

- **适用边界：** SFT 学的是示范分布的条件分布，质量上限受示范质量约束；示范有偏或不一致时，模型会学到平均化或多峰混淆的动作。
- **分布偏移：** 闭环执行时，策略自己的误差会把它带到示范没覆盖的状态，误差逐步累积；需要 DAgger 式交互纠错、RL 后训练或更宽的数据覆盖来缓解。
- **SFT 自身的短板（InstructGPT 原文）：** SFT 基线毒性最低但连贯性最低、最不受人类偏好，作者推测输出偏短或退化；在 FLAN / T0 等公开 NLP 数据集上微调也不如自采的多样示范，说明数据分布覆盖比数量更关键。
- **遗忘风险：** 在窄任务上过度微调可能削弱预训练获得的通用能力（语言理解、跨任务泛化），常见对策是混入通用数据或只训部分参数（本库归纳）。

## 关联页面

- [InstructGPT（RLHF 原典）](../entities/paper-instructgpt-rlhf.md) — SFT → 奖励模型 → PPO 三段管线的出处
- [通用后训练（Universal Post-Training）](./universal-post-training-robotics.md) — 机器人基础模型的后训练栈，SFT 是其第一段
- [行为克隆](../methods/behavior-cloning.md) — 动作空间上与 SFT 同形的模仿学习方法
- [DAgger](../methods/dagger.md) — 缓解 SFT / 行为克隆分布偏移的交互式数据收集
- [VLA](../methods/vla.md) — SFT 最常见的机器人应用对象
- [强化学习](../methods/reinforcement-learning.md) — SFT 之后的 RL 后训练
- [具身大模型分类学选型闭环（知识链枢纽）](../overview/hub-embodied-foundation-model.md) — VLM / VLA / 世界模型各家族的训练配方里，SFT 都是适配目标任务的常见一步

## 参考来源

- [InstructGPT 论文归档](../../sources/papers/instructgpt_arxiv_2203_02155.md) — 「SFT → 奖励模型 → PPO」管线
- [InstructGPT 通读归档（SFT 视角）](../../sources/papers/instructgpt_sft_reading_arxiv_2203_02155.md) — 数据规模、训练超参、过拟合与效果阶梯的原文核对
- [Karpathy《Deep Dive into LLMs》课程归档](../../sources/courses/karpathy_deep_dive_llms_youtube.md) — 预训练 → 后训练（对话 SFT）→ RL 的完整训练栈讲解
- [Qwen-VLA 论文归档](../../sources/papers/qwenvla_arxiv_2605_30280.md) — 机器人 VLA 中「CPT → SFT → RL」配方实例

## 推荐继续阅读

- Ouyang et al., *Training language models to follow instructions with human feedback*, arXiv:2203.02155, 2022. <https://arxiv.org/abs/2203.02155>
