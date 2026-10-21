# RoboFFT: Finetuning Generative Robot Policy via Online Reinforcement Learning with Forward Process（arXiv:2609.32236）

> 来源归档（ingest）

- **标题：** RoboFFT: Finetuning Generative Robot Policy via Online Reinforcement Learning with Forward Process
- **类型：** paper
- **原始链接：**
  - <https://arxiv.org/abs/2609.32236>（2026-09-26）
  - PDF：<https://arxiv.org/pdf/2609.32236>
  - 项目页：<https://student-of-holmes.github.io/RoboFFT/>
  - 代码：<https://github.com/shshsunny/RoboFFT>
  - Hugging Face：<https://huggingface.co/shshsunny/RoboFFT>
  - ModelScope：<https://modelscope.cn/models/shshsunny/RoboFFT/tree/master>
- **作者：** Yu Li、Shenghe Hu、Yuhan Wang、Yaoxiang Pu、Haotong Zhang、Yuanpei Chen、Yaodong Yang
- **机构：** 北京大学人工智能研究院、PKU-PsiBot 联合实验室、南京大学（项目页作者标注）
- **会议：** CoRL 2026（项目页标注录用）
- **入库日期：** 2026-10-16
- **一句话说明：** 面向扩散与 flow 生成式机器人策略的在线 RL 微调方法：前向加噪动作样本，以加权去噪/flow-matching 损失差构造 PPO surrogate ratio，避开把完整逆生成链建模成长时序决策过程。

## 核心论文摘录（MVP）

### 1) 问题：生成策略的似然难算

- **链接：** <https://arxiv.org/abs/2609.32236> 摘要
- **摘录要点：** 扩散与 flow 策略可从示范表达多模态动作分布，但普通 PPO 需要新旧策略概率比；对多步生成策略直接算动作似然通常不可 tractable。仅靠模仿学习也难消除次优示范与分布偏移。
- **对 wiki 的映射：** [RoboFFT 实体页](../../wiki/entities/paper-robofft.md)、[Diffusion Policy](../../wiki/methods/diffusion-policy.md)

### 2) 方法：前向过程构造策略比率

- **链接：** [项目页 Method](https://student-of-holmes.github.io/RoboFFT/#method)
- **摘录要点：** 对 rollout 采得的干净动作块按训练时相同的前向过程加噪；用加权 score/flow-matching loss 近似负对数似然。新旧策略的损失差指数化后成为 PPO 的 surrogate ratio，并结合 advantage 与 clip 目标更新。
- **对 wiki 的映射：** [RoboFFT 实体页](../../wiki/entities/paper-robofft.md)、[PPO](../../wiki/methods/ppo.md)

### 3) 统一扩散与 flow 策略

- **链接：** [项目页 Method](https://student-of-holmes.github.io/RoboFFT/#method)
- **摘录要点：** RoboFFT-F 用 flow-matching loss；RoboFFT-D 用按相邻扩散步 log-SNR 变化加权的噪声预测误差。论文还报告在 VLA 的 flow 动作头上微调，以及将方法接入真实机器人 RL 流程。
- **对 wiki 的映射：** [RoboFFT 实体页](../../wiki/entities/paper-robofft.md)、[Diffusion Policy](../../wiki/methods/diffusion-policy.md)

### 4) 评测与可复现边界

- **链接：** [项目页 Results](https://student-of-holmes.github.io/RoboFFT/#results)、[代码仓库 README](https://github.com/shshsunny/RoboFFT)
- **摘录要点：** 项目页报告 Robomimic 长时序/稀疏奖励任务上的改进，并显示训练迭代速度相较 ReinFlow、DPPO 提升；真实机器人表格报告四项任务从预训练到在线 RoboFFT 后的成功次数上升。当前 release 仓库覆盖 Robomimic flow 与 diffusion 仿真代码；README 明确论文里的 VLA 与真机实验未包含在此 release。
- **对 wiki 的映射：** [RoboFFT 实体页](../../wiki/entities/paper-robofft.md)

## 对 wiki 的映射（汇总）

- [`wiki/entities/paper-robofft.md`](../../wiki/entities/paper-robofft.md) — 论文、项目页与代码统一实体
- [`wiki/methods/diffusion-policy.md`](../../wiki/methods/diffusion-policy.md) — 扩散策略与前向加噪背景
- [`wiki/methods/ppo.md`](../../wiki/methods/ppo.md) — surrogate ratio 与 PPO clipping
