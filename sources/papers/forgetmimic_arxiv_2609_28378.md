# ForgetMimic: Motion Unlearning for Reinforcement Learning Humanoid Control

> 来源：[具身智能小站 · 13 篇盘点](../../sources/blogs/wechat_embodied_13_papers_forgetmimic_2026-09-24.md)（2026-09-24）

## 元数据

- **arXiv：** [2609.28378](https://arxiv.org/abs/2609.28378)
- **PDF：** https://arxiv.org/pdf/2609.28378
- **代码：** https://github.com/Zili1000/ForgetMimic
- **开源结论（2026-09-24）：** **已开源**

## 核心摘录

- **一句话：** 多技能人形策略可在 **动作级** 选择性遗忘指定 motion，而不必整策略重训。
- **机制：** 对目标动作施加 anti-reward，并关闭会补偿失败的训练机制；保留集 motion 的 tracking reward 与成功率维持。
- **指标：** Fight / FightAndSports1 成功率 **100%→0%**；非目标动作平均 tracking reward 降 **<2%**，成功率仍 **>85%**（G1、H2 等）。

## 对 wiki 的映射

- 实体页：[ForgetMimic](../../wiki/entities/paper-forgetmimic.md)
