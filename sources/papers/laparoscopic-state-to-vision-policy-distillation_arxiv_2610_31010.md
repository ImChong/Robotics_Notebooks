# Laparoscopic Distillation（arXiv:2610.31010）

> 来源归档（paper）

- **标题：** Revisiting Distribution Matching for State-to-Vision Policy Distillation in Laparoscopic Robotic Manipulation
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.31010>
- **PDF：** <https://arxiv.org/pdf/2610.31010>
- **作者：** Mingyu Sheng, Jianan Fan, Dongnan Liu, Guoyan Zheng, Ron Kikinis, Weidong Cai
- **arXiv 评论栏：** 8 pages, 3 figures, 2 tables

- **入库日期：** 2026-10-26
- **一句话说明：** 重新研究腹腔镜机器人操作中从状态到视觉的策略蒸馏分布匹配。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-10-26，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.19-10.23前沿论文动态】Manipulation 37篇](../blogs/wechat_senlanke_weekly_manipulation_2026-10-19_23.md)
2. **周报所述问题：** 使用完整状态训练的教师策略能够获得更多环境信息，但视觉学生策略只能根据相机观测执行动作。两者的输入信息和闭环状态分布存在差异。
3. **周报所述创新点：** 重新研究 State-to-Vision Policy Distillation 中的分布匹配问题，重点关注教师与学生之间的状态分布差异对蒸馏效果的影响。
4. **arXiv 摘要（英文节选）：** End-to-end control policies trained via Reinforcement Learning (RL) have emerged as an alternative to conventional surgical robotic control approaches, with recent studies exploring state-based policies trained on privileged state observations. However, such approaches rely on precise physical states that are difficult to obtain reliably in surgical environments. Vision-based RL provides an alternative by learning manipulation policies directly from RGB observations, avoiding explicit state estimation and manually engineered perception pipelines while enabling policies to acquire task-relevant visual representations.

**对 wiki 的映射**

- [paper-laparoscopic-state-to-vision-policy-distillation](../../wiki/entities/paper-laparoscopic-state-to-vision-policy-distillation.md)
