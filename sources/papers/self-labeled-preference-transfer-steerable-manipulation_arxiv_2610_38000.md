# SPT 自标注偏好迁移（arXiv:2610.38000）

> 来源归档（paper）

- **标题：** Self-Labeled Preference Transfer for Steerable Robot Manipulation
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.38000>
- **PDF：** <https://arxiv.org/pdf/2610.38000>
- **作者：** Hong, Kaiwen, Peng, Shaoting, Zeng, Xiangyu, Wang, Pengcheng, Tian, Boyuan, Chen, Haonan, Zhu, Yifan, Wang, Yixiao et al.
- **arXiv 评论栏：** （无）

- **入库日期：** 2026-11-02
- **一句话说明：** 面向可控机器人操作的自标注偏好迁移。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.26-10.30前沿论文动态】Manipulation 57篇](../blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md)
2. **周报所述问题：** VLA 可以从标注示范中学习抓取位置、绕障路径等执行偏好，但迁移到新任务后，普通微调容易提高任务成功率，却破坏原有偏好控制能力。
3. **周报所述创新点：** 提出 Self-Labeled Preference Transfer（SPT）。首先联合训练偏好识别器和偏好条件策略，然后冻结识别器，为新任务的未标注示范自动生成偏好标签。在真实机器人任务迁移中，偏好遵循率由 **52.5% 提升至 92.5%**。
4. **arXiv 摘要（英文节选）：** Vision-language-action (VLA) policies can learn to follow execution preferences, such as where to grasp an object or how to move around an obstacle, from demonstrations annotated with preference labels. Providing these annotations for every new task, however, is costly. We study whether learned preference control can transfer to new tasks whose demonstrations contain the desired behaviors but lack preference annotations.

**对 wiki 的映射**

- [paper-self-labeled-preference-transfer-steerable-manipulation](../../wiki/entities/paper-self-labeled-preference-transfer-steerable-manipulation.md)
