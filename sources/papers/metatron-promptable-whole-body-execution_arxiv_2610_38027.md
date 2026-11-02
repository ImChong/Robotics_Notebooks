# METATRON（arXiv:2610.38027）

> 来源归档（paper）

- **标题：** METATRON: A Promptable Whole-Body Execution Layer for Humanoids
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.38027>
- **PDF：** <https://arxiv.org/pdf/2610.38027>
- **作者：** Tessler, Chen, Jiang, Yifeng, Otto, Alec, Shi, Yi, Iqbal, Umar, Chechik, Gal, Peng, Xue Bin
- **arXiv 评论栏：** （无）

- **入库日期：** 2026-11-02
- **一句话说明：** METATRON：可提示的人形全身动作执行层。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.26-10.30前沿论文动态】人形／四足33篇](../blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md)
2. **周报所述问题：** 不同人形任务通常分别训练控制策略，重复学习平衡和全身协调能力。
3. **周报所述创新点：** 冻结一个接收身体部位目标、速度和风格提示的生成式基础策略；高层技能以有状态程序逐周期生成提示，使编码智能体能够通过修改程序而不是重新训练策略来构建新技能。
4. **arXiv 摘要（英文节选）：** Humanoid controllers often require task-specific training despite sharing the need for balance and whole-body coordination. We present Metatron, a promptable whole-body execution layer combining a generative base policy with a programmable skill layer. The base policy receives partial goals, collectively called a prompt, that can combine timed body-part targets, velocity commands, and style cues.

**对 wiki 的映射**

- [paper-metatron-promptable-whole-body-execution](../../wiki/entities/paper-metatron-promptable-whole-body-execution.md)
