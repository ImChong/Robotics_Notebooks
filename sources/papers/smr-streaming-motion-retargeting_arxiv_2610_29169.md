# SMR（arXiv:2610.29169）

> 来源归档（paper）

- **标题：** SMR: Learning Streaming Motion Retargeting for Humanoids from Human Semantic
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.29169>
- **PDF：** <https://arxiv.org/pdf/2610.29169>
- **作者：** Chun Gu, Juntong Zhang, Jingyu Li, Li Zhang
- **arXiv 评论栏：** 17 pages, 6 figures. Project page: this https URL
- **外链：** <https://logosintelligencelab.github.io/SMR/>
- **入库日期：** 2026-10-26
- **一句话说明：** SMR：从人体动作语义学习流式人形动作重定向。

## 开源状态

- arXiv 摘要页/评论给出项目页或外链（https://logosintelligencelab.github.io/SMR/）；未发现代码仓库链接（2026-10-26，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.19-10.23前沿论文动态】人形／四足39篇](../blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-19_23.md)
2. **周报所述问题：** 几何重定向不保证动力学可执行，而双向全序列网络无法用于实时流式输入。
3. **周报所述创新点：** 先训练物理专家跟踪机器人感知的任务空间语义目标，再将专家实际执行轨迹蒸馏到严格因果 Transformer；部署时只使用当前和历史人体观测生成动作参考。
4. **arXiv 摘要（英文节选）：** Retargeting human motion to humanoids requires preserving motion semantics while accounting for the robot's physical constraints. Kinematic pipelines prescribe a robot trajectory through geometric matching, without establishing whether it can be executed under the robot's dynamics. Prior physics-supervised neural methods train bidirectional full-sequence models on physics-refined kinematic trajectories, remaining tied to prescribed kinematic solutions and requiring noncausal context.

**对 wiki 的映射**

- [paper-smr-streaming-motion-retargeting](../../wiki/entities/paper-smr-streaming-motion-retargeting.md)
