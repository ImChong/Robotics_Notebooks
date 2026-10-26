# Fast Retargeting（arXiv:2610.28470）

> 来源归档（paper）

- **标题：** Fast Motion Retargeting Pipeline for Scalable Humanoid Robot Learning
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.28470>
- **PDF：** <https://arxiv.org/pdf/2610.28470>
- **作者：** Daniel Marew, Sangjun Park, Nisal Perera, Donghyun Kim
- **arXiv 评论栏：** （无）
- **外链：** <https://daroslab.github.io/frdv/>
- **入库日期：** 2026-10-26
- **一句话说明：** 面向规模化人形学习的快速动作重定向流程。

## 开源状态

- arXiv 摘要页/评论给出项目页或外链（https://daroslab.github.io/frdv/）；未发现代码仓库链接（2026-10-26，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.19-10.23前沿论文动态】人形／四足39篇](../blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-19_23.md)
2. **周报所述问题：** 将大型人体动作库转移到不同人形机器人时，传统轨迹优化速度慢，还需要逐动作人工修复。
3. **周报所述创新点：** 结合定制 SQP、实时迭代和分块并行，将速度提升至最高每秒 9156 帧；再用接触隐式 iLQR 筛除或修正动力学不可执行、悬脚和滑脚动作。
4. **arXiv 摘要（英文节选）：** While large-scale human motion capture datasets provide a rich source of demonstrations for humanoid robot learning, retargeting the diverse motions to different humanoid robots remains a nontrivial challenge. Existing kinodynamic trajectory optimization pipelines are computationally expensive and often require manual adjustments per motion. We present FRDV (Fast Retargeting with Dynamics Validation), a scalable framework consisting of rapid constrained kinematic retargeting and an additional dynamics validation pipeline.

**对 wiki 的映射**

- [paper-fast-motion-retargeting-pipeline-humanoid](../../wiki/entities/paper-fast-motion-retargeting-pipeline-humanoid.md)
