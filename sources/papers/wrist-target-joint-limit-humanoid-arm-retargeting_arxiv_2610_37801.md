# 腕目标与关节限位实证（arXiv:2610.37801）

> 来源归档（paper）

- **标题：** An Empirical Study of Wrist-Target Construction and Joint-Angle Limits in Humanoid Arm Retargeting
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.37801>
- **PDF：** <https://arxiv.org/pdf/2610.37801>
- **作者：** Chang, MinJun
- **arXiv 评论栏：** （无）

- **入库日期：** 2026-11-02
- **一句话说明：** 人形手臂重定向中腕部目标构造与关节限位的实证研究。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.26-10.30前沿论文动态】人形／四足33篇](../blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md)
2. **周报所述问题：** 人体腕部目标直接映射到机器人时，可能频繁把手臂关节推到限位，并增加跟踪误差和执行力矩。
3. **周报所述创新点：** 对比按人体肢段方向重建的腕部目标与按机器人可达距离归一化的目标；实验证明距离归一化能显著降低 Unitree H1-2 关节触限次数。
4. **arXiv 摘要（英文节选）：** Motion retargeting maps a demonstrated human movement onto a robot to compute its joint motion. This study examines how the construction of the desired wrist position affects whether computed arm joint angles reach their permitted limits on a Unitree H1-2. A reconstructed target is formed from demonstrated arm directions and robot segment lengths, so its shoulder-to-wrist distance depends on the demonstrated angle between the segments even when no posture-matching preference is applied.

**对 wiki 的映射**

- [paper-wrist-target-joint-limit-humanoid-arm-retargeting](../../wiki/entities/paper-wrist-target-joint-limit-humanoid-arm-retargeting.md)
