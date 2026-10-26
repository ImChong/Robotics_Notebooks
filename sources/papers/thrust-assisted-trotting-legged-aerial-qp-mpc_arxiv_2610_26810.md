# Thrust-Assisted Trotting（arXiv:2610.26810）

> 来源归档（paper）

- **标题：** Bio-inspired Thrust-Assisted Trotting on a Legged-Aerial Robot via Real-Time QP-MPC
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.26810>
- **PDF：** <https://arxiv.org/pdf/2610.26810>
- **作者：** Shreyansh Pitroda, Eric Sihite, Alireza Ramezani, Morteza Gharib
- **arXiv 评论栏：** To be published in IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2026)

- **入库日期：** 2026-10-26
- **一句话说明：** 基于实时 QP-MPC 的仿生推力辅助空地双足小跑。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-10-26，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.19-10.23前沿论文动态】人形／四足39篇](../blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-19_23.md)
2. **周报所述问题：** 空地双足机器人小跑时需要同时调节地面反力和推进器矢量，并满足摩擦锥和实时计算要求。
3. **周报所述创新点：** 将推力辅助弹簧倒立摆动力学凝聚成 500 Hz QP-MPC，矢状面由 MPC 控制、侧向由级联姿态控制器稳定；平均速度可跟踪指令且单次求解约 0.18 毫秒。
4. **arXiv 摘要（英文节选）：** This paper presents the hardware implementation of a quadratic program-based controller on Harpy: thruster-assisted bipedal robot to perform thruster-assisted trotting. This is inspired by wing-assisted locomotion observed in birds such as the chukar partridge, which employ thrust vectoring to manipulate the ground reaction force. The proposed controller formulates the thruster-assisted spring-loaded inverted pendulum dynamics as a condensed QP-based model predictive control problem, enabling simultaneous regulation of ground contact force and active thrust vectoring at 500 Hz.

**对 wiki 的映射**

- [paper-thrust-assisted-trotting-legged-aerial-qp-mpc](../../wiki/entities/paper-thrust-assisted-trotting-legged-aerial-qp-mpc.md)
