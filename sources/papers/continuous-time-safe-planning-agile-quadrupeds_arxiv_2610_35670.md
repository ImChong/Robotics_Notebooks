# 连续时间安全规划控制（arXiv:2610.35670）

> 来源归档（paper）

- **标题：** Continuous-Time Safe Planning and Control for Agile Quadrupeds
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.35670>
- **PDF：** <https://arxiv.org/pdf/2610.35670>
- **作者：** MacLin, Gage, Peterson, Mason, Tabasso, Camilla, Cichella, Venanzio, Migdal, Joshua, Brady, Tye, Padir, Taskin
- **arXiv 评论栏：** （无）

- **入库日期：** 2026-11-02
- **一句话说明：** 面向敏捷四足机器人的连续时间安全规划与控制。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.26-10.30前沿论文动态】人形／四足33篇](../blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md)
2. **周报所述问题：** 离散时刻避碰不能排除两个规划点之间的碰撞，简化为单轮车模型又会丢失四足机器人的横向运动能力。
3. **周报所述创新点：** 保留带朝向耦合速度边界的完整 SE(2) 模型，以分段 Bernstein 多项式表示轨迹；利用凸包性质在整个连续时间区间上验证避碰和控制边界。
4. **arXiv 摘要（英文节选）：** Agile quadruped navigation in dynamic environments demands trajectories that are both dynamically feasible and provably collision-free. Most real-time planners achieve tractability through a convex reduction that requires simplified kinematic models, discarding motion that the robot can readily execute, and certify safety only at discretization points, leaving open possible inter-sample collisions with moving obstacles. We present a real-time motion planning framework that retains the fully-actuated $SE(2)$ model of a quadruped, with its anisotropic, heading-coupled velocity bounds, and accepts the resulting nonconvexity.

**对 wiki 的映射**

- [paper-continuous-time-safe-planning-agile-quadrupeds](../../wiki/entities/paper-continuous-time-safe-planning-agile-quadrupeds.md)
