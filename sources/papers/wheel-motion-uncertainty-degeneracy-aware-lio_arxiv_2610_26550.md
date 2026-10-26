# Wheel Uncertainty LIO（arXiv:2610.26550）

> 来源归档（paper）

- **标题：** Learning Wheel Motion Uncertainty for Degeneracy-Aware LiDAR-Inertial Odometry on Quadrupedal Wheel-Legged Robots
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.26550>
- **PDF：** <https://arxiv.org/pdf/2610.26550>
- **作者：** Hairuo Wei, Yunfan Ren, Renda Yang, Qiangsheng Huang, Zhongyu Li
- **arXiv 评论栏：** （无）
- **外链：** <https://weihairuo.github.io/wheel-motion-uncertainty/>
- **入库日期：** 2026-10-26
- **一句话说明：** 面向四足轮腿机器人的轮运动不确定性学习与退化感知激光—惯性里程计。

## 开源状态

- arXiv 摘要页/评论给出项目页或外链（https://weihairuo.github.io/wheel-motion-uncertainty/）；未发现代码仓库链接（2026-10-26，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.19-10.23前沿论文动态】人形／四足39篇](../blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-19_23.md)
2. **周报所述问题：** 走廊等几何退化环境需要轮运动提供额外约束，但打滑、悬空会使轮速预测失真，二值接触状态不能反映不同方向上的可靠性。
3. **周报所述创新点：** 从本体历史为每个轮子学习方向相关过程协方差，在可微滤波器中训练；同时抑制激光雷达几何弱方向的约束，使轮运动和激光约束根据各自退化程度自适应加权。
4. **arXiv 摘要（英文节选）：** Wheel kinematics can support odometry for quadrupedal wheel-legged robots when LiDAR geometry provides weak motion constraints. However, slip and contact loss make rolling predictions unreliable, and contact state alone cannot determine their reliability across motion directions. We propose a tightly coupled LiDAR-inertial-kinematic odometry framework that retains analytical rolling predictions and learns per-wheel, direction-dependent process covariances from proprioceptive history.

**对 wiki 的映射**

- [paper-wheel-motion-uncertainty-degeneracy-aware-lio](../../wiki/entities/paper-wheel-motion-uncertainty-degeneracy-aware-lio.md)
