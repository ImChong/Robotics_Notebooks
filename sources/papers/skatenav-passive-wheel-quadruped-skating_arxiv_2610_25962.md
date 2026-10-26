# SkateNav（arXiv:2610.25962）

> 来源归档（paper）

- **标题：** SkateNav: Learning Energy-Efficient Navigation through Adaptive Skating on Passive-Wheel Quadrupedal Robots
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.25962>
- **PDF：** <https://arxiv.org/pdf/2610.25962>
- **作者：** Jeil Jeong, Taeyun Kim, Sangjin Kim, Jinyoung Choi, Sung-Eui Yoon
- **arXiv 评论栏：** 8 pages, 12 figures, 3 tables. Project page: this https URL
- **外链：** <https://ktw1404.github.io/SkateNav/>
- **入库日期：** 2026-10-26
- **一句话说明：** SkateNav：通过自适应滑行学习被动轮四足机器人的高能效导航。

## 开源状态

- arXiv 摘要页/评论给出项目页或外链（https://ktw1404.github.io/SkateNav/）；未发现代码仓库链接（2026-10-26，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.19-10.23前沿论文动态】人形／四足39篇](../blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-19_23.md)
2. **周报所述问题：** 速度跟踪奖励会抑制滑行动作固有的速度起伏，使被动轮机器人频繁蹬地而无法充分利用惯性。
3. **周报所述创新点：** 把导航改写为带时间预算的到达目标任务，使策略联合决定转向、蹬地和滑行时长；通过可逐控制周期调频的滑行动作循环适应环境，并零样本迁移到带被动轮的 Mini Cheetah。
4. **arXiv 摘要（英文节选）：** Passive-wheel quadrupedal robots offer energy-efficient locomotion through leg-driven propulsion and momentum-driven gliding. However, navigation requires coordinating steering with intermittent propulsion and adapting push-glide timing to task demands. Velocity-tracking objectives may suppress the speed variations inherent in skating, discouraging efficient gliding.

**对 wiki 的映射**

- [paper-skatenav-passive-wheel-quadruped-skating](../../wiki/entities/paper-skatenav-passive-wheel-quadruped-skating.md)
