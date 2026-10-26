# Traversal Tubes（arXiv:2610.27688）

> 来源归档（paper）

- **标题：** Adaptive Traversal Tubes for Posture-Aware Navigation of Bipedal Humanoids
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.27688>
- **PDF：** <https://arxiv.org/pdf/2610.27688>
- **作者：** Zijian Ge
- **arXiv 评论栏：** （无）
- **外链：** <http://github.com/lborogzj997/Humanoid-Navigation-in-Crowds-and-Narrow-Spaces>
- **入库日期：** 2026-10-26
- **一句话说明：** 面向双足人形姿态感知导航的自适应通行管道。

## 开源状态

- arXiv 摘要页/评论给出 GitHub 链接（http://github.com/lborogzj997/Humanoid-Navigation-in-Crowds-and-Narrow-Spaces；仅核对到链接存在，未核验仓库内容）（2026-10-26，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.19-10.23前沿论文动态】人形／四足39篇](../blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-19_23.md)
2. **周报所述问题：** 传统导航只规划路径和速度，无法主动利用侧身姿态穿越窄通道或为行人让路。
3. **周报所述创新点：** 用沿路径变化的通行宽度统一表示几何净空和社交距离，再将其映射成正向—侧向行走姿态及全向速度；配套训练侧步增强的 AMP 运动先验。当前版本的定量评测仍待补充。
4. **arXiv 摘要（英文节选）：** Bipedal humanoids can change their effective footprint by adjusting body yaw, enabling both narrow-passage traversal and socially aware yielding. This paper presents a posture-aware navigation framework centered on \emph{Adaptive Traversal Tubes} (ATT), \emph{Interval Velocity--Heading} (IVH) control, and a sidestep-enhanced AMP locomotion prior. ATT represents a behavior-feasible corridor width field along a reference path, unifying geometric clearance and social comfort into a continuous signal.

**对 wiki 的映射**

- [paper-adaptive-traversal-tubes-bipedal-navigation](../../wiki/entities/paper-adaptive-traversal-tubes-bipedal-navigation.md)
