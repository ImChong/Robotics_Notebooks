# Robot Hands for Modern AI and Real Work

> 来源归档（blog / Boston Dynamics 官方）

- **标题：** Robot Hands for Modern AI and Real Work
- **类型：** blog
- **作者：** Boston Dynamics（官方团队）
- **原始链接：** https://bostondynamics.com/blog/robot-hands-for-modern-ai-and-real-work/
- **入库日期：** 2026-10-08
- **一句话说明：** 介绍新一代 Atlas 13-DoF 直接驱动灵巧手如何在操作能力、力量、耐用、成本、维修和感知间取舍，并将仿真保真度作为 sim-to-real 强化学习的硬件设计目标。

## 核心摘录（归纳，非全文）

### 设计目标与结构
- 灵巧手的目标不仅是模仿人手；还需支持工具操作、适应工业环境、缩小人类演示到机器人本体的差距、可被可靠仿真，并能以低成本制造和维修。
- 官方介绍的新一代 Atlas 手为 **13 DoF、直接驱动**；演示目标包括精细捏取、三点抓握、物体重定位，以及持工具并按动触发器。
- 手部采用四指布局。团队没有采用小指，解释为新增的三个自由度和执行器带来尺寸、功耗与复杂度，而当前预期操作收益不足以抵偿成本。
- 设计决策结合仿真、3D 打印样机和实物试用，而非只依赖单一评估方法。

### 仿真与交互
- 官方将 sim-to-real 匹配拆为运动学（几何、连杆和关节）、动力学（摩擦、力矩、回差）和接触动力学（手与环境如何相互作用）。
- 可回驱性让环境施加在手指上的力、以及手指输出的运动，更直接反映在机器人与物体的交互中；碰撞时的顺应也有助于降低对齿轮箱的冲击。
- 官方将可仿真性视为使用强化学习训练手部操作、并向工具使用和装配任务扩展的基础。

## 对 wiki 的映射
- [Boston Dynamics Atlas 13-DoF 灵巧手](../../wiki/entities/boston-dynamics-atlas-13dof-hand.md)（统一节点）
- [Boston Dynamics](../../wiki/entities/boston-dynamics.md)（Atlas 产品线入口）

## 可信度与使用边界
- 这是厂商官方设计介绍，不是同行评审论文；任务视频与工程论述属于厂商自述。
- 文章没有可复核的量化成功率、基线对照或消融实验，适合作为设计动机和系统架构参考，不宜作为性能基准。
- 该文章未列手部 CAD、仿真资产、训练代码或数据集链接；这仅说明本文没有提供这些复现材料，不据此断言其他渠道不存在。

## Citation

```bibtex
@misc{bostondynamics_robot_hands_modern_ai,
  author = {{Boston Dynamics}},
  title = {Robot Hands for Modern AI and Real Work},
  howpublished = {Boston Dynamics Blog},
  url = {https://bostondynamics.com/blog/robot-hands-for-modern-ai-and-real-work/},
  note = {Accessed 2026-10-08}
}
```
