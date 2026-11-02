# FORGE（arXiv:2610.37974）

> 来源归档（paper）

- **标题：** FORGE: Feasible Online Reduced-Order Reference Generation and Whole-Body Execution for Agile Humanoid Locomotion
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.37974>
- **PDF：** <https://arxiv.org/pdf/2610.37974>
- **作者：** Ghansah, Adrian B., An, Yujin, Kim, Jeeseop, Chung, Soon-Jo
- **arXiv 评论栏：** （无）

- **入库日期：** 2026-11-02
- **一句话说明：** FORGE：面向敏捷人形运动的在线可行降阶参考生成与全身执行。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.26-10.30前沿论文动态】人形／四足33篇](../blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md)
2. **周报所述问题：** 降阶动力学模型可快速生成动作，但生成的参考不一定满足完整人形机器人的运动学和执行器约束。
3. **周报所述创新点：** 先在单刚体动力学和完整运动学组成的降阶环境中训练参考生成器，再用其在线输出训练全阶跟踪策略；单一系统覆盖前后、侧向、转向、单腿跳和连续技能转换。
4. **arXiv 摘要（英文节选）：** This paper presents a two-stage learning framework for versatile and agile humanoid locomotion that couples online reduced-order model (ROM) reference generation with whole-body tracking control. In the first stage, we train a command-conditioned whole-body reference generator using single rigid body dynamics combined with full humanoid kinematics. In the second stage, the frozen ROM policy is rolled out online to generate references for training a whole-body tracking policy on the full-order dynamics model.

**对 wiki 的映射**

- [paper-forge-humanoid-reduced-order-reference](../../wiki/entities/paper-forge-humanoid-reduced-order-reference.md)
