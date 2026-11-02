# 钩子拆卸力驱动 RL（arXiv:2610.38513）

> 来源归档（paper）

- **标题：** Force-Driven Hook Disassembly via Constraint-Aware Object-Centric Reinforcement Learning
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.38513>
- **PDF：** <https://arxiv.org/pdf/2610.38513>
- **作者：** Elhadad, Nadeen, Kiyokawa, Takuya, Harada, Kensuke
- **arXiv 评论栏：** （无）

- **入库日期：** 2026-11-02
- **一句话说明：** 通过约束感知的物体中心强化学习实现力驱动卡扣拆解。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.26-10.30前沿论文动态】Manipulation 57篇](../blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md)
2. **周报所述问题：** 工业卡扣连接通常被外壳遮挡，机器人难以仅根据视觉判断正确的拆解方向。
3. **周报所述创新点：** 使用物体中心状态表示和粒子化信念状态估计隐藏约束，再通过强化学习选择拆解动作。方法能够处理单轴和多阶段拆解，并在真实废旧空调部件拆解任务中验证。
4. **arXiv 摘要（英文节选）：** The growing volume of end-of-life products has intensified the need for automated disassembly, yet hook-type connections remain challengin: their orientations vary, they deform under load, and their geometry is hidden, so vision alone cannot identify a feasible non-destructive disassembly direction. This work proposes a constraint-aware reinforcement learning framework that infers feasible extraction directions online from contact forces. The policy is trained in a robot-agnostic, object-centric simulation on a simplified peg-in-hole pair and deployed zero-shot on a physical robot, where it transfers to real connections with unseen, multi-stage constraint structures.

**对 wiki 的映射**

- [paper-force-driven-hook-disassembly-object-centric-rl](../../wiki/entities/paper-force-driven-hook-disassembly-object-centric-rl.md)
