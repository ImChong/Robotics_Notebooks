# Embodied Human Simulation for Quantitative Design and Analysis of Interactive Robotics（arXiv:2603.09218）

> 来源归档（ingest）

- **标题：** Embodied Human Simulation for Quantitative Design and Analysis of Interactive Robotics
- **短名：** Embodied Human Simulation
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2603.09218>
- **项目页：** <https://lnsgroup.cc/research/HRI/>
- **发表信息：** IROS 2026（项目页标注）
- **作者：** Chenhui Zuo、Jinhao Xu、Michael Qian Vergnolle、Yanan Sui
- **机构：** 清华大学 LNS Group（MS-HRI）
- **提交日期：** 2026-03-10
- **一句话说明：** 以受强化学习控制的全身肌肉骨骼数字人为预测性代理，在 MuJoCo 中模拟人与外骨骼或人形机器人的耦合交互，并用预训练人体策略评估器支持结构参数与控制策略的协同优化。

## 核心摘录（面向 wiki 编译）

- **问题：** 交互机器人设计需要理解人体生物力学响应；仅看外部运动或实验中的间接指标，难以获得肌肉力、关节负载等内部量。
- **方法：** 将全身肌肉骨骼模型作为人体动力学代理，以强化学习控制器产生适应外力且受生理约束的运动。
- **优化流程：** 先训练人体运动策略，再固定该策略作为一致评估器；对人与机器人耦合仿真进行设计空间探索，协同调整机器人结构参数和控制策略。
- **展示任务：** 外骨骼辅助行走与人-人形机器人协作搬桌；外骨骼案例报告关节轴对齐改善、接触力下降。
- **边界：** 论文描述的是仿真驱动的量化设计分析；不能据此推断人体模型已在真实交互中完成临床或安全验证。

**对 wiki 的映射：** [Embodied Human Simulation](../../wiki/entities/embodied-human-simulation.md)
