# HuMBLE ingest

- **日期：** 2026-10-10
- **论文：** [HuMBLE: Human Motion-Driven Behavior Learning for Embodied Locomotion](https://arxiv.org/abs/2610.10489)
- **类型：** arXiv 论文 / 人形机器人 locomotion 方法
- **来源记录：** `sources/papers/humble_arxiv_2610_10489.md`
- **实体页：** `wiki/entities/paper-humble-human-motion-driven-behavior.md`
- **摘要：** 以人体步态 retarget 动捕训练全身参考 teacher，蒸馏为 SE(2) 速度指令 + proprioception 策略，再以参考模仿和目标跟踪双任务 PPO 微调；在 Atlas R1/D1 与 Unitree G1 验证。
- **状态核查：** 论文与补充材料公开；未找到作者公开代码/权重入口。G1 数据被描述为将发布，Atlas 数据因专有属性不公开。
- **交叉链接：** `wiki/tasks/humanoid-locomotion.md`、`wiki/overview/humanoid-amp-motion-prior-survey.md`
