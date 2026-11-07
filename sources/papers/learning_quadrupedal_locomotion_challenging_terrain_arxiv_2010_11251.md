# Learning Quadrupedal Locomotion over Challenging Terrain（Lee et al., Science Robotics 2020）

> 来源归档（部分通读 · 仅作 `wiki/methods/teacher-student-dagger-training.md` 的二次核对，未列为 original）

- **标题：** Learning Quadrupedal Locomotion over Challenging Terrain
- **作者：** Joonho Lee, Jemin Hwangbo, Lorenz Wellhausen, Vladlen Koltun, Marco Hutter
- **类型：** paper / legged-locomotion / privileged-learning
- **原文：** <https://arxiv.org/abs/2010.11251>（接受版，Science Robotics 5(47), eabc5986）
- **入库日期：** 2026-11-07
- **核对范围：** 读了摘要、引言中关于 TCN 与特权学习的段落、讨论与局限、方法概述（两阶段训练）、学生训练损失与 DAgger 段落、自适应地形课程开头。**没有读**：实验与野外测试章节、补充材料中的网络结构、奖励函数、仿真器与硬件细节、算法表（Table S8）。
- **通读：** 未完成

## 已核对的要点（均出自原文）

- 直接用 RL 训练粗糙地形盲走策略"没有成功"（监督信号稀疏，在合理时间内学不会）；因此分两阶段：先训练能读取地形与接触真值的**特权教师**，再用它监督只用本体感觉的**学生**（受 Learning by Cheating 启发）。
- 与 Chen 等的区别：教师**不依赖专家示教**，而是用 RL（文中使用 TRPO）训练；教师是 MLP，输出潜向量与动作；学生是时序卷积网络 TCN，输入 N 步本体感受历史。
- 学生损失为动作与潜向量两项平方误差之和，训练数据由学生自身 rollout 产生并由教师在访问到的状态上给出监督，即采用 **DAgger**。
- 两个 ANYmal-B 机器人完成四次各 60 分钟的任务，文中称零失败；网络在机载 CPU 上以 400 Hz 运行。
- 局限（原文）：只学到 trot 步态；只用本体感觉，盲走固有局限（例如被命令走下悬崖就会走下去），未来方向是本体感觉 + 外感知的混合控制器。
