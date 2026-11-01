# Learning Multi-Phase Robot Loco-Manipulation via Phase-Conditioned Policy Optimization

> 论文来源归档（ingest）

- **类型：** paper
- **论文：** <https://arxiv.org/abs/2610.36270>；[PDF](https://arxiv.org/pdf/2610.36270)
- **作者：** Simone Manni, Annika Wollschläger, Dario Bellicoso, Abel Gawel, Marco Hutter, Simon Le Cleac'h, Jan Brüdigam
- **机构：** 苏黎世联邦理工机器人系统实验室（ETH Zurich RSL）；机器人与人工智能研究院（RAI Institute）
- **版本：** arXiv v1，2026-10-26；入库 2026-10-29
- **代码与项目页：** 截至入库日，所查论文和 arXiv 页面未列本工作官方实现或项目页；开放状态待进一步核实。

## 核心摘录

1. **问题：** 抬箱、搬运等长时程任务包括靠近、下蹲和起身等不同动力学阶段；直接奖励塑形容易出现踢箱或拖箱等局部最优。每个环境训练一个高层策略，不依赖人体动作数据。
2. **阶段引导 PPO：** 训练时由环境状态机按条件划分阶段，奖励按阶段门控，并奖励前进、惩罚回退；actor 通过辅助分类预测阶段但不以真实阶段为输入。critic 为共享基础价值加阶段专属残差头；多阶段重置课程逐渐回到初始阶段。测试时 actor 不依赖阶段标签或状态机。
3. **控制与部署：** 分层控制，Spot 使用 ReLiC，G1 新训低层跟踪平面速度、骨盆高度和躯干俯仰；高层用增量命令并控制手臂。mjlab 训练，G1 导出 ONNX，经自定义 ROS 2 框架部署。常规实验使用动捕，G1 推箱和抬箱另以机载估计及 ArUco 实现无动捕演示。
4. **评测：** 三种随机种子仿真消融中，去掉阶段预测，Spot 滚胎成功率由 89% ± 2% 降至 44% ± 9%，G1 双箱推送由 92% ± 2% 降至 14% ± 6%；仅初始阶段重置时双箱任务训练崩溃。数值为论文仿真消融，不是真机成功率。
5. **迁移边界：** 论文报告训练随机化最大 25 ms 执行器延迟、使用较高摩擦课程，并限制目标与当前关节位置偏差以约束夹持力矩；接触仿真、手工阶段切分和关键帧维护仍限制扩展性。

## 对 wiki 的映射

- [阶段条件移动操作论文实体](../../wiki/entities/paper-phase-conditioned-loco-manipulation.md)
- [移动操作任务](../../wiki/tasks/loco-manipulation.md)
