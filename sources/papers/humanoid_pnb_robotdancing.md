# RobotDancing: Residual-Action Reinforcement Learning Enables Robust Long-Horizon Humanoid Motion Tracking

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** RobotDancing: Residual-Action Reinforcement Learning Enables Robust Long-Horizon Humanoid Motion Tracking
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/13_Physics-Based_Animation/RobotDancing__Residual-Action_RL_Enables_Robust_Long-Horizon_Motion_Tracking/RobotDancing__Residual-Action_RL_Enables_Robust_Long-Horizon_Motion_Tracking.html>
- **分类：** 13_Physics-Based_Animation
- **arXiv：** <https://arxiv.org/abs/2509.20717>
- **入库日期：** 2026-07-10
- **一句话说明：** 长时程、高动态的人形动作追踪之所以脆，是因为「绝对关节指令」无法补偿仿真-实机的动力学差异，误差会随时间累积。 RobotDancing 让策略不再输出绝对关节角，而是在参考轨迹之上预测残差修正量 q^tar = q^ref + a；再配合单阶段（single-stage）非对称 actor-critic PPO、统一的观测/奖励/超参，以及"只对髋/膝 pitch 关节加残差"的选择性残差化，就能把 LAFAN1 里三分钟一段的舞蹈（含跳跃、360° 旋转、侧手翻、冲刺）零样本部署到 Unitree G1 上。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-robotdancing-residual-action-rl-enables-robust-l](../../wiki/entities/paper-notebook-robotdancing-residual-action-rl-enables-robust-l.md).

## 对 wiki 的映射

- [paper-notebook-robotdancing-residual-action-rl-enables-robust-l](../../wiki/entities/paper-notebook-robotdancing-residual-action-rl-enables-robust-l.md)
- 分类父节点：[paper-notebook-category-13-physics-based-animation](../../wiki/overview/paper-notebook-category-13-physics-based-animation.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/13_Physics-Based_Animation/RobotDancing__Residual-Action_RL_Enables_Robust_Long-Horizon_Motion_Tracking/RobotDancing__Residual-Action_RL_Enables_Robust_Long-Horizon_Motion_Tracking.html>
- 论文：<https://arxiv.org/abs/2509.20717>
