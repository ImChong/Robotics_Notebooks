# Learning Robot Social Navigation By Sensing Human Legs（arXiv:2607.27922）

> 来源归档

- **arXiv：** <https://arxiv.org/abs/2607.27922>
- **PDF：** <https://arxiv.org/pdf/2607.27922>
- **代码 / 仿真：** <https://github.com/otr-ebla/LegNav-Sim>
- **预训练权重：** <https://github.com/otr-ebla/LegNav-Sim/tree/eb46ad1b6c3aae126542ad5a6ebc15439ef7aca0/checkpoints>（仓库内 `checkpoints/`：PPO / SAC / TQC / TAGD / NavRep / vanilla_ppo 等）
- **视频：** <https://youtu.be/P6gFTvi3k7w>
- **开源状态：** **已开源**（JAX 仿真 + CALF 训练/评测 + TurtleBot 4 部署脚本；IROS 2026 workshop 官方实现）
- **入库日期：** 2026-09-26
- **机构（README）：** 锡耶纳大学 DIISM（University of Siena, Department of Information Engineering and Mathematics）
- **一句话说明：** 踝高 2D LiDAR 只见腿部簇而非整人圆盘；LegNav 用 Planted-Foot 步态 + HSFM 人群，CALF（CNN-Attention-MLP）端到端 RL 导航；~30 min 单卡可训；TurtleBot 4 零样本真机。

## 核心摘录（对 wiki 编译）

1. **感知–建模错配：** 多数社交导航把行人当圆盘；10–20 cm 高度 LiDAR 实际看到 **双足独立运动簇** 与鞋部盲区。
   - **映射：** [paper-legnav-calf](../../wiki/entities/paper-legnav-calf.md)「为什么重要」。

2. **LegNav-Sim：** JAX 向量化 2D 射线 + Planted-Foot Gait + HSFM 动态人群；RTX 3080 ~**135k steps/s**，~**30 分钟** 训出可部署策略。
   - **映射：** 「流程总览」「工程实践」。

3. **CALF：** 堆叠 LiDAR 帧 → 共享 1D-CNN → 时序多头自注意力 → MLP 速度命令；奖励显式塑造 **yielding**，评测含 **Yielding Score**。
   - **映射：** 「核心原理」。

4. **基线与算法：** PPO / SAC / TQC + DWA、MPPI、HSFM planner、NavRep、TAGD、vanilla-MLP PPO；真机 **TurtleBot 4** 零样本部署。
   - **映射：** 「实验与评测」「源码运行时序图」。

**对 wiki 的映射：** [paper-legnav-calf](../../wiki/entities/paper-legnav-calf.md)
