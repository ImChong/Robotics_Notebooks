# junhengl/mpc-rl

> 来源归档（ingest）

- **标题：** MPC-guided RL for Humanoid Locomotion and Manipulation
- **类型：** repo
- **官方入口：** <https://github.com/junhengl/mpc-rl>
- **论文：** <https://arxiv.org/abs/2606.05687>（MPC-RL）；<https://arxiv.org/abs/2601.14414>（π MPC 基础）
- **视频：** <https://youtu.be/PrcbXkA1kYg>
- **机构：** Caltech；Johns Hopkins University
- **入库日期：** 2026-06-10
- **一句话说明：** MPC-RL 官方代码：质心 MPC 训练期奖励、πⁿ MPC GPU 批求解器、mjlab + rsl-rl PPO 人形 locomotion / loco-manipulation 训练与 Themis 真机部署管线。

## 仓库要点（公开 README 索引）

| 模块 | 说明 |
|------|------|
| **πⁿ MPC** | 并行于时域、无构造批 ADMM 求解器（PyTorch / JAX） |
| **CD-MPC** | 质心动力学 MPC 与 landmark guidance reward |
| **训练** | mjlab 仿真 + rsl-rl PPO；4096 并行环境配置 |
| **部署** | 学得的 MLP 策略直接下发关节 PD（无在线 MPC） |

## 对 wiki 的映射

- [MPC-RL 论文实体](../../wiki/entities/paper-mpc-rl-humanoid-locomotion-manipulation.md)
- [π MPC 方法页](../../wiki/methods/pi-mpc.md)
