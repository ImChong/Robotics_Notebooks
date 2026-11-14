# hukz18/Robot-Trains-Robot 官方代码仓库核对归档

> 来源归档（GitHub 源码实体核对）

- **仓库：** <https://github.com/hukz18/Robot-Trains-Robot>
- **类型：** repo
- **许可证：** MIT（根目录 `LICENSE`）
- **核对范围：** 核对 `README.md`、`toddlerbot/locomotion/train_mjx.py`、`toddlerbot/policies/run_policy.py`、`toddlerbot/policies/walk_finetune.py`、`toddlerbot/finetuning/remote_learner.py`、`launch/walk_finetune.sh` 与 `launch/swing.sh`。
- **通读：** 已完成
- **入库日期：** 2026-11-14

## 源码架构与运行链路核对

1. **仿真预训练入口（`toddlerbot/locomotion/train_mjx.py`）**：基于 Brax + MuJoCo MJX 并行仿真实现 Stage 1/Stage 2 训练，在域随机化环境中联合训练策略与动力学隐变量 FiLM 调制层，并优化通用隐变量初始值。
2. **真机一键启动脚本（`launch/walk_finetune.sh`、`launch/swing.sh`）**：分别编排「行走策略微调（学生 + UR5 引导）」与「荡秋千从零学习（学生 + 机械臂相位助推）」两套真机双机协同会话。
3. **机载执行与遥测闭环（`toddlerbot/policies/run_policy.py`、`walk_finetune.py`）**：`run_policy.py` 支持 `walk_finetune`、`at_leader`、`swing` 与 `swing_arm_leader` 模式；`walk_finetune.py` 冻结仿真主干与 FiLM 层并开放真机隐变量微调参数。
4. **远端学习服务端（`toddlerbot/finetuning/remote_learner.py`）**：`RemoteServer` 通过 ZMQ 接收真机 rollout 轨迹，在离板 GPU 工作站执行离线 critic 预训练或在线 PPO 梯度更新，并将刷新后的策略参数或隐变量实时下发回机端。

## 对 wiki 的映射

- [paper-notebook-robot-trains-robot-automatic-real-world-policy-a](../../wiki/entities/paper-notebook-robot-trains-robot-automatic-real-world-policy-a.md)
