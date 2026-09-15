# GO2 Backflip（Robot-Nav / PPO-backflip）

- **标题：** GO2 Backflip — 相位条件 PPO 四足后空翻训练与 Sim2Real 部署
- **类型：** repo
- **仓库：** <https://github.com/Robot-Nav/GO2_backflip/tree/PPO-backflip>
- **演示：** <https://www.bilibili.com/video/BV1HbtJ6AENK/>
- **维护：** Robot-Nav 社区（同 [legbot-MPC-WBC](./legbot-mpc-wbc.md)）
- **硬件：** Unitree Go2（12 DoF）
- **收录日期：** 2026-09-15
- **开源状态：** **已开源**（训练 / Sim2Sim / 真机部署 / 预训练 ONNX）

## 一句话摘要

Isaac Lab 上 **相位条件 PPO** 学 Go2 后空翻：60D 可部署 Actor、165D 特权 Critic、电机包络与时延 DR；导出 ONNX → MuJoCo 验证 → SDK2 **50 Hz 策略 + 500 Hz LowCmd** 真机状态机。

## 主要目录（PPO-backflip 分支）

| 路径 | 作用 |
|------|------|
| `isaaclab_backflip/` | Isaac Lab 环境、奖励、观测、DR、PPO 配置 |
| `isaaclab_backflip/tasks/go2_backflip/go2_backflip_env.py` | 环境核心 |
| `train.py` / `play.py` | 训练与回放 / ONNX 导出 |
| `mujoco/` | MuJoCo Sim2Sim |
| `deploy_real/` | 真机状态机、安全监控、ONNX 推理 |
| `model_*.onnx` | 预训练策略 |

## 训练栈

- Isaac Sim 5.1 + Isaac Lab + RSL-RL + PyTorch
- 4096 并行环境（文内）；相位 6D 谐波特征；非对称 Actor–Critic

## 部署要点

- RL 阶段主机侧 PD + 力矩裁剪后 **直接发 `motor.tau`**，避免双 PD 环
- 动作延迟与训练一致：ONNX 每 20 ms 更新 `next_action`，当前拍执行 `current_action`
- 触发前检查：直立、低速度、足端接触；`SELECT` 急停

## 对 wiki 的映射

- [repo-go2-backflip](../../wiki/entities/repo-go2-backflip.md)
- [PinkRobot 原理到代码长文](../blogs/wechat_pinkrobot_go2_backflip_ppo_2026-09-15.md)
