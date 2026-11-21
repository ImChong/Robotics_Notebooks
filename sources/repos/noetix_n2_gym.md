# noetix_n2_gym

> 来源归档（国内具身开源全景）

- **标题：** noetix_n2_gym
- **类型：** repo
- **机构：** 松延动力
- **链接：** https://github.com/Noetix-Robotics/noetix_n2_gym
- **分类：** 运动RL/技能训练
- **入库日期：** 2026-09-06
- **核对范围：** 已克隆 `Noetix-Robotics/noetix_n2_gym@153aad1`（核对 `README.md`、`humanoid/scripts/train.py`、`humanoid/scripts/play.py`、`humanoid/scripts/sim2sim.py` 目录与参数入口；未在本地物理运行 Isaac Gym Preview 4 或 MuJoCo 仿真）。
- **通读：** 已完成
- **一句话说明：** 松延动力面向 N2 双足人形机器人的 Isaac Gym RL 运动控制与动作模仿训练框架，覆盖 `n2`（18DoF）、`n2_10dof` 与 `n2_mimic` 任务，并内置 JIT/ONNX 导出及 MuJoCo Sim2Sim 验证管线。
- **沉淀到 wiki：** [`wiki/entities/cn-os-noetix-n2-gym.md`](../../wiki/entities/cn-os-noetix-n2-gym.md)

## 开源状态

- **已开源**：公开仓库（<https://github.com/Noetix-Robotics/noetix_n2_gym>，基于 `humanoid-gym` 衍生构建；具体许可条款以仓库文件为准）。

## 一手仓库要点摘录（`Noetix-Robotics/noetix_n2_gym@153aad1`）

- **环境栈**：NVIDIA Isaac Gym Preview 4 + PyTorch 1.13.1（CUDA 11.8）+ Python 3.8 + MuJoCo。
- **任务与导出链路**：`humanoid/scripts/train.py --task n2` / `n2_10dof` / `n2_mimic` 启动 PPO 训练；`play.py` 导出 JIT 与 ONNX 策略模型；`sim2sim.py` 在 MuJoCo 中进行跨仿真器策略验证。

## 对 wiki 的映射

- [wiki/entities/cn-os-noetix-n2-gym.md](../../wiki/entities/cn-os-noetix-n2-gym.md)
