# LadderMan 官方仓库（amazon-far/LadderMan）

> 来源归档（repo）

- **标题：** LadderMan: Learning Humanoid Perceptive Ladder Climbing
- **类型：** repo / humanoid / ladder climbing / depth distillation / sim2real
- **链接：** <https://github.com/amazon-far/LadderMan>
- **核对版本：** `main@9a8d5ed`（2026-11-08）
- **许可证：** Apache-2.0（`active_adaptation` 基于 MIT 协议的 HDMI 框架）
- **配套论文：** [arXiv:2606.05873](https://arxiv.org/abs/2606.05873)（CoRL 2026 Spotlight）
- **项目页：** <https://ladderman-robot.github.io/>
- **真机部署参考：** <https://github.com/EGalahad/sim2real>
- **入库日期：** 2026-11-21
- **核对范围：** 阅读 `amazon-far/LadderMan@9a8d5ed` 的 `README.md`、`scripts/train.py`、`scripts/play.py`、`active_adaptation/learning/ppo/ppo_me_dagger.py` 与仓内目录树；核实 Isaac Sim 4.5.0 + Isaac Lab v2.2.0 安装入口、两阶段教师/学生训练（7500 iters pure ME-DAgger → 2500 iters linear decay → 3000 iters pure PPO）、共享归一化文件 `data/vecnorm/ladder_vecnorm.pt` 与 `export_policy=true` ONNX 导出流程；未在本地 GPU 运行仿真训练。
- **通读：** 已完成
- **一句话说明：** LadderMan 官方训练与 ONNX 导出仓库（基于 HDMI `active_adaptation` + Isaac Lab v2.2.0）；包含单条参考动作 `ladder_data_1/motion.npz`、Warp 深度渲染与多专家 DAgger+PPO 学生训练入口。

## 仓库结构与入口（`main@9a8d5ed`）

| 路径 | 用途 |
|------|------|
| `cfg/` | Hydra 入口配置（`train.yaml`、`play.yaml`、`algo/{teacher,student}.yaml`、`task/{teacher,student}.yaml`） |
| `active_adaptation/envs/ladder_env.py` | Unitree G1 + 程序化生成梯子环境与 Warp 深度渲染（`warp_depth.py` / `warp_kernels.py`） |
| `active_adaptation/learning/ppo/` | `ppo_teacher.py`（Step 1 专家）与 `ppo_me_dagger.py`（Step 2 多专家 DAgger + PPO） |
| `scripts/train_teachers.sh` / `scripts/train.py` | 教师网格训练与学生蒸馏启动器（4096 envs on L40S 48 GB） |
| `scripts/play.py` | Isaac Sim 策略回放与 `export_policy=true` ONNX/JSON/YAML 部署包导出 |
| `data/` | `motion/ladder_data/ladder_data_1/motion.npz`（120 Hz 采样、50 Hz 重采样含 `object_contact`）与 `vecnorm/ladder_vecnorm.pt` |

## 对 wiki 的映射

- [LadderMan 论文实体](../../wiki/entities/paper-ladderman-humanoid-perceptive-ladder-climbing.md)
