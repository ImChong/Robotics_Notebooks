# OpenTrack / Any2Track

> 来源归档（国内具身开源全景 · 源码核对）

- **标题：** OpenTrack / Any2Track
- **类型：** repo
- **机构：** 银河通用（GALBOT · 清华大学）
- **链接：** <https://github.com/GalaxyGeneralRobotics/OpenTrack>
- **论文：** <https://arxiv.org/abs/2509.13833> · 项目页：<https://zzk273.github.io/Any2Track/>
- **分类：** 全身动作跟踪/技能训练
- **入库日期：** 2026-09-06
- **核查日期：** 2026-11-21
- **一句话说明：** Any2Track 的官方开源代码实现（基于 MuJoCo-MJX / Brax），已完整发布专家 PPO 训练、多 GPU DAgger 蒸馏、AnyAdapter（MBPPO）扰动微调、LAFAN1 Generalist v1/v2 预训练权重及带力矩投影的 C++/unitree_sdk2 真机部署框架。
- **核对范围：** 通读 `GalaxyGeneralRobotics/OpenTrack@cb9b751` 的 `README.md`、`track_mj/learning/train/train_ppo_track.py`、`track_mj/learning/train/train_dagger.py`、`track_mj/learning/train/train_adapter.py` 与 `deploy/README_deployment.md`；未在本机启动 GPU 训练或真机部署。
- **通读：** 已完成

## 开源状态

- **已开源**：README News 与 TODO 列表显示 **全部核心模块均已发布**（2025-09 跟踪代码与基础 DR → 2025-11 LAFAN1 Generalist v1 → 2026-04 DAgger 蒸馏 → 2026-05-18 LAFAN1 Generalist v2，单策略跟踪 40/40 LAFAN1 动作 → 2026-05-23 AnyAdapter 训练 → 2026-05-28 C++ 真机部署）。预训练权重托管于 README 所列 Google Drive 链接。

## 关键入口与实现要点（`cb9b751`）

- **环境与数据初始化：** `uv sync -i https://pypi.org/simple`，配置 `.env`（`GLI_PATH` 等）后运行 `python track_mj/app/mj_playground_init.py`；可选 `python scripts/process_motion/preprocess_motion.py` 对齐 50 Hz 控制频率并可选 OSQP QP 平滑起步过渡（`--smooth_start_end True`）。
- **三阶段训练入口：**
  1. 专家策略：`python -m track_mj.learning.train.train_ppo_track --task G1TrackingGeneralDR --exp_name <name>`（`tmj.registry` 获取 `G1Env` 与 config，调 `ppo.train` 并可选 `convert_jax2onnx` 导出 `policy.onnx`）。
  2. 通才蒸馏：`python -m track_mj.learning.train.train_dagger`（支持 `torchrun --use_ddp` 多卡；读取 `dagger_config_path` 簇采样权重，JAX `student_environment` 与 PyTorch DDP 协同运行 `dagger_horizon.dagger` 并调用 `convert_torch2onnx`）。
  3. 动力学适应：`python -m track_mj.learning.train.train_adapter --task G1TrackingGeneralDR --load-exp-name <base>`（`use_mbppo = args.use_adapter or args.use_world_model` 时默认 `history_len=79`，加载预训练 PPO 检查点并走 `mbppo.train`，导出带历史输入的 ONNX `convert_jax2onnx_with_history`）。
- **评测与真机部署：** `python -m track_mj.eval.dagger.mj_onnx_video` 验证 ONNX 策略；真机基于 `unitree_sdk2` + `onnxruntime` C++ 状态机（`WAIT_R2 → DAMPING → STAND → LOCO ↔ DANCE`），提供 `build_w_torque_projection.sh`（按训练期力矩上限重投影目标关节角以缩小 Sim2Real 差异）及 `sim_interface/main.py` 环回 MuJoCo 测试入口。

## 对 wiki 的映射

- [wiki/entities/paper-opentrack.md](../../wiki/entities/paper-opentrack.md)
- [wiki/methods/any2track.md](../../wiki/methods/any2track.md)
