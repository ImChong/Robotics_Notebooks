---
type: source
title: "NVlabs/SoftMimicGen — 源码结构与非刚性数据生成归档"
url: "https://github.com/NVlabs/SoftMimicGen"
date: 2026-11-14
status: ingested
tags: [repo, imitation-learning, data-generation, deformable-objects, isaac-lab, nvidia]
---

# NVlabs/SoftMimicGen — 源码结构与非刚性数据生成归档

- **仓库：** <https://github.com/NVlabs/SoftMimicGen>
- **开源状态：** 已开源（Apache-2.0，基于 Isaac Sim 5.1.0 + Isaac Lab 2.2.1，含完整任务环境、遥操作/标注/数据生成脚手架、robomimic 训练评估脚本及预标注示范下载脚本 `softmimicgen.sh`）。
- **核对范围：** 核对官方仓库 `NVlabs/SoftMimicGen` 源码：
  - `README.md`（安装与环境清单、Record / Annotate / Generate 三步工作流、10 个任务 Task ID 与 Mimic ID 映射表）
  - `scripts/imitation_learning/isaaclab_mimic/generate_dataset.py`、`annotate_demos.py`、`consolidated_demo.py`
  - `scripts/tools/record_demos.py`、`replay_demos.py` 与 `scripts/imitation_learning/robomimic/train.py`、`robust_eval.py`
  - `source/softmimicgen/softmimicgen/datagen/generation.py`（`env_loop` 与异步 `run_data_generator`）、`data_generator.py`（`DataGenerator.generate`、`generate_trajectory`、`transform_source_data_segment_using_nodal_registration` 与 `transform_source_data_segment_using_nodal_registration_scaled_tps_rpm_bij`）、`selection_strategy.py`（`RegistrationCostStrategy`）
- **通读：** 已完成

## 关键实现对齐细节

1. **三阶段标准入口：**
   - 录制与回放：`scripts/tools/record_demos.py`、`scripts/tools/replay_demos.py`；
   - 子任务分割标注：`scripts/imitation_learning/isaaclab_mimic/annotate_demos.py`（支持人工按键或 `--auto` 启发式标记，人形任务需追加 `--enable_pinocchio`）；
   - 并行异步生成：`scripts/imitation_learning/isaaclab_mimic/generate_dataset.py` 调用 `setup_async_generation` 与 `env_loop`，通过 `asyncio.Queue`（`env_reset_queue`、`env_action_queue`）驱动多环境步进并筛选成功轨迹写入 HDF5。
2. **非刚性配准核心代码（`data_generator.py` & `selection_strategy.py`）：**
   - 直接引入 `rapprentice` 库的薄板样条鲁棒点匹配（TPS-RPM-BIJ）：`RegistrationCostStrategy`（`NAME = "registration_cost"`）用 `tps_reg_cost(f) + tps_reg_cost(g)` 或 `tps_cost` 为每个子任务挑选最优源示范片段；
   - `transform_source_data_segment_using_nodal_registration_scaled_tps_rpm_bij` 先通过 `unit_boxify` 将源/目标软体节点位置归一化进单位盒，再执行 `tps_rpm_bij(...)` 与 `unscale_tps(...)` 还原物理尺度，使用 `f_unscaled.transform_points` 映射末端位置，并用 `f_unscaled.compute_jacobian` 左乘原旋转矩阵后通过 SVD（`U @ Vt` 并修正负行列式）正交化为合法 `SO(3)` 姿态。
3. **任务与资产包结构：** `source/softmimicgen_tasks/softmimicgen_tasks/` 按任务分设 `humanoid`、`franka_rope`、`franka_jenga`、`franka_towel`、`surgical_tissue`、`surgical_threading`、`yam_towel`、`yam_bag`，均继承 Isaac Lab 的 `ManagerBasedRLMimicEnv` 接口提供节点观测与 `target_eef_pose_to_action` 动作逆解。
