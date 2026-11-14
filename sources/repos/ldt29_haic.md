# ldt29/HAIC（官方源码与配置核查归档）

> 来源归档（ingest · 仓库核查）

- **仓库：** <https://github.com/ldt29/HAIC>
- **关联 Sim2Sim 仓库：** <https://github.com/Cybercal/HOIC-baseline>（仓库 README 声明于 2026-06-10 开源 MuJoCo Sim2Sim 部署链路）
- **类型：** repo
- **依赖栈：** Python 3.11、IsaacSim 5.1.0、IsaacLab 2.3.2、Hydra、TorchRL / TensorDict
- **入库日期：** 2026-11-14
- **核对范围：** 核查官方仓库 `LeCAR-Lab/HAIC`（`ldt29/HAIC`）的 `README.md`（全文）、`scripts/train.py`（第 1–214 行）、`scripts/play.py`（第 1–356 行）、`scripts/helpers.py`（`ObsOODDetector`）、`active_adaptation/learning/ppo/ppo_haic.py`（第 1–861 行 `PPOHAIC` 与 `TransformObject`）、`cfg/task/base/hdmi-base.yaml` 以及 `cfg/task/G1/haic/*.yaml`（11 个任务配置）。未在物理 GPU / IsaacSim 环境中运行训练或推理，未直接拉取外部 `HOIC-baseline` 仓库源码。
- **通读：** 已完成
- **一句话说明：** `ldt29/HAIC` 基于 `active_adaptation` 框架实现 `PPOHAIC` 两阶段训练（`ppo_haic_train` 联合训练+蒸馏、`ppo_haic_finetune` 非对称微调）与 11 个 G1 欠驱动交互/多地形任务配置，并支持导出带 `ObsOODDetector` 的 `.pt`/`.onnx` 部署策略。

## 核心源码结构与入口核查

### 1) 算法注册与配置入口（`active_adaptation/learning/ppo/ppo_haic.py`）
- `PPOHAIC` 通过 `ConfigStore` 注册了 5 个 Hydra 算法配置：`ppo_haic_train`（`phase="train"`）、`ppo_haic_adapt`（`phase="adapt"`）、`ppo_haic_finetune`（`phase="finetune"`）、`ppo_haic_train_est`（`phase="train_est"`）、`ppo_haic_adapt_est`（`phase="adapt_est"`）。
- `PPOHAIC` 核心子网络：
  - `object_adapt`：MLP（`[latent_dim, latent_dim]`，`latent_dim=256`），由 `[obs_, command_]` 预测物体状态 `object_pred`。
  - `TransformObject`（`object_transform` / `object_pred_transform`）：接收物体/额外物体 128 点静态点云模板 `object_geo_`，按位姿 `pos(3) + ori(3x3=9)` 投影到机器人坐标下生成 `object_trans` / `object_pred_trans`。
  - `encoder_priv`：编码 `[priv_, object_, object_trans]` 得到 `priv_feature`。
  - `adapt_module`（`MLP` 或 `GRUModule`）：从 `[object_pred, object_pred_trans, obs_, command_]` 预测 `priv_pred`（`train` 阶段先用真值 `object_, object_trans` 训练以加速收敛）。
  - `actor`（Teacher，`[512, 256, 256]`，支持残差 `RefJointPos`）、`actor_adapt`（Student）、`critic`（`[512, 256, 128]`，读取 `priv_, obs_, command_, object_`）。
  - EMA 稳定化：`adapt_ema` 与 `object_adapt_ema` 在每次 `train_adapt` 后以 `soft_copy_(..., 0.04)` 同步，并在 `finetune` 阶段作为 `get_rollout_policy()` 的前向推理模块。

### 2) 训练与部署脚本（`scripts/train.py`, `scripts/play.py`, `scripts/helpers.py`）
- **`scripts/train.py`：** 加载 Hydra 配置创建 `SimpleEnv`，实例化 `PPOHAIC`（或对应 `algo`），从 `checkpoint_path` 恢复权重；主循环收集 rollout 后调用 `policy.train_op(vec)`（在 `train` 阶段执行 `train_policy` + `train_adapt`，在 `finetune` 阶段仅以 `actor_adapt` 采样并联合更新 `actor_adapt`/`critic` 与 `adapt_module`/`object_adapt`），并定期保存 checkpoint。
- **`scripts/play.py`：** 支持 `command.replay_motion=true` 回放数据资产；支持 `export_policy=true` 时组装 `Sequential(obs_norm, ObsOODDetector(..., sigma=5.0), deploy_policy)` 并导出 `.pt`、`export_onnx` `.onnx` 与对应 `.yaml` 配置；MuJoCo Sim2Sim 部署代码由官方 README 指定的 [Cybercal/HOIC-baseline](https://github.com/Cybercal/HOIC-baseline) 承接。

### 3) 任务配置（`cfg/task/base/hdmi-base.yaml` 与 `cfg/task/G1/haic/*.yaml`）
- `hdmi-base.yaml` 定义了 `step_dt=0.02`（`50 Hz`）、`isaac_physics_dt=0.005`（`200 Hz`）、`mujoco_physics_dt=0.002`（`500 Hz`）、4096 并行环境与 `RobotObjectTracking` 跟踪指令。
- `cfg/task/G1/haic/` 包含 11 个任务配置：`skateboard`、`pull_cart`、`push_cart`、`pull_cart_box`、`push_cart_box`、`push_wheelchair`、`move_mybox`、`box_slope`、`box_stair`、`box_stairslope`、`box_slopestair`；不同任务在 YAML 中显式选择观测项（例如 `box_slopestair.yaml` 同时启用 `object_com_vel_b`、`object_com_acc_b` 与 `extra_object_*`，而 `skateboard.yaml` 默认仅启 `object_pos_b`、`object_ori_b`、`object_points`）。

## 对 wiki 的映射

- [paper-haic](../../wiki/entities/paper-haic.md)
- [haic](../../wiki/methods/haic.md)
