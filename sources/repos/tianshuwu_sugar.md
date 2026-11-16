# tianshuwu/SUGAR（官方源码与三阶段训练脚本核查归档）

> 来源归档（ingest · 仓库核查）

- **仓库：** <https://github.com/tianshuwu/SUGAR>（MIT 协议）
- **类型：** repo
- **依赖栈：** Python 3.11、IsaacSim 5.1.0、IsaacLab v2.3.0、RSL-RL（`BCPPO` 扩展）、`diffusers`（`DDIMScheduler`）、Hydra
- **入库日期：** 2026-11-14
- **核对范围：** 核查官方仓库 `tianshuwu/SUGAR` 的 `README.md`（全文）、`train.sh`（全文）、`inference.sh`（全文）、`scripts/sugar_rl/train.py`、`scripts/sugar_rl/play.py`（全文 259 行）、`source/sugar_rl/sugar_rl/utils/rsl_rl_bcppo.py`、`source/sugar_il/sugar_il/policy/generator.py` 与 `source/sugar_il/sugar_il/config/train_generator_workspace.yaml`。未在 IsaacSim GPU 环境执行训练脚本。
- **通读：** 已完成
- **一句话说明：** `tianshuwu/SUGAR` 在 `source/sugar_rl`（基于 `unitree_rl_lab` / `beyondmimic`）与 `source/sugar_il`（基于 `DexGraspVLA` 的 12 层 DiT 扩散生成器）中完整开源了 Refiner → Tracker → Generator 三阶段训练与六任务推理脚本，但 RGB-D 视频预处理与 Sim2Sim 管线仍在 README TODO 中尚未发布。

## 核心源码结构与入口核查

### 1) 一键三阶段流水线（`train.sh` & `inference.sh`）
- **`train.sh` 严格按顺序串联五个子步骤：**
  1. **Refiner RL 训练：** `scripts/sugar_rl/train.py --task Sugar-G129dof-${TASK_NAME}-Refiner --num_envs 4096 --max_iterations 30001`，导出 `refiner.pt`。
  2. **Refiner Rollout 与数据清洗：** `scripts/sugar_rl/play.py --task Sugar-G129dof-${TASK_NAME}-Refiner-Rollout --num_envs 1000` 生成 `raw_npz`，由 `scripts/sugar_rl/process_refiner_rollout.py` 转为 `rl_dataset`。
  3. **Tracker BC+RL 蒸馏训练：** `scripts/sugar_rl/train.py --task Sugar-G129dof-${TASK_NAME}-Tracker --teacher_ckpt ...`（加载 `BCPPO` 算法），导出 `tracker.pt`。
  4. **Tracker 闭环 Rollout 采集：** `scripts/sugar_rl/play.py --task Sugar-G129dof-${TASK_NAME}-Tracker-Rollout` 生成 `raw_npz`，由 `scripts/sugar_rl/process_tracker_rollout.py` 转为 `il_dataset`。
  5. **Command Generator IL 训练：** `scripts/sugar_il/train.py --config-name train_generator_workspace.yaml task="${TASK_NAME}" use_target=${USE_TARGET} num_epochs=1001`（其中 `CarryBox`/`PickBox`/`PushBox` 启用 `use_target=True`，`PickBottle`/`StandBottle`/`SitChair` 设 `use_target=False`）。
- **`inference.sh`：** 调用 `scripts/sugar_rl/play.py --task Sugar-G129dof-${TASK_NAME}-Inference --checkpoint "${TRACKER_CKPT}" --generator_checkpoint "${GENERATOR_CKPT}"` 在 16 个环境中同时加载高层生成器与底层跟踪器执行闭环评估。

### 2) 关键算法实现（`rsl_rl_bcppo.py` & `generator.py`）
- **`BCPPO`（`source/sugar_rl/sugar_rl/utils/rsl_rl_bcppo.py`）：** 继承 `PPO`，自动从 `teacher_ckpt` 解析并冻结 Teacher MLP 网络；配置阶段式蒸馏切换（`bc_only_steps=500`，`critic_warmup_steps=1000`，`full_ppo_warmup_steps=2000`）与 `distill_loss_coef=1.0`。
- **`Generator`（`source/sugar_il/sugar_il/policy/generator.py` & `train_generator_workspace.yaml`）：** 使用 `GeneratorStateObsEncoder`（`feature_dim=256`）编码状态条件，驱动 `12` 层 `8` 头 `TransformerForActionDiffusion` 与 `DDIMScheduler`（训练 `50` 步、推理 `16` 步）采样 `n_action_steps=8` 长度的运动命令序列。

### 3) 开源边界（`README.md` TODO List）
- **已开源：** 六任务（`CarryBox`、`KickBox`、`PushBox`、`SitChair`、`StandBottle`、`PickBottle`）预处理数据、描述与演示检查点（`gdown` 压缩包）、Refiner/Tracker/Generator 完整训练与推理脚本。
- **待开放：** 从原始 RGB-D 人类视频到训练用运动与接触先验的数据处理管线，以及独立的 Sim-to-Sim 迁移管线。

## 对 wiki 的映射

- [paper-loco-manip-161-076-sugar](../../wiki/entities/paper-loco-manip-161-076-sugar.md)
