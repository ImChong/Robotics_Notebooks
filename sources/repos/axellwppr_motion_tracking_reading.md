# Axellwppr/motion_tracking 源码与 HEFT 原文精读归档

- **类型：** 开源仓库精读（配合论文核对）
- **仓库：** <https://github.com/Axellwppr/motion_tracking>
- **本地检出提交：** `731c3c4`（`main` 分支）
- **对应论文：** HEFT: Heavy-Payload Full-size Humanoid Teleoperation with Privileged Motion Guidance（arXiv:2607.02332，15 页含附录 A–E）
- **通读：** 已完成
- **核对范围：** 论文原文 `arXiv:2607.02332` 全文（含附录 A 数据集统计、附录 B 观测维度 Table B.1、附录 C WPC Algorithm 1、附录 D 架构与 17 项奖励 Table D.1–D.3、附录 E 7 项 L7 真机高负载任务 Table E.1），以及 `Axellwppr/motion_tracking@731c3c4` 源码：`README.md`、`train.sh`、`generate_dataset.sh`、`scripts/train.py`、`scripts/eval.py`、`scripts/data_process/generate_dataset.py`、`scripts/utils/helpers.py`、`scripts/utils/play.py`、`active_adaptation/utils/window_load_capacity.py`、`active_adaptation/utils/export.py`、`cfg/load/l7_wpc.yaml`、`cfg/task/profile/G1_tracking.yaml`、`cfg/task/profile/L7_tracking.yaml`、`cfg/exp/train.yaml`、`cfg/exp/adapt.yaml`、`cfg/exp/finetune.yaml`、`active_adaptation/envs/base.py`（`_step`）、`active_adaptation/learning/ppo/ppo.py`（`get_rollout_policy`、`train_op`、`_train_ppo`、`_train_estimator`）。

---

## 一、论文核心设定与实验核对（arXiv:2607.02332）

1. **核心机制：Privileged Motion Guidance（PMG）与 Windowed Payload Curriculum（WPC）**
   - **PMG**：在训练期同时输入干净全身真值参考 `clean`（由 `SEED` 61,612 clips / 134.35 h、`100STYLE` 810 clips / 18.78 h、`LaFAN1` 40 clips / 2.19 h 与 `Paired VR` 643 clips / 6.08 h 经 RoHM 清洗及 GMR 50 Hz 重定向得到）和噪声 VR 稀疏观测参考 `raw`；teacher 使用 `encoder_priv` + `actor_teacher`，student 仅接收历史本体感知与目标（VR/未来步）观测，经 `adapt_module` 对齐特权潜表征并在后阶段使用 PPO 微调 `actor_student`。
   - **WPC（Algorithm 1）**：将长轨迹切分为 `5 s` 窗口，按成功窗容量查表动态调整手腕对称负载课程（从初始上限 `30 kg` 到 `0 kg`，步长 `5 kg`；每个窗记录最近一次可完成最高负载并按策略能力升降）。
   - **L7 与 G1 配置差异**：`cfg/task/profile/G1_tracking.yaml` 为 Unitree G1 29-DoF 跟踪；`cfg/task/profile/L7_tracking.yaml` 为全尺寸人形机器人 L7（含外部负重/WPC 接入点），主论文 Table E.1 在 L7 上报告 7 项最高 `2×12 kg = 24 kg` 的真机重负载任务。

2. **关键消融与基准结果（Table 1）**
   - `D_dynamic`（100 clips, 8.23 min）评测集：完整 HEFT 达 **0.73 成功率**、**0.057 m MPJPE**、**0.743 rad/s root vel** 跟踪误差与 **0.968 rad/s angvel**；去除干净参考教师监督（w/o expert）成功率降至 **0.64**；`TWIST2+FC` 基线仅 **0.10**。另有 `D_VR`（8 clips, 52.26 s）与 `D_random`（100 clips, 21.62 min）评测集。

---

## 二、仓库源码入口与执行流核对（`main@731c3c4`）

1. **数据集预处理（`generate_dataset.sh` → `scripts/data_process/generate_dataset.py`）**
   - 从 GMR 导出的 50 Hz `.npz` 与可选 `vr_paired/{clean,raw}` 目录读取轨迹，合并重采样为 `memdataset`（`.mem` + `meta.yaml`），并保留 `vr_paired` 配对关系供 PMG 采样。
2. **三阶段流水线（`train.sh` → `run_pipeline` → `scripts/train.py`）**
   - Phase 1（`+exp=train`）：`task.student_train=false`，`SimpleEnv` 启动 mjlab 仿真，`SymmetricVecNorm` 归一化观测，`PPOPolicy.get_rollout_policy()` 返回 `Seq(encoder_priv, actor_teacher)`，`train_op()` 同时执行 `_train_ppo(td, mode="teacher")` 与 `_train_estimator(td)`（用 `F.mse_loss(mb["priv_pred"], mb["priv_feature"])` 回归冻结的教师特权特征）。
   - Phase 2（`+exp=adapt`）：以 `checkpoint_path=run:<上一阶段>` 继承权重，`get_rollout_policy()` 切换为 `Seq(adapt_module, actor_student)`，`train_op()` 仅执行 `_train_estimator(td)`，冻结 `actor_student` 专训自适应估算器。
   - Phase 3（`+exp=finetune`）：继承 `adapt` 检查点，`train_op()` 仅执行 `_train_ppo(td, mode="student")`，端到端用 PPO 微调学生策略 `Seq(adapt_module, actor_student)`。
   - WPC 查表（`active_adaptation/utils/window_load_capacity.py`）：`WindowLoadCapacityLookup` 按 `cfg/load/l7_wpc.yaml` 读取/维护窗口负载标签。
3. **评测与导出（`scripts/eval.py` → `scripts/utils/play.py` / `active_adaptation/utils/export.py`）**
   - `load_run_cfg_and_checkpoint()` 加载 wandb run 或本地检查点；`--export` 时用 `ObsNorm.from_vecnorm()` 把归一化参数固化进 `policy.pt` 并调用 `export_onnx()` 导出至 `scripts/exports/<task>-<时间戳>/policy.onnx`。

---

## 三、本地检出覆盖边界与未核实项

- **未在本地 `main@731c3c4` 中包含的内容**：`README.md` 提及的 `sim2real`（已发布 checkpoint 与真机运行时）和 `compliance` 分支不在本地单分支检出 `731c3c4` 中，C++ 真机部署端未在本轮源码核对中展开。
- **外部数据资产**：`cfg/load/l7_wpc.yaml` 中引用的 `data/motion/paired_vr_22/window_labels` 窗标签目录及完整 `vr_paired/{clean,raw}` 原始动作包未随代码仓库内置，需按 README 指引下载或由 GMR 管线自建。
