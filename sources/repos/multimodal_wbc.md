# Renforce-Dynamics/MultiModalWBC（官方源码与任务配置核查归档）

> 来源归档（ingest · 仓库核查）

- **仓库：** <https://github.com/Renforce-Dynamics/MultiModalWBC>
- **类型：** repo
- **依赖栈：** Python 3.10、Isaac Sim 4.5.0、Isaac Lab 2.1.1、vendorized `third_party/rsl_rl`
- **入库日期：** 2026-11-14
- **核对范围：** 核查官方仓库 `MultiModalWBC` 的 `README.md`（全文）、`scripts/rsl_rl/train.py`、`scripts/rsl_rl/play.py`、`source/whole_body_control/whole_body_control/tasks/tracking/config/g1/__init__.py`、`source/whole_body_control/whole_body_control/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`、`tasks/tracking/mdp/curriculums.py`、`third_party/rsl_rl/rsl_rl/algorithms/triple_ae_ppo.py`（前 100 行）与 `third_party/rsl_rl/rsl_rl/modules/actor_critic_triple_ae.py`（前 100 行）。未在 Isaac Sim GPU 环境运行训练或真机脚本。
- **通读：** 已完成
- **一句话说明：** `MultiModalWBC` 基于 Isaac Lab 与魔改 `rsl_rl` 实现了 M3imic 的三模态自编码全身控制（代码中对应 `GAEMimic-Flat-G1-v0/v1`、`Actor_Triple_AE` 与 `Triple_AE_PPO`），并提供 SMPL-X / GMR 数据预处理及 JIT/ONNX 导出入口。

## 核心源码结构与入口核查

### 1) 注册任务与配置（`whole_body_control/tasks/tracking/config/g1/`）
- `__init__.py` 注册了 4 个 G1 任务：`Tracking-Flat-G1-v0`、`MultiTracking-Flat-G1-v0`、`GAEMimic-Flat-G1-v0`（标准版 `[512, 256, 128]`）与 `GAEMimic-Flat-G1-v1`（Large 版 `[1024, 512, 256]`）。
- `agents/rsl_rl_ppo_cfg.py` 中的 `GAEMimic_G1FlatPPORunnerCfg` 配置了三模态输入维度与超参数：`actor_sg_dim=290`（`29×10`）、`actor_sh_dim=1260`（`126×10`）、`actor_sk_dim=450`（`5×9×10`）、`latent_dim=64`、`activate_signals="robot"`、`empirical_normalization=False`，以及 `Triple_AE_PPO` 损失权重（`reconstruction_loss_coef_sg=0.5`、`sh=0.1`、`sk=0.1`、`alignment_loss_coef=1.0`、`consistency_loss_coef=5.0`）。

### 2) 三模态自编码策略与算法（`third_party/rsl_rl`）
- `actor_critic_triple_ae.py`（`Actor_Triple_AE` / `ActorCritic_Triple_AE`）：分别维护 `robot_encoder/decoder`、`human_encoder/decoder`、`keypoints_encoder/decoder`，通过 `activate_signals`（`"robot" | "smplx" | "keypoints"`）路由当前用于动作解码的潜变量 `z`，并对本体观测 `x_sp` 应用独立 `EmpiricalNormalization` 后与 `z` 拼接输入动作解码器。
- `triple_ae_ppo.py`（`Triple_AE_PPO`）：继承 `PPO`，在 `update()` 中联合计算标准 PPO 损失与三项表征损失（重构、三向潜变量对齐、跨模态解码一致性），并支持可选模态微调冻结 `freeze_for_finetune`。

### 3) 训练、导出与课程采样（`scripts/rsl_rl/` 与 `mdp/curriculums.py`）
- `scripts/rsl_rl/train.py`：`AppLauncher` 启动 Isaac Sim 后通过 `@hydra_task_config` 注入 `env_cfg` 与 `agent_cfg`，经 `RslRlVecEnvWrapper` 与 `OnPolicyRunner` 执行 `runner.learn(...)` 并持久化 `params/env.yaml`、`agent.yaml`。
- `scripts/rsl_rl/play.py`：加载训练检查点后调用 `export_policy_as_jit(..., "policy.pt")` 与 `export_policy_as_onnx(..., "policy.onnx")` 导出推理模型并在仿真循环中前向 rollout。
- `tasks/tracking/mdp/curriculums.py`：`adaptive_sampling_ratio` 根据跟踪奖励达标阈值逐步调节均匀采样与失败率自适应采样混合比 `adaptive_uniform_ratio`。
- 数据准备链位于 `scripts/data/`（`gmr_pkl_to_csv.py`、`smplx_interpolate.py`、`offline_csv_to_npz_datasets.py`、`auto_datasets_yaml.py`），预处理数据集托管在 ModelScope（`seulzx/gae_mimic_dataset`）。

## 对 wiki 的映射

- [paper-loco-manip-06-m3imic](../../wiki/entities/paper-loco-manip-06-m3imic.md)
