# nv-tlabs / ASE（Adversarial Skill Embeddings 官方仓）

> 来源归档（仓库源码核对）

- **标题：** Adversarial Skill Embeddings (ASE)
- **类型：** repo
- **链接：** <https://github.com/nv-tlabs/ASE>
- **配套论文：** Peng et al., *ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters*（[arXiv:2205.01906](https://arxiv.org/abs/2205.01906) → [ase_arxiv_2205_01906.md](../papers/ase_arxiv_2205_01906.md)）
- **后继仓库：** [xbpeng/MimicKit](https://github.com/xbpeng/MimicKit/)（本仓 README 顶部明确标记 deprecated，新实验推荐转用 MimicKit）
- **入库日期：** 2026-11-19
- **核对范围：** 通读克隆仓库 `README.md`、训练/测试统一入口 `ase/run.py` 与核心学习模块 `ase/learning/ase_agent.py`、`ase/learning/ase_players.py`、`ase/learning/ase_network_builder.py`、`ase/learning/amp_datasets.py`（核对 CLI 任务入口、PPO/AMP/ASE/HRL 注册逻辑、超球面潜变量采样/重置与网络前向结构；未在无 Isaac Gym 环境下运行仿真）。
- **通读：** 已完成
- **一句话说明：** 基于 NVIDIA Isaac Gym 与 `rl_games` 的 ASE / AMP / HRL 官方 PyTorch 实现，含剑盾人形预训练与四项下游任务配置、Reallusion 非商用动作切片及预训练权重。

## 开源核查（步骤 2.5）

**已开源（可运行，但作者标记已弃用）：**

| 项 | 状态与内容 |
|----|------------|
| 训练 / 测试代码 | 完整开源：`ase/run.py` 通过 `--task`、`--cfg_env`、`--cfg_train`、`--motion_file` 统一调度预训练（`HumanoidAMPGetup`）与高层任务（`HumanoidHeading` / `HumanoidReach` / `HumanoidLocation` / `HumanoidStrike`） |
| 预训练权重 | 仓库内置 `ase/data/models/ase_llc_reallusion_sword_shield.pth` 及四项下游任务 HLC `.pth` checkpoint |
| 动作数据 | `ase/data/motions/` 含 Reallusion 剑盾 `.npy` 片段及 `.yaml` 清单（**严格非商用**）；另有 `ase/poselib/retarget_motion.py` 重定向脚本 |
| 维护状态 | README 首行提示 **This codebase is now deprecated**，建议迁往 [MimicKit](https://github.com/xbpeng/MimicKit/) |

## 源码关键链路（实读摘录）

- **统一入口 `ase/run.py`：**
  - `main()` 解析参数与配置文件后，在 `build_alg_runner()` 向 `rl_games.torch_runner.Runner` 同时注册 `amp`（`AMPAgent`）、`ase`（`ASEAgent`/`ASEPlayer`/`ModelASEContinuous`/`ASEBuilder`）与 `hrl`（`HRLAgent`/`HRLPlayer`/`ModelHRLContinuous`/`HRLBuilder`）四件套工厂；环境经 `RLGPUEnv` 包装为 `vecenv`，同时透传 `observation_space` 与 `amp_observation_space`。
- **低层训练 `ASEAgent`（`ase/learning/ase_agent.py`，继承 `amp_agent.AMPAgent`）：**
  - `init_tensors()` 在 `experience_buffer` 中追加 `ase_latents` 张量，并为每个并行环境分配独立的切换倒计时 `_latent_reset_steps`（区间 `[_latent_steps_min, _latent_steps_max]`）。
  - `play_steps()` 每步通过 `_update_latents()` 仅对到期的环境重采样单位球面潜向量 $z$，收集 `(obses, actions, amp_obs, ase_latents)`；结束后调用 `_calc_amp_rewards()` 同时算判别器奖励 `disc_rewards` 与编码器奖励 `enc_rewards = clamp_min(dot(enc_pred, z), 0) * _enc_reward_scale`，再经 `_combine_rewards()` 按 `_task_reward_w / _disc_reward_w / _enc_reward_w` 加权并与 GAE 合并。
  - `calc_gradients()` 联合优化 PPO actor loss、critic loss、action bound loss、判别器损失 `_disc_loss`、vMF 编码器负余弦损失 `_enc_loss`（支持 `_enc_weight_decay` 与基于 `autograd.grad` 的 `_enc_grad_penalty`）以及可选多样性损失 `_diversity_loss`（以二次方惩罚逼近目标比率 `_amp_diversity_tar`，分母为余弦距离 $0.5(1 - z_{\text{new}}^\top z) + 10^{-5}$）。
- **网络结构 `ASEBuilder`（`ase/learning/ase_network_builder.py`）：**
  - `AMPStyleCatNet1` 将潜变量 `ase_latents` 先送入 `_style_mlp`（`[512, 256]` + `tanh` 输出 `style`），再与 `obs` 拼接输入 Actor MLP；Critic (`AMPMLPNet`) 直接拼接 `[obs, latent]`。
  - `_build_enc()` 可选择独立 `_enc_mlp` 或与判别器共享 `_disc_mlp`，经线性层后通过 `F.normalize(..., dim=-1)` 输出单位超球面均值 `eval_enc()`；`sample_latents()` 从标准正态采样并 L2 归一化。
- **推理与回放 `ASEPlayer` / `AMPDataset`（`ase/learning/ase_players.py`、`amp_datasets.py`）：**
  - `ASEPlayer.get_action()` 在评估回放时同样按倒计时定期重采样 `_ase_latents` 并更新角色颜色便于可视化区分技能；`AMPDataset` 继承 `PPODataset`，通过随机置换缓冲区 `_idx_buf` 按 minibatch 取样并在每个 epoch 末尾自动洗牌。

## 对 wiki 的映射

- [ASE 方法页](../../wiki/methods/ase.md) — 方法原理、源码运行时序图与工程实践
- [AMP Reward](../../wiki/methods/amp-reward.md) — 继承基类 `AMPAgent` 与共享判别器结构
- [MimicKit](../../wiki/entities/mimickit.md) — 官方推荐的继任框架
