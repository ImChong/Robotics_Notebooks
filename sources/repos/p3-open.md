# P3_Open（Probabilistic Policy Propagation）

> 来源归档

- **标题：** P3_Open
- **类型：** repo
- **来源：** 上海交通大学 / 同济大学 / 浙江大学 / 上海创智学院（Yan / Ma / Zhang / Fu / Cao / Zhu / Chen / Gao）
- **链接：** <https://github.com/ylyem9x/P3_Open>
- **论文：** <https://arxiv.org/abs/2607.25541>（Submitted 2026-07-28）
- **许可：** GitHub API 未返回 SPDX（截至 2026-08-13 无 LICENSE 文件元数据）
- **入库日期：** 2026-08-13
- **一句话说明：** Isaac Lab + 定制 `rl_p3` 的 Unitree G1 感知 locomotion 训练仓：矩匹配概率 Actor 主训，再切多样本潜变量微调（LSFT），并提供 Isaac 回放脚本。
- **沉淀到 wiki：** [`wiki/entities/paper-p3.md`](../../wiki/entities/paper-p3.md)

---

## 开源核查（2026-08-13）

| 项 | 结论 |
|----|------|
| 可见性 | 公开；`language=Python`；约 14 MB；3 stars |
| 可运行入口 | **是** — `run_train.sh` / `run_finetune.sh` / `run_play.sh` → `scripts/rsl_rl/train.py`、`play.py` |
| 任务 ID | 训练 `P3-G1-29DOF-v0`；回放 `P3-G1-29DOF-Play-v0` |
| 权重 | **未随仓分发**；需自训或自备 checkpoint |
| 真机栈 | 论文用 FAST-LIO + 高程图部署；README **未**单列 ROS 真机入口 |
| 依赖 | Isaac Sim **5.1.0**、Isaac Lab **v2.3.0**、Python 3.11、RSL-RL |

---

## 仓库入口（README）

| 组件 | 说明 |
|------|------|
| 主训 | `bash run_train.sh`：4096 env × 15,000 iter；cfg 中 `sample_times=1`、`probabilistic_actor=True`（MM） |
| LSFT | 改 `source/g1_locomotion/.../p3_rl_ppo_cfg.py` 的 `sample_times=15` 后 `bash run_finetune.sh`（`--resume`，默认 `--load_run P3_NoLSFT --checkpoint model_14000.pt`） |
| 回放 | `bash run_play.sh`：默认指向 `logs/rsl_rl/g1_parkour/P3_LSFT/model_16000.pt` |
| 配置 | `source/g1_locomotion/tasks/manager_based/g1_locomotion/agents/p3_rl_ppo_cfg.py` |
| 算法 | `rl_p3/modules/probabilistic_actor.py`、`probabilistic_layers.py`、`vae_encoders_decoders.py`、`actor_critic_vae_mlp_cnn.py`；`rl_p3/algorithms/ppo.py` |

**论文 vs 脚本：** 正文默认日程是 **7k MM + 1k LSFT（$N{=}15$）**；仓内 shell 按 **15k 主训**、微调脚本最多 3k iter 且硬编码 checkpoint 名。复现时以 cfg / `--load_run` 实际路径为准，不要照抄论文 epoch 数硬套脚本。

---

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [paper-p3](../../wiki/entities/paper-p3.md) | 论文实体：边缘策略估计、数据效率与真机表 |
| [Isaac Lab](../../wiki/entities/isaac-lab.md) | 训练仿真栈 |
| [Unitree G1](../../wiki/entities/unitree-g1.md) | 29-DoF 部署平台 |
| [PPO](../../wiki/methods/ppo.md) | clip 目标在随机潜空间的修正 |
