# robotic_world_model（Isaac Lab 扩展：RWM / RWM-U）

> 来源归档

- **标题：** Robotic World Model Extension for Isaac Lab
- **类型：** repo（Isaac Lab 扩展 + MBRL 训练脚本）
- **组织：** ETH Zurich — Robotic Systems Lab（RSL）/ Learning & Adaptive Systems（LAS）等（README 署名）
- **代码：** <https://github.com/leggedrobotics/robotic_world_model>
- **配套算法库：** model-based [RSL RL](https://github.com/leggedrobotics/rsl_rl_rwm)（README 要求替换 Isaac Lab 自带的 `rsl_rl_lib`）
- **论文 / 项目页：**
  - RWM：[arXiv:2501.10100](https://arxiv.org/abs/2501.10100)，[项目页](https://sites.google.com/view/roboticworldmodel/home)
  - RWM-U：[arXiv:2504.16680](https://arxiv.org/abs/2504.16680)，[项目页](https://sites.google.com/view/uncertainty-aware-rwm)
- **入库日期：** 2026-05-17
- **一句话说明：** 在 **Isaac Lab** 上实现 **Robotic World Model（RWM）** 与 **Uncertainty-Aware RWM（RWM-U）** 的参考管线：在线阶段用仿真中 PPO 采集的经验 **联训** 神经动力学模型；随后支持 **在线想象**（持续真机/仿真数据刷新模型，README 对应 RWM + MBPO-PPO）与 **纯离线想象**（静态模型 + 初始状态 CSV，对应 RWM-U + MOPO-PPO），并提供动力学 **自回归可视化** 与策略 `play` 入口。
- **沉淀到 wiki：** [Robotic World Model（ETH RSL，RWM / RWM-U）](../../wiki/entities/robotic-world-model-eth-rsl.md)

---

## README 归纳（版本、安装、任务形态）

1. **版本徽章（README）：** Isaac Sim **4.5.0**，Isaac Lab **2.1.0**，Python **3.10**，Linux / Windows，MIT License；启用 pre-commit。
2. **安装概要：** 先按官方指南装 Isaac Lab（纯离线策略训练可跳过）；再安装 **leggedrobotics/rsl_rl_rwm**；本仓克隆在 Isaac Lab **目录外**；`python -m pip install -e source/mbrl`；可用 `Template-Isaac-Velocity-Flat-Anymal-D-Init-v0` 冒烟验证。
3. **动力学预训练：** `ObservationsCfg_PRETRAIN` 等配置在 `flat_env_cfg.py`；网络与 MBRL-PPO 超参在 `rsl_rl_ppo_cfg.py`（含 `ensemble_size`、`history_horizon`、`system_dynamics_forecast_horizon` 等）。训练入口示例：`Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0`；可视化：`visualize.py` + `Template-Isaac-Velocity-Flat-Anymal-D-Visualize-v0`，需传入 `--system_dynamics_load_path`。
4. **模型基策略：** **Option 1 在线想象**：`Template-Isaac-Velocity-Flat-Anymal-D-Finetune-v0`，可带或不带 `--checkpoint`。**Option 2 离线想象**：`scripts/reinforcement_learning/model_based/train.py --task anymal_d_flat`，架构与数据路径在 `anymal_d_flat_cfg.py`；仓库附带 `assets/models/pretrain_rnn_ens.pt` 与 `assets/data/state_action_data_0.csv`。
5. **代码结构索引（README）：** 在线侧核心含 `anymal_d_manager_based_mbrl_env.py`、`anymal_d_manager_based_visualize_env.py`；离线侧为 `model_based/envs/anymal_d_flat.py` 与对应 `configs/`。

---

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Model-Based RL](../../wiki/methods/model-based-rl.md) | 显式 **学习动力学 + 在想象中训练策略** 的 MBRL 工程锚点；与 Dreamer 系（潜空间 RSSM）并列但此处强调 **状态/特权头监督的 RNN 集成动力学** 与 **在线 / 离线想象** 两分支 |
| [Isaac Gym / Isaac Lab](../../wiki/entities/isaac-gym-isaac-lab.md) | 作为 **Isaac Lab 扩展** 安装与任务注册；版本需与 README 徽章对齐 |
| [Generative World Models](../../wiki/methods/generative-world-models.md) | 同名「世界模型」但本线聚焦 **低维状态–动作神经模拟器** 服务 MBRL，而非像素视频生成式世界模型 |
| [ANYmal](../../wiki/entities/anymal.md) | 参考实现平台为 **ANYmal D** 平地速度跟踪任务族 |

---

## 对 wiki 的映射

- 新建/主挂 **`wiki/entities/robotic-world-model-eth-rsl.md`**：合并叙述 RWM 与 RWM-U、完整扩展与 **Lite** 仓的分工、流程图与论文链接。
- 轻量交叉更新 **`wiki/methods/model-based-rl.md`**、**`wiki/methods/generative-world-models.md`**、**`wiki/entities/isaac-gym-isaac-lab.md`**、**`index.md`**、**`sources/README.md`**。

---

## 外部参考（便于复核）

- Li et al., *Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in Robotics*, [arXiv:2501.10100](https://arxiv.org/abs/2501.10100)
- Li et al., *Uncertainty-Aware Robotic World Model Makes Offline Model-Based Reinforcement Learning Work on Real Robots*, [arXiv:2504.16680](https://arxiv.org/abs/2504.16680)
- [leggedrobotics/robotic_world_model（GitHub）](https://github.com/leggedrobotics/robotic_world_model)
