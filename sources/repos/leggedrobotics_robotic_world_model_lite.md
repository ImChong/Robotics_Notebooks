# robotic_world_model_lite（无仿真依赖的 RWM / RWM-U 轻量管线）

> 来源归档

- **标题：** Robotic World Model Lite
- **类型：** repo（纯 Python / PyTorch；**无 Isaac Sim 依赖**）
- **组织：** ETH Zurich — RSL / LAS 等（与完整扩展同一作者群，见 README）
- **代码：** <https://github.com/leggedrobotics/robotic_world_model_lite>
- **快速体验：** [Google Colab 笔记本](https://colab.research.google.com/drive/1SRL0ss59RxMp-MwY38Pi6iRW-VTFeku6?usp=sharing)（README 链接）
- **完整扩展对照：** [robotic_world_model](https://github.com/leggedrobotics/robotic_world_model)（Isaac Lab 在线采集 + 预训练 + 评测全链路）
- **论文 / 项目页：** 与完整版相同 — RWM [arXiv:2501.10100](https://arxiv.org/abs/2501.10100)；RWM-U [arXiv:2504.16680](https://arxiv.org/abs/2504.16680)
- **入库日期：** 2026-05-17
- **一句话说明：** 仅保留 **离线想象** 路径：用 **集成 RNN 动力学**（仓库内 `pretrain_rnn_ens.pt`）在 **CSV 初始状态** 上 rollout 虚构环境，训练 **模型基策略**；支持 **Weights & Biases** 日志；适合无法安装 Isaac Lab 时对齐论文 **MOPO-PPO / RWM-U** 行为，策略部署与 `play` 仍需回到 **Isaac Lab 扩展或官方任务注册表**（README 所述）。
- **沉淀到 wiki：** [Robotic World Model（ETH RSL，RWM / RWM-U）](../../wiki/entities/robotic-world-model-eth-rsl.md)

---

## README 归纳（安装、入口、代码布局）

1. **定位：** 明确写给 **不想装完整机器人仿真器** 的用户；强调 **无仿真器依赖**。
2. **环境：** Conda 创建 `python=3.10`；`python -m pip install -e .` 安装包名 **`rwm_lite`**（README）。
3. **训练入口：** `wandb login` 后 `python scripts/train.py --task anymal_d_flat`；日志在 `logs/`。默认分支上主 README 文件名为 **`readme.md`**（小写），与常见 `README.md` 不同。
4. **评测说明：** README 给出的 `play.py` 示例路径属于 **Isaac Lab 任务注册表** 调用方式，**本 Lite 仓内不包含** `scripts/reinforcement_learning/rsl_rl/play.py`；真机或仿真上回放需按 README 指引使用 **完整 RWM 扩展** 或 **Isaac Lab**。
5. **关键文件（README）：** `scripts/envs/anymal_d_flat.py`（Lite 路径下为 `scripts/envs/`，非 `model_based/envs/`）、`scripts/configs/anymal_d_flat_cfg.py`、`assets/models/pretrain_rnn_ens.pt`、`assets/data/state_action_data_0.csv`。

---

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [robotic_world_model 完整扩展归档](./leggedrobotics_robotic_world_model.md) | **Lite = 离线子集**；动力学预训练与在线微调在完整扩展中 |
| [Model-Based RL](../../wiki/methods/model-based-rl.md) | 离线模型基 RL（静态模型 + 想象 rollout）教学与复现入口 |
| [Isaac Gym / Isaac Lab](../../wiki/entities/isaac-gym-isaac-lab.md) | 策略 **部署/回放** 仍依赖 Isaac Lab 生态（README 自述） |

---

## 对 wiki 的映射

- 与完整仓共用 **`wiki/entities/robotic-world-model-eth-rsl.md`**，在文中分节对比 **Full vs Lite**。

---

## 外部参考（便于复核）

- [leggedrobotics/robotic_world_model_lite（GitHub）](https://github.com/leggedrobotics/robotic_world_model_lite)
- Li et al., *Robotic World Model…* [arXiv:2501.10100](https://arxiv.org/abs/2501.10100)
- Li et al., *Uncertainty-Aware Robotic World Model…* [arXiv:2504.16680](https://arxiv.org/abs/2504.16680)
