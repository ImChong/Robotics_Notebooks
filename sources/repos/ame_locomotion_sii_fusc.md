# AME_Locomotion（SII-FUSC/AME_Locomotion）

> 来源归档（ingest）

- **标题：** AME Locomotion Reproduction (Isaac Lab + Unitree G1)
- **类型：** repo（**非官方**社区复现）
- **组织：** SII-FUSC
- **代码：** <https://github.com/SII-FUSC/AME_Locomotion>
- **论文：** [Attention-Based Map Encoding（arXiv:2506.09588）](../papers/ame_arxiv_2506_09588.md) · [Science Robotics 10(105), eadv3604](https://doi.org/10.1126/scirobotics.adv3604)
- **许可：** 见仓库根目录（MIT 类开源栈；含自定义 `rsl_rl` fork）
- **入库日期：** 2026-09-02
- **一句话说明：** 社区在 **Isaac Sim 5.1 + Isaac Lab 2.3** 上复现 AME 的 **CNN + 本体条件 MHA 高程编码** 与 **两阶段 PPO**；目标平台为 **Unitree G1 29-DoF**（非论文原平台 ANYmal-D / Fourier GR-1）；提供 `ame1.pt` / `ame2.pt` 预训练权重与 `run_train.sh` / `run_play.sh` 入口。

## 开源状态（步骤 2.5，截至 2026-09-02）

| 项 | 状态 |
|----|------|
| 与 ETH 官方关系 | **非官方** — 论文作者未发布训练/部署仓库 |
| 可运行入口 | **有** — `run_train.sh`、`run_play.sh`；`rsl_rl/rsl_rl/modules/actor_critic_encoder.py` 为 AME 核心 |
| 预训练权重 | **有** — `pretrained/ame1.pt`、`ame2.pt` |
| 平台差异 | 论文为 **ANYmal-D + GR-1**；本仓为 **G1 + Isaac Lab** |
| 上游对照 | [senlanke/mimic](senlanke_mimic.md) 正将本仓迁到 mjlab（**未完成/未验证**） |

判定：**社区已开源（可跑 G1 仿真）**；引用论文方法时请区分 **官方 Zenodo 数据** 与本仓实现边界。

## 技术要点摘录

- **栈：** Isaac Lab task `ame_locomotion` + 本地 `pip install -e rsl_rl`（自定义 Actor-Critic 地形编码器）。
- **两阶段：** `velocity_env_cfg_29dof.py` 中 `FINETUNE=False` 跑 Stage 1，完成后改 `True` 再 Stage 2（与论文两阶段课程一致）。
- **ame1 vs ame2：** `ame1` 为默认 33×21 高程 + CNN 下采样；`ame2` 额外 `attach_global=True`（全局上下文，更接近 AME-2 方向但仍是 AME-1 论文方法的社区扩展）。

## 对 wiki 的映射

- [`wiki/entities/paper-ame-attention-based-map-encoding.md`](../../wiki/entities/paper-ame-attention-based-map-encoding.md)
- [`wiki/entities/smp-g1-mjlab.md`](../../wiki/entities/smp-g1-mjlab.md) — mjlab 移植未完成
- [`sources/repos/senlanke_mimic.md`](senlanke_mimic.md) — AME 迁移源指针
