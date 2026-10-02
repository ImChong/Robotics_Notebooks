# LocoWM（CASIA 高精度行走 + 世界模型残差）

> 来源归档

- **标题：** LocoWM
- **类型：** repo
- **来源：** 中国科学院自动化研究所（CASIA）等
- **链接：** <https://github.com/zhaozijie2022/LocoWM>
- **入库日期：** 2026-10-01
- **核查日期：** 2026-10-02
- **项目页归档：** [LocoWM 项目页](../sites/locowm.md)
- **论文归档：** [LocoWM 论文](../papers/locowm_arxiv_2609_39179.md)
- **一句话说明：** LocoWM 官方仓库：Isaac Sim 5.1 / Isaac Lab 固定 commit 上的 Go2-W 两阶段 RSL-RL 训练、world model、残差 adapter 与 payload 成功率评测脚本。
- **沉淀到 wiki：** [`wiki/entities/paper-locowm.md`](../../wiki/entities/paper-locowm.md)

---

## 核心定位

**LocoWM** 是 [arXiv:2609.39179](https://arxiv.org/abs/2609.39179) 的官方代码。Base policy + action-conditioned world model 在 Stage 1 共训；Stage 2 冻结二者并训练 residual adapter，推理时 \(a=a^b+a^r\)，用 **预测的未来 task substate** 做 preactive 修正。

---

## 仓库结构要点（README，2026-10-01）

| 路径 / 入口 | 作用 |
|-------------|------|
| `locowm/` | Isaac Lab 任务与 `scripts/train`、`list_envs` |
| `loco_rl/` | world model、residual adapter、RSL-RL 扩展 |
| `python -m locowm.scripts.train --task Isaac-LocomotionGo2W-v1` | Stage 1：base + WM |
| `python -m locowm.scripts.train --task Isaac-TransportGo2W-Adapter-v1` | Stage 2：adapter（需 policy + WM checkpoint） |
| `python -m locowm.scripts.eval` / `succ_eval` | 指标与载荷成功率评测 |
| `Isaac-TransportGo2W-v1` 等 | End-to-End、React（NoWM）、ReconWM 消融任务 |

**依赖：** Python 3.11、Isaac Sim 5.1.0、Isaac Lab `c91a125c73`、PyTorch 2.7 / CUDA 12.8、RSL-RL 2.3.3；基于 [robot_lab](https://github.com/fan-ziqi/robot_lab) 与 [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)。

---

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [paper-locowm](../../wiki/entities/paper-locowm.md) | 论文实体与结论 |
| [paper-notebook-steadytray](../../wiki/entities/paper-notebook-steadytray.md) | 同为托盘/载荷高精度行走；SteadyTray 为 **反应式** 残差，LocoWM 强调 **WM 预测 preactive** |
| [paper-wm-loco](../../wiki/entities/paper-wm-loco.md) | 同名缩写不同工作：WM-LOCO 为人形落脚 RSSM+PPO，非本文 |

## 2026-10-02 复核补充

官方 README 的 `succ_eval` 会重试初始加速阶段掉载荷，并将其排除出成功/失败计数；复现时应报告重试/排除数量。Stage 1 policy 与 world model 需来自同一 run、同一迭代。公开任务入口主要为 Go2-W，不能视为现成 G1 真机包。

**对 wiki 的映射：** [LocoWM](../../wiki/entities/paper-locowm.md)、[残差策略学习](../../wiki/methods/residual-policy-learning.md)。
