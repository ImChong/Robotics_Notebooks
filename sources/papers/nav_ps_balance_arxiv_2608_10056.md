# Navigating the Proximity-Safety Balance（arXiv:2608.10056）

> 来源归档（ingest）

- **标题：** Navigating the Proximity-Safety Balance: Constraint Decomposition for Human Following in Pedestrian Crowds
- **缩写：** 本文用 **PPO-Lagrangian + DtACI** 做接近–安全分解；项目页/仓名 **nav-ps-balance**
- **类型：** paper / social-navigation / human-following / constrained-rl
- **arXiv：** <https://arxiv.org/abs/2608.10056>
- **会议：** IROS 2026
- **项目页：** <https://nav-ps-balance.github.io/>（归档见 [`sources/sites/nav-ps-balance.md`](../sites/nav-ps-balance.md)）
- **代码：** <https://github.com/tasl-lab/nav-ps-balance>（归档见 [`sources/repos/nav-ps-balance.md`](../repos/nav-ps-balance.md)）
- **作者：** Shiting Gong、Jianpeng Yao、Jinfeng Wang、Marco Pavone、Jiachen Li
- **机构：** 宾夕法尼亚大学；加州大学河滨分校；斯坦福；NVIDIA Research；佐治亚理工
- **入库日期：** 2026-08-18
- **一句话说明：** 人群跟随拆成稀疏任务奖励与独立 cost 约束（有行为含义的阈值），并把行人预测不确定性写进 cost；ROSMASTER X3 真机零样本。

## 开源状态（步骤 2.5）

- **项目页核查（2026-08-18）：** Paper / Video / Code 指向 [tasl-lab/nav-ps-balance](https://github.com/tasl-lab/nav-ps-balance)。
- **仓库：** MIT；`train.py` / `test.py` / `trained_models/`；CrowdNav 扩展 + 静态障碍 + DtACI。
- **结论：** **已开源、可运行**（预训练评测 + 再训练）。

## 摘录：数字（项目页 Table 1/3）

| 设定 | 本文 SR | 本文总体 CR | 对照最强（RL+ACI）SR |
|------|---------|-------------|----------------------|
| ID | **78.08%** | 16.16% | 71.60% |
| OOD 走廊 | **89.76%** | 8.64% | 82.96% |
| OOD 15% 奔跑 | **70.56%** | 22.72% | 68.48% |

调阈值（δF/δH）比改 dense reward 权重更能显式切换「更安全 / 跟更紧」。

**对 wiki 的映射：** [`wiki/entities/paper-nav-ps-balance.md`](../../wiki/entities/paper-nav-ps-balance.md)；交叉 [PGIF-MPPI](../../wiki/entities/paper-pgif-mppi.md)、[iCrowdNav](../../wiki/entities/paper-icrowdnav.md)、[PPO](../../wiki/methods/ppo.md)。

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 开源状态核查（可运行）
