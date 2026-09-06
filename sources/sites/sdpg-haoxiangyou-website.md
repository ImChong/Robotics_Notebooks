# SDPG — Yale 视觉 RL 项目页

- **来源：** https://haoxiangyou.github.io/sdpg-website/
- **类型：** site
- **机构：** Yale University（机械工程 / 计算机科学）；合作 SJTU、University of Sydney
- **论文：** arXiv:2605.26478 — *Efficient On-policy Visual-RL via Stochastic Decoupled Policy Gradient*
- **代码：** https://github.com/HaoxiangYou/SDPG（**已开源**，项目页 Footer 链出）
- **归档日期：** 2026-09-06

## 一句话说明

官方项目页：SDPG 混合 batch-rendered 与 physics-only 环境估计视觉 on-policy 梯度；Visual MuJoCo 上训练时间与显存优于 DrQ-v2 / DreamerV3 / 视觉 PPO；发布 egocentric 任务套件与 Unitree Go2 sim2real 视频。

## 项目页核查（步骤 2.5）

| 项 | 结论 |
|----|------|
| GitHub | **已开源** — [HaoxiangYou/SDPG](https://github.com/HaoxiangYou/SDPG) |
| arXiv | [2605.26478](https://arxiv.org/abs/2605.26478) |
| 数据集 / 权重 | 随仓库 README 安装与训练流程；无单独 HF 模型卡 |
| 状态 | Under review（页眉标注） |

## 关键数字（项目页表格）

- 显存（GB，64 batched env）：SDPG ~10.2–10.5；视觉 PPO† ~48–50；DrQ-v2 / DreamerV3 / Distillation ~8–11
- † PPO 按 4096 env + 状态超参估计

## 交叉链接

- 论文专档：[sources/papers/sdpg_visual_rl_arxiv_2605_26478.md](../papers/sdpg_visual_rl_arxiv_2605_26478.md)
- 仓库归档：[sources/repos/sdpg-haoxiangyou.md](../repos/sdpg-haoxiangyou.md)
- Wiki：[wiki/entities/paper-sdpg-visual-rl-stochastic-decoupled.md](../../wiki/entities/paper-sdpg-visual-rl-stochastic-decoupled.md)
