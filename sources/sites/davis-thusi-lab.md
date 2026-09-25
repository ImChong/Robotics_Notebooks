# DAVIS — 项目页（thusi-lab.github.io）

> 来源归档（sites）

- **标题：** DAVIS: A Depth-Only End-to-End Active-Vision Framework for Humanoid Soccer Skills
- **URL：** <https://thusi-lab.github.io/DAVIS/>
- **论文：** [arXiv:2609.28175](https://arxiv.org/abs/2609.28175)
- **机构：** Noetix Robotics；清华大学（Tsinghua University）
- **入库日期：** 2026-09-25
- **代码：** **待发布** — 项目页 **未列 GitHub**（步骤 2.5，2026-09-25 复核；仅学术模板外链）

## 页面要点（相对公众号摘要的增量）

- **部署接口：** 168×80 对齐深度（0.3–5 m 裁剪）+ **5 步**本体历史 + 可选低维指令 → **25-DoF** 关节残差（23 身体 + 2 头）；`qdes = q0 + s ⊙ a`。
- **训练：** 非对称 critic；LightDepthEncoder（32-D）+ HIM 历史；**可见性门控**辅助几何头；GT→prediction annealing；AMP 先验；PPO。
- **技能：** 射门与带球 **分 checkpoint**，共享 depth-to-control 接口。
- **指标（站页）：** 点球/任意球仿真 SR ~0.85；真机点球分档 SR 0.55–0.68；Repeated-S 带球 mean SR **0.65**（900 trials）。

## 交叉归档

- [`davis-humanoid-soccer_arxiv_2609_28175.md`](../papers/davis-humanoid-soccer_arxiv_2609_28175.md)
- [`paper-davis-humanoid-soccer.md`](../../wiki/entities/paper-davis-humanoid-soccer.md)
