# MPC 脚手架灵巧 RL（arXiv:2609.14878）

> 来源归档（paper）

- **标题：** Real-World Reinforcement Learning with MPC Scaffolding for Dexterous Manipulation
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.14878>
- **PDF：** <https://arxiv.org/pdf/2609.14878>
- **入库日期：** 2026-09-20
- **一句话说明：** Sampling MPC 初始化 buffer 并预训练；在线 SAC 与 MPC 共训，逐渐交权；16-DoF Allegro 手内旋转 7 min 达 5/5，20 min 速度超 MPC 5×、1000 次旋转。

## 开源状态

- **待发布**（步骤 2.5 核查，2026-09-20）

## 核心摘录

1. **策展来源：** [senlanke 周更](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-14_18.md)
2. **机制：** MPC trajectories init replay + online SAC with gradual handoff from MPC to policy.

**对 wiki 的映射**

- [paper-mpc-scaffolding-dex-rl](../../wiki/entities/paper-mpc-scaffolding-dex-rl.md)
