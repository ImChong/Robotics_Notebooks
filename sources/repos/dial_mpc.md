# DIAL-MPC 仓库

> 来源归档（repo）

- **标题：** DIAL-MPC
- **类型：** repo
- **链接：** <https://github.com/LeCAR-Lab/dial-mpc>
- **入库日期：** 2026-11-01
- **线索来源：** [wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md](../blogs/wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md)（古月居转载，原作者敢敢AUTOHUB）
- **一句话说明：** DIAL-MPC（Diffusion-Inspired Annealing for Legged MPC）是 training-free 的全阶、力矩级采样式 MPC，直接在 Brax 物理仿真中采样 rollout，不需要降阶建模、线性化或预设接触序列（ICRA 2025 Best Paper Finalist）。
- **最近提交：** 2025-05-28
- **核查：** 仓库可达（`git ls-remote`，浅克隆读 README，不运行代码）
- **文章星标（文章表述）：** 952
- **沉淀到 wiki：** [`wiki/entities/dial-mpc.md`](../../wiki/entities/dial-mpc.md)

## README 摘录（入库日人工阅读）

README 标注 ICRA 2025 Best Paper Finalist，对应论文「Full-Order Sampling-Based MPC for Torque-Level Locomotion Control via Diffusion-Style Annealing」（arXiv:2409.15610），Apache-2.0 徽章，基于 JAX。
