# Quadruped-PyMPC 仓库

> 来源归档（repo）

- **标题：** Quadruped-PyMPC
- **类型：** repo
- **链接：** <https://github.com/iit-DLSLab/Quadruped-PyMPC>
- **入库日期：** 2026-11-01
- **线索来源：** [wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md](../blogs/wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md)（古月居转载，原作者敢敢AUTOHUB）
- **一句话说明：** Quadruped-PyMPC（IIT DLS Lab）是基于单刚体模型、用 Python 写的 MPC：梯度式（acados）与采样式（JAX）两种实现，已在真机测试并兼容 MuJoCo。
- **最近提交：** 2026-10-15
- **核查：** 仓库可达（`git ls-remote`，浅克隆读 README，不运行代码）
- **文章星标（文章表述）：** 446
- **沉淀到 wiki：** [`wiki/entities/quadruped-pympc.md`](../../wiki/entities/quadruped-pympc.md)

## README 摘录（入库日人工阅读）

README：梯度式 MPC 在 i7-13700H 上计算不到 5 ms，可选积分器、足端力平滑、落脚点优化、实时迭代等；采样式 MPC 在 NVIDIA 4050 mobile GPU 上 10000 并行 rollout 不到 2 ms。
