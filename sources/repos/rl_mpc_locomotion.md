# rl-mpc-locomotion 仓库

> 来源归档（repo）

- **标题：** rl-mpc-locomotion
- **类型：** repo
- **链接：** <https://github.com/silvery107/rl-mpc-locomotion>
- **入库日期：** 2026-11-01
- **线索来源：** [wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md](../blogs/wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md)（古月居转载，原作者敢敢AUTOHUB）
- **一句话说明：** rl-mpc-locomotion 提供快速仿真与 RL 训练框架：高层策略网络动态预测 MPC 控制器的权重参数，低层是参考 MIT Cheetah Software、用 Python 重写的 MPC。
- **最近提交：** 2026-04-26
- **核查：** 仓库可达（`git ls-remote`，浅克隆读 README，不运行代码）
- **文章星标（文章表述）：** 946
- **沉淀到 wiki：** [`wiki/entities/rl-mpc-locomotion.md`](../../wiki/entities/rl-mpc-locomotion.md)

## README 摘录（入库日人工阅读）

README 称 RL 训练使用 NVIDIA Isaac Gym 并行、使用 Unitree Aliengo 模型，并在真实 Aliengo 上实现 sim2real（**sim2real 不在本代码库内**）；依赖 Python 3.8、PyTorch 1.10、Isaac Gym Preview 4。
