# 无监督环境设计基准（arXiv:2610.40364）

> 来源归档（paper）

- **标题：** Is It the Curriculum or Just the Easy Start? Benchmarking Unsupervised Environment Design on Quadrupeds
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.40364>
- **PDF：** <https://arxiv.org/pdf/2610.40364>
- **作者：** Gilewicz, Jakub, Galus, Dominik, Wojdas, Piotr
- **arXiv 评论栏：** （无）

- **入库日期：** 2026-11-02
- **一句话说明：** 究竟是课程有效，还是起点简单？四足无监督环境设计基准。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.26-10.30前沿论文动态】人形／四足33篇](../blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md)
2. **周报所述问题：** 无监督环境设计方法主要在二维迷宫上比较，而四足地形课程使用不同机器人、预算和指标，结论无法直接对照。
3. **周报所述创新点：** 在五种三维四足机器人上，以相同步数预算比较 14 种环境设计、域随机化和地形课程；结果表明没有一种方法在所有机器人及测试地形上占优，腿式机器人常用的成功率驱动课程进入最优组的次数最多。
4. **arXiv 摘要（英文节选）：** Unsupervised environment design (UED) methods such as PLR, ACCEL, SFL and PAIRED are compared mostly on 2D mazes, grid worlds and BipedalWalker, while terrain curricula for legged robots are hand-designed and evaluated against a few baselines. We present a benchmark that runs fourteen UED, domain-randomisation and robotics curricula under one protocol on five 3D quadrupeds in MuJoCo MJX: the closed-chain W01-TEK and the open-chain Unitree Go1 and Go2, Google Barkour vB and Boston Dynamics Spot. Levels are vectors of eight interpretable terrain parameters, and all curricula drive the same PPO student at matched environment-step budgets.

**对 wiki 的映射**

- [paper-unsupervised-environment-design-quadruped-benchmark](../../wiki/entities/paper-unsupervised-environment-design-quadruped-benchmark.md)
