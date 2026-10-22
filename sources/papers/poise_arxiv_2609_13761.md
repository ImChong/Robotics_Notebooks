# POISE（arXiv:2609.13761）

> 来源归档（paper）

- **标题：** Learning In-Hand Object Reaching to General 6D Poses
- **arXiv：** https://arxiv.org/abs/2609.13761（v1 2026-09-12）
- **作者：** Junxiao Lin, Tianyue Wu, Jie Yin, Jia Pan, Kaifeng Zhang, Weiming Zhi
- **机构：** The University of Sydney、Sharpa、The University of Hong Kong（据项目页）
- **项目页：** https://junxiaolin.github.io/poise-website/ （见 [sources/sites/poise.md](../sites/poise.md)）
- **入库日期：** 2026-10-22

## 摘要要点（据 arXiv 摘要页）

- 把「手内 6D 位姿到达」形式化：从已有抓取出发，手指协同把物体移到掌心相对的目标位姿。
- POISE（Palm-relative Object reaching In SE(3)）：sim-to-real 强化学习，含多样稳定抓取初始化、目标与几何条件控制、自适应 6D 目标课程、紧凑的位姿到达 + 抓取保持奖励。
- 仿真：多样初始化使留出抓取成功率 40.1% → 51.5%，掉落后恢复 33.8% → 72.9%；课程使全范围成功率 6.2% → 59.5%。
- 真机：抓取保持奖励使连续三目标成功率 20% → 80%。
- 代码：摘要称将在项目页发布。

## 对 wiki 的映射

- [paper-poise-in-hand-6d-pose-reaching](../../wiki/entities/paper-poise-in-hand-6d-pose-reaching.md)
