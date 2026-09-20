# Learning sim-to-real humanoid locomotion in 15 minutes

> 来源归档（paper / 自由度FreeDof Sim2Real 44 篇参考文献 [42/44]）

- **标题：** Learning sim-to-real humanoid locomotion in 15 minutes
- **类型：** paper
- **出处：** arXiv 2025
- **章节：** 训练成本（[四条路线梳理](https://mp.weixin.qq.com/s/K_6MibGXWwh9OL9eSZxOMg)）
- **arXiv：** <https://arxiv.org/abs/2512.01996>
- **入库日期：** 2026-09-20
- **开源状态：** 待核实
- **一句话说明：** 单卡 RTX 4090 约 15 分钟训出可 sim2real 的人形 locomotion，改变辨识 vs DR 的性价比计算。
- **沉淀到 wiki：** [`wiki/entities/paper-notebook-learning-sim-to-real-humanoid-locomotion-in-15-m.md`](../../wiki/entities/paper-notebook-learning-sim-to-real-humanoid-locomotion-in-15-m.md)

## 核心摘录（归纳）

- 文内「训练成本塌缩」代表；低敏捷任务可能多跑 DR 而非做 SysID。
- off-policy 算法 + 数千并行环境 + 极简 reward。

## 对 wiki 的映射

- [paper-notebook-learning-sim-to-real-humanoid-locomotion-in-15-m](../../wiki/entities/paper-notebook-learning-sim-to-real-humanoid-locomotion-in-15-m.md)
- [freedof-sim2real-44-papers-technology-map](../../wiki/overview/freedof-sim2real-44-papers-technology-map.md)
- [sim2real-four-routes-identifiability](../../wiki/comparisons/sim2real-four-routes-identifiability.md)
