# Distilled Tree Search Rearrangement（arXiv:2610.25785）

> 来源归档（paper）

- **标题：** Long-Horizon Multi-Object Rearrangement via Distilled Tree Search over Continuous Actions
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.25785>
- **PDF：** <https://arxiv.org/pdf/2610.25785>
- **作者：** Kallol Saha, Yining Hong, Amber Li, Yejin Choi, Maxim Likhachev, David Held
- **arXiv 评论栏：** Accepted at CoRL 2026 (Oral). Project page: this https URL
- **外链：** <https://distilled-search.github.io>
- **入库日期：** 2026-10-26
- **一句话说明：** 通过连续动作空间树搜索蒸馏实现长程多物体重排。

## 开源状态

- arXiv 摘要页/评论给出项目页或外链（https://distilled-search.github.io）；未发现代码仓库链接（2026-10-26，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.19-10.23前沿论文动态】Manipulation 37篇](../blogs/wechat_senlanke_weekly_manipulation_2026-10-19_23.md)
2. **周报所述问题：** 长程物体重排需要考虑当前动作对后续操作的影响。树搜索能够规划未来，但在连续高维动作空间中直接搜索计算成本很高。
3. **周报所述创新点：** 提出 DISCO（连续动作树搜索蒸馏）。使用 Progressive-Widening MCTS 在物理仿真中生成长程计划，由学习到的策略提出连续动作，价值函数评估任务进度；随后将搜索结果周期性蒸馏到 Policy-Value Model，使下一轮搜索获得更好的动作候选和价值估计。最终部署时直接执行蒸馏后的策略，无需在线仿真搜索。在仿真和真实多物体重排任务中，DISCO 优于论文比较的行为克隆、强化学习及搜索基线。论文获 CoRL 2026 Oral。
4. **arXiv 摘要（英文节选）：** Despite recent advances in policy learning, long-horizon manipulation remains difficult because learned policies must avoid compounding errors while preserving future feasibility. While search-based planning can explicitly reason over such future consequences, it quickly becomes expensive in high-dimensional continuous action spaces. This raises a central question: how can long-horizon reasoning from search be transferred into policies in continuous action spaces?

**对 wiki 的映射**

- [paper-distilled-tree-search-multi-object-rearrangement](../../wiki/entities/paper-distilled-tree-search-multi-object-rearrangement.md)
