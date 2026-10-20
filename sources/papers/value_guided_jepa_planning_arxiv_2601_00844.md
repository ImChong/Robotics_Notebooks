# Value-guided action planning with JEPA world models（arXiv:2601.00844）

> 来源归档（ingest）

- **标题：** Value-guided action planning with JEPA world models
- **类型：** paper / JEPA / goal-conditioned value / planning
- **arXiv：** <https://arxiv.org/abs/2601.00844>
- **代码：** 页面未列出代码仓库
- **作者（arXiv 页）：** Matthieu Destrade；Oumayma Bounou；Quentin Le Lidec；Jean Ponce；Yann LeCun
- **机构：** arXiv 页未显示
- **提交 / 会议：** 2025-12-28 提交 arXiv（cs.LG / cs.AI / cs.RO）；文内标注为 World Modeling Workshop 2026（Mila）海报
- **入库日期：** 2026-10-20
- **来源文章：** [wechat_deepblue_lecun_jepa_11_papers_2026-10-20.md](../blogs/wechat_deepblue_lecun_jepa_11_papers_2026-10-20.md)

## 开源核查（步骤 2.5，2026-10-20）

arXiv 页仅提供 PDF / HTML / TeX 源，无代码链接 → **未发现官方代码**，仅写「截至入库日未列代码」。

## 摘录要点（归纳）

- 普通 JEPA 规划只看潜空间里哪个未来「看起来更近」，但欧氏距离近不等于可达（例如目标在墙后）。
- 方法：用 goal-conditioned value 训练潜空间，使状态嵌入之间的距离近似「到达目标的代价」。
- 摘要称在若干简单控制任务上优于标准 JEPA 规划。

## 对 wiki 的映射

- 主节点：[paper-value-guided-jepa-planning](../../wiki/entities/paper-value-guided-jepa-planning.md)
