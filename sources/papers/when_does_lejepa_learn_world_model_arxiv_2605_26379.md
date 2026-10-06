# When Does LeJEPA Learn a World Model?（arXiv:2605.26379）

> 来源归档（paper）

- **标题：** When Does LeJEPA Learn a World Model?
- **arXiv：** <https://arxiv.org/abs/2605.26379>
- **HTML：** <https://arxiv.org/html/2605.26379v1>
- **作者：** David Klindt、Yann LeCun、Randall Balestriero
- **机构：** Cold Spring Harbor Laboratory；纽约大学；布朗大学（以论文 HTML 署名为准）
- **提交日期：** 2026-05-25（v1）
- **许可：** arXiv 页面列 CC BY 4.0
- **入库日期：** 2026-10-06
- **代码：** arXiv 页面未列出代码链接（核查日期：2026-10-06）
- **一句话说明：** 分析 LeJEPA 何时能恢复世界潜变量，给出线性可辨识性结果，并以像素输入机器人控制实验检验潜空间规划。

## 论文要点

论文研究一个有明确假设范围的问题：当潜变量具有高斯分布并按平稳、加性噪声转移时，alignment 加高斯正则的 LeJEPA 可从非线性观测中线性恢复真实潜变量（至旋转等简单变换）。作者还讨论近似可辨识性，并报告潜空间规划实验。

这个定理不代表任意数据、任意 latent 或任意 JEPA 都会学到真实世界状态。是否满足高斯潜变量、转移过程和优化假设，是把理论用于工程判断时的边界。

## 对本文的关系

VideoDB 长文借此强调，latent embedding 要保留与预测和规划相关的结构。该理论补充已有 [LeJEPA](./lejepa_arxiv_2511_08544.md) 方法页：原论文介绍训练目标与 SIGReg，这篇后续工作研究特定条件下的可辨识性。

## 对 wiki 的映射

- [When Does LeJEPA Learn a World Model? 实体页](../../wiki/entities/paper-when-does-lejepa-learn-world-model.md)
- [LeJEPA 实体页](../../wiki/entities/paper-lejepa.md)
- [VideoDB JEPA 长文实体页](../../wiki/entities/article-videodb-jepa-world-models.md)
