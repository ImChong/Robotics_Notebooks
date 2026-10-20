# R²-WAM（arXiv:2610.13024）

> 来源归档（paper；2026-10-20 核对）

- **标题：** What Visual Representations Do World Action Models Need?
- **类型：** paper
- **论文：** <https://arxiv.org/abs/2610.13024>
- **PDF：** <https://arxiv.org/pdf/2610.13024>
- **作者：** Kai Liu、Jiangran Lyu、Tingyu Deng、Yixin Zheng、Jingtian Wan、Wenbo Cui、Mi Yan、Yizhou Zhou、Haoran Li、Dongbin Zhao、Zhizheng Zhang、He Wang
- **机构：** 中国科学院自动化研究所、银河通用机器人、北京大学、中国科学技术大学
- **项目页：** <https://kailiu18.github.io/R-2WAM.github.io/>
- **提交日期：** 2026-10-09
- **一句话说明：** 在统一的 WAM 训练协议下受控比较视觉表征、特征集成机制与组合方式，给出 R²-WAM 表征配方。
- **项目页归档：** [R²-WAM 项目页核查](../sites/r2-wam-visual-representations-project.md)
- **对 wiki 的映射：** [R²-WAM](../../wiki/entities/paper-r2-wam-visual-representations.md)

## 核心摘录

1. **研究问题：** WAM 常继承视频生成模型的 VAE 潜变量，但该潜变量主要针对视觉重建优化；论文考察哪些外部视觉表征更适合机器人操作，以及如何接入。
2. **受控比较：** 在固定训练设置下比较 13 种预训练视觉表征、5 种特征集成机制及多源特征组合，尝试将表征选择与集成机制的影响分开分析。
3. **表征结论：** 强通用视觉表征带来较广泛的鲁棒性收益；在实验设置中，它们可匹配或超过面向特定视觉能力预训练的表征。
4. **集成结论：** 通道维联合预测取得最高平均成功率。分析指出，预测未来特征以及动作专家实际利用新增视觉信息，可能共同解释收益。
5. **组合结论：** 附加表征应按其相对现有输入提供的任务相关增量筛选；同一来源与不同表征组合时可能带来正负不同的变化。
6. **结果与边界：** R²-WAM 在 RoboCasa GR1 Tabletop 达到 84.7%，在 RoboTwin 2.0 达到 92.8%；相对各自 VAE-only 基线分别提升 5.9 和 2.2 个百分点。论文和项目页未提供官方研究训练代码或权重。

## 对 wiki 的映射

- 实体页：[paper-r2-wam-visual-representations](../../wiki/entities/paper-r2-wam-visual-representations.md)
- WAM 概念页：[world-action-models](../../wiki/concepts/world-action-models.md)
