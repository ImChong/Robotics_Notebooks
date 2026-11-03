# Beyond Validation Loss（机器人策略预训练的覆盖率与可靠性）

- **标题：** Beyond Validation Loss: Coverage and Reliability Dynamics in Robot Policy Pretraining
- **类型：** paper
- **原文：** <https://arxiv.org/pdf/2611.01565v1>（v1，2026-11-01）
- **作者：** Zhanyi Sun、Max Simchowitz、Shuran Song
- **机构：** 斯坦福大学、卡内基梅隆大学
- **项目页：** <https://zhanyisun.github.io/robot-policy-learning-dynamics/>
- **入库日期：** 2026-11-03
- **核对范围：** 摘要、正文中的定义、方法和主要实验；附录尚待完整核对。此收录为草稿。

## 核心摘录

1. 固定初始条件的 pass@1 衡量平均成功率，较大的 pass@k 测量重复独立采样时至少成功一次的概率；预训练后期二者可能背离。[知识节点](../../wiki/entities/paper-beyond-validation-loss.md)
2. 从留出数据的动作误差分位数构造覆盖与可靠性归一化风险，最小化两者中的较大者以离线选取检查点。[知识节点](../../wiki/entities/paper-beyond-validation-loss.md)
3. 联合生成观测与动作以扩增监督；大模型实验采用依据所选策略动作方差加噪的动作侧变体。[知识节点](../../wiki/entities/paper-beyond-validation-loss.md)
4. 真机 Push-T 采用 20 个初始状态、每个 5 次 rollout；所选和末尾检查点的 pass@1 为 0.62 / 0.52，pass@5 为 0.95 / 0.75。[知识节点](../../wiki/entities/paper-beyond-validation-loss.md)
