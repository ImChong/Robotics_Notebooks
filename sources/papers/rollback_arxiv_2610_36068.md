# Rollback: Experience Backtracking in World Models for Policy Improvement

- **类型：** paper
- **论文：** <https://arxiv.org/abs/2610.36068>
- **项目：** <https://rollback-wm.github.io/>
- **日期：** 2026-10-26（v1）；2026-10-29（入库）
- **关联 wiki：** [Rollback](../../wiki/entities/paper-rollback-world-model.md)

## 摘要摘录与映射

1. 在关键决策点回溯并探索反事实动作分支，将结果转为动作级监督，无需独立学习的价值模型。见 [核心机制](../../wiki/entities/paper-rollback-world-model.md)。
2. 异步真机执行和世界模型搜索形成循环；真实转移通过轻量 LoRA 更新世界模型动力学。见核心机制。
3. 三项真机任务相对最强基线分别提高 10.67、11.11、11.85 个百分点；抛球和桌面冰壶在 8 小时训练内超过人类平均成功率。见实验与评测。

项目页访问失败，代码、权重与数据开放状态待核实，见 [项目页记录](../sites/rollback-wm.md)。
