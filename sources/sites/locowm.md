# LocoWM 项目页

- **URL：** <https://zhaozijie2022.github.io/LocoWM>
- **关联论文：** [locowm_arxiv_2609_39179](../papers/locowm_arxiv_2609_39179.md)
- **实体页：** [paper-locowm](../../wiki/entities/paper-locowm.md)
- **代码：** <https://github.com/zhaozijie2022/LocoWM> — 归档见 [`sources/repos/locowm.md`](../repos/locowm.md)
- **核查日期：** 2026-10-02

## 开源状态（步骤 2.5）

| 组件 | 状态 |
|------|------|
| 项目页 | 已上线（teaser、三任务真机、仿真曲线、G1 扩展说明） |
| Code 按钮 | 指向 `github.com/zhaozijie2022/LocoWM` |
| 预训练权重 | 页上未列；以仓库 README 训练流程为准 |

**结论：已开源**（官方仓库可复现两阶段训练与 succ_eval）。

## 2026-10-02 复核补充

官方 README 的 `succ_eval` 会重试初始加速阶段掉载荷，并将其排除出成功/失败计数；复现时应报告重试/排除数量。Stage 1 policy 与 world model 需来自同一 run、同一迭代。公开任务入口主要为 Go2-W，不能视为现成 G1 真机包。

**对 wiki 的映射：** [LocoWM](../../wiki/entities/paper-locowm.md)、[残差策略学习](../../wiki/methods/residual-policy-learning.md)。
