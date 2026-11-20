# STRATA: Self-Maintaining Structured Experience Graphs for Self-Evolving Language Agents

- **类型：** paper
- **原始链接：** https://arxiv.org/html/2610.19386v1
- **入库日期：** 2026-11-20
- **核对范围：** arXiv v1 正文：方法三层结构、五种维护机制、AppWorld/WebArena/ALFWorld/ScienceWorld 实验、消融、案例和局限；版本 2026-10-14
- **通读：** 已完成

## 原始资料要点（归纳，非全文转载）

- 作者：Guanhua Zhu、Hanwen Li、Zhi Wei、Mengnan Du、Chenxi Yuan；NJIT 与香港中文大学（深圳）。arXiv:2610.19386。
- 经验分成原子技能/代码片段、任务级工作流、跨任务元框架；以 call、generalize、merge、contrast 等类型边连通。
- 检索结合内容、目标及 operation/target 标签；任务结束按成功或失败实施 typed reflection，决定创建、扩展或跳过经验。
- 社区检测归纳元框架；压力触发清理，同时保护相似性图中的连接桥；contrast 边阻止方向相反任务的错误迁移。
- 双冻结底座、四个交互基准与七种基线。作者报告 SciWorld 在移除 contrast 边时 78.7、保留时 96.7；这不是实际机器人部署实验。
- 原文未提供可确认的官方可运行代码仓库，不能把算法描述当作已开源复现。

## 对 wiki 的映射

- [STRATA: Self-Maintaining Structured Experience Graphs for Self-Evolving Language Agents](../../wiki/entities/paper-strata-experience-graphs.md)
