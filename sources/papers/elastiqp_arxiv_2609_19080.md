# ElastiQP（arXiv:2609.19080）

> 来源归档（paper）

- **标题：** ElastiQP: An Always-Feasible QP Solver for Constrained Robot Control
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.19080>
- **PDF：** <https://arxiv.org/pdf/2609.19080>
- **代码：** https://github.com/StanfordASL/elastiqp
- **入库日期：** 2026-09-20
- **一句话说明：** 不等式约束带精确 L1 软化并折进凝聚 QP，等式动力学保持硬约束；不可行时违约集中到冲突不等式，微秒级控制循环仍返回控制量。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-09-20）

## 核心摘录

1. **策展来源：** [具身智能小站 11 篇盘点](../../sources/blogs/wechat_embodied_station_11_papers_constraint_control_2026-09-20.md)
2. **机制：** 每个不等式约束精确 L1 惩罚；等式硬约束；消元把松弛变量折进凝聚系统；违约位置与幅度可解释。

**对 wiki 的映射**

- [paper-elastiqp](../../wiki/entities/paper-elastiqp.md)
