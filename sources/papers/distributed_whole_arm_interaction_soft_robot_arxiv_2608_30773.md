# Learning to infer and manipulate through distributed whole-arm interaction in a soft robot（arXiv:2608.30773）

> 来源归档（ingest）

- **标题：** Learning to infer and manipulate through distributed whole-arm interaction in a soft robot
- **短名：** Distributed Whole-Arm Interaction
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2608.30773>
- **PDF：** <https://arxiv.org/pdf/2608.30773>
- **作者：** Chuhan Zhang、Ebrahim Shahabi、Kseniia Khomenko、Wei Pan、Cosimo Della Santina
- **机构：** 代尔夫特理工大学（TU Delft）、纽卡斯尔大学（Newcastle University）
- **领域：** 软体机器人、强化学习、全臂操作、Sim2Real
- **提交日期：** 2026-08-31
- **入库日期：** 2026-10-09
- **项目/代码：** 当前 arXiv 记录及本次提交资料未提供官方项目页或代码仓库链接。
- **一句话说明：** 将软臂与物体的分布式接触视为信息来源，以嵌入柔顺结构的 IMU 历史驱动循环策略，实现盲式全臂抓取。

## 核心摘录（面向 wiki 编译）

- 论文把物理交互从“需要抑制的扰动”转为信息获取与动作组织过程：任务相关信息不可直接测量，策略需要从连续交互历史中推断。
- 方法以端到端、带记忆的强化学习策略为核心：先训练覆盖工作空间的探索策略，再联合优化探索与抓取目标，并通过观测映射、策略微调两阶段适配真实机器人。
- 实验在混合刚性—柔性机械臂上进行，柔顺结构内嵌 IMU 作为本体感觉来源；策略自主完成工作空间探索、物体接触与定位、抓取相关属性推断及全臂包裹抓取。

**对 wiki 的映射：** [paper-distributed-whole-arm-interaction-soft-robot](../../wiki/entities/paper-distributed-whole-arm-interaction-soft-robot.md)

## 开源与复现状态

截至 2026-10-09，本次提交只给出 arXiv 预印本链接；arXiv 摘要页未列项目主页、代码或数据仓库入口。本文按可核实的论文摘要归纳方法，不将未公开的训练配置、精确成功率或复现步骤当作已知事实；若作者后续发布实现，再补充源码归档与运行时序。

## 参考来源

- Zhang et al., [arXiv:2608.30773](https://arxiv.org/abs/2608.30773) — 论文元数据与摘要
- 作者提交信息 — 机构中文名称与论文分类信息
