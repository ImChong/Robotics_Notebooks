# 六篇机器人研究论文：一手资料待深入归档

> 状态：资料入口归档，尚未完成 ingest。以下仅依据 arXiv v1 摘要；方法、实验、局限和相关附录尚需通读。不得将本页视作完整论文解析。

## Hierarchical World Models with Implicit Dynamics（ID-WM）

- 论文：[arXiv:2610.42146v1](https://arxiv.org/abs/2610.42146v1)，2026-10-30。
- 项目页：<https://id-wm.github.io/>（本次访问失败；代码与数据开放状态待核查）。
- 摘要：分层模型将状态空间可达性与局部控制分离；高层采样可行状态转移，低层世界模型作为逆动力学求解器。

## RoboArchitect: Automating the Design of Robotic Agents with Persistent Memory

- 论文：[arXiv:2610.42141v1](https://arxiv.org/abs/2610.42141v1)，2026-10-30。
- 项目页：<https://roboarchitect-ai.github.io/>（开放状态待核查）。
- 摘要：编码模型根据机器人轨迹诊断失败并迭代修改冻结策略周围的 agent harness；论文报告未见场景和任务的成功率提升。

## REA: A Routed Embodied Agent Runtime for Asynchronous Brain-Cerebellum Control

- 论文：[arXiv:2610.42053v1](https://arxiv.org/abs/2610.42053v1)，2026-10-30。
- 摘要：运行时通过路由记忆、策略边界判断及有预算的升级机制，协调推理模型和快速策略；运行中的策略不等待推理模型。
- 项目和代码开放状态待核查。

## ROOF: Learning Humanoid Fall Recovery under Overhead Constraints

- 论文：[arXiv:2610.42097v1](https://arxiv.org/abs/2610.42097v1)，2026-10-30。
- 项目页：<https://lu-yidan.github.io/ROOF/>（本次访问失败；代码与数据开放状态待核查）。
- 摘要：利用接触支撑几何逃逸引导和学习得到的运动先验训练单一恢复策略，仅使用本体感觉。

## Nexus-VLA: Bridging Human and Robot Experiences via a Shared Hierarchical Semantic Latent Space

- 论文：[arXiv:2610.42021v1](https://arxiv.org/abs/2610.42021v1)，2026-10-30。
- 摘要：端到端联合训练人和机器人的数据，通过分层跨本体语义对齐促进人类操作经验迁移。
- 项目和代码开放状态待核查。

## MolmoSpaces-CommonSense: Evaluating Commonsense in Robotic Manipulation Policies

- 论文：[arXiv:2610.41377v1](https://arxiv.org/abs/2610.41377v1)，2026-10-30。
- 相关生态仓库：<https://github.com/allenai/molmospaces>；具体 benchmark 实现及入口待核实，不能仅凭生态仓库推断已发布。
- 摘要：提出操作常识推理分类和基准，覆盖遮挡、包含、支撑关系、语义抓取点、空间可供性及隐藏物体。

## 待办

- 通读六篇全文及理解方法和实验所需的附录，核对用户提供的定量结果、基线、评测条件和局限。
- 核查项目页、代码、权重与数据的开放范围；对有可运行实现的项目阅读运行入口和相关源码。
- 为符合知识库范围的项目建立或更新唯一 wiki 实体、交叉链接，并根据已核实内容绘制 Mermaid 流程图及源码运行时序图。
- Movens-Zero 由已有草稿 PR [#3323](https://github.com/ImChong/Robotics_Notebooks/pull/3323) 跟踪，避免重复建档。
