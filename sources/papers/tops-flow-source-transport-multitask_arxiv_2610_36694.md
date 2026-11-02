# TOPS（arXiv:2610.36694）

> 来源归档（paper）

- **标题：** TOPS: Reshaping Flow Source and Transport for Multi-Task Robot Manipulation
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.36694>
- **PDF：** <https://arxiv.org/pdf/2610.36694>
- **作者：** An, Tuo, Zhou, Chuhao, Dong, Jiahua, Li, Jingliang, Bai, Jiaqi, Jia, Jindou, Yang, Jianfei
- **arXiv 评论栏：** （无）

- **入库日期：** 2026-11-02
- **一句话说明：** TOPS：通过重构 Flow 源分布与传输路径实现多任务机器人操作。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.26-10.30前沿论文动态】Manipulation 57篇](../blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md)
2. **周报所述问题：** 多任务 Flow Matching Policy 通常对所有任务使用相同的高斯噪声分布和线性插值路径，但不同操作任务的动作分布差异很大。
3. **周报所述创新点：** 提出 Task-Oriented Priors and Schedules（TOPS）。首先从专家动作反向映射构建任务专用的多模态源分布，再根据源分布与目标动作分布的协方差差异，设计任务专用的 Flow 插值时间调度。在 LIBERO-10 上，单步推理成功率达到 **69.67%**，超过标准 Flow Matching 十步推理的 **53.44%**，推理时间约为基线的六分之一。
4. **arXiv 摘要（英文节选）：** Flow-based policies for language-conditioned multi-task manipulation commonly share a single $\textit{Gaussian}$ source prior and a common linear transport path across tasks. However, reliably executing multiple tasks with distinct motion requirements remains challenging for a shared policy. We introduce $\textbf{TOPS}$ ($\textbf{T}$ask-$\textbf{O}$riented $\textbf{P}$riors and $\textbf{S}$chedules), which adapts the flow source prior and interpolation schedule to each task without modifying the network architecture.

**对 wiki 的映射**

- [paper-tops-flow-source-transport-multitask](../../wiki/entities/paper-tops-flow-source-transport-multitask.md)
