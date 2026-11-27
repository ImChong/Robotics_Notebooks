# Hierarchical LLM-Based Multi-Agent Framework with Prompt Optimization for Multi-Robot Task Planning

- **作者：** Tomoya Kawabe, Rin Takano
- **机构：** Data Science Laboratories, NEC Corporation
- **版本 / 载体：** arXiv:2602.21670v2 [cs.RO]（2026-02-25 提交 v1，2026-02-26 修订 v2，8 页，2 图；已录用于 IEEE ICRA 2026）
- **原文：** [arXiv HTML v2](https://arxiv.org/html/2602.21670) / [arXiv Abs](https://arxiv.org/abs/2602.21670)
- **项目页 / 代码：** 未确认开源代码（arXiv:2602.21670v2 正文及附录未列出 GitHub 仓库、独立项目页 URL 或代码发布链接；正文 §V-D 仅提及补充材料视频）
- **核对范围：** arXiv:2602.21670v2 全文（Abstract、§I–§VI、Table I–III、Algorithm 1–2、Fig. 1–2 文字说明、Appendix A 及 References [1]–[23]）
- **通读：** 已完成
- **入库日期：** 2026-11-27

## 原文核对笔记

- **问题定义（§III-A）：** 将多机器人合作任务规划形式化为八元组 $\Pi = \langle R, \mathcal{T}, \mathsf{type}, \mathsf{cap}, P, I, G, \{A^r\}_{r \in R} \rangle$，其中 $R = \{r_1, \dots, r_N\}$ 为物理机器人集合，$\mathcal{T}$ 为机器人类型集合（$M := |\mathcal{T}|$），每类对应一个 PDDL 域 $D_\tau$ 与技能集合 $\mathsf{cap}(\tau) \subseteq \Sigma$；多机器人计划表示为带偏序约束的动作实例集 $\Pi_g = \langle \Delta, \prec \rangle$，无序动作可并发执行，目标是在满足可执行性与终态条件 $G$ 的前提下最小化步数或并行 makespan。
- **三层多智能体架构（§IV-A, §IV-C, §IV-F, §V-C）：** 将逻辑推理智能体集 $\mathcal{E} = \bigcup_{l=0}^{L-1} \mathcal{E}_l$ 划分为 $L$ 层（实验采用 $L=3$ 层，均基于 GPT-4o）：第 0 层为全局规划层（分解自然语言指令 $u$），第 1 层为机器人类型层（按技能描述向对应类型分配子任务），第 2 层为单机器人叶节点层（生成对应 PDDL 域与问题文件）。每个智能体 $E_{l,i}$ 维护当前任务 $\Psi(E_{l,i})$、私有提示 $\theta_{E_{l,i}}$ 以及同层共享元提示 $\hat{\theta}_l$。
- **经典规划器验证与层级重规划决策（§IV-B, §IV-D, Algorithm 1）：** 叶节点生成的 PDDL 规范交由经典规划器（Fast Downward，采用 LAMA 启发式）求解并验证从初态 $I$ 到目标 $G$ 的逻辑可执行性。若任一子计划失败，从失败叶节点 $E_{\mathrm{src}}$ 起沿层级向上调用重规划提示 $\breve{\theta}_{E'}$ 判定 `"self"`（由当前节点修改子任务）或 `"parent"`（回溯给父节点重新分解），直至决策为 `"self"` 或到达根节点 $E_{0,0}$，最大外层迭代次数 $K_{\max} = 5$。
- **双层文本梯度提示优化（§IV-E, Algorithm 2）：** 受 TextGrad 与 MAML 启发，在离散文本空间执行两级更新：
  1. **智能体级内层更新（Agent-level）：** 将规划器与下游失败反馈映射为文本损失 $\mathrm{loss}^{\mathrm{pre}}_{E_{l,i}}$，生成文本梯度 $g^{(k)}_{E_{l,i}}$，经 $\mathrm{TGD.step}$ 对 $\theta_{E_{l,i}}$ 实施排序编辑更新后重算 $\mathrm{loss}^{\mathrm{post}}_{E_{l,i}}$。
  2. **层级外层更新（Layer-level）：** 通过 LLM 聚合算子 $\mathcal{A}_l$ 对同层各智能体的更新后损失去重、归一化并提炼共性修正得到 $\mathrm{loss}_l^{(k+1)}$，计算元梯度 $\hat{g}_l^{(k+1)}$ 并更新层共享元提示 $\hat{\theta}_l$，使同类失败经验在同一层不同智能体与后续任务间迁移。
- **实验与关键数值（§V, Table II–III）：**
  - 在 AI2-THOR 底座的 **MAT-THOR** 基准上评测 70 个任务（30 Compound / 20 Complex / 20 Vague，跨 5 个户型、2–4 台异构机器人，每任务 5 次随机初始化取均值）。
  - 主实验（Table II，均为 GPT-4o）：本文方法在 Compound / Complex / Vague 三类任务的成功率（SR）分别达到 **0.95 / 0.84 / 0.60**（LaMMA-P 为 0.93 / 0.77 / 0.45，分别提升 +2 / +7 / +15 个百分点），目标条件召回率（GCR）为 **0.95 / 0.84 / 0.60**，机器人利用率（RU）均为 **1.00**，时间效率（Eff）为 **0.90 / 0.75 / 0.75**（LaMMA-P 为 0.90 / 0.67 / 0.65）。
  - 消融实验（Table III，全数据集均值）：完整方法（H+P+M）SR 为 **0.84**、平均耗时 **173 s**；去掉层级结构改为单 LLM（–H）降至 **0.25**（-59.3%）、耗时 **140 s**；去掉层共享元提示优化（–M）降至 **0.80**（-4.2%）、耗时 **116 s**；同时去掉提示优化与元提示优化（–(P, M)）降至 **0.47**（-37.0%）、耗时 **32 s**。
- **案例分析（§V-D, Appendix A）：** 在 *"put the tomato in the fridge and turn off the room light"* 任务中，第 0 轮因未开门导致 `Put` 前置条件失败，文本梯度先在提示与层元提示中总结出「向带开合属性容器放入物体前需插入 `Open`」；第 1 轮因开门机器人停在冰箱前阻塞路径，进一步迭代出「开门后移动至非阻塞路点（non-blocking waypoint）让出通道」，第 2 轮（第三次迭代）规划成功。
- **核查边界与局限（§VI）：** 论文明确指出当前系统受限于固定层级结构、完全可观测性（full observability）假设，以及文本梯度优化在收敛速度与稳定性上的局限（平均规划耗时 173 s 显著高于无提示优化变体的 32 s）；此外实验仅在仿真器（MAT-THOR / AI2-THOR）完成且未公开官方实现代码。
- **对 wiki 的映射：** [Hierarchical LLM-Based Multi-Agent Framework with Prompt Optimization for Multi-Robot Task Planning](../../wiki/entities/paper-agentic-2602-21670.md)
