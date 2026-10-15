# What Happens to the Goal? Distinct Failures of Policy Alignment in Goal Misgeneralization（arXiv:2610.19731）

> 来源归档（ingest）

- **标题：** What Happens to the Goal? Distinct Failures of Policy Alignment in Goal Misgeneralization
- **缩写 / 框架：** 目标泛化失效（goal misgeneralization）的策略对齐分析；文中区分三类失效
- **类型：** paper / goal-misgeneralization / policy-alignment / procgen / robot-manipulation / analysis
- **arXiv：** <https://arxiv.org/abs/2610.19731>（Submitted 2026-10-14，v1；PDF：<https://arxiv.org/pdf/2610.19731>）
- **代码（arXiv 页列出）：** <https://github.com/Julbarth/what-happens-to-the-goal> — 仓库内容未能核实（2026-10-15 入库时 GitHub 对本会话返回 403，未能读取 README / LICENSE）
- **项目页：** 未核实（论文页未列）
- **作者：** Julia Barth、Armin Sommer、Kunvar Thaman、Jannik Schilling
- **机构：** 未核实（arXiv 摘要页未列出机构）
- **入库日期：** 2026-10-15
- **来源文章：** 微信「具身站 31 篇论文」摘要，见 [sources/raw/wechat_embodied_station_robot_31_papers_2026-10-15.md](../raw/wechat_embodied_station_robot_31_papers_2026-10-15.md)（文章一句话：区分目标替代、策略捷径和线索竞争三种失效，恢复措施的有效性取决于失效发生的层次）
- **一句话说明：** 在三个 Procgen 环境与五个机器人任务中，区分目标替代（functional target substitution）、策略捷径（policy shortcut）与线索竞争（cue competition）三类策略-目标错配，并追踪目标信息在模型中的位置与影响深度。

## 开源状态（步骤 2.5，2026-10-15）

- **论文 / 文章声明：** arXiv 摘要页与文章均给出 `Julbarth/what-happens-to-the-goal` 仓库链接。
- **仓库核查：** 未能核实。curl 与 GitHub 只读接入均未能读取该仓库；未在本次入库中改动会话的仓库挂载。
- **暂定结论：** 已列出代码链接，**已开源程度未核实**（待能读取仓库后补充训练 / 评测入口与许可证）。

## 摘录 1：问题与主张（摘要）

- **痛点：** 智能体在分布外情形下行为能力仍在，但追求的目标与训练时不同（目标泛化失效）。以往恢复方法多把它当作单一问题处理。
- **主张：** 目标泛化失效至少包含三种不同的策略-目标错配，失效发生在策略的不同层次，因此恢复措施需按失效类型选择。
- **三类失效：**
  1. 功能性目标替代：行为跟随另一个替代目标。
  2. 策略捷径：行为跟随习得的动作策略，而非目标。
  3. 线索竞争：目标与代理目标同时影响决策，相互争夺。

## 摘录 2：实验设置与结果（摘要，WebFetch 摘要转述，未逐字核对原文）

- **环境：** 三个 Procgen 环境与五个机器人任务。
- **诊断方法：** 追踪目标信息残留在模型何处、对策略的影响强度，以及它在动作选择中承担的角色。
- **干预结果（摘要数字）：**
  - 目标已到达决策层：把策略导向目标位置后，目标可达率由 16.0% 升至 95.3%。
  - 目标到达策略层但未到决策层：仅新增下游连线有效，捷径环境中可达率由 31.6% 升至 84.3%。
  - 目标完全惰性：两种干预均无效。

## 摘录 3：对本库的意义

- 与 [OOD 泛化](../../wiki/concepts/out-of-distribution-generalization.md) 页的讨论直接相关：分布外失败不一定是能力缺失，也可能是目标信息在网络中的位置错误。
- 对 RL / 模仿学习的奖励设计与评测（如 [奖励设计](../../wiki/concepts/reward-design.md)）有提醒作用：只看任务成功率可能掩盖目标错配。

## 对 wiki 的映射

- 升格页面：[paper-goal-misgeneralization-policy-alignment](../../wiki/entities/paper-goal-misgeneralization-policy-alignment.md)
