# Physical RSI on RoboCasa

- **作者：** Rushi Qiang, Huijie Tang, Chenxiao Gao, Yizhou Zhao, Changhao Li, Na Li, Bo Dai
- **机构：** STACX Lab（Georgia Tech CSE；Bo Dai 兼任 Google DeepMind；Na Li 隶属 Harvard SEAS）
- **版本 / 载体：** STACX Lab 官方交互式项目页（run `S365best`）及配套架构专文《Who Improves the Improver?》（2026 年 10 月）
- **原文：** [Physical RSI on RoboCasa 项目页](https://stacx-lab.github.io/physical-rsi/)；[配套架构文章：Who Improves the Improver?](https://stacx-lab.github.io/blog/who-improves-the-improver/)
- **项目页 / 代码：** [项目页](https://stacx-lab.github.io/physical-rsi/)；未确认开源代码（项目页与 [STACX Lab Projects](https://stacx-lab.github.io/projects/) 仅提供 Website 入口，未公布控制循环、蒸馏主控与基准评测脚本仓库；项目页内嵌了最终 `V7` 版本 `rc` Python 包 70 个辅助函数静态源码与运行统计数据；其提及的通用物理智能体框架 Kinex 另行发布于 [kinex-agent.org](https://kinex-agent.org/)）
- **核对范围：** 通读项目页正文、内嵌 `D` 数据集（`tasks` 365 项、`versions` V0–V7 变更记录与预测裁决、`skills` 7 项元数据、`usage` 与 `final` 评测统计、`analysis.lib` 70 个 `rc` 函数清单）及 SVG 管线说明代码，并通读配套架构全文《Who Improves the Improver?》（§00–§08）
- **通读：** 已完成

## 原文核对笔记

1. **核心问题与架构主张（《Who Improves the Improver?》§01–§06）：**
   - 传统自改进智能体存在“无限递归（Infinite Regress）”难题：若将智能体视为整体，改进智能体 A 需要外部智能体 B，改进 B 又需要 C。
   - STACX Lab 提出**“一切皆技能（Everything is a skill）”**的微内核式设计：将执行核心压缩为极简、永不更改的“原始智能体（primitive agent）”主循环（接收观测 → 调用模型 → 执行返回 → 重复），其余能力（包括上下文管理、技能管理与自改进器本身）全部以技能形式放在核心之外。
   - 技能按统一代码接口（Python）组织，底层可以是三种形态之一并可由自改进器互相改写：**书面指令（Markdown）**、**固定代码（Python 函数）**、**子智能体（Subagents）**。由于自改进器自身也是技能之一，它可以改写包括自身代码/提示在内的全部技能，同时不动固定核心，从而以冯·诺依曼自复制自动机（固定构造器 + 可改写蓝图）的结构性拆分阻断无限递归，无需在系统之上叠设监控层。

2. **RoboCasa 上的 Physical RSI 运行机制（项目页 §01–§04 及 `pipeLoopsSVG`）：**
   - **固定控制器（Controller）：** 对厨房或机器人领域零先验，仅掌握 365 个任务 ID、试验台账（trial ledgers）与仿真器成功判定（verdicts）。
   - **工作智能体（Workers）：** 针对每个开放且未解决的任务并行启动（最多 120 并发），每个工作智能体在独立容器内运行，以 `Claude Opus 5.5` 为后端，通过编写并执行调用 `rc` 包的 Python 代码控制仿真机器人（Franka Panda 机械臂 + Omron 移动底盘）。每轮每任务分配 1 次试验（trial），最多 3 幕（episodes）；同一任务的所有幕共享同一初始场景、同一工作目录（跨幕保留工作智能体自写函数、技能与备忘录）以及全局库 `Vk`（`rc` 包、领域 Markdown 技能与 <=15 行的通用 brief）；对前一轮失败的任务额外下发 `task brief`（含已知有效步骤与 `Step-k` 变体）。
   - **主控自改进器（Principal / Self-improver）：** 与工作智能体并行运行（异步非阻塞）。主控读取截至当前所有试验索引后，按有新幕的任务派发子智能体分析员（sub-agent analysts）逐幕总结结构化行记录，将协议/技能库既有假设与日志逐一核对，按“导致结束的失败幕数”排序并仅修复最高频失败原因；每项修改必须附带一条可证伪的编号预测，在下一轮新数据上裁决（`KEPT` / `REFUTED` / `NOT MET` / `VOID`）。修改后的库需通过 `validate_library` 门禁（检查清单、技能结构与代码编译，失败最多触发 2 次短修复），通过后发布为 `Vk+1`。
   - **不训练模型权重：** 整个 365 任务求解过程完全不更新语言模型或策略网络权重（`No model is trained`），经验完全沉淀在技能文档与 `rc` 函数库中。

3. **核心实验结果（项目页 `D` 数据集核验）：**
   - **在线累计求解（Run `S365best`）：** 覆盖 RoboCasa 全部 365 个任务（65 个原子任务 + 300 个复合任务），分批推进（`10 / 20 / 40 / 80 / 160 / 55`）及后续重测，单机耗时约 `26.5` 小时（`95,531.6` s，页面显示 `27 hours`），使用 `Claude Opus 5.5` 累计花费 `493` 幕（`385` 次试验，平均每任务 `1.35` 幕）解决全部 `365 / 365` 任务；首次遭遇即在第 1 幕解决 `305` 项，首次试验（<=3 幕）内解决 `347` 项，其余 `18` 项经后续版本重测解决。
   - **最终库（`V7`）单轮复测对比空库（`V0`）：** 在相同 365 个场景上各跑 1 次试验（<=3 幕）：`V0` 空库在第 1 幕解决 `252 / 365`（`69%`）、1 次试验内解决 `325 / 365`（`89%`）、消耗 `545` 幕；`V7` 在第 1 幕解决 `309 / 365`（`85%`）、1 次试验内解决 `354 / 365`（`97%`，其中第 1/2/3 幕分别解决 `309 / 33 / 12` 项，仍有 `11` 项未过）、总计消耗 `444` 幕（平均 `1.22` 幕/任务）。
   - **最终知识库规模（`V7`）：** 共 `6` 个 Markdown 领域技能（`robocasa-basics`、`robocasa-pick-place`、`robocasa-doors`、`robocasa-controls`、`robocasa-navigate`、`robocasa-appliances`，合计 `1,411` 行）以及封装于 `rc-code` 技能（`2,338` 行）中的 `70` 个 `rc` Python 函数（`2,321` 行代码，分属 8 大类别）；在 `375` 次使用 `rc` 的试验中共被调用 `114,281` 次（实际使用其中 `61` 个函数）。

对 wiki 的映射：[Physical RSI on RoboCasa](../../wiki/entities/project-agentic-project-physical-rsi-robocasa.md)
