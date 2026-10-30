# Kairos（arXiv:2610.39454）

> 来源归档（paper）

- **标题：** Kairos: Semantics-Aware Orchestration of Planning Engines across Classical, Numeric, Temporal and Hybrid PDDL
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.39454>
- **PDF：** <https://arxiv.org/pdf/2610.39454>
- **代码：** <https://github.com/PlanSys2/kairos>
- **入库日期：** 2026-10-30
- **线索来源：** [机器人做错了，奖励为什么还在增加？｜附代码与项目入口](../blogs/wechat_embodied_station_28_papers_event2reward_progress_rewards_2026-10-30.md)（具身智能小站，2026-10-30）
- **一句话说明：** 运行时规划既要覆盖时序、数值与连续变化，又要解释失败意味着什么。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-10-30）：仓库可达（git ls-remote）。

## 核心摘录（公众号转述，未核验原文表格）

- 让规划器按任务语义选择搜索引擎。Kairos 按 PDDL 语义调度异构引擎，并独立验证候选计划，为机器人系统提供更明确的规划接口。
- 运行时规划既要覆盖时序、数值与连续变化，又要解释失败意味着什么。
- 指标：公众号转述：失败语义进入接口：区分有证明的不可解与预算内未找到计划，避免上层把搜索超时误判成任务无解。

## 对 wiki 的映射

- [paper-kairos-pddl-planning-engine-orchestration](../../wiki/entities/paper-kairos-pddl-planning-engine-orchestration.md)
- [进度奖励、接触监督与规划验证：具身智能小站 10-30 期 28 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-30-28-papers-technology-map.md)
