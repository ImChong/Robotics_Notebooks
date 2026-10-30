---
type: entity
project_id: kairos-pddl-planning-engine-orchestration
tags:
  - paper
  - planning
  - perception
status: complete
updated: 2026-10-30
arxiv: "2610.39454"
code: https://github.com/PlanSys2/kairos
related:
  - ../tasks/locomotion.md
  - ../tasks/vision-language-navigation.md
  - ../tasks/manipulation.md
  - ../overview/embodied-station-2026-10-30-28-papers-technology-map.md
sources:
  - ../../sources/papers/kairos_pddl_planning_engine_orchestration_arxiv_2610_39454.md
  - ../../sources/repos/kairos_pddl_planning_engine_orchestration.md
  - ../../sources/blogs/wechat_embodied_station_28_papers_event2reward_progress_rewards_2026-10-30.md
summary: "Kairos（arXiv:2610.39454）：运行时规划既要覆盖时序、数值与连续变化，又要解释失败意味着什么。"
---

# Kairos（arXiv:2610.39454）

**Kairos**（*Kairos: Semantics-Aware Orchestration of Planning Engines across Classical, Numeric, Temporal and Hybrid PDDL*，[arXiv:2610.39454](https://arxiv.org/abs/2610.39454)，[代码](https://github.com/PlanSys2/kairos)）来自 [具身智能小站 2026-10-30 盘点](../../sources/blogs/wechat_embodied_station_28_papers_event2reward_progress_rewards_2026-10-30.md)，属「规划、感知与建图」组。

## 一句话定义

**运行时规划既要覆盖时序、数值与连续变化，又要解释失败意味着什么。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Kairos | Kairos: Semantics-Aware Orchestration of Planning Engines across Classical, Numeric, Temporal and Hybrid PDDL | 本文方法 / 基准的简称 |

## 为什么重要

- 让规划器按任务语义选择搜索引擎。Kairos 按 PDDL 语义调度异构引擎，并独立验证候选计划，为机器人系统提供更明确的规划接口。
- 开源结论：**已开源**（步骤 2.5，2026-10-30）。仓库可达（git ls-remote）。
- 同期同组工作见 [进度奖励、接触监督与规划验证：具身智能小站 10-30 期 28 篇阅读坐标](../overview/embodied-station-2026-10-30-28-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.39454](https://arxiv.org/abs/2610.39454) |
| **开源** | **已开源** |
| **要点** | 运行时规划既要覆盖时序、数值与连续变化，又要解释失败意味着什么。 |
| **文内指标** | 公众号转述：失败语义进入接口：区分有证明的不可解与预算内未找到计划，避免上层把搜索超时误判成任务无解。 |

## 源码运行时序图

以下为按仓库入口（`PlanSys2/kairos`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as PlanSys2/kairos
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 Kairos 模型
  U->>R: 运行评测脚本
  R->>E: 重置环境、取观测
  E-->>M: 观测
  M-->>E: 动作 / 预测
  R-->>U: 指标与日志
```

## 实验与评测

- 公众号转述：失败语义进入接口：区分有证明的不可解与预算内未找到计划，避免上层把搜索超时误判成任务无解。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [进度奖励、接触监督与规划验证：具身智能小站 10-30 期 28 篇阅读坐标](../overview/embodied-station-2026-10-30-28-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**让规划器按任务语义选择搜索引擎：失败语义进入接口：区分有证明的不可解与预算内未找到计划，避免上层把搜索超时误判成任务无解。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。
2. 核心机制：运行时规划既要覆盖时序、数值与连续变化，又要解释失败意味着什么。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [locomotion](../tasks/locomotion.md)
- [vision-language-navigation](../tasks/vision-language-navigation.md)
- [manipulation](../tasks/manipulation.md)
- [embodied-station-2026-10-30-28-papers-technology-map](../overview/embodied-station-2026-10-30-28-papers-technology-map.md)

## 参考来源

- [kairos_pddl_planning_engine_orchestration_arxiv_2610_39454.md](../../sources/papers/kairos_pddl_planning_engine_orchestration_arxiv_2610_39454.md)
- [wechat_embodied_station_28_papers_event2reward_progress_rewards_2026-10-30.md](../../sources/blogs/wechat_embodied_station_28_papers_event2reward_progress_rewards_2026-10-30.md)
- [arXiv:2610.39454](https://arxiv.org/abs/2610.39454)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.39454)
- [https://github.com/PlanSys2/kairos](https://github.com/PlanSys2/kairos)
