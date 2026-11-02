---
type: entity
project_id: logiv
tags:
  - paper
  - planning
  - perception
status: complete
updated: 2026-10-30
arxiv: "2610.40016"
project: https://enqiao-lu.github.io/logiv-project/
code: https://github.com/Enqiao-Lu/LOGIV
related:
  - ../queries/robot-perception-stack-selection-loop.md
  - ../tasks/locomotion.md
  - ../tasks/vision-language-navigation.md
  - ../tasks/manipulation.md
  - ../concepts/llm-robotics-control-interfaces.md
  - ../overview/embodied-station-2026-10-30-28-papers-technology-map.md
sources:
  - ../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md
  - ../../sources/papers/logiv_arxiv_2610_40016.md
  - ../../sources/repos/logiv.md
  - ../../sources/sites/logiv.md
  - ../../sources/blogs/wechat_embodied_station_28_papers_event2reward_progress_rewards_2026-10-30.md
summary: "LOGIV（arXiv:2610.40016）：把动作完成与条件仍然有效分开记录，沿任务图定位需要恢复的支持条件。"
---

# LOGIV（arXiv:2610.40016）

**LOGIV**（*LOGIV: Graph-Verified Execution with Causal Local Repair for Long-Horizon Robot Tasks*，[arXiv:2610.40016](https://arxiv.org/abs/2610.40016)，[项目页](https://enqiao-lu.github.io/logiv-project/)，[代码](https://github.com/Enqiao-Lu/LOGIV)）来自 [具身智能小站 2026-10-30 盘点](../../sources/blogs/wechat_embodied_station_28_papers_event2reward_progress_rewards_2026-10-30.md)，属「规划、感知与建图」组。

## 一句话定义

**把动作完成与条件仍然有效分开记录，沿任务图定位需要恢复的支持条件。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| LOGIV | LOGIV: Graph-Verified Execution with Causal Local Repair for Long-Horizon Robot Tasks | 本文方法 / 基准的简称 |

## 为什么重要

- 长任务出错后，只修复失效的依赖。LOGIV 在已有策略之上加入验证和局部修复；真机提升仍受底层接触精度约束。
- 开源结论：**已开源**（步骤 2.5，2026-10-30）。仓库可达（git ls-remote）。项目页 HTTP 200。
- 同期同组工作见 [进度奖励、接触监督与规划验证：具身智能小站 10-30 期 28 篇阅读坐标](../overview/embodied-station-2026-10-30-28-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.40016](https://arxiv.org/abs/2610.40016) |
| **开源** | **已开源** |
| **要点** | 把动作完成与条件仍然有效分开记录，沿任务图定位需要恢复的支持条件。 |
| **文内指标** | 公众号转述：LOGIV 在已有策略之上加入验证和局部修复；真机提升仍受底层接触精度约束。 |

## 源码运行时序图

以下为按仓库入口（`Enqiao-Lu/LOGIV`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as Enqiao-Lu/LOGIV
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 LOGIV 模型
  U->>R: 运行评测脚本
  R->>E: 重置环境、取观测
  E-->>M: 观测
  M-->>E: 动作 / 预测
  R-->>U: 指标与日志
```

## 实验与评测

- 公众号转述：LOGIV 在已有策略之上加入验证和局部修复；真机提升仍受底层接触精度约束。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [进度奖励、接触监督与规划验证：具身智能小站 10-30 期 28 篇阅读坐标](../overview/embodied-station-2026-10-30-28-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**长任务出错后，只修复失效的依赖：LOGIV 在已有策略之上加入验证和局部修复；真机提升仍受底层接触精度约束。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。项目页 HTTP 200。
2. 核心机制：把动作完成与条件仍然有效分开记录，沿任务图定位需要恢复的支持条件。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [locomotion](../tasks/locomotion.md)
- [vision-language-navigation](../tasks/vision-language-navigation.md)
- [manipulation](../tasks/manipulation.md)
- [llm-robotics-control-interfaces](../concepts/llm-robotics-control-interfaces.md)
- [embodied-station-2026-10-30-28-papers-technology-map](../overview/embodied-station-2026-10-30-28-papers-technology-map.md)
- [Query：机器人视觉感知栈选型](../queries/robot-perception-stack-selection-loop.md)

## 参考来源

- [logiv_arxiv_2610_40016.md](../../sources/papers/logiv_arxiv_2610_40016.md)
- [wechat_embodied_station_28_papers_event2reward_progress_rewards_2026-10-30.md](../../sources/blogs/wechat_embodied_station_28_papers_event2reward_progress_rewards_2026-10-30.md)
- [arXiv:2610.40016](https://arxiv.org/abs/2610.40016)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.40016)
- [项目页](https://enqiao-lu.github.io/logiv-project/)
- [https://github.com/Enqiao-Lu/LOGIV](https://github.com/Enqiao-Lu/LOGIV)
