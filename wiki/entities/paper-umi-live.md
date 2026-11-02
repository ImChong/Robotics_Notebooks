---
type: entity
project_id: umi-live
tags:
  - paper
  - data-collection
  - evaluation
status: complete
updated: 2026-10-30
arxiv: "2610.39385"
project: https://umi-live.github.io/
code: https://github.com/AfterQuery-Research/umi-live
related:
  - ../queries/embodied-eval-benchmark-selection-loop.md
  - ../concepts/embodied-data-collection-four-layers-taxonomy.md
  - ../methods/imitation-learning.md
  - ../concepts/simulation-evaluation-infrastructure.md
  - ../tasks/teleoperation.md
  - ../tasks/bimanual-manipulation.md
  - ../overview/embodied-station-2026-10-30-28-papers-technology-map.md
sources:
  - ../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md
  - ../../sources/papers/umi_live_arxiv_2610_39385.md
  - ../../sources/repos/umi_live.md
  - ../../sources/sites/umi-live.md
  - ../../sources/blogs/wechat_embodied_station_28_papers_event2reward_progress_rewards_2026-10-30.md
summary: "UMI-Live（arXiv:2610.39385）：UMI-Live 集成相机、激光惯性定位和异步重建，连接手持采集、数据处理与策略训练。"
---

# UMI-Live（arXiv:2610.39385）

**UMI-Live**（*UMI-Live for In-The-Wild Bimanual Robotics Data Collection*，[arXiv:2610.39385](https://arxiv.org/abs/2610.39385)，[项目页](https://umi-live.github.io/)，[代码](https://github.com/AfterQuery-Research/umi-live)）来自 [具身智能小站 2026-10-30 盘点](../../sources/blogs/wechat_embodied_station_28_papers_event2reward_progress_rewards_2026-10-30.md)，属「示范数据与评测」组。

## 一句话定义

**UMI-Live 集成相机、激光惯性定位和异步重建，连接手持采集、数据处理与策略训练。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| UMI-Live | UMI-Live for In-The-Wild Bimanual Robotics Data Collection | 本文方法 / 基准的简称 |
| UMI | Universal Manipulation Interface | 手持通用操作采集接口 |

## 为什么重要

- 把双手示范采集接到训练数据流水线。双手搬运验证每个模型仅十次测试，应把它作为数据管线证据理解。
- 开源结论：**已开源**（步骤 2.5，2026-10-30）。仓库可达（git ls-remote）。项目页 HTTP 200。
- 同期同组工作见 [进度奖励、接触监督与规划验证：具身智能小站 10-30 期 28 篇阅读坐标](../overview/embodied-station-2026-10-30-28-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.39385](https://arxiv.org/abs/2610.39385) |
| **开源** | **已开源** |
| **要点** | UMI-Live 集成相机、激光惯性定位和异步重建，连接手持采集、数据处理与策略训练。 |
| **文内指标** | 公众号转述：双手搬运验证每个模型仅十次测试，应把它作为数据管线证据理解。 |

## 源码运行时序图

以下为按仓库入口（`AfterQuery-Research/umi-live`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as AfterQuery-Research/umi-live
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 UMI-Live 模型
  U->>R: 运行评测脚本
  R->>E: 重置环境、取观测
  E-->>M: 观测
  M-->>E: 动作 / 预测
  R-->>U: 指标与日志
```

## 实验与评测

- 公众号转述：双手搬运验证每个模型仅十次测试，应把它作为数据管线证据理解。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [进度奖励、接触监督与规划验证：具身智能小站 10-30 期 28 篇阅读坐标](../overview/embodied-station-2026-10-30-28-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**把双手示范采集接到训练数据流水线：双手搬运验证每个模型仅十次测试，应把它作为数据管线证据理解。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。项目页 HTTP 200。
2. 核心机制：UMI-Live 集成相机、激光惯性定位和异步重建，连接手持采集、数据处理与策略训练。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [embodied-data-collection-four-layers-taxonomy](../concepts/embodied-data-collection-four-layers-taxonomy.md)
- [imitation-learning](../methods/imitation-learning.md)
- [simulation-evaluation-infrastructure](../concepts/simulation-evaluation-infrastructure.md)
- [teleoperation](../tasks/teleoperation.md)
- [bimanual-manipulation](../tasks/bimanual-manipulation.md)
- [embodied-station-2026-10-30-28-papers-technology-map](../overview/embodied-station-2026-10-30-28-papers-technology-map.md)
- [Query：具身大模型评测基准选型](../queries/embodied-eval-benchmark-selection-loop.md)

## 参考来源

- [umi_live_arxiv_2610_39385.md](../../sources/papers/umi_live_arxiv_2610_39385.md)
- [wechat_embodied_station_28_papers_event2reward_progress_rewards_2026-10-30.md](../../sources/blogs/wechat_embodied_station_28_papers_event2reward_progress_rewards_2026-10-30.md)
- [arXiv:2610.39385](https://arxiv.org/abs/2610.39385)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.39385)
- [项目页](https://umi-live.github.io/)
- [https://github.com/AfterQuery-Research/umi-live](https://github.com/AfterQuery-Research/umi-live)
