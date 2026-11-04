---
type: entity
project_id: dexarena
tags:
  - paper
  - benchmark
  - evaluation
status: complete
updated: 2026-11-04
arxiv: "2611.02629"
project: https://sipair-elastic.github.io/DexArena-website/
code: https://github.com/SIPAIR-Elastic/DexArena
related:
  - ../concepts/simulation-evaluation-infrastructure.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../overview/embodied-station-2026-11-04-57-papers-technology-map.md
sources:
  - ../../sources/papers/dexarena_arxiv_2611_02629.md
  - ../../sources/repos/dexarena.md
  - ../../sources/sites/dexarena.md
  - ../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md
summary: "DexArena（arXiv:2611.02629）：面向双臂灵巧操作提供分阶段任务定义，同时衡量中间进度与整段完成；覆盖家庭和工业场景，有利于区分单技能能力和跨阶段衔接的失败来源。"
---

# DexArena（arXiv:2611.02629）

**DexArena**（*DexArena: A Unified Benchmark for Long-Horizon Dexterous Manipulation*，[arXiv:2611.02629](https://arxiv.org/abs/2611.02629)，[项目页](https://sipair-elastic.github.io/DexArena-website/)，[代码](https://github.com/SIPAIR-Elastic/DexArena)）来自 [具身智能小站 2026-11-04 盘点](../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)，属「评测、数据与基准」组。

## 一句话定义

**面向双臂灵巧操作提供分阶段任务定义，同时衡量中间进度与整段完成；覆盖家庭和工业场景，有利于区分单技能能力和跨阶段衔接的失败来源。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DexArena | DexArena: A Unified Benchmark for Long-Horizon Dexterous Manipulation | 本文方法 / 基准的简称 |

## 为什么重要

- 把灵巧手长任务拆成可检验阶段。
- 开源结论：**已开源**（步骤 2.5，2026-11-04）。仓库可达（git ls-remote）。项目页 HTTP 200。
- 同期同组工作见 [VLM 奖励评估、动作动力学与开放权重机器人智能体：具身智能小站 11-04 期 57 篇阅读坐标](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2611.02629](https://arxiv.org/abs/2611.02629) |
| **开源** | **已开源** |
| **要点** | 面向双臂灵巧操作提供分阶段任务定义，同时衡量中间进度与整段完成；覆盖家庭和工业场景，有利于区分单技能能力和跨阶段衔接的失败来源。 |
| **文内指标** | 公众号未给具体数字，本页不引用。 |

## 源码运行时序图

以下为按仓库入口（`SIPAIR-Elastic/DexArena`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as SIPAIR-Elastic/DexArena
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 DexArena 模型
  U->>R: 运行评测脚本
  R->>E: 重置环境、取观测
  E-->>M: 观测
  M-->>E: 动作 / 预测
  R-->>U: 指标与日志
```

## 实验与评测

- 公众号未给具体数字，本页不引用。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [VLM 奖励评估、动作动力学与开放权重机器人智能体：具身智能小站 11-04 期 57 篇阅读坐标](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**把灵巧手长任务拆成可检验阶段：面向双臂灵巧操作提供分阶段任务定义，同时衡量中间进度与整段完成；覆盖家庭和工业场景，有利于区分单技能能力和跨阶段衔接的失败来源。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。项目页 HTTP 200。
2. 核心机制：面向双臂灵巧操作提供分阶段任务定义，同时衡量中间进度与整段完成；覆盖家庭和工业场景，有利于区分单技能能力和跨阶段衔接的失败来源。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [simulation-evaluation-infrastructure](../concepts/simulation-evaluation-infrastructure.md)
- [vla](../methods/vla.md)
- [manipulation](../tasks/manipulation.md)
- [embodied-station-2026-11-04-57-papers-technology-map](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)

## 参考来源

- [dexarena_arxiv_2611_02629.md](../../sources/papers/dexarena_arxiv_2611_02629.md)
- [wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md](../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)
- [arXiv:2611.02629](https://arxiv.org/abs/2611.02629)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.02629)
- [项目页](https://sipair-elastic.github.io/DexArena-website/)
- [https://github.com/SIPAIR-Elastic/DexArena](https://github.com/SIPAIR-Elastic/DexArena)
