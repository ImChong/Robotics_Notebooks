---
type: entity
project_id: bidex-100
tags:
  - paper
  - benchmark
  - evaluation
status: complete
updated: 2026-11-01
arxiv: "2610.25459"
project: https://robbyant-research.github.io/BiDex-100/
code: https://github.com/robbyant-research/BiDex-100
related:
  - ../concepts/simulation-evaluation-infrastructure.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/bimanual-manipulation.md
  - ../overview/embodied-station-2026-11-01-56-papers-technology-map.md
  - ../queries/embodied-eval-benchmark-selection-loop.md
sources:
  - ../../sources/papers/bidex_100_arxiv_2610_25459.md
  - ../../sources/repos/bidex_100.md
  - ../../sources/sites/bidex-100.md
  - ../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md
summary: "BiDex-100（arXiv:2610.25459）：BiDex-100用双臂多指手任务分别测记忆、长程、指令、少示范和泛化，并配对域内外布局；统一评测多种VLA，结果仍属于刚体仿真范围。"
---

# BiDex-100（arXiv:2610.25459）

**BiDex-100**（*BiDex-100: A Comprehensive Benchmark for Bimanual Dexterous Manipulation Across 100+ Simulation Tasks*，[arXiv:2610.25459](https://arxiv.org/abs/2610.25459)，[项目页](https://robbyant-research.github.io/BiDex-100/)，[代码](https://github.com/robbyant-research/BiDex-100)）来自 [具身智能小站 2026-11-01 盘点](../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md)，属「评测与基准」组。

## 一句话定义

**BiDex-100用双臂多指手任务分别测记忆、长程、指令、少示范和泛化，并配对域内外布局；统一评测多种VLA，结果仍属于刚体仿真范围。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BiDex-100 | BiDex-100: A Comprehensive Benchmark for Bimanual Dexterous Manipulation Across 100+ Simulation Tasks | 本文方法 / 基准的简称 |

## 为什么重要

- 双手灵巧能力，分轴测而非只看均分。
- 开源结论：**已开源**（步骤 2.5，2026-11-01）。仓库可达（git ls-remote）。项目页 HTTP 200。
- 同期同组工作见 [动态评测、技能编译与环境潜变量适配：具身智能小站 11-01 期 56 篇阅读坐标](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.25459](https://arxiv.org/abs/2610.25459) |
| **开源** | **已开源** |
| **要点** | BiDex-100用双臂多指手任务分别测记忆、长程、指令、少示范和泛化，并配对域内外布局；统一评测多种VLA，结果仍属于刚体仿真范围。 |
| **文内指标** | 公众号未给具体数字，本页不引用。 |

## 源码运行时序图

以下为按仓库入口（`robbyant-research/BiDex-100`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as robbyant-research/BiDex-100
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 BiDex-100 模型
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

- 横向索引见 [动态评测、技能编译与环境潜变量适配：具身智能小站 11-01 期 56 篇阅读坐标](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**双手灵巧能力，分轴测而非只看均分：BiDex-100用双臂多指手任务分别测记忆、长程、指令、少示范和泛化，并配对域内外布局；统一评测多种VLA，结果仍属于刚体仿真范围。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。项目页 HTTP 200。
2. 核心机制：BiDex-100用双臂多指手任务分别测记忆、长程、指令、少示范和泛化，并配对域内外布局；统一评测多种VLA，结果仍属于刚体仿真范围。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [simulation-evaluation-infrastructure](../concepts/simulation-evaluation-infrastructure.md)
- [vla](../methods/vla.md)
- [manipulation](../tasks/manipulation.md)
- [bimanual-manipulation](../tasks/bimanual-manipulation.md)
- [embodied-station-2026-11-01-56-papers-technology-map](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)
- [具身大模型评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md) — 评测基准在选型闭环中的定位与取舍

## 参考来源

- [bidex_100_arxiv_2610_25459.md](../../sources/papers/bidex_100_arxiv_2610_25459.md)
- [wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md](../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md)
- [arXiv:2610.25459](https://arxiv.org/abs/2610.25459)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.25459)
- [项目页](https://robbyant-research.github.io/BiDex-100/)
- [https://github.com/robbyant-research/BiDex-100](https://github.com/robbyant-research/BiDex-100)
