---
type: entity
project_id: sensorium
tags:
  - paper
  - benchmark
  - evaluation
status: complete
updated: 2026-11-03
arxiv: "2610.41946"
code: https://github.com/nct-tso-robotics/tso_sensorium
related:
  - ../concepts/simulation-evaluation-infrastructure.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../overview/embodied-station-2026-11-03-49-papers-technology-map.md
sources:
  - ../../sources/papers/sensorium_arxiv_2610_41946.md
  - ../../sources/repos/sensorium.md
  - ../../sources/blogs/wechat_embodied_station_49_papers_sacq_action_aware_quantization_2026-11-03.md
summary: "Sensorium（arXiv:2610.41946）：Sensorium 基于 ROS 与 ROS 2 串联记录、处理、语义标注和回合数据导出，覆盖模型、离体与动物实验数据；主要贡献是可配置的数据工程流程。"
---

# Sensorium（arXiv:2610.41946）

**Sensorium**（*Sensorium: An Open-Source Multimodal Data Collection and Curation Library for Embodied Surgical AI*，[arXiv:2610.41946](https://arxiv.org/abs/2610.41946)，[代码](https://github.com/nct-tso-robotics/tso_sensorium)）来自 [具身智能小站 2026-11-03 盘点](../../sources/blogs/wechat_embodied_station_49_papers_sacq_action_aware_quantization_2026-11-03.md)，属「评测、数据与基准」组。

## 一句话定义

**Sensorium 基于 ROS 与 ROS 2 串联记录、处理、语义标注和回合数据导出，覆盖模型、离体与动物实验数据；主要贡献是可配置的数据工程流程。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sensorium | Sensorium: An Open-Source Multimodal Data Collection and Curation Library for Embodied Surgical AI | 本文方法 / 基准的简称 |

## 为什么重要

- 统一手术机器人多模态数据流程。
- 开源结论：**已开源**（步骤 2.5，2026-11-03）。仓库可达（git ls-remote）。
- 同期同组工作见 [动作感知量化、失败定位与可验证自主：具身智能小站 11-03 期 49 篇阅读坐标](../overview/embodied-station-2026-11-03-49-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.41946](https://arxiv.org/abs/2610.41946) |
| **开源** | **已开源** |
| **要点** | Sensorium 基于 ROS 与 ROS 2 串联记录、处理、语义标注和回合数据导出，覆盖模型、离体与动物实验数据；主要贡献是可配置的数据工程流程。 |
| **文内指标** | 公众号未给具体数字，本页不引用。 |

## 源码运行时序图

以下为按仓库入口（`nct-tso-robotics/tso_sensorium`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as nct-tso-robotics/tso_sensorium
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 Sensorium 模型
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

- 横向索引见 [动作感知量化、失败定位与可验证自主：具身智能小站 11-03 期 49 篇阅读坐标](../overview/embodied-station-2026-11-03-49-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**统一手术机器人多模态数据流程：Sensorium 基于 ROS 与 ROS 2 串联记录、处理、语义标注和回合数据导出，覆盖模型、离体与动物实验数据；主要贡献是可配置的数据工程流程。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。
2. 核心机制：Sensorium 基于 ROS 与 ROS 2 串联记录、处理、语义标注和回合数据导出，覆盖模型、离体与动物实验数据；主要贡献是可配置的数据工程流程。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [simulation-evaluation-infrastructure](../concepts/simulation-evaluation-infrastructure.md)
- [vla](../methods/vla.md)
- [manipulation](../tasks/manipulation.md)
- [embodied-station-2026-11-03-49-papers-technology-map](../overview/embodied-station-2026-11-03-49-papers-technology-map.md)

## 参考来源

- [sensorium_arxiv_2610_41946.md](../../sources/papers/sensorium_arxiv_2610_41946.md)
- [wechat_embodied_station_49_papers_sacq_action_aware_quantization_2026-11-03.md](../../sources/blogs/wechat_embodied_station_49_papers_sacq_action_aware_quantization_2026-11-03.md)
- [arXiv:2610.41946](https://arxiv.org/abs/2610.41946)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.41946)
- [https://github.com/nct-tso-robotics/tso_sensorium](https://github.com/nct-tso-robotics/tso_sensorium)
