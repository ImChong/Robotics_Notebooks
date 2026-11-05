---
type: entity
project_id: auriga-technical-report
tags:
  - paper
  - vla
  - policy-deployment
status: complete
updated: 2026-11-03
arxiv: "2611.01112"
project: https://auriga.stelledge.com/
code: https://github.com/lose4578/Auriga
related:
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../methods/reinforcement-learning.md
  - ../overview/embodied-station-2026-11-03-49-papers-technology-map.md
sources:
  - ../../sources/papers/auriga_technical_report_arxiv_2611_01112.md
  - ../../sources/repos/auriga_technical_report.md
  - ../../sources/sites/auriga-technical-report.md
  - ../../sources/blogs/wechat_embodied_station_49_papers_sacq_action_aware_quantization_2026-11-03.md
summary: "Auriga Technical Report（arXiv:2611.01112）：Auriga 将三维测量、动作预览与执行检查组织成统一控制接口，使模型在偏差发生后原地恢复。"
---

# Auriga Technical Report（arXiv:2611.01112）

**Auriga Technical Report**（*Auriga Technical Report: A Task-Agnostic Agent Harness for Zero-Shot Embodied Manipulation*，[arXiv:2611.01112](https://arxiv.org/abs/2611.01112)，[项目页](https://auriga.stelledge.com/)，[代码](https://github.com/lose4578/Auriga)）来自 [具身智能小站 2026-11-03 盘点](../../sources/blogs/wechat_embodied_station_49_papers_sacq_action_aware_quantization_2026-11-03.md)，属「策略学习与部署」组。

## 一句话定义

**Auriga 将三维测量、动作预览与执行检查组织成统一控制接口，使模型在偏差发生后原地恢复。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Auriga Technical Report | Auriga Technical Report: A Task-Agnostic Agent Harness for Zero-Shot Embodied Manipulation | 本文方法 / 基准的简称 |

## 为什么重要

- 让机器人在失败现场继续执行。当前报告的主要成绩来自仿真操作基准。
- 开源结论：**已开源**（步骤 2.5，2026-11-03）。仓库可达（git ls-remote）。项目页 HTTP 200。
- 同期同组工作见 [动作感知量化、失败定位与可验证自主：具身智能小站 11-03 期 49 篇阅读坐标](../overview/embodied-station-2026-11-03-49-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2611.01112](https://arxiv.org/abs/2611.01112) |
| **开源** | **已开源** |
| **要点** | Auriga 将三维测量、动作预览与执行检查组织成统一控制接口，使模型在偏差发生后原地恢复。 |
| **文内指标** | 公众号转述：当前报告的主要成绩来自仿真操作基准。 |

## 源码运行时序图

以下为按仓库入口（`lose4578/Auriga`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as lose4578/Auriga
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 Auriga Technical Report 模型
  U->>R: 运行评测脚本
  R->>E: 重置环境、取观测
  E-->>M: 观测
  M-->>E: 动作 / 预测
  R-->>U: 指标与日志
```

## 实验与评测

- 公众号转述：当前报告的主要成绩来自仿真操作基准。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [动作感知量化、失败定位与可验证自主：具身智能小站 11-03 期 49 篇阅读坐标](../overview/embodied-station-2026-11-03-49-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**让机器人在失败现场继续执行：当前报告的主要成绩来自仿真操作基准。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。项目页 HTTP 200。
2. 核心机制：Auriga 将三维测量、动作预览与执行检查组织成统一控制接口，使模型在偏差发生后原地恢复。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [vla](../methods/vla.md)
- [imitation-learning](../methods/imitation-learning.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)
- [embodied-station-2026-11-03-49-papers-technology-map](../overview/embodied-station-2026-11-03-49-papers-technology-map.md)

## 参考来源

- [auriga_technical_report_arxiv_2611_01112.md](../../sources/papers/auriga_technical_report_arxiv_2611_01112.md)
- [wechat_embodied_station_49_papers_sacq_action_aware_quantization_2026-11-03.md](../../sources/blogs/wechat_embodied_station_49_papers_sacq_action_aware_quantization_2026-11-03.md)
- [arXiv:2611.01112](https://arxiv.org/abs/2611.01112)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.01112)
- [项目页](https://auriga.stelledge.com/)
- [https://github.com/lose4578/Auriga](https://github.com/lose4578/Auriga)
