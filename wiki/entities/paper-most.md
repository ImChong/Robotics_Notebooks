---
type: entity
project_id: most
tags:
  - paper
  - world-model
  - memory
status: complete
updated: 2026-11-03
arxiv: "2611.00734"
project: https://most-vla.github.io/
code: https://github.com/diqihe666/MOST-vla
related:
  - ../concepts/world-action-models.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../overview/embodied-station-2026-11-03-49-papers-technology-map.md
sources:
  - ../../sources/papers/most_arxiv_2611_00734.md
  - ../../sources/repos/most.md
  - ../../sources/sites/most.md
  - ../../sources/blogs/wechat_embodied_station_49_papers_sacq_action_aware_quantization_2026-11-03.md
summary: "MOST（arXiv:2611.00734）：MOST 用物体状态和子任务进度监督时序记忆，使动作生成能够利用已经离开视野的信息。"
---

# MOST（arXiv:2611.00734）

**MOST**（*MOST: Memory of Object States and Task Progress for Long-Horizon Mobile Manipulation*，[arXiv:2611.00734](https://arxiv.org/abs/2611.00734)，[项目页](https://most-vla.github.io/)，[代码](https://github.com/diqihe666/MOST-vla)）来自 [具身智能小站 2026-11-03 盘点](../../sources/blogs/wechat_embodied_station_49_papers_sacq_action_aware_quantization_2026-11-03.md)，属「世界模型、记忆与交互」组。

## 一句话定义

**MOST 用物体状态和子任务进度监督时序记忆，使动作生成能够利用已经离开视野的信息。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MOST | MOST: Memory of Object States and Task Progress for Long-Horizon Mobile Manipulation | 本文方法 / 基准的简称 |

## 为什么重要

- 长任务记忆要记住状态变化。阅读时应区分任务成功率和最大任务进度两种指标。
- 开源结论：**已开源**（步骤 2.5，2026-11-03）。仓库可达（git ls-remote）。项目页 HTTP 200。
- 同期同组工作见 [动作感知量化、失败定位与可验证自主：具身智能小站 11-03 期 49 篇阅读坐标](../overview/embodied-station-2026-11-03-49-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2611.00734](https://arxiv.org/abs/2611.00734) |
| **开源** | **已开源** |
| **要点** | MOST 用物体状态和子任务进度监督时序记忆，使动作生成能够利用已经离开视野的信息。 |
| **文内指标** | 公众号转述：阅读时应区分任务成功率和最大任务进度两种指标。 |

## 源码运行时序图

以下为按仓库入口（`diqihe666/MOST-vla`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as diqihe666/MOST-vla
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 MOST 模型
  U->>R: 运行评测脚本
  R->>E: 重置环境、取观测
  E-->>M: 观测
  M-->>E: 动作 / 预测
  R-->>U: 指标与日志
```

## 实验与评测

- 公众号转述：阅读时应区分任务成功率和最大任务进度两种指标。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [动作感知量化、失败定位与可验证自主：具身智能小站 11-03 期 49 篇阅读坐标](../overview/embodied-station-2026-11-03-49-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**长任务记忆要记住状态变化：阅读时应区分任务成功率和最大任务进度两种指标。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。项目页 HTTP 200。
2. 核心机制：MOST 用物体状态和子任务进度监督时序记忆，使动作生成能够利用已经离开视野的信息。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [world-action-models](../concepts/world-action-models.md)
- [generative-world-models](../methods/generative-world-models.md)
- [vla](../methods/vla.md)
- [embodied-station-2026-11-03-49-papers-technology-map](../overview/embodied-station-2026-11-03-49-papers-technology-map.md)

## 参考来源

- [most_arxiv_2611_00734.md](../../sources/papers/most_arxiv_2611_00734.md)
- [wechat_embodied_station_49_papers_sacq_action_aware_quantization_2026-11-03.md](../../sources/blogs/wechat_embodied_station_49_papers_sacq_action_aware_quantization_2026-11-03.md)
- [arXiv:2611.00734](https://arxiv.org/abs/2611.00734)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.00734)
- [项目页](https://most-vla.github.io/)
- [https://github.com/diqihe666/MOST-vla](https://github.com/diqihe666/MOST-vla)
