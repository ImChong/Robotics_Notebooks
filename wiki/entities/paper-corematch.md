---
type: entity
project_id: corematch
tags:
  - paper
  - world-model
  - perception
status: complete
updated: 2026-11-04
arxiv: "2611.03932"
code: https://github.com/edge-cv/CoReMatch
related:
  - ../concepts/world-action-models.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../overview/embodied-station-2026-11-04-57-papers-technology-map.md
sources:
  - ../../sources/papers/corematch_arxiv_2611_03932.md
  - ../../sources/repos/corematch.md
  - ../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md
summary: "CoReMatch（arXiv:2611.03932）：不再只在固定网格上预测匹配置信度，而是循环细化连续坐标；面向定位、建图与导航，在室内外基准和边缘硬件上比较精度与运行效率。"
---

# CoReMatch（arXiv:2611.03932）

**CoReMatch**（*CoReMatch: Recurrent Continuous Coordinate Refinement for Real-Time Robotic Feature Matching*，[arXiv:2611.03932](https://arxiv.org/abs/2611.03932)，[代码](https://github.com/edge-cv/CoReMatch)）来自 [具身智能小站 2026-11-04 盘点](../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)，属「世界模型与感知」组。

## 一句话定义

**不再只在固定网格上预测匹配置信度，而是循环细化连续坐标；面向定位、建图与导航，在室内外基准和边缘硬件上比较精度与运行效率。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CoReMatch | CoReMatch: Recurrent Continuous Coordinate Refinement for Real-Time Robotic Feature Matching | 本文方法 / 基准的简称 |

## 为什么重要

- 在连续坐标里做机器人特征匹配。
- 开源结论：**已开源**（步骤 2.5，2026-11-04）。仓库可达（git ls-remote）。
- 同期同组工作见 [VLM 奖励评估、动作动力学与开放权重机器人智能体：具身智能小站 11-04 期 57 篇阅读坐标](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2611.03932](https://arxiv.org/abs/2611.03932) |
| **开源** | **已开源** |
| **要点** | 不再只在固定网格上预测匹配置信度，而是循环细化连续坐标；面向定位、建图与导航，在室内外基准和边缘硬件上比较精度与运行效率。 |
| **文内指标** | 公众号未给具体数字，本页不引用。 |

## 源码运行时序图

以下为按仓库入口（`edge-cv/CoReMatch`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as edge-cv/CoReMatch
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 CoReMatch 模型
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

**在连续坐标里做机器人特征匹配：不再只在固定网格上预测匹配置信度，而是循环细化连续坐标；面向定位、建图与导航，在室内外基准和边缘硬件上比较精度与运行效率。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。
2. 核心机制：不再只在固定网格上预测匹配置信度，而是循环细化连续坐标；面向定位、建图与导航，在室内外基准和边缘硬件上比较精度与运行效率。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [world-action-models](../concepts/world-action-models.md)
- [generative-world-models](../methods/generative-world-models.md)
- [vla](../methods/vla.md)
- [embodied-station-2026-11-04-57-papers-technology-map](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)

## 参考来源

- [corematch_arxiv_2611_03932.md](../../sources/papers/corematch_arxiv_2611_03932.md)
- [wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md](../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)
- [arXiv:2611.03932](https://arxiv.org/abs/2611.03932)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.03932)
- [https://github.com/edge-cv/CoReMatch](https://github.com/edge-cv/CoReMatch)
