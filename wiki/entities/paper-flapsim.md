---
type: entity
project_id: flapsim
tags:
  - paper
  - planning
  - optimization
status: complete
updated: 2026-11-03
arxiv: "2610.40902"
code: https://github.com/MingshanHe/FlapSim
related:
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
  - ../concepts/sim2real.md
  - ../overview/embodied-station-2026-11-03-49-papers-technology-map.md
sources:
  - ../../sources/papers/flapsim_arxiv_2610_40902.md
  - ../../sources/repos/flapsim.md
  - ../../sources/blogs/wechat_embodied_station_49_papers_sacq_action_aware_quantization_2026-11-03.md
summary: "FlapSim（arXiv:2610.40902）：FlapSim 将机器人描述、翼几何和多层次气动模型组合进 MuJoCo，支持可重复实验。"
---

# FlapSim（arXiv:2610.40902）

**FlapSim**（*FlapSim: A Composable MuJoCo Simulation Platform and Reduced-Order Aerodynamic Model Hierarchy for Flapping-Wing Robotics*，[arXiv:2610.40902](https://arxiv.org/abs/2610.40902)，[代码](https://github.com/MingshanHe/FlapSim)）来自 [具身智能小站 2026-11-03 盘点](../../sources/blogs/wechat_embodied_station_49_papers_sacq_action_aware_quantization_2026-11-03.md)，属「规划、优化与感知基础」组。

## 一句话定义

**FlapSim 将机器人描述、翼几何和多层次气动模型组合进 MuJoCo，支持可重复实验。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| FlapSim | FlapSim: A Composable MuJoCo Simulation Platform and Reduced-Order Aerodynamic Model Hierarchy for Flapping-Wing Robotics | 本文方法 / 基准的简称 |

## 为什么重要

- 扑翼仿真要兼顾气动与控制。报告的定量气动比较主要使用固定基座，不应等同自由飞行验证。
- 开源结论：**已开源**（步骤 2.5，2026-11-03）。仓库可达（git ls-remote）。
- 同期同组工作见 [动作感知量化、失败定位与可验证自主：具身智能小站 11-03 期 49 篇阅读坐标](../overview/embodied-station-2026-11-03-49-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.40902](https://arxiv.org/abs/2610.40902) |
| **开源** | **已开源** |
| **要点** | FlapSim 将机器人描述、翼几何和多层次气动模型组合进 MuJoCo，支持可重复实验。 |
| **文内指标** | 公众号转述：报告的定量气动比较主要使用固定基座，不应等同自由飞行验证。 |

## 源码运行时序图

以下为按仓库入口（`MingshanHe/FlapSim`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as MingshanHe/FlapSim
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 FlapSim 模型
  U->>R: 运行评测脚本
  R->>E: 重置环境、取观测
  E-->>M: 观测
  M-->>E: 动作 / 预测
  R-->>U: 指标与日志
```

## 实验与评测

- 公众号转述：报告的定量气动比较主要使用固定基座，不应等同自由飞行验证。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [动作感知量化、失败定位与可验证自主：具身智能小站 11-03 期 49 篇阅读坐标](../overview/embodied-station-2026-11-03-49-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**扑翼仿真要兼顾气动与控制：报告的定量气动比较主要使用固定基座，不应等同自由飞行验证。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。
2. 核心机制：FlapSim 将机器人描述、翼几何和多层次气动模型组合进 MuJoCo，支持可重复实验。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [manipulation](../tasks/manipulation.md)
- [locomotion](../tasks/locomotion.md)
- [sim2real](../concepts/sim2real.md)
- [embodied-station-2026-11-03-49-papers-technology-map](../overview/embodied-station-2026-11-03-49-papers-technology-map.md)

## 参考来源

- [flapsim_arxiv_2610_40902.md](../../sources/papers/flapsim_arxiv_2610_40902.md)
- [wechat_embodied_station_49_papers_sacq_action_aware_quantization_2026-11-03.md](../../sources/blogs/wechat_embodied_station_49_papers_sacq_action_aware_quantization_2026-11-03.md)
- [arXiv:2610.40902](https://arxiv.org/abs/2610.40902)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.40902)
- [https://github.com/MingshanHe/FlapSim](https://github.com/MingshanHe/FlapSim)
