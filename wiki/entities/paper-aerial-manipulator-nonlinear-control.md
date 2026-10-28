---
type: entity
project_id: aerial-manipulator-nonlinear-control
tags:
  - paper
  - control
  - planning
status: complete
updated: 2026-10-28
arxiv: "2610.34454"
code: https://github.com/sandeshthapa/Cartesian_Impedance_Control_Aerial_Manipulator
related:
  - ../tasks/manipulation.md
  - ../concepts/impedance-control.md
  - ../concepts/llm-robotics-control-interfaces.md
  - ../overview/embodied-station-2026-10-28-59-papers-technology-map.md
sources:
  - ../../sources/papers/aerial_manipulator_nonlinear_control_arxiv_2610_34454.md
  - ../../sources/repos/aerial_manipulator_nonlinear_control.md
  - ../../sources/blogs/wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md
summary: "Aerial Manipulator Nonlinear Control（arXiv:2610.34454）：为带双自由度机械臂的四旋翼建立完整动力学，比较笛卡尔阻抗与自适应滑模控制。"
---

# Aerial Manipulator Nonlinear Control（arXiv:2610.34454）

**Aerial Manipulator Nonlinear Control**（*Nonlinear Control of Aerial Manipulator*，[arXiv:2610.34454](https://arxiv.org/abs/2610.34454)，[代码](https://github.com/sandeshthapa/Cartesian_Impedance_Control_Aerial_Manipulator)）来自 [具身智能小站 2026-10-28 盘点](../../sources/blogs/wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md)，属「控制、规划与系统」组。

## 一句话定义

**为带双自由度机械臂的四旋翼建立完整动力学，比较笛卡尔阻抗与自适应滑模控制。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Aerial Manipulator Nonlinear Control | Nonlinear Control of Aerial Manipulator | 本文方法 / 基准的简称 |

## 为什么重要

- 比较空中机械臂的抗扰控制。实验聚焦相同初始条件下的受扰悬停，适合控制实现参考，不宜外推到复杂接触任务。
- 开源结论：**已开源**（步骤 2.5，2026-10-28）。仓库可达（git ls-remote）。
- 同期同组工作见 [潜在动作、免渲染事件仿真与机器人评测：具身智能小站 10-28 期 59 篇阅读坐标](../overview/embodied-station-2026-10-28-59-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.34454](https://arxiv.org/abs/2610.34454) |
| **开源** | **已开源** |
| **要点** | 为带双自由度机械臂的四旋翼建立完整动力学，比较笛卡尔阻抗与自适应滑模控制。 |
| **文内指标** | 公众号转述：实验聚焦相同初始条件下的受扰悬停，适合控制实现参考，不宜外推到复杂接触任务。 |

## 源码运行时序图

以下为按仓库入口（`sandeshthapa/Cartesian_Impedance_Control_Aerial_Manipulator`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as sandeshthapa/Cartesian_Impedance_Control_Aerial_Manipulator
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 Aerial Manipulator Nonlinear Control 模型
  U->>R: 运行评测脚本
  R->>E: 重置环境、取观测
  E-->>M: 观测
  M-->>E: 动作 / 预测
  R-->>U: 指标与日志
```

## 实验与评测

- 公众号转述：实验聚焦相同初始条件下的受扰悬停，适合控制实现参考，不宜外推到复杂接触任务。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [潜在动作、免渲染事件仿真与机器人评测：具身智能小站 10-28 期 59 篇阅读坐标](../overview/embodied-station-2026-10-28-59-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**比较空中机械臂的抗扰控制：实验聚焦相同初始条件下的受扰悬停，适合控制实现参考，不宜外推到复杂接触任务。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。
2. 核心机制：为带双自由度机械臂的四旋翼建立完整动力学，比较笛卡尔阻抗与自适应滑模控制。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [manipulation](../tasks/manipulation.md)
- [impedance-control](../concepts/impedance-control.md)
- [llm-robotics-control-interfaces](../concepts/llm-robotics-control-interfaces.md)
- [embodied-station-2026-10-28-59-papers-technology-map](../overview/embodied-station-2026-10-28-59-papers-technology-map.md)

## 参考来源

- [aerial_manipulator_nonlinear_control_arxiv_2610_34454.md](../../sources/papers/aerial_manipulator_nonlinear_control_arxiv_2610_34454.md)
- [wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md](../../sources/blogs/wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md)
- [arXiv:2610.34454](https://arxiv.org/abs/2610.34454)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.34454)
- [https://github.com/sandeshthapa/Cartesian_Impedance_Control_Aerial_Manipulator](https://github.com/sandeshthapa/Cartesian_Impedance_Control_Aerial_Manipulator)
