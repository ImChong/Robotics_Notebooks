---
type: entity
project_id: crowdmaps-2611-15078
tags: [paper, imitation-learning, teleoperation, embodied-data]
status: complete
updated: 2026-11-16
arxiv: "2611.15078"
project: https://yilongsong.github.io/CrowdMAPS/
original:
  - ../../sources/papers/crowdmaps-2611-15078.md
sources:
  - ../../sources/papers/crowdmaps-2611-15078.md
  - ../../sources/sites/crowdmaps-2611-15078.md
related:
  - ../overview/embodied-frontier-29-papers-2026-11-16.md
  - ./paper-maniladder-manipulation-benchmark.md
summary: "不增加真机执行分支，而是让众包标注人员为已经采集的关键状态提供多个可行动作。"
---

# CrowdMAPS: Crowdsourcing Multiple Actions per State for Real-World Robot Imitation Learning｜CrowdMAPS：真机状态多动作众包模仿学习

## 一句话定义

**CrowdMAPS：真机状态多动作众包模仿学习**：不增加真机执行分支，而是让众包标注人员为已经采集的关键状态提供多个可行动作。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| MAPS | Multiple Actions per State | 同状态多动作监督 |
| RGB-D | Red Green Blue and Depth | 彩色与深度同步 |
| IL | Imitation Learning | 模仿学习 |

## 为什么重要

不增加真机执行分支，而是让众包标注人员为已经采集的关键状态提供多个可行动作。 比起单一 benchmark 排名，研究与工程落地需要分清楚是否具有动作控制闭环、是否能复现以及硬件/数据前提。

## 核心信息

| 条目 | 原始证据 |
|---|---|
| 论文 | [arXiv:2611.15078v1](https://arxiv.org/abs/2611.15078) |
| 官方项目 | [主页](https://yilongsong.github.io/CrowdMAPS/) |
| 开源入口 | 未核实该研究完整可运行官方代码 |

## 核心原理

从真实示教提取 RGB-D 与机器人状态，在 Isaac Lab 使用 CAD 资产重建状态；远程标注者操控虚拟机器人提出其他末端目标姿态，由研究员校验后与实际执行目标组成 MAPS；分别用于 keypose 和 trajectory 模仿学习。

### 流程总览

```mermaid
flowchart LR
  A0["真实操作轨迹关键帧"]
  A1["RGB-D 与 CAD 场景重建"]
  A2["网页众包多候选目标姿态"]
  A3["人工质量筛选"]
  A4["多动作标签模仿学习"]
  A0 --> A1
  A1 --> A2
  A2 --> A3
  A3 --> A4
```

流程图来自一手论文方法链；与实际源码包之间的映射如下。

## 源码运行时序图

**不适用：** 当前官方材料未提供经核实的可运行源码入口，暂不编造训练或推理代码调用关系。

## 工程实践

1. 依据原文方法核对输入传感器、控制/评估接口、样本分布及实验硬件。
2. 没有核实的官方实现时不把其他仓库替代成作者实际源码。
3. 将论文成绩、真机误差、系统延迟与安全要求分开验收，复现前明确失败标准。

## 实验与评测

五个真机操作任务，动作标签宽度从 1 扩展到 10；项目页报告 keypose/trajectory 策略平均成功率分别提升 35.6/33.0 个百分点。

## 结论

**不增加真机执行分支，而是让众包标注人员为已经采集的关键状态提供多个可行动作。**

1. 保留相同模型与数据预算下的可比口径，不跨基准简单求平均。
2. 优先复核最可能出现数据泄漏、接触失败或部署延迟的环节。
3. 对论文未检验的部署条件单列验证计划。

## 局限与风险

替代动作从未真机执行，仅经人工验证；依赖已知 CAD 资产和目标姿态形式，研究以准静态单臂桌面任务为主。项目页标 Code Coming Soon。

## 关联页面

- [本期 29 篇具身论文选型地图](../overview/embodied-frontier-29-papers-2026-11-16.md)
- [ManiLadder：任务难度分层评测](./paper-maniladder-manipulation-benchmark.md)

## 参考来源

- [原论文核对归档](../../sources/papers/crowdmaps-2611-15078.md)
- [项目页开放状态](../../sources/sites/crowdmaps-2611-15078.md)

## 推荐继续阅读

- [arXiv HTML](https://arxiv.org/html/2611.15078v1)
- [论文 PDF](https://arxiv.org/pdf/2611.15078)
