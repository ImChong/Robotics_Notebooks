---
type: entity
project_id: dexremo-2611-15981
tags: [paper, dexterous-manipulation, tactile, force-control]
status: complete
updated: 2026-11-16
arxiv: "2611.15981"
project: https://dexremo.github.io/
original:
  - ../../sources/papers/dexremo-2611-15981.md
sources:
  - ../../sources/papers/dexremo-2611-15981.md
  - ../../sources/sites/dexremo-2611-15981.md
related:
  - ../overview/embodied-frontier-29-papers-2026-11-16.md
  - ./paper-maniladder-manipulation-benchmark.md
summary: "三指力觉统一到腕部坐标系，帮助操作策略处理粘附、表面清理和埋藏物体揭除。"
---

# DexRemo: A Force-Sensing Hand Platform and Task Suite for Contact-Rich Removal Manipulation｜DexRemo：面向揭除清理操作的力觉灵巧手

## 一句话定义

**DexRemo：面向揭除清理操作的力觉灵巧手**：三指力觉统一到腕部坐标系，帮助操作策略处理粘附、表面清理和埋藏物体揭除。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| ACT | Action Chunking Transformer | 动作分块 Transformer |
| DoF | Degrees of Freedom | 关节自由度 |
| RGB | Red Green Blue | 彩色成像 |

## 为什么重要

三指力觉统一到腕部坐标系，帮助操作策略处理粘附、表面清理和埋藏物体揭除。 比起单一 benchmark 排名，研究与工程落地需要分清楚是否具有动作控制闭环、是否能复现以及硬件/数据前提。

## 核心信息

| 条目 | 原始证据 |
|---|---|
| 论文 | [arXiv:2611.15981v1](https://arxiv.org/abs/2611.15981) |
| 官方项目 | [主页](https://dexremo.github.io/) |
| 开源入口 | 未核实该研究完整可运行官方代码 |

## 核心原理

三指 12DoF 手将各指端三轴力通过正向运动学变换到共同腕坐标系，融合三路 RGB、关节位置、指尖位姿及九维合力，条件化 ACT 短动作块策略；双臂 leader-follower 采集示教。

### 流程总览

```mermaid
flowchart LR
  A0["多视角 RGB 与关节状态"]
  A1["三指力传感器"]
  A2["正运动学变换到腕系"]
  A3["多模态 ACT 动作块预测"]
  A4["局部接触揭除/清理任务"]
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

项目页报告六个核心任务 pooled success，从仅视觉状态的 24% 提升至融合手指位置及腕坐标力输入时 83%；论文另含 9 项任务、3 类清理情形。

## 结论

**三指力觉统一到腕部坐标系，帮助操作策略处理粘附、表面清理和埋藏物体揭除。**

1. 保留相同模型与数据预算下的可比口径，不跨基准简单求平均。
2. 优先复核最可能出现数据泄漏、接触失败或部署延迟的环节。
3. 对论文未检验的部署条件单列验证计划。

## 局限与风险

研究聚焦手已定位到局部作业区的揭除与清理，不涵盖大范围导航抓取；项目首页尚未核实完整可运行代码。

## 关联页面

- [本期 29 篇具身论文选型地图](../overview/embodied-frontier-29-papers-2026-11-16.md)
- [ManiLadder：任务难度分层评测](./paper-maniladder-manipulation-benchmark.md)

## 参考来源

- [原论文核对归档](../../sources/papers/dexremo-2611-15981.md)
- [项目页开放状态](../../sources/sites/dexremo-2611-15981.md)

## 推荐继续阅读

- [arXiv HTML](https://arxiv.org/html/2611.15981v1)
- [论文 PDF](https://arxiv.org/pdf/2611.15981)
