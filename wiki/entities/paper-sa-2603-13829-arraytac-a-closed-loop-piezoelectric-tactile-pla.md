---
type: entity
tags: [paper, curated-index, awesome-touch, sun254667-touch]
status: complete
updated: 2026-10-06
arxiv: "2603.13829"
venue: "arXiv 2026"
summary: "Piezoelectric-driven tactile display capable of simultaneously rendering shape, stiffness, and friction to reproduce realistic haptic signals."
related:
  - ../entities/awesome-touch.md
  - ../overview/sun-awesome-touch-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/bimanual-manipulation.md
sources:
  - ../../sources/papers/sun_awesome_touch_2603_13829_arraytac-a-closed-loop-piezoelectric-tac.md
  - ../../sources/papers/sun_awesome_touch_catalog.md
  - ../../sources/repos/awesome-touch.md
---

# ArrayTac

**ArrayTac: A Closed-loop Piezoelectric Tactile Platform for Continuously Tunable Rendering of Shape, Stiffness, and Friction** 收录于 [Awesome Touch](https://github.com/sun254667/awesome-touch) **第 040/065** 篇，分组 **Tactile Sensors & Hardware**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Piezoelectric-driven tactile display capable of simultaneously rendering shape, stiffness, and friction to reproduce realistic haptic signals.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VTLA | Vision-Tactile-Language-Action | 视–触–语言–动作策略 |
| WM | World Model | 视触觉前向预测 |
| WAM | World Action Model | 世界–动作联合建模 |
| Sim2Real | Simulation to Real | 仿真到真机迁移 |

## 为什么重要

- Piezoelectric-driven tactile display capable of simultaneously rendering shape, stiffness, and friction to reproduce realistic haptic signals.
- 想横向对照同一分组的其他工作，可以从 [Awesome Touch 技术地图](../overview/sun-awesome-touch-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Touch](../entities/awesome-touch.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    SRC["多源触觉信息<br/>物理仿真 / 由图像做 AI 推理 / 实时远端传感器"] --> WIN["用户通过 XYZ 滑动平台<br/>以 sliding window 实时采样感兴趣区域"]
    WIN --> SEM["多维触觉语义算法<br/>形状、刚度、摩擦"]
    SEM --> HOST["主机经串口把采样的触觉信息<br/>发送给驱动电路"]
    HOST --> DRV["驱动电路驱动 4×4 压电执行器阵列"]
    DRV --> UNIT["每个单元：压电陶瓷 + 三级微杠杆放大<br/>末端位移最大约 5 mm"]
    UNIT --> HALL["末端 Hall 传感器反馈位移"]
    HALL --> PID["PID 闭环：跟踪目标高度（形状）"]
    PID --> UNIT
    SEM -. "刚度指令 k：两阶段控制（PID + 前馈惩罚项）" .-> PID
    SEM -. "摩擦指令 f：在高度指令上叠加正弦振动" .-> PID
    UNIT --> OUT["指尖感受到形状、刚度、摩擦的触觉渲染"]
```

图依据：论文 Results「Tactile Platform Architecture」/ Fig.1a-b，「Multidimensional Tactile Rendering Principles」/ Fig.3（该论文为硬件系统论文，图描述其系统信息流与单元级控制）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 040/065 |
| 分组 | Tactile Sensors & Hardware |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2603.13829> |

## 核心机制（归纳）

### 策展导读要点

Piezoelectric-driven tactile display capable of simultaneously rendering shape, stiffness, and friction to reproduce realistic haptic signals.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2603.13829) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-touch-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-touch-technology-map.md) 的 **Tactile Sensors & Hardware** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「ArrayTac」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Tactile Sensors & Hardware**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Highlights 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome Touch](../entities/awesome-touch.md)
- 技术地图：[Awesome Touch 技术地图](../overview/sun-awesome-touch-technology-map.md)
- 方法/任务：[vla.md](../methods/vla.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/sun_awesome_touch_2603_13829_arraytac-a-closed-loop-piezoelectric-tac.md`](../../sources/papers/sun_awesome_touch_2603_13829_arraytac-a-closed-loop-piezoelectric-tac.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_touch_catalog.md`](../../sources/papers/sun_awesome_touch_catalog.md) — 列表总表
- [`sources/repos/awesome-touch.md`](../../sources/repos/awesome-touch.md)
- 论文：<https://arxiv.org/abs/2603.13829>
- [arXiv PDF](https://arxiv.org/pdf/2603.13829)：已通读方法部分（实验与附录以略读为准）：Results 中 Tactile Platform Architecture、Multidimensional Tactile Rendering Principles 及 Fig.1 / Fig.3 图注（Methods 章节与应用实验仅略读）

## 推荐继续阅读

- [Awesome Touch 仓库](https://github.com/sun254667/awesome-touch)
- [原文](https://arxiv.org/abs/2603.13829)
