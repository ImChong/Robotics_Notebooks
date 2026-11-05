---
type: entity
tags: [paper, curated-index, awesome-touch, sun254667-touch]
status: complete
updated: 2026-10-06
arxiv: "2408.06506"
venue: "IEEE TRO 2025"
summary: "GPU-based visuotactile sensor simulation over 200× faster than prior SOTA within Isaac Simulator; asymmetric actor-critic distillation (AACD) for sim-to-real transfer."
related:
  - ../entities/awesome-touch.md
  - ../overview/sun-awesome-touch-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/bimanual-manipulation.md
sources:
  - ../../sources/papers/sun_awesome_touch_2408_06506_tacsl-a-library-for-visuotactile-sensor.md
  - ../../sources/papers/sun_awesome_touch_catalog.md
  - ../../sources/repos/awesome-touch.md
---

# TacSL

**TacSL: A Library for Visuotactile Sensor Simulation and Learning** 收录于 [Awesome Touch](https://github.com/sun254667/awesome-touch) **第 051/065** 篇，分组 **Tactile Simulation & Sim2Real**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

GPU-based visuotactile sensor simulation over 200× faster than prior SOTA within Isaac Simulator; asymmetric actor-critic distillation (AACD) for sim-to-real transfer.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VTLA | Vision-Tactile-Language-Action | 视–触–语言–动作策略 |
| WM | World Model | 视触觉前向预测 |
| WAM | World Action Model | 世界–动作联合建模 |
| Sim2Real | Simulation to Real | 仿真到真机迁移 |

## 为什么重要

- GPU-based visuotactile sensor simulation over 200× faster than prior SOTA within Isaac Simulator; asymmetric actor-critic distillation (AACD) for sim-to-real transfer.
- 想横向对照同一分组的其他工作，可以从 [Awesome Touch 技术地图](../overview/sun-awesome-touch-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Touch](../entities/awesome-touch.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph SIM["Isaac 仿真器中的 GPU 并行触觉仿真"]
        PHYS["接触仿真：PhysX 求解器 + Kelvin-Voigt 软接触约束"] --> DEPTH["传感器内相机渲染深度图 Idepth"]
        DEPTH --> RGB["查找表映射 Irgb = F(Idepth)：触觉 RGB 图像"]
        PHYS --> SDF["触觉点查询接触物体 SDF：穿透深度 d 与梯度"]
        SDF --> FF["罚函数模型：法向力 + 切向（剪切）力场"]
    end
    subgraph LEARN["策略学习"]
        EXPERT["特权状态 PPO 专家（低维输入）"] --> DIST["策略蒸馏：BC 离线 / DAgger 在线（专家概率 β 递减）"]
        EXPERT --> AACD["AACD：预训练低维 critic 引导高维 actor"]
        AAC["非对称 actor-critic：actor 用触觉图像，critic 用特权状态"]
    end
    subgraph S2R["Sim-to-real"]
        RAND["软接触刚度/阻尼随机化"] --> TWO["两阶段：低维专家（物理随机化）→ 蒸馏到触觉策略"]
        AUG["触觉图像增强：平移/缩放 + 颜色随机化"] --> TWO
    end
    RGB --> DIST
    FF --> DIST
    RGB --> AACD
    DIST --> TWO
    AACD --> TWO
    TWO --> REAL["部署到真实机器人"]
```

图依据：论文 §III 快速视触觉仿真（Fig.3、Fig.7）、§IV 高效视触觉策略学习（Algorithm 2、3，Fig.4、Fig.5）、§V 视触觉 sim-to-real 迁移。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 051/065 |
| 分组 | Tactile Simulation & Sim2Real |
| 出处 | IEEE TRO 2025 |
| 论文 | <https://arxiv.org/abs/2408.06506> |

## 核心机制（归纳）

### 策展导读要点

GPU-based visuotactile sensor simulation over 200× faster than prior SOTA within Isaac Simulator; asymmetric actor-critic distillation (AACD) for sim-to-real transfer.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2408.06506) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-touch-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-touch-technology-map.md) 的 **Tactile Simulation & Sim2Real** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「TacSL」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Tactile Simulation & Sim2Real**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_touch_2408_06506_tacsl-a-library-for-visuotactile-sensor.md`](../../sources/papers/sun_awesome_touch_2408_06506_tacsl-a-library-for-visuotactile-sensor.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_touch_catalog.md`](../../sources/papers/sun_awesome_touch_catalog.md) — 列表总表
- [`sources/repos/awesome-touch.md`](../../sources/repos/awesome-touch.md)
- 论文：<https://arxiv.org/abs/2408.06506>
- [arXiv PDF](https://arxiv.org/pdf/2408.06506)：已通读方法部分（实验与附录以略读为准）：§III 快速视触觉仿真、§IV 策略学习、§V sim-to-real 迁移；§VI 实验略读

## 推荐继续阅读

- [Awesome Touch 仓库](https://github.com/sun254667/awesome-touch)
- [原文](https://arxiv.org/abs/2408.06506)
