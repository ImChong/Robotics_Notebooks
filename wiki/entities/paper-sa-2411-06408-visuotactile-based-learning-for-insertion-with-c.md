---
type: entity
tags: [paper, curated-index, awesome-touch, sun254667-touch]
status: complete
updated: 2026-10-06
arxiv: "2411.06408"
venue: "IEEE RA-L 2025"
summary: "Simulation-based multimodal policy learning with all-around tactile sensing and depth camera; transformer-based policy via teacher-student distillation; zero-shot transfer."
related:
  - ../entities/awesome-touch.md
  - ../overview/sun-awesome-touch-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/bimanual-manipulation.md
sources:
  - ../../sources/papers/sun_awesome_touch_2411_06408_visuotactile-based-learning-for-insertio.md
  - ../../sources/papers/sun_awesome_touch_catalog.md
  - ../../sources/repos/awesome-touch.md
---

# Visuotactile-Based Learning for Insertion with Compliant Hands

**Visuotactile-Based Learning for Insertion with Compliant Hands** 收录于 [Awesome Touch](https://github.com/sun254667/awesome-touch) **第 036/065** 篇，分组 **53 Diffusion Policies for Tactile Manipulation**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Simulation-based multimodal policy learning with all-around tactile sensing and depth camera; transformer-based policy via teacher-student distillation; zero-shot transfer.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VTLA | Vision-Tactile-Language-Action | 视–触–语言–动作策略 |
| WM | World Model | 视触觉前向预测 |
| WAM | World Action Model | 世界–动作联合建模 |
| Sim2Real | Simulation to Real | 仿真到真机迁移 |

## 为什么重要

- Simulation-based multimodal policy learning with all-around tactile sensing and depth camera; transformer-based policy via teacher-student distillation; zero-shot transfer.
- 想横向对照同一分组的其他工作，可以从 [Awesome Touch 技术地图](../overview/sun-awesome-touch-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Touch](../entities/awesome-touch.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph SimT["仿真阶段 1：教师策略（PPO）"]
        PI["特权信息 e_t：任务相关量 + 物理属性"] --> PE["特权编码器 → 潜变量 z_t"]
        PE --> BP["基础策略 π_θ(o_t, z_t)，o_t 含 EE 位姿与上一动作"]
    end
    BP -->|"teacher rollout 产生示范"| BC
    subgraph SimS["仿真阶段 2：学生蒸馏"]
        TA["三路触觉图像 → 卷积编码"] --> CAT
        PC["分割的物体 / 插座点云 → PointNet → MLP 得联合特征"] --> CAT
        OB["观测 o_t → MLP"] --> CAT
        CAT["拼接 f_t，堆叠 k 个历史特征"] --> TF["Transformer 学生策略"]
        TF --> BC["先 Behavior Cloning 预热，再 DAgger 以概率 β 混合教师"]
    end
    BC --> DR["域随机化：动作延迟、插座扰动、触觉与点云增强"]
    DR --> DEP["零样本部署真实机器人"]
    subgraph Real["真实感知"]
        RT["触觉：灰度化 + 背景减除 + 取 ROI"]
        RV["深度 + TrackingSAM 分割 → 点云（转到机器人基座系）"]
    end
    RT --> TF2["已训练 Transformer 策略输出下一步动作"]
    RV --> TF2
    DEP --> TF2
```

图依据：论文 §III-B、§III-C、§III-D / Fig. 2、Fig. 4。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 036/065 |
| 分组 | 53 Diffusion Policies for Tactile Manipulation |
| 出处 | IEEE RA-L 2025 |
| 论文 | <https://arxiv.org/abs/2411.06408> |

## 核心机制（归纳）

### 策展导读要点

Simulation-based multimodal policy learning with all-around tactile sensing and depth camera; transformer-based policy via teacher-student distillation; zero-shot transfer.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2411.06408) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-touch-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-touch-technology-map.md) 的 **53 Diffusion Policies for Tactile Manipulation** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Visuotactile-Based Learning for Insertion with Compliant Hands」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **53 Diffusion Policies for Tactile Manipulation**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_touch_2411_06408_visuotactile-based-learning-for-insertio.md`](../../sources/papers/sun_awesome_touch_2411_06408_visuotactile-based-learning-for-insertio.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_touch_catalog.md`](../../sources/papers/sun_awesome_touch_catalog.md) — 列表总表
- [`sources/repos/awesome-touch.md`](../../sources/repos/awesome-touch.md)
- 论文：<https://arxiv.org/abs/2411.06408>
- [arXiv PDF](https://arxiv.org/pdf/2411.06408)：已通读方法部分（实验与附录以略读为准）：§II–§III（方法）

## 推荐继续阅读

- [Awesome Touch 仓库](https://github.com/sun254667/awesome-touch)
- [原文](https://arxiv.org/abs/2411.06408)
