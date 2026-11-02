---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
arxiv: "2609.04911"
venue: "2026"
summary: "TourPhysics renders persistent camera tours and object manipulations from a single image plus a declared physical scene. A simulator fixes each trajectory before diffusion generates its appearance; accepted windows publi"
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_2609_04911_tourphysics-bringing-physics-to-world-mo.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# TourPhysics

**TourPhysics: Bringing Physics to World Models for Exploration and Manipulation from a Single Image** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 056/564** 篇，分组 **Benchmarks & simulators**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

TourPhysics renders persistent camera tours and object manipulations from a single image plus a declared physical scene. A simulator fixes each trajectory before diffusion generates its appearance; accepted windows publish physical endpoints and visual memory together. The strongest evidence concerns prescribed moti...

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- TourPhysics renders persistent camera tours and object manipulations from a single image plus a declared physical scene. A simulator fixes each trajectory before diffusion generates its appearance; accepted windows publish physical endpoints and visual memory together. The strongest evidence concerns prescribed moti...
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["单张图像 I0 + 物理场景声明 Θdecl"] --> B["场景提升（仅一次）：SAM 3 / SAM 3D Objects 得到实例 mesh<br/>LaMa 修补出背景板 B0；拟合参考相机与仿射深度"]
    B --> R["只读参考记录 R（I0 作永久外观锚点）"]
    R --> C["窗口 n 的已提交状态 S_n（物理 / 相机 / 外观记忆 K_n / 尾部证据）"]
    C --> D["仿真器与相机控制器克隆状态<br/>按动作 a_n 生成固定轨迹 + 几何记录 G_n"]
    D --> E["生成器侧相对深度控制 D_gen<br/>（由 G_n 构造，并融合已接受的尾部证据）"]
    E --> F["扩散视频模型合成窗口 Y_Wn<br/>条件：G_n、D_gen、I0、K_n"]
    F --> G{"质量门（边界 / 黑帧 / 冻结运动）"}
    G -- 通过 --> H["原子提交：终端仿真/相机状态 + 外观页 + 尾部证据一并发布"]
    G -- 未通过 --> I["回滚候选，重试：仅可换采样种子，轨迹与控制不变"]
    I --> F
    H --> C
```

图依据：论文 §4.1–4.6 / Fig. 2。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 056/564 |
| 分组 | Benchmarks & simulators |
| 出处 | 2026 |
| 论文 | <https://arxiv.org/abs/2609.04911> |
| 子类 / 象限 | 神经世界模拟器 · 不适用 |

## 核心机制（归纳）

### 策展导读要点

TourPhysics renders persistent camera tours and object manipulations from a single image plus a declared physical scene. A simulator fixes each trajectory before diffusion generates its appearance; accepted windows publish physical endpoints and visual memory together. The strongest evidence concerns prescribed moti...

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2609.04911) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **Benchmarks & simulators** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「TourPhysics」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Benchmarks & simulators**，以及 Contribution 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Contribution 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome World-Action Models（RCL）](paper-rcl-wam-robot-learning-control-survey.md)
- 技术地图：[RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md)
- 方法/任务：[generative-world-models.md](../methods/generative-world-models.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/rcl_awesome_wam_2609_04911_tourphysics-bringing-physics-to-world-mo.md`](../../sources/papers/rcl_awesome_wam_2609_04911_tourphysics-bringing-physics-to-world-mo.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://arxiv.org/abs/2609.04911>
- [arXiv PDF](https://arxiv.org/pdf/2609.04911)：已通读方法部分（实验与附录以略读为准）：§4 方法（4.1–4.6）与 §5.1 实验设置

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://arxiv.org/abs/2609.04911)
