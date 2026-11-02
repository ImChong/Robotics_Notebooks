---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
arxiv: "2606.01205"
venue: "2026"
summary: "To address this, we propose ImagineUAV, an imagination-driven framework leveraging cascaded world-action modeling. With only 1.3B parameters, ImagineUAV outperforms prior VLN and VLA baselines on benchmarks and real-worl"
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_2606_01205_imagineuav-aerial-vision-language-naviga.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# ImagineUAV

**ImagineUAV: Aerial Vision-Language Navigation via World-Action Modeling and Kinodynamic Planning** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 389/564** 篇，分组 **WAMs**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

To address this, we propose ImagineUAV, an imagination-driven framework leveraging cascaded world-action modeling. With only 1.3B parameters, ImagineUAV outperforms prior VLN and VLA baselines on benchmarks and real-world flights, validating the practicality of imagination-driven aerial navigation.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- To address this, we propose ImagineUAV, an imagination-driven framework leveraging cascaded world-action modeling. With only 1.3B parameters, ImagineUAV outperforms prior VLN and VLA baselines on benchmarks and real-world flights, validating the practicality of imagination-driven aerial navigation.
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    IN["自然语言指令 l + UAV 第一人称观测 o_t（无全局地图）"] --> W["想象模块 W：视频扩散生成式世界模型，生成指令一致的未来第一视角视频"]
    W --> VID["想象视频片段 V̂_(t:t+T)"]
    VID --> D["视觉里程计式动作提取器 Dψ（Transformer）：回归相对平移 / 旋转序列 a_(t:t+T)"]
    D --> REF["累加得到世界系参考轨迹"]
    MAP["局部障碍图 O"] --> K
    REF --> K["运动动力学规划器 K：B 样条梯度轨迹优化（平滑 / 避障 / 可行性 / 参考偏离）"]
    K --> OUT["平滑、无碰撞、动力学可行轨迹 τ*（滚动时域重规划）"]
    subgraph DEPLOY["部署加速"]
        T["教师扩散模型"] --> S["DMD 步数蒸馏 + 蒸馏 CFG → 学生 10 步 DiT，无 CFG"]
    end
    S -.-> W
```

图依据：论文 §III-B 系统概述（式 1–3）/ Fig.2、§III-D 动作提取与运动动力学规划、§III-E 步数蒸馏部署。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 389/564 |
| 分组 | WAMs |
| 出处 | 2026 |
| 论文 | <https://arxiv.org/abs/2606.01205> |
| 子类 / 象限 | 导航 · 视觉规划与IDM · Q4 · Dual-system × IDM |

## 核心机制（归纳）

### 策展导读要点

To address this, we propose ImagineUAV, an imagination-driven framework leveraging cascaded world-action modeling. With only 1.3B parameters, ImagineUAV outperforms prior VLN and VLA baselines on benchmarks and real-world flights, validating the practicality of imagination-driven aerial navigation.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2606.01205) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **WAMs** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「ImagineUAV」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **WAMs**，以及 Contribution 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/rcl_awesome_wam_2606_01205_imagineuav-aerial-vision-language-naviga.md`](../../sources/papers/rcl_awesome_wam_2606_01205_imagineuav-aerial-vision-language-naviga.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://arxiv.org/abs/2606.01205>
- [arXiv PDF](https://arxiv.org/pdf/2606.01205)：已通读方法部分（实验与附录以略读为准）：§III 方法（A–E）；§IV 实验仅略读，想象模块的网络细节（§III-C 1) 部分）略读

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://arxiv.org/abs/2606.01205)
