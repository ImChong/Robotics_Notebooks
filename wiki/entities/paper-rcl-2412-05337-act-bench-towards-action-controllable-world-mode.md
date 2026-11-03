---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
arxiv: "2412.05337"
venue: "2024"
summary: "RCL Awesome WAM 清单收录（Benchmarks & simulators）；细节以原文 PDF / 项目页为准。"
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_2412_05337_act-bench-towards-action-controllable-wo.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# ACT-Bench

**ACT-Bench: Towards Action Controllable World Models for Autonomous Driving** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 002/564** 篇，分组 **Benchmarks & simulators**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

RCL Awesome WAM 清单收录（Benchmarks & simulators）；细节以原文 PDF / 项目页为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- RCL Awesome WAM 清单收录（Benchmarks & simulators）；细节以原文 PDF / 项目页为准。
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph DS["基准数据集 (§3.1)"]
        V["nuScenes CAM_FRONT 的 10 帧上下文视频"]
        TT["9 类轨迹模板 (取自 CoVLA, 共 36 种) -> 2286 个视频-轨迹对"]
    end
    subgraph WM["被测世界模型 (如基线 Terra, §3.4 / 附录 D)"]
        TK["Open-MAGVIT2 图像 tokenizer -> 离散 token"]
        AR["自回归 Transformer: 图像 token 与轨迹交错输入 (c1, a1, c2, a2, ...), 预测下一帧 token"]
        TK --> AR
        AR --> DEC["帧解码器 + Video Refiner (微调 SVD) -> 生成视频"]
    end
    V --> TK
    TT -->|"逐帧轨迹指令 a_ins / τ_ins"| AR
    DEC --> EST
    subgraph EST["ACT-Estimator (§3.3)"]
        I3D["I3D 骨干 + 自注意力"]
        I3D --> CLS["分类头 MLP -> 高层动作 a_est"]
        I3D --> TRJ["轨迹头 GRU + 交叉注意力 -> 轨迹 τ_est"]
    end
    CLS --> IEC["IEC: 指令与执行动作是否一致"]
    TRJ --> TA["TA: τ_ins 与 τ_est 距离 (ADE / FDE)"]
    TT -.-> IEC
    TT -.-> TA
```

图依据：论文 §3.1-§3.4、附录 D / Fig.3

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 002/564 |
| 分组 | Benchmarks & simulators |
| 出处 | 2024 |
| 论文 | <https://arxiv.org/abs/2412.05337> |
| 子类 / 象限 | 世界模型评测基准 · 自动驾驶基准 · 评测协议与诊断 · 不适用 |

## 核心机制（归纳）

### 策展导读要点

RCL Awesome WAM 清单收录（Benchmarks & simulators）；细节以原文 PDF / 项目页为准。

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2412.05337) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **Benchmarks & simulators** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「ACT-Bench」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/rcl_awesome_wam_2412_05337_act-bench-towards-action-controllable-wo.md`](../../sources/papers/rcl_awesome_wam_2412_05337_act-bench-towards-action-controllable-wo.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://arxiv.org/abs/2412.05337>
- [arXiv PDF](https://arxiv.org/pdf/2412.05337)：已通读方法部分（实验与附录以略读为准）：§3 ACT-Bench（3.1-3.4）与附录 D（Terra 世界模型设计）

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://arxiv.org/abs/2412.05337)
