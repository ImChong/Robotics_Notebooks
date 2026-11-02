---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
arxiv: "2608.10107"
summary: "To alleviate this issue, we present 4D-WAM, a model that leverages geometric foundation models for training-time supervision to enable 4D consistent world modeling. Extensive experiments demonstrate that 4D-WAM effective"
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_2608_10107_4d-wam-4d-consistent-world-modeling-for.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# 4D-WAM

**4D-WAM: 4D Consistent World Modeling for Autonomous Driving** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 270/564** 篇，分组 **WAMs**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

To alleviate this issue, we present 4D-WAM, a model that leverages geometric foundation models for training-time supervision to enable 4D consistent world modeling. Extensive experiments demonstrate that 4D-WAM effectively models 4D consistent scene evolution and achieves state-of-the-art performance on challenging...

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- To alleviate this issue, we present 4D-WAM, a model that leverages geometric foundation models for training-time supervision to enable 4D consistent world modeling. Extensive experiments demonstrate that 4D-WAM effectively models 4D consistent scene evolution and achieves state-of-the-art performance on challenging...
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    IN["历史帧 v(-Th:0)、当前自车状态 e0、文本驾驶指令 q"] --> ENC["视频 VAE 编码历史帧 → 干净历史 latent（条件前缀）<br/>T5 编码指令；自车状态线性投影并接到文本 token 后"]
    ENC --> MOT
    subgraph MOT["MoT 骨干（Stage I：联合视频-动作流匹配训练）"]
        VE["视频专家<br/>由 Wan2.2 video DiT 初始化<br/>输入：历史 latent + 噪声未来视频 latent"]
        AEX["动作专家<br/>结构对齐的 action DiT<br/>输入：噪声轨迹线性投影的动作 token"]
        VE <-->|"非对称掩码混合注意力<br/>条件 token 只看自己；视频/动作 token 可看三组"| AEX
    end
    VE --> FV["预测未来视频"]
    AEX --> TRJ["自车轨迹（4 s、2 Hz）"]
    subgraph SUP["Stage II：4D 一致性监督（仅训练时，推理无额外开销）"]
        VE -.-> CLEAN["由速度预测恢复干净 latent 估计<br/>VAE 解码得到预测未来帧"]
        CLEAN --> GEO["冻结几何基础模型 VGGT-Ω<br/>输出多层特征与深度"]
        GT["真值未来帧"] --> GEO
        GEO --> L4D["L_4D = λ_feat·L_feat（特征余弦）+ λ_depth·L_depth（对数深度 SmoothL1）"]
    end
    DS["决策导向时间步采样<br/>噪声水平向高噪声（决策区 θ ≥ 0.90）偏置"] -.->|"决定训练时间步"| SUP
    L4D -.->|"反传至骨干"| MOT
```

图依据：论文 §Method（Overview 与 MoT 非对称注意力、4D Consistency Supervision、Decision-Oriented Timestep Sampling）、Implementation Details / Fig.2、Fig.3。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 270/564 |
| 分组 | WAMs |
| 出处 | 见清单 / 原文 |
| 论文 | <https://arxiv.org/abs/2608.10107> |
| 子类 / 象限 | 三维多视角建模 · 自动驾驶 · Q3 · Dual-system × 联合预测 |

## 核心机制（归纳）

### 策展导读要点

To alleviate this issue, we present 4D-WAM, a model that leverages geometric foundation models for training-time supervision to enable 4D consistent world modeling. Extensive experiments demonstrate that 4D-WAM effectively models 4D consistent scene evolution and achieves state-of-the-art performance on challenging...

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2608.10107) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **WAMs** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「4D-WAM」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/rcl_awesome_wam_2608_10107_4d-wam-4d-consistent-world-modeling-for.md`](../../sources/papers/rcl_awesome_wam_2608_10107_4d-wam-4d-consistent-world-modeling-for.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://arxiv.org/abs/2608.10107>
- [arXiv PDF](https://arxiv.org/pdf/2608.10107)：已通读方法部分（实验与附录以略读为准）：Method（MoT 骨干、4D 一致性监督、决策导向时间步采样）与 Implementation Details

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://arxiv.org/abs/2608.10107)
