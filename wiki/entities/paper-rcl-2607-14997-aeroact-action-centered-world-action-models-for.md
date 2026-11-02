---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
arxiv: "2607.14997"
summary: "We present AeroAct, an action-centered world-action model (WAM) for quadrotor navigation. To obtain aligned visual, state, language, and dynamically feasible action data, we build a DiffAero-based pipeline with complemen"
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_2607_14997_aeroact-action-centered-world-action-mod.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# AeroAct

**AeroAct: Action-Centered World-Action Models for Language-Conditioned Quadrotor Flight** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 283/564** 篇，分组 **WAMs**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

We present AeroAct, an action-centered world-action model (WAM) for quadrotor navigation. To obtain aligned visual, state, language, and dynamically feasible action data, we build a DiffAero-based pipeline with complementary Isaac Lab and 3D Gaussian splatting renderers.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- We present AeroAct, an action-centered world-action model (WAM) for quadrotor navigation. To obtain aligned visual, state, language, and dynamically feasible action data, we build a DiffAero-based pipeline with complementary Isaac Lab and 3D Gaussian splatting renderers.
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    DATA["训练数据：DiffAero 动力学 + Isaac Lab / 3DGS 渲染的仿真数据，以及手持设备采集的真实数据（同步 RGB、本体状态、语言、动作块）"] --> ENC
    OBS["步长 Δ 的第一人称历史帧 O_t + 本体状态 s_t + 语言 l"] --> ENC["冻结视频 VAE → 参考 token T_o；MLP 编码 → 状态 token T_s、动作 token T_a；文本编码器 → 语言 token T_l（cross-attention 上下文）"]
    ENC --> SEQ["拼接单一序列 [T_s; T_o; T_a; T_f]（Wan2.1-1.3B 视频 DiT，无独立专家）"]
    SEQ --> MASK["块因果掩码：状态与参考 token 不读预测 token；动作 token 只读状态、参考、动作；未来帧 token 读全部"]
    MASK --> ACTDEC["动作解码：局部五阶轨迹参数的动作块 a_t:t+p-1（p=24）"]
    MASK --> FUT["未来帧预测 T_f（K=8 帧）：仅训练期的后果监督 L_vis"]
    ACTDEC --> LOSS["训练损失 L = L_act + λ_vis·L_vis"]
    FUT --> LOSS
    SEQ --> INFER["部署：不含 T_f，仅解码动作流（4 步去噪）"]
    INFER --> SG["自引导：重叠窗口内，用上一块动作的后缀约束新块前缀，其余部分正常采样"]
    SG --> CTRL["轨迹控制器执行，四旋翼飞行"]
```

图依据：论文 §3 Methodology（§3.1 Problem Formulation、§3.2 Model Architecture、§3.3 Data、§3.4 Self-Guidance）与 Fig. 1、Fig. 2、Fig. 4。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 283/564 |
| 分组 | WAMs |
| 出处 | 见清单 / 原文 |
| 论文 | <https://arxiv.org/abs/2607.14997> |
| 子类 / 象限 | 导航 · 联合视频动作建模 · 高效推理与实时控制 · Q1 · One Model × 联合预测 |

## 核心机制（归纳）

### 策展导读要点

We present AeroAct, an action-centered world-action model (WAM) for quadrotor navigation. To obtain aligned visual, state, language, and dynamically feasible action data, we build a DiffAero-based pipeline with complementary Isaac Lab and 3D Gaussian splatting renderers.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2607.14997) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **WAMs** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「AeroAct」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/rcl_awesome_wam_2607_14997_aeroact-action-centered-world-action-mod.md`](../../sources/papers/rcl_awesome_wam_2607_14997_aeroact-action-centered-world-action-mod.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://arxiv.org/abs/2607.14997>
- [arXiv PDF](https://arxiv.org/pdf/2607.14997)：已通读方法部分（实验与附录以略读为准）：Introduction、§3 Methodology（§3.1–§3.4）、§4.1 实现细节

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://arxiv.org/abs/2607.14997)
