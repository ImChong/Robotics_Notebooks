---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
arxiv: "2608.29937"
summary: "We introduce AcrossWAM1.0, a modularization and scaling study of this latent world-action stack."
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_2608_29937_acrosswam1-0-a-modular-latent-world-acti.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# AcrossWAM1.0

**AcrossWAM1.0:A Modular Latent World-Action Stack for Compact Robot Policies** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 279/564** 篇，分组 **WAMs**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

We introduce AcrossWAM1.0, a modularization and scaling study of this latent world-action stack.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- We introduce AcrossWAM1.0, a modularization and scaling study of this latent world-action stack.
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph S1["Stage 1：潜在世界模型（继承自 LaWAM）"]
        A["转移对 (o_t, o_t+Δ) → 冻结 DINOv3 → 补丁特征 h_t, h_t+Δ"]
        B["逆动力学 Q-Former → 32 维潜动作 z_t（训练期教师）"]
        C["潜在世界解码器 D_ω（AdaLN 受 z_t 调制）→ 预测未来特征"]
        D["状态落地：g_ξ(s_t, z_t) 预测状态位移"]
        A --> B --> C
        B --> D
    end
    subgraph S2["Stage 2：策略学习与推理"]
        E["指令 + 主视角 + 腕部/辅助视角 → Qwen-VL 或 AcrossVAM 骨干适配器"]
        F["ACT 查询（8 个）→ 交叉注意力映射 → 预测潜动作 ẑ_t"]
        G["流查询（8 个）→ 动作专家的骨干上下文"]
        H["D_ω(h_t, ẑ_t) → 潜在视觉子目标 ĥ_t+Δ"]
        I["Flow-DiT 动作专家：条件 = 骨干上下文 + (h_t, ĥ_t+Δ) + 可选状态"]
        J["动作块"]
        E --> F --> H --> I
        E --> G --> I
        I --> J
    end
    B -. "蒸馏目标 z*（仅训练）" .-> F
    C -. "保留解码器" .-> H
```

图依据：论文 §3.3–3.5 / Fig. 1。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 279/564 |
| 分组 | WAMs |
| 出处 | 见清单 / 原文 |
| 论文 | <https://arxiv.org/abs/2608.29937> |
| 子类 / 象限 | 潜空间预测与JEPA · 视觉规划与IDM · 高效推理与实时控制 · Q4 · Dual-system × IDM |

## 核心机制（归纳）

### 策展导读要点

We introduce AcrossWAM1.0, a modularization and scaling study of this latent world-action stack.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2608.29937) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **WAMs** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「AcrossWAM1.0」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/rcl_awesome_wam_2608_29937_acrosswam1-0-a-modular-latent-world-acti.md`](../../sources/papers/rcl_awesome_wam_2608_29937_acrosswam1-0-a-modular-latent-world-acti.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://arxiv.org/abs/2608.29937>
- [arXiv PDF](https://arxiv.org/pdf/2608.29937)：已通读方法部分（实验与附录以略读为准）：§3 方法（3.1–3.5）；实验与部署导出验证仅略读

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://arxiv.org/abs/2608.29937)
