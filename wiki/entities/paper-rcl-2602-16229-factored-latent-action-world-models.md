---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
arxiv: "2602.16229"
venue: "2026"
summary: "FLAM learns a video world model whose slots each carry a latent action while sharing interaction-aware dynamics networks. Prediction-trained factorization improves rollouts supplied with future-inferred actions and can s"
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_2602_16229_factored-latent-action-world-models.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# Factored Latent Action World Models

**Factored Latent Action World Models** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 206/564** 篇，分组 **Related resources**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

FLAM learns a video world model whose slots each carry a latent action while sharing interaction-aware dynamics networks. Prediction-trained factorization improves rollouts supplied with future-inferred actions and can supply pseudo action labels for behavior cloning. Its strongest evidence concerns multi-entity vid...

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- FLAM learns a video world model whose slots each carry a latent action while sharing interaction-aware dynamics networks. Prediction-trained factorization improves rollouts supplied with future-inferred actions and can supply pseudo action labels for behavior cloning. Its strongest evidence concerns multi-entity vid...
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph S1["阶段 (a)：预训练 VQ-VAE 编码器"]
        A["无动作标签视频帧 o"] --> B["CNN Encoder → patch 特征 z"]
        B --> C["FSQ 量化 → Decoder 重建<br/>重建损失 L_VQ-VAE"]
    end
    subgraph S2["阶段 (b)：Factored 潜动作模型（各模块联合训练）"]
        D["特征 z_t"] --> E["Factorizer：slot attention + 因果时间注意力<br/>得到 K 个 slot"]
        E --> F["共享 IDM：输入全部当前 slot 与该 slot 的下一帧值<br/>每个 slot 一个潜动作 a_i（KL 约束到单位正态先验）"]
        F --> G["共享 FDM：输入全部当前 slot + a_i<br/>预测各 slot 下一帧值"]
        G --> H["Aggregator：cross-attention 并结合 z_t<br/>映射回特征空间 → 预测下一帧特征"]
        H --> I["下一帧预测损失，联合优化"]
    end
    C --> D
    subgraph S3["第三阶段：潜动作的使用"]
        J["可控视频生成：从先验采样或指定各 slot 潜动作"]
        K["策略学习：小规模带动作标注示教训练 Action Decoder f<br/>给无标注视频的潜动作打伪动作标签 → 行为克隆策略 π"]
    end
    F --> J
    F --> K
```

图依据：论文 §4.1–4.3 / Fig.2、Fig.3、Algorithm 1

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 206/564 |
| 分组 | Related resources |
| 出处 | 2026 |
| 论文 | <https://arxiv.org/abs/2602.16229> |
| 子类 / 象限 | Latent-action & representation methods · 不适用 |

## 核心机制（归纳）

### 策展导读要点

FLAM learns a video world model whose slots each carry a latent action while sharing interaction-aware dynamics networks. Prediction-trained factorization improves rollouts supplied with future-inferred actions and can supply pseudo action labels for behavior cloning. Its strongest evidence concerns multi-entity vid...

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2602.16229) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **Related resources** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Factored Latent Action World Models」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Related resources**，以及 Contribution 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/rcl_awesome_wam_2602_16229_factored-latent-action-world-models.md`](../../sources/papers/rcl_awesome_wam_2602_16229_factored-latent-action-world-models.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://arxiv.org/abs/2602.16229>
- [arXiv PDF](https://arxiv.org/pdf/2602.16229)：已通读方法部分（实验与附录以略读为准）：§3 Preliminaries、§4（4.1–4.3）、Fig.2、Fig.3、Alg.1；§5 实验与附录略读

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://arxiv.org/abs/2602.16229)
