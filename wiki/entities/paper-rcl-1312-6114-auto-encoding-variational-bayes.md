---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
arxiv: "1312.6114"
venue: "ICLR 2014"
summary: "RCL Awesome WAM 清单收录（Components of WAMs）；细节以原文 PDF / 项目页为准。"
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_1312_6114_auto-encoding-variational-bayes.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# Auto-Encoding Variational Bayes

**Auto-Encoding Variational Bayes** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 065/564** 篇，分组 **Components of WAMs**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

RCL Awesome WAM 清单收录（Components of WAMs）；细节以原文 PDF / 项目页为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- RCL Awesome WAM 清单收录（Components of WAMs）；细节以原文 PDF / 项目页为准。
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    X["数据点 x（minibatch，M=100）"] --> ENC["识别模型 / 概率编码器 q_φ(z|x)<br/>MLP 输出 μ、σ（对角高斯）"]
    EPS["辅助噪声 ε ~ N(0, I)"] --> RP["重参数化 z = μ + σ ⊙ ε"]
    ENC --> RP
    RP --> DEC["生成模型 / 概率解码器 p_θ(x|z)<br/>MLP，高斯或伯努利"]
    PRI["先验 p_θ(z) = N(0, I)"] --> KL["KL(q_φ(z|x) ‖ p_θ(z))<br/>高斯情形有解析式，起正则作用"]
    ENC --> KL
    DEC --> REC["重构项 log p_θ(x|z)"]
    KL --> ELBO["SGVB 下界估计 L̃ = −KL + 重构项"]
    REC --> ELBO
    ELBO --> GRAD["对 θ、φ 联合求梯度<br/>(SGD / Adagrad 更新，AEVB 算法)"]
    GRAD -.-> ENC
    GRAD -.-> DEC
```

图依据：论文 §2.3（SGVB 估计器与 Algorithm 1）、§2.4（重参数化）、§3（VAE 示例）、Fig. 1。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 065/564 |
| 分组 | Components of WAMs |
| 出处 | ICLR 2014 |
| 论文 | <https://arxiv.org/abs/1312.6114> |
| 子类 / 象限 | Generative modeling & tokenizers · 视觉编码器与表征 · 不适用 |

## 核心机制（归纳）

### 策展导读要点

RCL Awesome WAM 清单收录（Components of WAMs）；细节以原文 PDF / 项目页为准。

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/1312.6114) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **Components of WAMs** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Auto-Encoding Variational Bayes」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Components of WAMs**，以及 Contribution 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/rcl_awesome_wam_1312_6114_auto-encoding-variational-bayes.md`](../../sources/papers/rcl_awesome_wam_1312_6114_auto-encoding-variational-bayes.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://arxiv.org/abs/1312.6114>
- [arXiv PDF](https://arxiv.org/pdf/1312.6114)：已通读方法部分（实验与附录以略读为准）：§2.1 问题设定、§2.2 变分下界、§2.3 SGVB 估计器与 AEVB 算法、§2.4 重参数化技巧、§3 变分自编码器示例

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://arxiv.org/abs/1312.6114)
