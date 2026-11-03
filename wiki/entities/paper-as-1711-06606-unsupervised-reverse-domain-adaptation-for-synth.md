---
type: entity
tags: [paper, curated-index, awesome-sim2real, longchao-sim2real]
status: complete
updated: 2026-10-06
arxiv: "1711.06606"
venue: "IEEE transactions on medical imaging"
summary: "LongchaoDa AwesomeSim2Real 收录；分组 Observation / Domain Adaptation。 本页是清单索引，细节以原文为准。"
related:
  - ../entities/awesome-sim2real.md
  - ../overview/lc-awesome-sim2real-technology-map.md
  - ../concepts/sim2real.md
  - ../methods/reinforcement-learning.md
  - ../tasks/locomotion.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/lc_awesome_sim2real_1711_06606_unsupervised-reverse-domain-adaptation-f.md
  - ../../sources/papers/lc_awesome_sim2real_catalog.md
  - ../../sources/repos/awesome-sim2real.md
---

# Unsupervised reverse domain adaptation for synthetic medical images via adver...

**Unsupervised reverse domain adaptation for synthetic medical images via adversarial training** 收录于 [AwesomeSim2Real](https://github.com/LongchaoDa/AwesomeSim2Real) **第 041/139** 篇，分组 **Observation / Domain Adaptation**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

LongchaoDa AwesomeSim2Real 收录；分组 Observation / Domain Adaptation。 本页是清单索引，细节以原文为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 仿真策略迁移到真机 |
| MDP | Markov Decision Process | 状态–动作–转移–奖励形式化 |
| DR | Domain Randomization | 域随机化 |
| FM | Foundation Model | 大模型/基础模型增强迁移 |

## 为什么重要

- LongchaoDa AwesomeSim2Real 收录；分组 Observation / Domain Adaptation。 本页是清单索引，细节以原文为准。
- 想横向对照同一分组的其他工作，可以从 [AwesomeSim2Real 技术地图](../overview/lc-awesome-sim2real-technology-map.md) 逐条展开。
- 顺着列表实体 [AwesomeSim2Real](../entities/awesome-sim2real.md) 与站内 Sim2Real 方法/任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    SYN["合成内窥镜数据：前向模型 + 解剖结构结肠模型 + 渲染，附带真值深度"] --> D["判别器 D：区分合成图像与变换后的真实图像"]
    REAL["真实图像 x"] --> T["变换网络 T：真实 → 类合成表示 x'"]
    T --> D
    D -->|"对抗损失反馈"| T
    T --> REG["自正则项：约束 x' 与 x 的逐像素偏差，保留诊断特征"]
    REG --> T
    T --> XP["类合成真实图像"]
    XP --> DEPTH["仅在合成数据上训练的 CNN-CRF 深度估计网络"]
    SYN --> DEPTH
    DEPTH --> OUT["单目内窥镜深度估计"]
```

图依据：论文 §3（合成医学数据，Fig.2）、§4（反向域适应，Fig.3、Algorithm 1）、§5（CNN-CRF 深度估计）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 041/139 |
| 分组 | Observation / Domain Adaptation |
| 出处 | IEEE transactions on medical imaging |
| 论文 | <https://arxiv.org/abs/1711.06606> |

## 核心机制（归纳）

### 策展导读要点

LongchaoDa AwesomeSim2Real 收录；分组 Observation / Domain Adaptation。 本页是清单索引，细节以原文为准。

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/1711.06606) 为准。
- 横向对照请回到 [技术地图](../overview/lc-awesome-sim2real-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/lc-awesome-sim2real-technology-map.md) 的 **Observation / Domain Adaptation** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome 列表可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Unsupervised reverse domain adaptation for synthetic medical images via adver...」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Observation / Domain Adaptation**；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的分组标签当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[AwesomeSim2Real](../entities/awesome-sim2real.md)
- 技术地图：[AwesomeSim2Real 技术地图](../overview/lc-awesome-sim2real-technology-map.md)
- 方法/任务：[sim2real.md](../concepts/sim2real.md)、[locomotion.md](../tasks/locomotion.md)

## 参考来源

- [`sources/papers/lc_awesome_sim2real_1711_06606_unsupervised-reverse-domain-adaptation-f.md`](../../sources/papers/lc_awesome_sim2real_1711_06606_unsupervised-reverse-domain-adaptation-f.md) — 本条目策展摘录
- [`sources/papers/lc_awesome_sim2real_catalog.md`](../../sources/papers/lc_awesome_sim2real_catalog.md) — 列表总表
- [`sources/repos/awesome-sim2real.md`](../../sources/repos/awesome-sim2real.md)
- 论文：<https://arxiv.org/abs/1711.06606>
- [arXiv PDF](https://arxiv.org/pdf/1711.06606)：已通读方法部分（实验与附录以略读为准）：§1-2 引言与相关工作、§3 合成医学数据、§4 反向域适应、§5 深度估计（式 (6)-(9) 前后）

## 推荐继续阅读

- [AwesomeSim2Real 仓库](https://github.com/LongchaoDa/AwesomeSim2Real)
- [原文](https://arxiv.org/abs/1711.06606)
