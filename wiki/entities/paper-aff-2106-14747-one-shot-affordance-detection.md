---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-one-shot-affordance-detection
arxiv: "2106.14747"
venue: "IJCAI 2021"
summary: "IJCAI 2021 的 affordance 论文，清单分组：Perception（Scope: Object；Perception Subcategory: Visual · Generalizable）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/affordance_awesome_one-shot-affordance-detection.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# OS-AD

**One-Shot Affordance Detection**（IJCAI 2021）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

One-Shot Affordance Detection——在清单中归入 **Perception（Scope: Object；Perception Subcategory: Visual · Generalizable）**（清单未附方法摘要；方法流程见「流程总览」，贡献概括以原文为准）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Affordance | Action possibility (Gibson) | 环境/物体对 agent 提供的可交互机会 |
| HOI | Human-Object Interaction | 人–物交互，常作 affordance 监督来源 |
| VLM | Vision-Language Model | 视觉–语言模型，用于 affordance 推理与接地 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |

## 为什么重要

- 综述把 affordance 学习拆成感知、推理、引导动作三段（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），本条目是其中一段的代表文献之一。
- 可顺着 [技术地图](../overview/affordance-awesome-technology-map.md) 的同分组条目横向阅读同方向工作。

## 流程总览

```mermaid
flowchart TD
    S["support 图像 I_sup（人-物交互）<br/>+ 人与物体的 bounding box"] --> R1["ResNet50 提取特征 X_sup"]
    Q["n 张 query 图像（文中 n=5）"] --> R2["ResNet50 提取特征 X_1..X_n"]
    R1 --> PLM
    subgraph PLM["PLM 目的学习模块"]
        B["按框取人/物特征 X_H、X_O → GMP 得 f_H、f_O"]
        B --> MH["f_H、f_O 分别激活 X_sup 得 M_H、M_O"]
        B --> MHO["f_O 引导 X_H 得人-物交互区域 M_HO"]
        MH --> FS["M_HO 分别作用于 M_H、M_O，相加后 max pooling<br/>得到动作目的编码 F_sup"]
        MHO --> FS
    end
    FS --> PTM
    R2 --> PTM
    subgraph PTM["PTM 目的迁移模块"]
        AT["F_sup 与 query 各位置做相关并归一化，得注意力图<br/>X_T = X + Softmax(X ⊗ F_sup) ⊗ X"]
    end
    AT --> CEM
    subgraph CEM["CEM 协同增强模块"]
        EM["对 query 集合特征做 E-M 迭代（K 个基，3 步）<br/>用基重建特征 F̃，卷积后残差加回：X̂ = X_T + Conv(F̃)"]
    end
    EM --> DEC["解码器：多层上采样，5 个侧输出，交叉熵监督"]
    DEC --> OUT["query 图像中具有共同 affordance 的物体掩码"]
```

图依据：论文 §3 / Fig.3、Fig.4（总体 §3.1，PLM §3.2，PTM §3.3，CEM §3.4，解码器与损失 §3.5）

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | IJCAI 2021 |
| 方法名（清单标注） | OS-AD |
| 分组 1 | Perception · Scope: Object；Perception Subcategory: Visual · Generalizable |
| 论文 | <https://arxiv.org/abs/2106.14747> |

## 核心机制（归纳）

清单本身只给出标题、出处与分类标签；方法流程见上文「流程总览」，实验细节请读 [原文](https://arxiv.org/abs/2106.14747)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2106.14747) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「OS-AD」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的有标题、出处与分类标签，以及「流程总览」所述方法流程（依据论文方法部分）；实验结果未经本页核实。
- 开源状态清单未标注，入库时未打开项目页核查，复现前请自行确认。
- 要深读：从原文入手，再回综述实体页对照 perception → reasoning → action 的位置。

## 常见误区

1. 不要把分类标签当作论文结论；标签来自清单维护者的归类。
2. 不要把本页当作方法证明；它只是清单入口。

## 关联页面

- [Affordance 学习综述](../entities/paper-affordance-learning-embodied-ai-survey.md)
- [Awesome-Affordance-Learning 技术地图](../overview/affordance-awesome-technology-map.md)
- [manipulation.md](../tasks/manipulation.md)、[vla.md](../methods/vla.md)

## 参考来源

- [`sources/papers/affordance_awesome_one-shot-affordance-detection.md`](../../sources/papers/affordance_awesome_one-shot-affordance-detection.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2106.14747>
- [arXiv PDF](https://arxiv.org/pdf/2106.14747)：已通读方法部分（实验与附录以略读为准）：§1 Introduction、§3 Method（3.1–3.5）、§4.1–4.2 数据集与实验设置（略读）

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2106.14747)
