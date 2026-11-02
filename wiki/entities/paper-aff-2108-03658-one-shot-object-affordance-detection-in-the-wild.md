---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-one-shot-object-affordance-detection-in-the-wild
arxiv: "2108.03658"
venue: "IJCV 2022"
summary: "IJCV 2022 的 affordance 论文，清单分组：Perception（Scope: Object；Perception Subcategory: Visual · Generalizable）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/affordance_awesome_one-shot-object-affordance-detection-in-the-wild.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# OSAD-Net

**One-Shot Object Affordance Detection in the Wild**（IJCV 2022）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

One-Shot Object Affordance Detection in the Wild——在清单中归入 **Perception（Scope: Object；Perception Subcategory: Visual · Generalizable）**（清单未附方法摘要，此处不替原文概括贡献）。

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
    S["支持样本 S：support 图像 + 人体框 B_h + 物体框 B_o + 人体姿态 P"] --> BB["ResNet50 骨干提取 support 特征 X_S"]
    Q["N 张 query 图像"] --> BQ["ResNet50 骨干提取 query 特征 X_Q"]
    BB --> APL
    subgraph APL["APL 动作目的学习模块"]
        G["姿态分支：4 层 GCN（带可学习边权 M，残差）+ GAP<br/>与 X_S 拼接卷积得 F_P"]
        OB["物体分支：物体框掩码 → 卷积 + GAP 得 f_o<br/>softmax 相关性注意力得 F_o"]
        RP["相对位置：人/物框拼接卷积得 S_ho"]
        OB --> HO["F_ho = Conv(concat(F_o, S_ho))"]
        RP --> HO
        G --> SUM["F_S = F_P + F_ho（动作目的特征）"]
        HO --> SUM
    end
    SUM --> MPT
    BQ --> MPT
    subgraph MPT["MPT 混合目的迁移模块"]
        EM["对 F_S 做 EM 迭代（E 步 / M 步）<br/>得到一组紧凑基 μ（高斯混合）"]
        EM --> RC["用 μ 与注意力 Z_Q 重建 query 特征 X̃_Q<br/>拼接 X_Q 与 X̃_Q 卷积得 F_Q"]
    end
    RC --> DCE
    subgraph DCE["DCE 密集协同增强模块"]
        CS["query 两两逐像素余弦相似度，取最大值<br/>min-max 归一化得相关性掩码 M_ij"]
        CS --> AV["对 N-1 个掩码取平均 M̄_i，乘 F_i 后与 F_i 拼接卷积得 F̂_i"]
    end
    AV --> DEC["解码器：多层上采样 + skip connection<br/>5 个侧输出，交叉熵监督"]
    DEC --> OUT["query 图像中具有相同 affordance 的物体分割结果"]
```

图依据：论文 §3 / Fig.3（总体流程 §3.2，APL §3.3 Fig.4，MPT §3.4 Fig.5，DCE §3.5 Fig.6，解码器与损失 §3.6）

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | IJCV 2022 |
| 方法名（清单标注） | OSAD-Net |
| 分组 1 | Perception · Scope: Object；Perception Subcategory: Visual · Generalizable |
| 论文 | <https://arxiv.org/abs/2108.03658> |

## 核心机制（归纳）

清单本身只给出标题、出处与分类标签；方法流程见上文「流程总览」，实验细节请读 [原文](https://arxiv.org/abs/2108.03658)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2108.03658) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「OSAD-Net」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的只有标题、出处与分类标签；方法与结果未经本页核实。
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

- [`sources/papers/affordance_awesome_one-shot-object-affordance-detection-in-the-wild.md`](../../sources/papers/affordance_awesome_one-shot-object-affordance-detection-in-the-wild.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2108.03658>
- [arXiv PDF](https://arxiv.org/pdf/2108.03658)：已通读方法部分（实验与附录以略读为准）：§1 Introduction、§3 Method（3.1–3.6）、§4.1 数据集构建（略读）

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2108.03658)
