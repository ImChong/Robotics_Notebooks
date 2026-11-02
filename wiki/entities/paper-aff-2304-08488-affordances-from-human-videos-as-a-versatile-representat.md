---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-affordances-from-human-videos-as-a-versatile-representat
arxiv: "2304.08488"
venue: "CVPR 2023"
summary: "CVPR 2023 的 affordance 论文，清单分组：Action（Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/papers/affordance_awesome_affordances-from-human-videos-as-a-versatile-representat.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# VRB

**Affordances from Human Videos as a Versatile Representation for Robotics**（CVPR 2023）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Affordances from Human Videos as a Versatile Representation for Robotics——在清单中归入 **Action（Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action）**（清单未附方法摘要，此处不替原文概括贡献）。

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
    subgraph EX["从人类第一视角视频提取标签"]
        V["视频 V = {I_1..I_T}"] --> HD["手-物检测模型：每帧手框 h_t 与接触变量 o_t"]
        HD --> TC["取首次接触时刻 t_contact"]
        TC --> CP["肤色分割：手周边与接触物体框相交的点 → 拟合 GMM 得接触点 c"]
        TC --> TJ["t_contact 之后的手框序列 → 后接触轨迹 τ"]
        TJ --> HG["单应矩阵补偿相机自运动，映射回起始坐标系"]
        CP --> HL["映射回人未入镜的首帧（消除人-机视觉域差）"]
        HG --> HL
    end
    HL --> TR["训练 affordance 模型 f_θ：ResNet 编码 + 反卷积出 K 个热图（空间 softmax）+ Transformer 预测轨迹；局部裁剪作输入"]
    TR --> AF["(c, τ) = f_θ(I_t)：接触点 + 后接触轨迹"]
    AF --> P1["离线数据收集 → k-NN / 行为克隆"]
    AF --> P2["无奖励探索：精英轨迹拟合分布 h 引导采样"]
    AF --> P3["目标条件学习：以到目标图像的距离排序轨迹"]
    AF --> P4["affordance 作为动作空间：GMM 得离散 (c, τ) 集合供搜索"]
```

图依据：论文 §3.1–3.3（Fig.2、Fig.3）

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | CVPR 2023 |
| 方法名（清单标注） | VRB |
| 分组 1 | Action · Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action |
| 论文 | <https://arxiv.org/abs/2304.08488> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，**没有**方法摘要。方法流程见上方「流程总览」（依据论文方法部分）；需要机制与实验请读 [原文](https://arxiv.org/abs/2304.08488)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2304.08488) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「VRB」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的只有标题、出处与分类标签；方法与结果未经本页核实。
- 开源状态清单未标注，入库时未打开项目页核查，复现前请自行确认。
- 要深读：从原文入手，再回综述实体页对照 perception → reasoning → action 的位置。

## 常见误区

1. 不要把分类标签当作论文结论；标签来自清单维护者的归类。
2. 不要把本页当作方法证明；它只是清单入口。

## 关联页面

- [Affordance 学习综述](../entities/paper-affordance-learning-embodied-ai-survey.md)
- [Awesome-Affordance-Learning 技术地图](../overview/affordance-awesome-technology-map.md)
- [manipulation.md](../tasks/manipulation.md)、[imitation-learning.md](../methods/imitation-learning.md)

## 参考来源

- [`sources/papers/affordance_awesome_affordances-from-human-videos-as-a-versatile-representat.md`](../../sources/papers/affordance_awesome_affordances-from-human-videos-as-a-versatile-representat.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2304.08488>
- [arXiv PDF](https://arxiv.org/pdf/2304.08488)：已通读方法部分（实验与附录以略读为准）：§3 方法（3.1 可操作表示、3.2 从第一视角视频学习 affordance、3.3 四种机器人学习范式）、§4 实验设置开头

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2304.08488)
