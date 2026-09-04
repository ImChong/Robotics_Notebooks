---
type: overview
tags: [hub, hub-vla, vision-language-action, foundation, manipulation]
status: complete
updated: 2026-09-04
summary: "VLA 与基础策略知识链汇总：视觉-语言-动作统一建模、OpenVLA/π0/GR00T 等开源谱系，以及 BFM 身体接口与 loco-manip 任务接口；含社区多基准榜入口与驾驶 VLA（S²-VLA）锚点。"
---

# VLA 与基础策略（知识链汇总）

> **知识链汇总**：本页是相关概念/方法的统一入口；对应策展纵深见图谱 [路线视图](../../docs/graph.html?depth=vla) 与 [路线页](../../roadmap/depth-vla.md)。

## 一句话定义

**VLA（Vision-Language-Action）** 把 **视觉观测、自然语言指令与机器人动作** 统一到同一策略或基础模型中，面向多任务操作与 loco-manip 的「一个模型多种技能」。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作策略 |
| BFM | Behavior Foundation Model | 可复用/可调用的身体行为基座 |
| VLM | Vision-Language Model | 视觉-语言预训练骨干 |
| RT | Robotics Transformer | 早期 transformer 机器人策略代表 |
| OXE | Open X-Embodiment | 跨具身开源数据集倡议 |

## 为什么重要

- **降低任务专用策略成本**：语言与视觉提供泛化接口。
- **与人形产业叙事同向**：「运控基座 + 任务头」分层部署。
- **与 WBT / 抓取 / Sim2Real 交叉**：VLA 常作为高层，WBC/低层控制负责执行。

## 本知识链覆盖什么

| 层次 | 典型问题 | 站内入口 |
|------|----------|----------|
| 对比 | 五大模型分类 | [VLM/VLN/VLA/VLX/WM 分类](../comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md) |
| 方法 | VLA 定义与路线 | [VLA](../methods/vla.md) |
| 概念 | 行为基础模型 | [Behavior Foundation Model](../concepts/behavior-foundation-model.md) |
| 概念 | Foundation Policy | [Foundation Policy](../concepts/foundation-policy.md) |
| 地图 | BFM 41 篇技术地图 | [BFM 技术地图](./bfm-41-papers-technology-map.md) |
| 地图 | VLA/WM 14 篇阅读路线 | [CLIP→RT→OpenVLA→π₀ + 四篇 WM](./vla-wm-reading-roadmap-14-papers-technology-map.md) |
| 概念 | BT 编排 VLA 部署 | [行为树 × VLA 编排](../concepts/behavior-tree-vla-orchestration.md) |
| 实体 | ROBOTIS Physical AI 栈 | [Cyclo Intelligence](../entities/cyclo-intelligence.md) |
| 榜站 | VLA / 灵巧手多基准相对位次 | [VLA SOTA Leaderboard](../entities/vla-sota-leaderboard.md) |
| 实体 | Perceptron 开源通才（非 NVIDIA） | [Isaac 0.5](../entities/perceptron-isaac-05.md) |
| 实体 | 相机系锚点几何通才 | [UCAG-P](../entities/paper-ucag-p.md) — 人手与机器人共享 \(p_0/p_1\)；代码待发布 |
| 驾驶 VLA | 语义∥空间双流规划（NAVSIM） | [S²-VLA](../entities/paper-s-squared-vla.md) |
| 数据 | 跨具身数据倡议 | [Open X-Embodiment](../concepts/open-x-embodiment.md) |

## 与其他知识链的关系

- **[IL/RL](./hub-learning.md)**：VLA 训练常混合 IL 与 RLHF/微调。
- **[视觉骨干](./hub-vision-backbone.md)**：感知表征质量影响 VLA 上限。
- **[WBT](./hub-wbt.md)**：全身技能与 VLA 任务接口的分层。

## 关联页面

- [VLA / 世界模型 14 篇阅读路线](./vla-wm-reading-roadmap-14-papers-technology-map.md)
- [统一机器人学习综述](../entities/paper-unified-robot-learning-survey.md) — 表征–VLA–WM 耦合诊断（TMLR 2026）
- [VLA Open-Source Landscape 2025](./vla-open-source-repro-landscape-2025.md)
- [Perceptron Isaac 0.5](../entities/perceptron-isaac-05.md)
- [Whole-Body VLA 相关实体](../entities/paper-hrl-stack-30-wholebodyvla.md)
- [World Action Models](../concepts/world-action-models.md)
- [UCAG-P](../entities/paper-ucag-p.md) — 相机系腕/抓取锚点几何通才操作

## 参考来源

- 本库归纳自 [VLA](../methods/vla.md)、[Behavior Foundation Model](../concepts/behavior-foundation-model.md)、[BFM 技术地图](./bfm-41-papers-technology-map.md)
- [UCAG-P 论文摘录](../../sources/papers/ucag_p_arxiv_2608_26058.md) — 相机系锚点几何通才操作
- 知识链定义：[docs/depth-filters.js](../../docs/depth-filters.js)（`vla` 命中规则）
- 上游原始资料（本链概念页共同的 ingest 来源）：[BFM 综述（arXiv:2506.20487）](../../sources/papers/bfm_survey_arxiv_2506_20487.md)、[BFM：人形机器人行为基础模型（arXiv:2509.13780）](../../sources/papers/bfm_humanoid_arxiv_2509_13780.md)、[awesome-bfm-papers](../../sources/repos/awesome_bfm_papers.md)
