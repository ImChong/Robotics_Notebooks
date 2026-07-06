---
type: overview
tags: [topic, topic-vla, vision-language-action, foundation, manipulation]
status: complete
updated: 2026-07-05
summary: "VLA 与基础策略专题汇总：视觉-语言-动作统一建模、OpenVLA/π0/GR00T 等开源谱系，以及 BFM 身体接口与 loco-manip 任务接口。"
---

# VLA 与基础策略（专题汇总）

> **图谱专题视图**：本页是知识图谱「👀 视觉-语言-动作 (VLA)」专题的统一入口；在 [图谱专题视图](../../docs/graph.html?topic=vla) 筛选时，本节点为汇总锚点。

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

## 本专题覆盖什么

| 层次 | 典型问题 | 站内入口 |
|------|----------|----------|
| 对比 | 五大模型分类 | [VLM/VLN/VLA/VLX/WM 分类](../comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md) |
| 方法 | VLA 定义与路线 | [VLA](../methods/vla.md) |
| 概念 | 行为基础模型 | [Behavior Foundation Model](../concepts/behavior-foundation-model.md) |
| 概念 | Foundation Policy | [Foundation Policy](../concepts/foundation-policy.md) |
| 地图 | BFM 41 篇技术地图 | [BFM 技术地图](./bfm-41-papers-technology-map.md) |
| 概念 | BT 编排 VLA 部署 | [行为树 × VLA 编排](../concepts/behavior-tree-vla-orchestration.md) |
| 实体 | ROBOTIS Physical AI 栈 | [Cyclo Intelligence](../entities/cyclo-intelligence.md) |
| 数据 | 跨具身数据倡议 | [Open X-Embodiment](../concepts/open-x-embodiment.md) |

## 与其他专题的关系

- **[IL/RL](./topic-learning.md)**：VLA 训练常混合 IL 与 RLHF/微调。
- **[视觉骨干](./topic-vision-backbone.md)**：感知表征质量影响 VLA 上限。
- **[WBT](./topic-wbt.md)**：全身技能与 VLA 任务接口的分层。

## 关联页面

- [VLA Open-Source Landscape 2025](./vla-open-source-repro-landscape-2025.md)
- [Whole-Body VLA 相关实体](../entities/paper-hrl-stack-30-wholebodyvla.md)
- [World Action Models](../concepts/world-action-models.md)

## 参考来源

- 本库归纳自 [VLA](../methods/vla.md)、[Behavior Foundation Model](../concepts/behavior-foundation-model.md)、[BFM 技术地图](./bfm-41-papers-technology-map.md)
- 图谱专题定义：[docs/topic-filters.js](../../docs/topic-filters.js)（`vla` 命中规则）
