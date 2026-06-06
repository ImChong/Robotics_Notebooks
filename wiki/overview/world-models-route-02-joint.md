---
type: overview
tags: [world-models, category-hub, joint-architecture, diffusion, autoregressive, wam, shenlan-survey]
status: complete
updated: 2026-06-03
summary: "深蓝世界模型 15 项目 · 02 联合架构（6 篇）— 未来状态与动作在同一扩散/自回归骨干中联合建模，减少级联误差传递；代表 GR-1、UWM、WorldVLA、UVA。"
related:
  - ./world-models-15-open-source-technology-map.md
  - ./world-models-route-01-cascade.md
  - ./world-models-route-03-virtual-sandbox.md
  - ../concepts/world-action-models.md
  - ../methods/vla.md
  - ../entities/paper-shenlan-wm-07-worldvla.md
  - ../entities/paper-shenlan-wm-08-uwm.md
  - ../entities/paper-shenlan-wm-09-gr1.md
  - ../entities/paper-shenlan-wm-10-uva.md
  - ../entities/paper-shenlan-wm-11-cosmos-policy.md
  - ../entities/paper-shenlan-wm-12-f1-vla.md
sources:
  - ../../sources/blogs/wechat_shenlan_world_models_15_open_source_2026.md
  - ../../sources/papers/shenlan_world_models_15_reference_catalog.md
---

# 世界模型路线 02：联合架构

> **图谱分类节点**：**02 联合架构**；总地图见 [世界模型 15 项目技术地图](./world-models-15-open-source-technology-map.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| UWM | Unified World Models | 统一世界模型，联合建模观测与动作 |
| UVA | Unified Video Action Model | 统一视频-动作模型 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态基础策略方向 |
| WM | World Model | 学习环境动态以供想象/规划的世界模型 |
| MoE | Mixture-of-Experts | 门控网络加权组合多个专家子网络 |

## 核心问题

**能否让「想象未来」与「决定动作」在同一次前向传播中完成？** 联合架构用 **共享 Transformer/扩散骨干** 同时建模未来观测与动作——通过联合去噪、联合 token 预测或模态特定扩散时间步，减少级联架构的 **误差传递**。代价通常是 **训练与推理复杂度更高**，且需在 **实时性** 与 **生成质量** 间权衡（UVA 等工程化工作重点解决后者）。

**代表机制（策展）：** GPT 风格联合 token（GR-1）→ 模态特定扩散时间步（UWM）→ VLA+WM 双向增强（WorldVLA）→ 视频基础模型微调（Cosmos Policy）→ MoE 预见+控制（F1-VLA）

## 本组论文（6 篇）

| # | 工作 | Wiki 实体 | Source |
|---|------|-----------|--------|
| 07 | WorldVLA / RynnVLA-002 | [paper-shenlan-wm-07-worldvla.md](../entities/paper-shenlan-wm-07-worldvla.md) | [source](../../sources/papers/shenlan_wm_survey_07_worldvla.md) |
| 08 | UWM | [paper-shenlan-wm-08-uwm.md](../entities/paper-shenlan-wm-08-uwm.md) | [source](../../sources/papers/shenlan_wm_survey_08_uwm.md) |
| 09 | GR-1 | [paper-shenlan-wm-09-gr1.md](../entities/paper-shenlan-wm-09-gr1.md) | [source](../../sources/papers/shenlan_wm_survey_09_gr1.md) |
| 10 | UVA | [paper-shenlan-wm-10-uva.md](../entities/paper-shenlan-wm-10-uva.md) | [source](../../sources/papers/shenlan_wm_survey_10_uva.md) |
| 11 | Cosmos Policy | [paper-shenlan-wm-11-cosmos-policy.md](../entities/paper-shenlan-wm-11-cosmos-policy.md) | [source](../../sources/papers/shenlan_wm_survey_11_cosmos-policy.md) |
| 12 | F1-VLA | [paper-shenlan-wm-12-f1-vla.md](../entities/paper-shenlan-wm-12-f1-vla.md) | [source](../../sources/papers/shenlan_wm_survey_12_f1-vla.md) |

## 在 15 项目地图中的位置

| 字段 | 内容 |
|------|------|
| 分组 | 02 联合架构 |
| 篇数 | 6/15 |
| 概念对照 | [World Action Models](../concepts/world-action-models.md) — Joint 族文献坐标 |
| 姊妹路线 | [01 级联架构](./world-models-route-01-cascade.md)、[03 虚拟沙盒](./world-models-route-03-virtual-sandbox.md) |

## 关联页面

- [世界模型 15 项目技术地图](./world-models-15-open-source-technology-map.md)
- [World Action Models](../concepts/world-action-models.md)
- [VLA](../methods/vla.md)
- [机器人世界模型训练闭环 taxonomy](./robot-world-models-training-loop-taxonomy.md)

## 参考来源

- [wechat_shenlan_world_models_15_open_source_2026.md](../../sources/blogs/wechat_shenlan_world_models_15_open_source_2026.md) — <https://mp.weixin.qq.com/s/KZT8sI4n7GvHWyM20wN3gg>
- [shenlan_world_models_15_reference_catalog.md](../../sources/papers/shenlan_world_models_15_reference_catalog.md)

## 推荐继续阅读

- [GR-1（arXiv:2312.13139）](https://arxiv.org/abs/2312.13139) — 联合架构重要基线
- [UWM（arXiv:2504.02792）](https://arxiv.org/abs/2504.02792) — 模态特定扩散时间步
