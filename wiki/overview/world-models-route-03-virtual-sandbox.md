---
type: overview
tags: [world-models, category-hub, virtual-sandbox, model-based-rl, policy-evaluation, dreamer, shenlan-survey]
status: complete
updated: 2026-06-03
summary: "深蓝世界模型 15 项目 · 03 虚拟沙盒（3 篇）— 世界模型作 RL 想象环境或策略评估靶场，用 rollout 替代昂贵真机试错；代表 DreamerV3、RLVR-World、WorldGym。"
related:
  - ./world-models-15-open-source-technology-map.md
  - ./world-models-route-01-cascade.md
  - ./world-models-route-02-joint.md
  - ../methods/model-based-rl.md
  - ../concepts/latent-imagination.md
  - ../entities/paper-shenlan-wm-13-dreamerv3.md
  - ../entities/paper-shenlan-wm-14-rlvr-world.md
  - ../entities/paper-shenlan-wm-15-worldgym.md
sources:
  - ../../sources/blogs/wechat_shenlan_world_models_15_open_source_2026.md
  - ../../sources/papers/shenlan_world_models_15_reference_catalog.md
---

# 世界模型路线 03：虚拟沙盒

> **图谱分类节点**：**03 虚拟沙盒**；总地图见 [世界模型 15 项目技术地图](./world-models-15-open-source-technology-map.md)。

## 核心问题

**能否在「脑海」里试错，而不是每次都在真实物理世界碰壁？** 虚拟沙盒路线不（仅）把 WM 当作 **动作前的预测模块**，而是将其提升为 **可交互的想象环境**——用于 **RL 微调**（在学到的动力学中 rollout 优化策略）或 **策略评估**（蒙特卡洛模拟 + 可验证奖励，保持策略/checkpoint 相对排名）。与 [Model-Based RL](../methods/model-based-rl.md) 和 [Latent Imagination](../concepts/latent-imagination.md) 直接同构。

**代表机制（策展）：** 通用想象 RL（DreamerV3）→ RLVR 对齐任务指标（RLVR-World）→ VLM 奖励的策略评估靶场（WorldGym）

## 本组论文（3 篇）

| # | 工作 | Wiki 实体 | Source |
|---|------|-----------|--------|
| 13 | DreamerV3 | [paper-shenlan-wm-13-dreamerv3.md](../entities/paper-shenlan-wm-13-dreamerv3.md) | [source](../../sources/papers/shenlan_wm_survey_13_dreamerv3.md) |
| 14 | RLVR-World | [paper-shenlan-wm-14-rlvr-world.md](../entities/paper-shenlan-wm-14-rlvr-world.md) | [source](../../sources/papers/shenlan_wm_survey_14_rlvr-world.md) |
| 15 | WorldGym | [paper-shenlan-wm-15-worldgym.md](../entities/paper-shenlan-wm-15-worldgym.md) | [source](../../sources/papers/shenlan_wm_survey_15_worldgym.md) |

## 在 15 项目地图中的位置

| 字段 | 内容 |
|------|------|
| 分组 | 03 虚拟沙盒 |
| 篇数 | 3/15 |
| 方法对照 | [Model-Based RL](../methods/model-based-rl.md) · [Latent Imagination](../concepts/latent-imagination.md) |
| 学术对照 | [robot-world-models-training-loop-taxonomy](./robot-world-models-training-loop-taxonomy.md) 中「学习型模拟器」主线 |
| 姊妹路线 | [01 级联架构](./world-models-route-01-cascade.md)、[02 联合架构](./world-models-route-02-joint.md) |

## 关联页面

- [世界模型 15 项目技术地图](./world-models-15-open-source-technology-map.md)
- [Model-Based RL](../methods/model-based-rl.md)
- [机器人世界模型训练闭环 taxonomy](./robot-world-models-training-loop-taxonomy.md)
- [Video-as-Simulation](../concepts/video-as-simulation.md)

## 参考来源

- [wechat_shenlan_world_models_15_open_source_2026.md](../../sources/blogs/wechat_shenlan_world_models_15_open_source_2026.md) — <https://mp.weixin.qq.com/s/KZT8sI4n7GvHWyM20wN3gg>
- [shenlan_world_models_15_reference_catalog.md](../../sources/papers/shenlan_world_models_15_reference_catalog.md)

## 推荐继续阅读

- [DreamerV3（arXiv:2301.04104）](https://arxiv.org/abs/2301.04104) — 通用世界模型 RL 里程碑
- [WorldGym（arXiv:2506.00613）](https://arxiv.org/abs/2506.00613) — WM 策略评估虚拟靶场
