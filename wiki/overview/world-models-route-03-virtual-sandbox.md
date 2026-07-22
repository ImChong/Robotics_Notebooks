---
type: overview
tags: [world-models, category-hub, virtual-sandbox, model-based-rl, policy-evaluation, dreamer, shenlan-survey]
status: complete
updated: 2026-07-22
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
  - ../entities/paper-oscar.md
  - ../entities/paper-driftworld.md
  - ../entities/paper-masked-visual-actions.md
sources:
  - ../../sources/blogs/wechat_shenlan_world_models_15_open_source_2026.md
  - ../../sources/papers/shenlan_world_models_15_reference_catalog.md
  - ../../sources/papers/masked_visual_actions_arxiv_2607_19343.md
---

# 世界模型路线 03：虚拟沙盒

> **图谱分类节点**：**03 虚拟沙盒**；总地图见 [世界模型 15 项目技术地图](./world-models-15-open-source-technology-map.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| DreamerV3 | Dreamer version 3 | 在潜空间想象中训练、单一超参跨 150+ 任务通用的世界模型智能体 |
| WM | World Model | 学习环境动态以供想象/规划的世界模型 |
| VLM | Vision-Language Model | 视觉-语言多模态理解模型，VLA 的上游 |

## 核心问题

**能否在「脑海」里试错，而不是每次都在真实物理世界碰壁？** 虚拟沙盒路线不（仅）把 WM 当作 **动作前的预测模块**，而是将其提升为 **可交互的想象环境**——用于 **RL 微调**（在学到的动力学中 rollout 优化策略）或 **策略评估**（蒙特卡洛模拟 + 可验证奖励，保持策略/checkpoint 相对排名）。与 [Model-Based RL](../methods/model-based-rl.md) 和 [Latent Imagination](../concepts/latent-imagination.md) 直接同构。

**代表机制（策展）：** 通用想象 RL（DreamerV3）→ RLVR 对齐任务指标（RLVR-World）→ VLM 奖励的策略评估靶场（WorldGym）

## 本组论文（3 篇）

| # | 工作 | Wiki 实体 | Source |
|---|------|-----------|--------|
| 13 | DreamerV3 | [paper-shenlan-wm-13-dreamerv3.md](../entities/paper-shenlan-wm-13-dreamerv3.md) | [source](../../sources/papers/shenlan_wm_survey_13_dreamerv3.md) |
| 14 | RLVR-World | [paper-shenlan-wm-14-rlvr-world.md](../entities/paper-shenlan-wm-14-rlvr-world.md) | [source](../../sources/papers/shenlan_wm_survey_14_rlvr-world.md) |
| 15 | WorldGym | [paper-shenlan-wm-15-worldgym.md](../entities/paper-shenlan-wm-15-worldgym.md) | [source](../../sources/papers/shenlan_wm_survey_15_worldgym.md) |

**路线外延（非 15 项目策展）：**

- [OSCAR](../entities/paper-oscar.md)（arXiv:2606.04463）— 跨具身 **2D 骨架条件** 视频 WM，在 [RoboArena](../methods/roboarena.md) 七策略上验证虚拟 rollout 与真机排名相关性（Pearson **ρ +0.750**）。
- [DriftWorld](../entities/paper-driftworld.md)（arXiv:2607.15065）— **1-step drifting** 动作条件 WM（30+ fps），用快想象做 **GPC-RANK 推理时改进** 与离线策略评估（与 GT 相关性最高约 **0.99**）。
- [Masked Visual Actions](../entities/paper-masked-visual-actions.md)（arXiv:2607.19343）— **像素掩码动作** 统一前向仿真与逆向行为合成；RoboCasa 策略评估 **r=0.982**，Best-of-N 规划 + 真机演示进度对齐。

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
- [OSCAR](../entities/paper-oscar.md) — RoboArena 虚拟策略评估代理
- [DriftWorld](../entities/paper-driftworld.md) — 1-step drifting：推理时搜索 + 离线评估
- [Masked Visual Actions](../entities/paper-masked-visual-actions.md) — 掩码视觉动作：规划 / 评估 + 前向/逆向统一

## 参考来源

- [wechat_shenlan_world_models_15_open_source_2026.md](../../sources/blogs/wechat_shenlan_world_models_15_open_source_2026.md) — <https://mp.weixin.qq.com/s/KZT8sI4n7GvHWyM20wN3gg>
- [shenlan_world_models_15_reference_catalog.md](../../sources/papers/shenlan_world_models_15_reference_catalog.md)

## 推荐继续阅读

- [DreamerV3（arXiv:2301.04104）](https://arxiv.org/abs/2301.04104) — 通用世界模型 RL 里程碑
- [WorldGym（arXiv:2506.00613）](https://arxiv.org/abs/2506.00613) — WM 策略评估虚拟靶场
