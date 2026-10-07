---
type: overview
tags: [overview, humanoid, world-model, world-action-model]
status: complete
updated: 2026-10-07
related:
  - ./humanoid-motion-intelligence-day4-loco-manipulation.md
  - ../concepts/whole-body-control.md
sources:
  - ../../sources/blogs/humanoid_motion_intelligence_day5_world_models_decision_2026_10_06.md
summary: "Day 5 导读：整理原文 30 项工作，逐项链接其独立详情节点。"
---

# 具身智能从入门到精通 Day 5：世界模型与决策

> **文章节点**：Yuanxq（具身智能研究室）原文的站内导读。论文和项目各自链接到独立详情。

## 一句话观点

世界模型的控制价值取决于预测是否保留任务变化、能否约束可执行动作，以及新观测到来后能否及时修正。

## 英文缩写速查

| 缩写 | 英文全称 | 说明 |
|---|---|---|
| WM | World Model | 预测行动后的环境状态变化。 |
| WAM | World Action Model | 联合建模未来视觉变化和机器人动作。 |
| VLA | Vision-Language-Action | 将视觉和指令映射到机器人动作。 |
| MPC | Model Predictive Control | 滚动预测、执行短动作并依据新观测重规划。 |

## 方法关系

```mermaid
flowchart LR
    A["潜在动力学或视频预测"] --> B["评估候选未来"]
    B --> C["生成动作块"]
    C --> D["机器人执行"]
    D --> E["新观测、日志和反馈"]
    E --> A
```

潜在动力学、视频生成和联合动作生成提供不同的规划效率与预测保真取舍；异步执行还需在策略推理时保持已承诺动作时序。

## 论文与项目独立详情

| 工作 | 独立详情 | 作用 |
|---|---|---|
| Learning Latent Dynamics for Planning from Pixels | [独立详情](../entities/paper-planet-latent-dynamics.md) | latent dynamics for planning |
| Dream to Control | [独立详情](../entities/paper-dreamer-latent-imagination.md) | behavior learning through latent imagination |
| Learning Universal Policies via Text-Guided Video Generation | [独立详情](../entities/paper-rcl-2302-00111-learning-universal-policies-via-text-guided-vide.md) | text-guided video policy transfer |
| Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation | [独立详情](../entities/paper-rcl-ref-e1a2abbaffcea1e2e971-unleashing-large-scale-video-generative-pre-trai.md) | large-scale video pretraining |
| Video Prediction Policy | [独立详情](../entities/paper-shenlan-wm-02-vpp.md) | predictive visual representations |
| Unified World Models | [独立详情](../entities/paper-shenlan-wm-08-uwm.md) | joint video and action diffusion |
| Cosmos Policy | [独立详情](../entities/paper-shenlan-wm-11-cosmos-policy.md) | video-model fine-tuning for control |
| DreamZero | [独立详情](../entities/paper-notebook-dreamzero-world-action-models-are-zero-shot-poli.md) | joint action and future-video generation |
| Causal World Modeling for Robot Control | [独立详情](../entities/paper-sa-2601-21998-lingbot-va-causal-video-action-world-model-for-g.md) | causal video-action world model |
| Being-H0.7 | [独立详情](../entities/paper-rcl-2605-00078-being-h0-7-a-latent-world-action-model-from-egoc.md) | egocentric latent world-action pretraining |
| IronMind | [独立详情](../entities/paper-ironmind.md) | camera-space humanoid pretraining |
| RoboAssist | [独立详情](../entities/paper-roboassist.md) | interactive long-horizon planning |
| ASENA: Self-evolving Agents for Embodied Navigation | [独立详情](../entities/paper-asena-self-evolving-agents.md) | self-evolving embodied navigation |
| Systematically Exploring the Capabilities of GPT-6 Astra as Embodied Policies | [独立详情](../entities/paper-gpt-6-astra-embodied-policy.md) | foundation-agent embodiment study |
| WB-WAM | [独立详情](../entities/paper-wb-wam.md) | heterogeneous body-hand pretraining |
| Humanoid Loco-Manipulation With Discrete VLA Model | [独立详情](../entities/paper-holo-m.md) | discrete body-part action tokenization |
| DualManip: Agentic Dynamic Manipulation via Dual-Path Semantic Reasoning and Geometric Adaptation | [独立详情](../entities/paper-dualmanip.md) | semantic reasoning plus geometric adaptation |
| InternW0-Δ | [独立详情](../entities/paper-internw0-delta.md) | mixture-of-transformers world-action model |
| DyMD | [独立详情](../entities/paper-dymd.md) | few-step interaction-preserving video distillation |
| VLaRL | [独立详情](../entities/paper-vlarl.md) | simulation-trained residual reinforcement learning |
| Kintsugi-VLA | [独立详情](../entities/paper-kintsugi-vla.md) | interventional recovery data |
| Causeway | [独立详情](../entities/paper-causeway.md) | task re-entry for instruction switching |
| Imp-ACT | [独立详情](../entities/paper-imp-act.md) | adaptive impedance with action chunking |
| Interaction-Centric Two-Finger Manipulation | [独立详情](../entities/paper-interaction-centric-gripper.md) | cross-domain gripper representation |
| Fast Plans, Faithful Actions | [独立详情](../entities/paper-fast-plans-faithful-actions.md) | reducing hierarchical planning-execution gap |
| Linear Representation Hypothesis for VLA | [独立详情](../entities/paper-linear-representation-vla.md) | linear structure in VLA representations |
| StarWM | [独立详情](../entities/paper-starwm.md) | attention routing for robust world models |
| Rolling-WAM | [独立详情](../entities/paper-rolling-wam.md) | reusing denoising state across replanning |
| Streaming-WAM | [独立详情](../entities/paper-streaming-wam.md) | asynchronous action-conditioned world-action model |
| RACaP | [独立详情](../entities/paper-racap.md) | evolvable typed robot skills |

## 评读边界

分别看预测质量、规划质量、动作执行和真实机器人闭环；基准分数或视频清晰度不能替代跨物体与受扰动真机测试。

## 结论

阅读世界模型工作时，检查预测中的状态变化如何约束动作，以及动作执行后如何用新观测修正计划。

## 关联页面

- [Day 4：移动操作](./humanoid-motion-intelligence-day4-loco-manipulation.md)
- [全身控制](../concepts/whole-body-control.md)

## 参考来源

- [Day 5 原文与索引归档](../../sources/blogs/humanoid_motion_intelligence_day5_world_models_decision_2026_10_06.md)
- [humanoid-motion-intelligence 项目](https://github.com/RealXiaoze/humanoid-motion-intelligence)

## 推荐继续阅读

- [Cosmos Policy 项目页](https://research.nvidia.com/labs/cosmos-lab/cosmos-policy/)
