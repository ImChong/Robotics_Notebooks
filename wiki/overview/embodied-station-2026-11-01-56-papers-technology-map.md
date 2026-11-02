---
type: overview
tags: [overview, survey, embodied-ai, vla, world-model, benchmark, technology-map, wechat]
status: complete
updated: 2026-11-01
related:
  - ../entities/paper-libero-dynamic.md
  - ../entities/rax-scaling-online-exploration.md
  - ../entities/paper-t2-z-environment-latents.md
  - ../methods/vla.md
  - ../concepts/world-action-models.md
sources:
  - ../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md
summary: "具身智能小站 2026-11-01 盘点：LIBERO-Dynamic 把推理延迟放进动态操作评测（深读）、RAX 在线探索编译技能、T2-z 测试时适配环境潜变量，以及评测基准、世界模型、VLA 策略、导航、灵巧接触与人形控制等 56 篇；56/56 独立节点（新建 43 / 复用 13）。"
---

# 动态评测、技能编译与环境潜变量适配：具身智能小站 11-01 期 56 篇阅读坐标

> **本页定位**：为 [具身智能小站 · 模型还在思考，目标已经移动：机器人评测该把时间算进来吗？](https://mp.weixin.qq.com/s/1XR48_smtmSoYQgYSXPKaw)（2026-11-01）提供按研究方向组织的阅读坐标；56 篇全部有独立详情节点。

## 一句话观点

**本期主线是评测与适配要把真实时间与隐藏环境因素算进来：LIBERO-Dynamic 让推理延迟与目标运动同台评测，RAX 把探索经验编译成可复用程序，T2-z 在测试时适配摩擦与质量等环境潜变量；其余覆盖鲁棒与个性化基准、世界模型、VLA 后训练、导航、灵巧接触与人形控制。深读 LIBERO-Dynamic，跟进 RAX 与 T2-z。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| WAM | World Action Model | 世界–动作联合模型 |
| VLM | Vision-Language Model | 视觉语言模型 |

## 为什么单独做这张地图

- 公众号一次列出 56 篇（1 深读 + 2 跟进 + 53 扫读）、方向跨度大，需要横切面索引；每篇只保留一个判断，细节在各节点。
- 节点内的数字为公众号转述，未核验原文表格；开源结论为入库日核查结果，见各节点「为什么重要」。

## 流程总览

```mermaid
flowchart TB
  G0["评测与基准（7）"] --> GOAL
  G1["具身智能体与导航（8）"] --> GOAL
  G2["世界模型与在线适配（10）"] --> GOAL
  G3["VLA 与策略学习（10）"] --> GOAL
  G4["灵巧、接触与精细操作（7）"] --> GOAL
  G5["感知、规划与系统（9）"] --> GOAL
  G6["人形与腿足控制（5）"] --> GOAL
  GOAL[可部署具身系统]
```

## 分组索引

### 评测与基准（7 篇）

| 论文 | 节点 | 开源 |
|------|------|------|
| LIBERO-Dynamic | [paper-libero-dynamic](../entities/paper-libero-dynamic.md) | 已开源 |
| LIBERO-Pro2.0 | [paper-libero-pro2-0](../entities/paper-libero-pro2-0.md) | 已开源 |
| BiDex-100 | [paper-bidex-100](../entities/paper-bidex-100.md) | 已开源 |
| ContactBench 1.0 | [paper-contactbench-1-0](../entities/paper-contactbench-1-0.md) | 待发布 |
| GPT-6-Astra Embodied Intelligence | [paper-gpt-6-astra-embodied-intelligence](../entities/paper-gpt-6-astra-embodied-intelligence.md) | 已开源 |
| LIBERO-PeRM | [paper-libero-perm-personalized-manipulation](../entities/paper-libero-perm-personalized-manipulation.md)（复用） | 已开源 |
| Interventional Memory Evaluation VLA | [paper-interventional-memory-evaluation-vla](../entities/paper-interventional-memory-evaluation-vla.md) | 项目页未列 GitHub |

### 具身智能体与导航（8 篇）

| 论文 | 节点 | 开源 |
|------|------|------|
| RAX | [rax-scaling-online-exploration](../entities/rax-scaling-online-exploration.md)（复用） | 已开源 |
| ScaleNav | [paper-scalenav](../entities/paper-scalenav.md) | 已开源 |
| Learning to Compose | [paper-learning-to-compose](../entities/paper-learning-to-compose.md) | 已开源 |
| HexaShell | [paper-hexashell](../entities/paper-hexashell.md)（复用） | 已开源 |
| REAP | [paper-reap](../entities/paper-reap.md) | 已开源 |
| Spatial Memory by Agents | [paper-spatial-memory-by-agents](../entities/paper-spatial-memory-by-agents.md) | 待发布 |
| HAVANA | [paper-havana](../entities/paper-havana.md) | 已开源 |
| Structure over Reasoning | [paper-structure-over-reasoning](../entities/paper-structure-over-reasoning.md) | 项目页未列 GitHub |

### 世界模型与在线适配（10 篇）

| 论文 | 节点 | 开源 |
|------|------|------|
| T2-z | [paper-t2-z-environment-latents](../entities/paper-t2-z-environment-latents.md)（复用） | 已开源 |
| Dream What I Can | [paper-dreamcan-sparse-visual-planning](../entities/paper-dreamcan-sparse-visual-planning.md)（复用） | 已开源 |
| Fréchet Forcing | [paper-frechet-forcing](../entities/paper-frechet-forcing.md) | 已开源 |
| LAW | [paper-law](../entities/paper-law.md)（复用） | 已开源 |
| BeJEPA | [paper-bejepa](../entities/paper-bejepa.md)（复用） | 已开源 |
| UMI-WM | [paper-umi-wm](../entities/paper-umi-wm.md) | 项目页未列 GitHub |
| ORD-WM | [paper-ord-wm](../entities/paper-ord-wm.md) | 项目页未列 GitHub |
| DefFlow | [paper-defflow](../entities/paper-defflow.md) | 待发布 |
| ACTS | [paper-acts](../entities/paper-acts.md) | 项目页未列 GitHub |
| LY-GWM-RoboCasa-Human300 | [paper-ly-gwm-robocasa-human300](../entities/paper-ly-gwm-robocasa-human300.md) | 已开源 |

### VLA 与策略学习（10 篇）

| 论文 | 节点 | 开源 |
|------|------|------|
| Goal Images Are Control Targets | [paper-goal-images-are-control-targets](../entities/paper-goal-images-are-control-targets.md) | 已开源 |
| MambaAct | [paper-mambaact](../entities/paper-mambaact.md) | 已开源 |
| Value as Memory | [paper-vam-value-as-memory](../entities/paper-vam-value-as-memory.md)（复用） | 已开源 |
| VLA-Fabric | [paper-vla-fabric](../entities/paper-vla-fabric.md) | 已开源 |
| Noise-Invariant Spatial Encoding | [paper-noise-invariant-spatial-encoding](../entities/paper-noise-invariant-spatial-encoding.md) | 项目页未列 GitHub |
| SteerAct | [paper-steeract-few-shot-temporal-steering](../entities/paper-steeract-few-shot-temporal-steering.md)（复用） | 已开源 |
| Flow-Reversed Action Anchors | [paper-flow-reversed-action-anchors](../entities/paper-flow-reversed-action-anchors.md) | 待发布 |
| NestedVLA | [paper-nestedvla](../entities/paper-nestedvla.md) | 已开源 |
| TILDE | [paper-tilde](../entities/paper-tilde.md) | 已开源 |
| Relational Spatial Alignment VLA | [paper-relational-spatial-alignment-vla](../entities/paper-relational-spatial-alignment-vla.md) | 已开源 |

### 灵巧、接触与精细操作（7 篇）

| 论文 | 节点 | 开源 |
|------|------|------|
| Blind Precise Assembly | [paper-blind-proprioceptive-assembly](../entities/paper-blind-proprioceptive-assembly.md)（复用） | 已开源 |
| ESCAPE | [paper-escape](../entities/paper-escape.md) | 已开源 |
| MoDex | [paper-modex](../entities/paper-modex.md) | 已开源 |
| No Eyes, No Problem | [paper-no-eyes-no-problem](../entities/paper-no-eyes-no-problem.md) | 已开源 |
| Contact-Aware Adaptive Impedance | [paper-contact-aware-adaptive-impedance](../entities/paper-contact-aware-adaptive-impedance.md) | 待发布 |
| Embodiment-Aware Heterogeneous Bimanual | [paper-embodiment-aware-heterogeneous-bimanual](../entities/paper-embodiment-aware-heterogeneous-bimanual.md) | 项目页未列 GitHub |
| Pedicel Strawberry Harvesting | [paper-pedicel-strawberry-harvesting](../entities/paper-pedicel-strawberry-harvesting.md) | 已开源 |

### 感知、规划与系统（9 篇）

| 论文 | 节点 | 开源 |
|------|------|------|
| CusADi2 | [paper-cusadi2](../entities/paper-cusadi2.md) | 已开源 |
| PrismPlan | [paper-prismplan](../entities/paper-prismplan.md) | 已开源 |
| Structured Perception Off-road Dynamics | [paper-structured-perception-off-road-dynamics](../entities/paper-structured-perception-off-road-dynamics.md) | 已开源 |
| MicroSplat | [paper-microsplat](../entities/paper-microsplat.md) | 项目页未列 GitHub |
| Shape-Aware Neural Time Fields | [paper-shape-aware-neural-time-fields](../entities/paper-shape-aware-neural-time-fields.md) | 已开源 |
| SpaRV | [paper-sparv](../entities/paper-sparv.md) | 已开源 |
| Fourier Transmission Error Compensation | [paper-fourier-transmission-error-compensation](../entities/paper-fourier-transmission-error-compensation.md) | 项目页未列 GitHub |
| Gearbox Lubricant Temperature Observer | [paper-gearbox-lubricant-temperature-observer](../entities/paper-gearbox-lubricant-temperature-observer.md) | 项目页未列 GitHub |
| Multi-Robot Authoring | [paper-multi-robot-authoring](../entities/paper-multi-robot-authoring.md) | 项目页未列 GitHub |

### 人形与腿足控制（5 篇）

| 论文 | 节点 | 开源 |
|------|------|------|
| Humanoid SRL | [paper-humanoid-srl](../entities/paper-humanoid-srl.md)（复用） | 已开源 |
| FCTS | [paper-fcts-failure-guided-curriculum](../entities/paper-fcts-failure-guided-curriculum.md)（复用） | 已开源 |
| Compositional Motion Styles | [paper-compositional-motion-styles](../entities/paper-compositional-motion-styles.md) | 待发布 |
| Giant Rubik's Cube Quadruped | [paper-rubik-skills](../entities/paper-rubik-skills.md)（复用） | 已开源 |
| Lightweight Contact Feedback Teleop | [paper-lightweight-contact-feedback-teleop](../entities/paper-lightweight-contact-feedback-teleop.md) | 项目页未列 GitHub |

## 关联页面

- [VLA](../methods/vla.md)
- [World Action Models](../concepts/world-action-models.md)

## 参考来源

- [wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md](../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md)

## 推荐继续阅读

- [LIBERO-Dynamic](../entities/paper-libero-dynamic.md)
- [RAX](../entities/rax-scaling-online-exploration.md)
