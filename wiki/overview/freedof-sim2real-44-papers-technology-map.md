---
type: overview
tags: [overview, survey, sim2real, system-identification, domain-randomization, technology-map]
status: complete
updated: 2026-09-20
related:
  - ../comparisons/sim2real-four-routes-identifiability.md
  - ../concepts/sim2real.md
  - ../methods/sim2real-joint-sysid-experiment-design.md
  - ../overview/hub-sim2real.md
  - ../entities/paper-pace-sim2real-legged-robots.md
  - ../entities/paper-survey-sim2real-rl-foundation-models.md
sources:
  - ../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md
  - ../../sources/papers/freedof_sim2real_44_catalog.md
summary: "自由度FreeDof 四条 Sim2Real 路线梳理：44 篇参考文献独立节点索引（22 新建 + 22 复用）。"
---

# Sim2Real 四条路线：44 篇参考文献阅读坐标

> **本页定位**：为 [自由度FreeDof · 从域随机化到残差学习](https://mp.weixin.qq.com/s/K_6MibGXWwh9OL9eSZxOMg) 文末 **44 篇参考文献** 提供按章节组织的独立详情节点索引。

## 一句话观点

**Sim2Real 选型先看「参数能否辨识、剩余误差如何处理」，再按 SysID → 窄 DR → 残差/适应 分层组合；44 篇文献是四条立场在工程上的证据链，而非时间线摘要堆叠。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| SysID | System Identification | 系统辨识 |
| DR | Domain Randomization | 域随机化 |
| RMA | Rapid Motor Adaptation | 在线快速适应 |
| OOD | Out-of-Distribution | 分布外/失配检测 |

## 为什么单独做这张地图

- 原文按 **可辨识性** 串联系统辨识、DR、在线适应、残差学习，参考文献跨四十年与多子领域。
- **44/44 独立节点**：**22 新建** + **22 复用** 已有 canonical 页；**0 重复 arXiv 节点**。
- 与 [四条路线对比](../comparisons/sim2real-four-routes-identifiability.md) 配合：对比页讲立场与组合，本页讲 **逐篇入口**。

## 流程总览

```mermaid
flowchart LR
  subgraph id["§2 系统辨识"]
    PACE[PACE]
    SPI[SPI-Active]
  end
  subgraph dr["§3 域随机化"]
    Peng[Peng DR]
    Poly[PolySim]
  end
  subgraph ad["§4 在线适应"]
    RMA[RMA]
    UP[UP-OSI]
  end
  subgraph res["§5 残差学习"]
    AN[Actuator Net]
    ASAP[ASAP]
  end
  id --> dr --> ad --> res
```

## 分组索引

### 系统辨识

| # | 论文 | 节点 | 开源 |
|---|------|------|------|
| 01 | Parameter identification of robot dynamics | [paper-khosla-robot-dynamics-parameter-identification-1985](../entities/paper-khosla-robot-dynamics-parameter-identification-1985.md) | 不适用 |
| 02 | On the identification of the inertial parameters | [paper-gautier-khalil-inertial-parameter-identification-1988](../entities/paper-gautier-khalil-inertial-parameter-identification-1988.md) | 不适用 |
| 03 | Towards bridging the gap | [paper-pace-sim2real-legged-robots](../entities/paper-pace-sim2real-legged-robots.md) | 已开源 |
| 04 | Impact of static friction on Sim2Real in robotic | [paper-sa-2503-01255-impact-of-static-friction-on-sim2real-in-robotic](../entities/paper-sa-2503-01255-impact-of-static-friction-on-sim2real-in-robotic.md) | 待核实 |
| 05 | Sampling-based system identification with active | [paper-notebook-sampling-based-system-identification-with-active](../entities/paper-notebook-sampling-based-system-identification-with-active.md) | 已开源 |
| 06 | Identification and the information matrix | [paper-gevers-identification-information-matrix-2009](../entities/paper-gevers-identification-information-matrix-2009.md) | 不适用 |
| 07 | Achieving precise and reliable locomotion with d | [paper-kovalev-differentiable-simulation-locomotion-sysid](../entities/paper-kovalev-differentiable-simulation-locomotion-sysid.md) | 待核实 |
| 08 | Simulator adaptation for sim-to-real learning of | [paper-notebook-simulator-adaptation-via-proprioceptive-distribu](../entities/paper-notebook-simulator-adaptation-via-proprioceptive-distribu.md) | 待核实 |

### 域随机化

| # | 论文 | 节点 | 开源 |
|---|------|------|------|
| 09 | Domain randomization for transferring deep neura | [paper-notebook-domain-randomization-for-transferring-deep-neura](../entities/paper-notebook-domain-randomization-for-transferring-deep-neura.md) | 待核实 |
| 10 | Sim-to-real transfer of robotic control with dyn | [paper-peng-dynamics-randomization-sim2real](../entities/paper-peng-dynamics-randomization-sim2real.md) | 待核实 |
| 11 | Sim-to-real | [paper-tan-quadruped-agile-locomotion-sim2real](../entities/paper-tan-quadruped-agile-locomotion-sim2real.md) | 待核实 |
| 12 | Learning dexterous in-hand manipulation | [paper-pai-1808-00177-learningdexterousinhandmanipulat](../entities/paper-pai-1808-00177-learningdexterousinhandmanipulat.md) | 待核实 |
| 13 | Closing the sim-to-real loop | [paper-pai-1910-13325-simopt](../entities/paper-pai-1910-13325-simopt.md) | 待核实 |
| 14 | BayesSim | [paper-pai-1906-01728-bayessim](../entities/paper-pai-1906-01728-bayessim.md) | 待核实 |
| 15 | Data-efficient domain randomization with Bayesia | [paper-muratore-bayesian-optimization-domain-randomization](../entities/paper-muratore-bayesian-optimization-domain-randomization.md) | 待核实 |
| 16 | Solving Rubik's cube with a robot hand | [paper-as-1910-07113-solving-rubik-s-cube-with-a-robot-hand](../entities/paper-as-1910-07113-solving-rubik-s-cube-with-a-robot-hand.md) | 待核实 |
| 17 | EPOpt | [paper-epopt-robust-policies-model-ensembles](../entities/paper-epopt-robust-policies-model-ensembles.md) | 待核实 |
| 18 | Robust adversarial reinforcement learning | [paper-rarl-robust-adversarial-rl](../entities/paper-rarl-robust-adversarial-rl.md) | 待核实 |
| 19 | PolySim | [paper-polysim-multi-simulator-humanoid-sim2real](../entities/paper-polysim-multi-simulator-humanoid-sim2real.md) | 待核实 |
| 20 | Simulation tools for model-based robotics | [paper-erez-simulation-tools-comparison-icra-2015](../entities/paper-erez-simulation-tools-comparison-icra-2015.md) | 不适用 |
| 21 | Validating robotics simulators on real-world imp | [paper-acosta-validating-simulators-real-world-impacts](../entities/paper-acosta-validating-simulators-real-world-impacts.md) | 待核实 |
| 22 | Contact models in robotics | [paper-le-lidec-contact-models-comparative-analysis](../entities/paper-le-lidec-contact-models-comparative-analysis.md) | 待核实 |

### 在线适应

| # | 论文 | 节点 | 开源 |
|---|------|------|------|
| 23 | Preparing for the unknown | [paper-up-osi-universal-policy-online-sysid](../entities/paper-up-osi-universal-policy-online-sysid.md) | 待核实 |
| 24 | RMA | [paper-rma-rapid-motor-adaptation](../entities/paper-rma-rapid-motor-adaptation.md) | 已开源 |
| 25 | Rapid locomotion via reinforcement learning | [paper-rapid-locomotion-rl](../entities/paper-rapid-locomotion-rl.md) | 待核实 |
| 26 | Real-world humanoid locomotion with reinforcemen | [paper-digit-humanoid-locomotion-rl](../entities/paper-digit-humanoid-locomotion-rl.md) | 待核实 |
| 27 | Learning quadrupedal locomotion over challenging | [paper-notebook-learning-quadrupedal-locomotion-over-challenging](../entities/paper-notebook-learning-quadrupedal-locomotion-over-challenging.md) | 待核实 |
| 28 | Learning to walk in minutes using massively para | [paper-anymal-walk-minutes-parallel-drl](../entities/paper-anymal-walk-minutes-parallel-drl.md) | 已开源 |

### 残差学习

| # | 论文 | 节点 | 开源 |
|---|------|------|------|
| 29 | Learning agile and dynamic motor skills for legg | [paper-notebook-learning-agile-and-dynamic-motor-skills-for-legg](../entities/paper-notebook-learning-agile-and-dynamic-motor-skills-for-legg.md) | 待核实 |
| 30 | Sim-to-real transfer with neural-augmented robot | [paper-golemo-neural-augmented-robot-simulation](../entities/paper-golemo-neural-augmented-robot-simulation.md) | 待核实 |
| 31 | Bridging the sim-to-real gap for athletic loco-m | [paper-notebook-bridging-the-sim-to-real-gap-for-athletic-loco-m](../entities/paper-notebook-bridging-the-sim-to-real-gap-for-athletic-loco-m.md) | 待核实 |
| 32 | Residual reinforcement learning for robot contro | [paper-residual-rl-robot-control](../entities/paper-residual-rl-robot-control.md) | 待核实 |
| 33 | ASAP | [paper-hrl-stack-25-asap](../entities/paper-hrl-stack-25-asap.md) | 待核实 |
| 34 | MOSAIC | [paper-loco-manip-161-014-mosaic](../entities/paper-loco-manip-161-014-mosaic.md) | 待核实 |
| 35 | Off-dynamics reinforcement learning | [paper-eysenbach-off-dynamics-rl](../entities/paper-eysenbach-off-dynamics-rl.md) | 待核实 |
| 36 | Legged robots that keep on learning | [paper-smith-legged-robots-keep-learning](../entities/paper-smith-legged-robots-keep-learning.md) | 待核实 |

### 监控与评测

| # | 论文 | 节点 | 开源 |
|---|------|------|------|
| 37 | RAPT | [paper-rapt-sim2real-ood-detection](../entities/paper-rapt-sim2real-ood-detection.md) | 待核实 |
| 38 | Sim2Real predictivity | [paper-kadian-sim2real-predictivity](../entities/paper-kadian-sim2real-predictivity.md) | 待核实 |

### 视觉 Sim2Real

| # | 论文 | 节点 | 开源 |
|---|------|------|------|
| 39 | SplatSim | [paper-splatsim-gaussian-splatting-sim2real](../entities/paper-splatsim-gaussian-splatting-sim2real.md) | 待核实 |
| 40 | GaussGym | [paper-notebook-gaussgym-an-open-source-real-to-sim-framework-fo](../entities/paper-notebook-gaussgym-an-open-source-real-to-sim-framework-fo.md) | 待核实 |

### 可微仿真

| # | 论文 | 节点 | 开源 |
|---|------|------|------|
| 41 | Learning deployable locomotion control via diffe | [paper-schwarke-differentiable-simulation-locomotion-corl](../entities/paper-schwarke-differentiable-simulation-locomotion-corl.md) | 待核实 |

### 训练成本

| # | 论文 | 节点 | 开源 |
|---|------|------|------|
| 42 | Learning sim-to-real humanoid locomotion in 15 m | [paper-notebook-learning-sim-to-real-humanoid-locomotion-in-15-m](../entities/paper-notebook-learning-sim-to-real-humanoid-locomotion-in-15-m.md) | 待核实 |

### 综述

| # | 论文 | 节点 | 开源 |
|---|------|------|------|
| 43 | A survey of sim-to-real methods in RL | [paper-survey-sim2real-rl-foundation-models](../entities/paper-survey-sim2real-rl-foundation-models.md) | 已开源 |

### 资源

| # | 论文 | 节点 | 开源 |
|---|------|------|------|
| 44 | Awesome Humanoid Robot Learning | [paper-awesome-humanoid-robot-learning](../entities/paper-awesome-humanoid-robot-learning.md) | 已开源 |

## 关联页面

- [Sim2Real 四条路线（可辨识性）](../comparisons/sim2real-four-routes-identifiability.md)
- [Sim2Real](../concepts/sim2real.md)
- [Hub: Sim2Real](../overview/hub-sim2real.md)
- [Sim2Real RL 综述（2502.13187）](../entities/paper-survey-sim2real-rl-foundation-models.md)

## 参考来源

- [wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md](../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md)
- [freedof_sim2real_44_catalog.md](../../sources/papers/freedof_sim2real_44_catalog.md)

## 推荐继续阅读

- [PACE](../entities/paper-pace-sim2real-legged-robots.md)
- [ASAP](../entities/paper-hrl-stack-25-asap.md)
- [AwesomeSim2Real 综述](../entities/paper-survey-sim2real-rl-foundation-models.md)
