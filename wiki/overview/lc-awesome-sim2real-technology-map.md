---
type: overview
tags: [overview, curated-index, awesome-sim2real, longchao-sim2real, technology-map]
status: complete
updated: 2026-09-18
summary: "AwesomeSim2Real 技术地图：为清单内论文提供独立详情节点索引（新建 134，复用已有 5）。"
related:
  - ../entities/awesome-sim2real.md
  - ../methods/reinforcement-learning.md
  - ../tasks/locomotion.md
  - ../entities/paper-survey-sim2real-rl-foundation-models.md
sources:
  - ../../sources/papers/lc_awesome_sim2real_catalog.md
  - ../../sources/repos/awesome-sim2real.md
---

# AwesomeSim2Real 技术地图

> 本页把 [AwesomeSim2Real](https://github.com/LongchaoDa/AwesomeSim2Real) 清单中的论文条目映射为站内 **独立详情节点**（`wiki/entities/paper-as-*` 或已有 canonical 页），供图谱与 `detail.html` 检索。配套综述见 [Sim2Real RL Survey（2502.13187）](../entities/paper-survey-sim2real-rl-foundation-models.md)。

## 一句话定义

**AwesomeSim2Real 技术地图** = LongchaoDa 维护的 Sim2Real RL 论文策展列表的站内节点化索引（按 MDP 四要素 + 领域分组浏览）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 仿真策略迁移到真机 |
| MDP | Markov Decision Process | 状态–动作–转移–奖励形式化 |
| DR | Domain Randomization | 域随机化 |
| FM | Foundation Model | 大模型/基础模型增强迁移 |

## 为什么重要

- Awesome 列表本身不是知识图谱节点；若不升格论文实体，首页/图谱无法挂上具体工作。
- 本地图 **优先复用** 库内已有 arXiv canonical 页，仅对缺失条目新建索引级 `paper-as-*` 节点。
- 统计：清单可解析条目 **139**（新建详情节点 **134**，复用已有 **5**）。

## 覆盖范围

| 项 | 值 |
|----|-----|
| 上游仓库 | <https://github.com/LongchaoDa/AwesomeSim2Real> |
| 列表实体 | [AwesomeSim2Real](../entities/awesome-sim2real.md) |
| 配套综述 | [paper-survey-sim2real-rl-foundation-models.md](../entities/paper-survey-sim2real-rl-foundation-models.md) |
| 目录 source | [lc_awesome_sim2real_catalog.md](../../sources/papers/lc_awesome_sim2real_catalog.md) |

## 分组索引

### Action / Action Delay

| # | 论文 | 详情节点 |
|---|------|----------|
| 001 | Acting in delayed environments with non-stationary markov policies | [paper-as-2101-11992-acting-in-delayed-environments-with-non-stationa](../entities/paper-as-2101-11992-acting-in-delayed-environments-with-non-stationa.md) |
| 002 | At human speed: Deep reinforcement learning with action delay | [paper-as-1810-07286-at-human-speed-deep-reinforcement-learning-with](../entities/paper-as-1810-07286-at-human-speed-deep-reinforcement-learning-with.md) |
| 003 | Challenges of real-world reinforcement learning | [paper-as-1904-12901-challenges-of-real-world-reinforcement-learning](../entities/paper-as-1904-12901-challenges-of-real-world-reinforcement-learning.md) |
| 004 | Control delay in reinforcement learning for real-time dynamic systems: A memoryless approa | [paper-as-004-control-delay-in-reinforcement-learning-for-real](../entities/paper-as-004-control-delay-in-reinforcement-learning-for-real.md) |
| 005 | Delay-aware VNF scheduling: A reinforcement learning approach with variable action set | [paper-as-005-delay-aware-vnf-scheduling-a-reinforcement-learn](../entities/paper-as-005-delay-aware-vnf-scheduling-a-reinforcement-learn.md) |
| 006 | Dynamic Modeling for Reinforcement Learning with Random Delay | [paper-as-006-dynamic-modeling-for-reinforcement-learning-with](../entities/paper-as-006-dynamic-modeling-for-reinforcement-learning-with.md) |
| 007 | Dynamic collaborative optimization of end-to-end delay and power consumption in wireless s | [paper-as-007-dynamic-collaborative-optimization-of-end-to-end](../entities/paper-as-007-dynamic-collaborative-optimization-of-end-to-end.md) |
| 008 | Habits, action sequences and reinforcement learning | [paper-as-008-habits-action-sequences-and-reinforcement-learni](../entities/paper-as-008-habits-action-sequences-and-reinforcement-learni.md) |
| 009 | Hierarchical decision and control for continuous multitarget problem: Policy evaluation wi | [paper-as-009-hierarchical-decision-and-control-for-continuous](../entities/paper-as-009-hierarchical-decision-and-control-for-continuous.md) |
| 010 | Mvfst-rl: An asynchronous rl framework for congestion control with delayed actions | [paper-as-1910-04054-mvfst-rl-an-asynchronous-rl-framework-for-conges](../entities/paper-as-1910-04054-mvfst-rl-an-asynchronous-rl-framework-for-conges.md) |
| 011 | Reinforcement learning based VNF scheduling with end-to-end delay guarantee | [paper-as-011-reinforcement-learning-based-vnf-scheduling-with](../entities/paper-as-011-reinforcement-learning-based-vnf-scheduling-with.md) |
| 012 | Reinforcement learning for pivoting task | [paper-as-1703-00472-reinforcement-learning-for-pivoting-task](../entities/paper-as-1703-00472-reinforcement-learning-for-pivoting-task.md) |
| 013 | Reinforcement learning framework for delay sensitive energy harvesting wireless sensor net | [paper-as-013-reinforcement-learning-framework-for-delay-sensi](../entities/paper-as-013-reinforcement-learning-framework-for-delay-sensi.md) |
| 014 | Reinforcement learning with random delays | [paper-as-2010-02966-reinforcement-learning-with-random-delays](../entities/paper-as-2010-02966-reinforcement-learning-with-random-delays.md) |
| 015 | Revisiting state augmentation methods for reinforcement learning with stochastic delays | [paper-as-2108-07555-revisiting-state-augmentation-methods-for-reinfo](../entities/paper-as-2108-07555-revisiting-state-augmentation-methods-for-reinfo.md) |
| 016 | Utilizing reinforcement learning to autonomously mange buffers in a delay tolerant network | [paper-as-016-utilizing-reinforcement-learning-to-autonomously](../entities/paper-as-016-utilizing-reinforcement-learning-to-autonomously.md) |

### Action / Action Space Scale

| # | 论文 | 详情节点 |
|---|------|----------|
| 017 | Fear Field: Adaptive constraints for safe environment transitions in Shielded Reinforcemen | [paper-as-017-fear-field-adaptive-constraints-for-safe-environ](../entities/paper-as-017-fear-field-adaptive-constraints-for-safe-environ.md) |
| 018 | Safe reinforcement learning via shielding | [paper-as-1708-08611-safe-reinforcement-learning-via-shielding](../entities/paper-as-1708-08611-safe-reinforcement-learning-via-shielding.md) |
| 019 | Safety-Driven Deep Reinforcement Learning Framework for Cobots: A Sim2Real Approach | [paper-as-2407-02231-safety-driven-deep-reinforcement-learning-framew](../entities/paper-as-2407-02231-safety-driven-deep-reinforcement-learning-framew.md) |
| 020 | Sim-to-real transfer for vision-and-language navigation | [paper-as-2011-03807-sim-to-real-transfer-for-vision-and-language-nav](../entities/paper-as-2011-03807-sim-to-real-transfer-for-vision-and-language-nav.md) |

### Action / Action Uncertainty

| # | 论文 | 详情节点 |
|---|------|----------|
| 021 | Action advising with advice imitation in deep reinforcement learning | [paper-as-2104-08441-action-advising-with-advice-imitation-in-deep-re](../entities/paper-as-2104-08441-action-advising-with-advice-imitation-in-deep-re.md) |
| 022 | Action robust reinforcement learning and applications in continuous control | [paper-as-1901-09184-action-robust-reinforcement-learning-and-applica](../entities/paper-as-1901-09184-action-robust-reinforcement-learning-and-applica.md) |
| 023 | Efficient action robust reinforcement learning with probabilistic policy execution uncerta | [paper-as-2307-07666-efficient-action-robust-reinforcement-learning-w](../entities/paper-as-2307-07666-efficient-action-robust-reinforcement-learning-w.md) |
| 024 | Safe reinforcement learning with model uncertainty estimates | [paper-as-1810-08700-safe-reinforcement-learning-with-model-uncertain](../entities/paper-as-1810-08700-safe-reinforcement-learning-with-model-uncertain.md) |
| 025 | Uncertainty-aware action advising for deep reinforcement learning agents | [paper-as-025-uncertainty-aware-action-advising-for-deep-reinf](../entities/paper-as-025-uncertainty-aware-action-advising-for-deep-reinf.md) |

### Action / Foundation Models

| # | 论文 | 详情节点 |
|---|------|----------|
| 026 | Local Policies Enable Zero-shot Long-horizon Manipulation | [paper-as-2410-22332-local-policies-enable-zero-shot-long-horizon-man](../entities/paper-as-2410-22332-local-policies-enable-zero-shot-long-horizon-man.md) |
| 027 | RLingua: Improving Reinforcement Learning Sample Efficiency in Robotic Manipulations With  | [paper-as-2403-06420-rlingua-improving-reinforcement-learning-sample](../entities/paper-as-2403-06420-rlingua-improving-reinforcement-learning-sample.md) |
| 028 | robosuite: A modular simulation framework and benchmark for robot learning | [paper-as-2009-12293-robosuite-a-modular-simulation-framework-and-ben](../entities/paper-as-2009-12293-robosuite-a-modular-simulation-framework-and-ben.md) |

### Observation / Domain Adaptation

| # | 论文 | 详情节点 |
|---|------|----------|
| 029 | Bi-directional domain adaptation for sim2real transfer of embodied navigation agents | [paper-as-2011-12421-bi-directional-domain-adaptation-for-sim2real-tr](../entities/paper-as-2011-12421-bi-directional-domain-adaptation-for-sim2real-tr.md) |
| 030 | Coupled real-synthetic domain adaptation for real-world deep depth enhancement | [paper-as-030-coupled-real-synthetic-domain-adaptation-for-rea](../entities/paper-as-030-coupled-real-synthetic-domain-adaptation-for-rea.md) |
| 031 | Da4event: towards bridging the sim-to-real gap for event cameras using domain adaptation | [paper-as-2103-12768-da4event-towards-bridging-the-sim-to-real-gap-fo](../entities/paper-as-2103-12768-da4event-towards-bridging-the-sim-to-real-gap-fo.md) |
| 032 | Domain adaption as auxiliary task for sim-to-real transfer in vision-based neuro-robotic c | [paper-as-032-domain-adaption-as-auxiliary-task-for-sim-to-rea](../entities/paper-as-032-domain-adaption-as-auxiliary-task-for-sim-to-rea.md) |
| 033 | MIC: Masked image consistency for context-enhanced domain adaptation | [paper-as-033-mic-masked-image-consistency-for-context-enhance](../entities/paper-as-033-mic-masked-image-consistency-for-context-enhance.md) |
| 034 | Retinagan: An object-aware approach to sim-to-real transfer | [paper-as-034-retinagan-an-object-aware-approach-to-sim-to-rea](../entities/paper-as-034-retinagan-an-object-aware-approach-to-sim-to-rea.md) |
| 035 | Rl-cyclegan: Reinforcement learning aware simulation-to-real | [paper-as-2011-03148-rl-cyclegan-reinforcement-learning-aware-simulat](../entities/paper-as-2011-03148-rl-cyclegan-reinforcement-learning-aware-simulat.md) |
| 036 | Self-supervised sim-to-real adaptation for visual robotic manipulation | [paper-as-036-self-supervised-sim-to-real-adaptation-for-visua](../entities/paper-as-036-self-supervised-sim-to-real-adaptation-for-visua.md) |
| 037 | Sensor transfer: Learning optimal sensor effect image augmentation for sim-to-real domain  | [paper-as-1809-06256-sensor-transfer-learning-optimal-sensor-effect-i](../entities/paper-as-1809-06256-sensor-transfer-learning-optimal-sensor-effect-i.md) |
| 038 | Sim-to-real visual grasping via state representation learning based on combining pixel-lev | [paper-as-038-sim-to-real-visual-grasping-via-state-representa](../entities/paper-as-038-sim-to-real-visual-grasping-via-state-representa.md) |
| 039 | Unsupervised adversarial domain adaptation for sim-to-real transfer of tactile images | [paper-as-039-unsupervised-adversarial-domain-adaptation-for-s](../entities/paper-as-039-unsupervised-adversarial-domain-adaptation-for-s.md) |
| 040 | Unsupervised pixel-level domain adaptation with generative adversarial networks | [paper-as-040-unsupervised-pixel-level-domain-adaptation-with](../entities/paper-as-040-unsupervised-pixel-level-domain-adaptation-with.md) |
| 041 | Unsupervised reverse domain adaptation for synthetic medical images via adversarial traini | [paper-as-1711-06606-unsupervised-reverse-domain-adaptation-for-synth](../entities/paper-as-1711-06606-unsupervised-reverse-domain-adaptation-for-synth.md) |
| 042 | Using simulation and domain adaptation to improve efficiency of deep robotic grasping | [paper-as-042-using-simulation-and-domain-adaptation-to-improv](../entities/paper-as-042-using-simulation-and-domain-adaptation-to-improv.md) |
| 043 | Vr-goggles for robots: Real-to-sim domain adaptation for visual control | [paper-as-1802-00265-vr-goggles-for-robots-real-to-sim-domain-adaptat](../entities/paper-as-1802-00265-vr-goggles-for-robots-real-to-sim-domain-adaptat.md) |

### Observation / Domain Randomization

| # | 论文 | 详情节点 |
|---|------|----------|
| 044 | Asymmetric Actor Critic for Image-Based Robot Learning | [paper-as-044-asymmetric-actor-critic-for-image-based-robot-le](../entities/paper-as-044-asymmetric-actor-critic-for-image-based-robot-le.md) |
| 045 | Bridging the Reality Gap Between Virtual and Physical Environments Through Reinforcement L | [paper-as-045-bridging-the-reality-gap-between-virtual-and-phy](../entities/paper-as-045-bridging-the-reality-gap-between-virtual-and-phy.md) |
| 046 | DROPO: Sim-to-real transfer with offline domain randomization | [paper-as-2201-08434-dropo-sim-to-real-transfer-with-offline-domain-r](../entities/paper-as-2201-08434-dropo-sim-to-real-transfer-with-offline-domain-r.md) |
| 047 | Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real Wor | [paper-notebook-domain-randomization-for-transferring-deep-neura](../entities/paper-notebook-domain-randomization-for-transferring-deep-neura.md) |
| 048 | Learning Vision-Based Bipedal Locomotion for Challenging Terrain | [paper-as-2309-14594-learning-vision-based-bipedal-locomotion-for-cha](../entities/paper-as-2309-14594-learning-vision-based-bipedal-locomotion-for-cha.md) |
| 049 | Learning to Manipulate Anywhere: A Visual Generalizable Framework For Reinforcement Learni | [paper-as-2407-15815-learning-to-manipulate-anywhere-a-visual-general](../entities/paper-as-2407-15815-learning-to-manipulate-anywhere-a-visual-general.md) |
| 050 | Solving Rubik's Cube with a Robot Hand | [paper-as-1910-07113-solving-rubik-s-cube-with-a-robot-hand](../entities/paper-as-1910-07113-solving-rubik-s-cube-with-a-robot-hand.md) |

### Observation / Foundation Models

| # | 论文 | 详情节点 |
|---|------|----------|
| 051 | ChatGPT Label: Comparing the Quality of Human-Generated and LLM-Generated Annotations in L | [paper-as-051-chatgpt-label-comparing-the-quality-of-human-gen](../entities/paper-as-051-chatgpt-label-comparing-the-quality-of-human-gen.md) |
| 052 | Dinov2: Learning robust visual features without supervision | [paper-dinov2](../entities/paper-dinov2.md) |
| 053 | Gpt-4 technical report | [paper-as-2303-08774-gpt-4-technical-report](../entities/paper-as-2303-08774-gpt-4-technical-report.md) |
| 054 | InfiniteWorld: A Unified Scalable Simulation Framework for General Visual-Language Robot I | [paper-as-2412-05789-infiniteworld-a-unified-scalable-simulation-fram](../entities/paper-as-2412-05789-infiniteworld-a-unified-scalable-simulation-fram.md) |
| 055 | LLM-Optic: Unveiling the Capabilities of Large Language Models for Universal Visual Ground | [paper-as-2405-17104-llm-optic-unveiling-the-capabilities-of-large-la](../entities/paper-as-2405-17104-llm-optic-unveiling-the-capabilities-of-large-la.md) |
| 056 | Natural Language Can Help Bridge the Sim2Real Gap | [paper-as-2405-10020-natural-language-can-help-bridge-the-sim2real-ga](../entities/paper-as-2405-10020-natural-language-can-help-bridge-the-sim2real-ga.md) |
| 057 | Segment as You Wish--Free-Form Language-Based Segmentation for Medical Images | [paper-as-2410-12831-segment-as-you-wish-free-form-language-based-seg](../entities/paper-as-2410-12831-segment-as-you-wish-free-form-language-based-seg.md) |
| 058 | Synthetic Vision: Training Vision-Language Models to Understand Physics | [paper-as-2412-08619-synthetic-vision-training-vision-language-models](../entities/paper-as-2412-08619-synthetic-vision-training-vision-language-models.md) |

### Observation / Sensor Fusion

| # | 论文 | 详情节点 |
|---|------|----------|
| 059 | A sim2real deep learning approach for the transformation of images from multiple vehicle-m | [paper-as-2005-04078-a-sim2real-deep-learning-approach-for-the-transf](../entities/paper-as-2005-04078-a-sim2real-deep-learning-approach-for-the-transf.md) |
| 060 | LiDAR Object Detection and-Sensor Fusion in Simulation Environments Sensor modelling towar | [paper-as-060-lidar-object-detection-and-sensor-fusion-in-simu](../entities/paper-as-060-lidar-object-detection-and-sensor-fusion-in-simu.md) |
| 061 | Longitudinal vehicle speed estimation for four-wheel-independently-actuated electric vehic | [paper-as-061-longitudinal-vehicle-speed-estimation-for-four-w](../entities/paper-as-061-longitudinal-vehicle-speed-estimation-for-four-w.md) |
| 062 | Quantifying the sim2real gap for GPS and IMU sensors | [paper-as-2403-11000-quantifying-the-sim2real-gap-for-gps-and-imu-sen](../entities/paper-as-2403-11000-quantifying-the-sim2real-gap-for-gps-and-imu-sen.md) |
| 063 | Sensor fusion for robot control through deep reinforcement learning | [paper-as-063-sensor-fusion-for-robot-control-through-deep-rei](../entities/paper-as-063-sensor-fusion-for-robot-control-through-deep-rei.md) |
| 064 | World model based sim2real transfer for visual navigation | [paper-as-2310-18847-world-model-based-sim2real-transfer-for-visual-n](../entities/paper-as-2310-18847-world-model-based-sim2real-transfer-for-visual-n.md) |

### Other Benchmarks

| # | 论文 | 详情节点 |
|---|------|----------|
| 065 | Benchmarking Safe Exploration in Deep Reinforcement Learning | [paper-as-065-benchmarking-safe-exploration-in-deep-reinforcem](../entities/paper-as-065-benchmarking-safe-exploration-in-deep-reinforcem.md) |
| 066 | EnergyPlus: creating a new-generation building energy simulation program | [paper-as-066-energyplus-creating-a-new-generation-building-en](../entities/paper-as-066-energyplus-creating-a-new-generation-building-en.md) |

### Other Environments

| # | 论文 | 详情节点 |
|---|------|----------|
| 067 | AI2-THOR: An Interactive 3D Environment for Visual AI | [paper-as-1712-05474-ai2-thor-an-interactive-3d-environment-for-visua](../entities/paper-as-1712-05474-ai2-thor-an-interactive-3d-environment-for-visua.md) |
| 068 | DeepMind Lab | [paper-as-1612-03801-deepmind-lab](../entities/paper-as-1612-03801-deepmind-lab.md) |
| 069 | OpenAI Gym Retro | [paper-as-069-openai-gym-retro](../entities/paper-as-069-openai-gym-retro.md) |
| 070 | The Arcade Learning Environment: An Evaluation Platform for General Agents | [paper-as-1207-4708-the-arcade-learning-environment-an-evaluation-pl](../entities/paper-as-1207-4708-the-arcade-learning-environment-an-evaluation-pl.md) |

### Recommender System Benchmarks

| # | 论文 | 详情节点 |
|---|------|----------|
| 071 | KuaiSim: A Comprehensive Simulator for Recommender Systems | [paper-as-2309-12645-kuaisim-a-comprehensive-simulator-for-recommende](../entities/paper-as-2309-12645-kuaisim-a-comprehensive-simulator-for-recommende.md) |
| 072 | RL4RS: A Real-World Dataset for Reinforcement Learning based Recommender System | [paper-as-2110-11073-rl4rs-a-real-world-dataset-for-reinforcement-lea](../entities/paper-as-2110-11073-rl4rs-a-real-world-dataset-for-reinforcement-lea.md) |
| 073 | Sim-to-Real Interactive Recommendation via Off-Dynamics Reinforcement Learning | [paper-as-073-sim-to-real-interactive-recommendation-via-off-d](../entities/paper-as-073-sim-to-real-interactive-recommendation-via-off-d.md) |

### Recommender System Environments

| # | 论文 | 详情节点 |
|---|------|----------|
| 074 | RecoGym: A Reinforcement Learning Environment for the problem of Product Recommendation in | [paper-as-1808-00720-recogym-a-reinforcement-learning-environment-for](../entities/paper-as-1808-00720-recogym-a-reinforcement-learning-environment-for.md) |
| 075 | Recsim: A configurable simulation platform for recommender systems | [paper-as-1909-04847-recsim-a-configurable-simulation-platform-for-re](../entities/paper-as-1909-04847-recsim-a-configurable-simulation-platform-for-re.md) |
| 076 | Reinforcement Learning for Slate-based Recommender Systems: A Tractable Decomposition and  | [paper-as-1905-12767-reinforcement-learning-for-slate-based-recommend](../entities/paper-as-1905-12767-reinforcement-learning-for-slate-based-recommend.md) |
| 077 | Virtual-Taobao: Virtualizing Real-world Online Retail Environment for Reinforcement Learni | [paper-as-1805-10000-virtual-taobao-virtualizing-real-world-online-re](../entities/paper-as-1805-10000-virtual-taobao-virtualizing-real-world-online-re.md) |

### Reward / LLM-Based Reward Design

| # | 论文 | 详情节点 |
|---|------|----------|
| 078 | Accessing gpt-4 level mathematical olympiad solutions via monte carlo tree self-refine wit | [paper-as-2406-07394-accessing-gpt-4-level-mathematical-olympiad-solu](../entities/paper-as-2406-07394-accessing-gpt-4-level-mathematical-olympiad-solu.md) |
| 079 | CurricuLLM: Automatic Task Curricula Design for Learning Complex Robot Skills using Large  | [paper-as-2409-18382-curricullm-automatic-task-curricula-design-for-l](../entities/paper-as-2409-18382-curricullm-automatic-task-curricula-design-for-l.md) |
| 080 | EvoPrompting: language models for code-level neural architecture search | [paper-as-2302-14838-evoprompting-language-models-for-code-level-neur](../entities/paper-as-2302-14838-evoprompting-language-models-for-code-level-neur.md) |

### Reward / Reward Shaping

| # | 论文 | 详情节点 |
|---|------|----------|
| 081 | A simple framework for intrinsic reward-shaping for rl using llm feedback | [paper-as-081-a-simple-framework-for-intrinsic-reward-shaping](../entities/paper-as-081-a-simple-framework-for-intrinsic-reward-shaping.md) |
| 082 | Adaptive Reinforcement Learning with LLM-augmented Reward Functions | [paper-as-082-adaptive-reinforcement-learning-with-llm-augment](../entities/paper-as-082-adaptive-reinforcement-learning-with-llm-augment.md) |
| 083 | Reward design with language models | [paper-as-2303-00001-reward-design-with-language-models](../entities/paper-as-2303-00001-reward-design-with-language-models.md) |

### Robotics Benchmarks

| # | 论文 | 详情节点 |
|---|------|----------|
| 084 | DISCOVERSE: Efficient Robot Simulation in Complex High-Fidelity Environments | [paper-as-084-discoverse-efficient-robot-simulation-in-complex](../entities/paper-as-084-discoverse-efficient-robot-simulation-in-complex.md) |
| 085 | Humanoid-Gym: Reinforcement Learning for Humanoid Robot with Zero-Shot Sim2Real Transfer | [humanoid-gym](../entities/humanoid-gym.md) |
| 086 | ManipulaTHOR: A Framework for Visual Object Manipulation | [paper-as-2104-11213-manipulathor-a-framework-for-visual-object-manip](../entities/paper-as-2104-11213-manipulathor-a-framework-for-visual-object-manip.md) |
| 087 | NeuronsGym: A Hybrid Framework and Benchmark for Robot Tasks with Sim2Real Policy Learning | [paper-as-087-neuronsgym-a-hybrid-framework-and-benchmark-for](../entities/paper-as-087-neuronsgym-a-hybrid-framework-and-benchmark-for.md) |
| 088 | RLBench: The Robot Learning Benchmark & Learning Environment | [paper-as-1909-12271-rlbench-the-robot-learning-benchmark-learning-en](../entities/paper-as-1909-12271-rlbench-the-robot-learning-benchmark-learning-en.md) |
| 089 | RRLS : Robust Reinforcement Learning Suite | [paper-as-2406-08406-rrls-robust-reinforcement-learning-suite](../entities/paper-as-2406-08406-rrls-robust-reinforcement-learning-suite.md) |
| 090 | Robust Gymnasium: A Unified Modular Benchmark for Robust Reinforcement Learning | [paper-as-090-robust-gymnasium-a-unified-modular-benchmark-for](../entities/paper-as-090-robust-gymnasium-a-unified-modular-benchmark-for.md) |

### Robotics Environments

| # | 论文 | 详情节点 |
|---|------|----------|
| 091 | Assistive Gym: A Physics Simulation Framework for Assistive Robotics | [paper-as-1910-04700-assistive-gym-a-physics-simulation-framework-for](../entities/paper-as-1910-04700-assistive-gym-a-physics-simulation-framework-for.md) |
| 092 | CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipu | [paper-as-2112-03227-calvin-a-benchmark-for-language-conditioned-poli](../entities/paper-as-2112-03227-calvin-a-benchmark-for-language-conditioned-poli.md) |
| 093 | Continuous Adaptation via Meta-Learning in Nonstationary and Competitive Environments | [paper-as-1710-03641-continuous-adaptation-via-meta-learning-in-nonst](../entities/paper-as-1710-03641-continuous-adaptation-via-meta-learning-in-nonst.md) |
| 094 | Delving Deeper into Out-of-Distribution Detection in Deep Neural Networks | [paper-as-2301-04195-delving-deeper-into-out-of-distribution-detectio](../entities/paper-as-2301-04195-delving-deeper-into-out-of-distribution-detectio.md) |
| 095 | Design and use paradigms for Gazebo, an open-source multi-robot simulator | [paper-as-095-design-and-use-paradigms-for-gazebo-an-open-sour](../entities/paper-as-095-design-and-use-paradigms-for-gazebo-an-open-sour.md) |
| 096 | Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning | [paper-as-1910-10897-meta-world-a-benchmark-and-evaluation-for-multi](../entities/paper-as-1910-10897-meta-world-a-benchmark-and-evaluation-for-multi.md) |
| 097 | MuJoCo: A physics engine for model-based control | [paper-as-097-mujoco-a-physics-engine-for-model-based-control](../entities/paper-as-097-mujoco-a-physics-engine-for-model-based-control.md) |
| 098 | OpenAI Gym | [paper-as-1606-01540-openai-gym](../entities/paper-as-1606-01540-openai-gym.md) |
| 099 | PyBullet: Real-Time Physics Simulation | [paper-as-099-pybullet-real-time-physics-simulation](../entities/paper-as-099-pybullet-real-time-physics-simulation.md) |
| 100 | SoftGym: Benchmarking Deep Reinforcement Learning for Deformable Object Manipulation | [paper-as-2011-07215-softgym-benchmarking-deep-reinforcement-learning](../entities/paper-as-2011-07215-softgym-benchmarking-deep-reinforcement-learning.md) |
| 101 | dm_control: Software and tasks for continuous control | [paper-as-101-dm-control-software-and-tasks-for-continuous-con](../entities/paper-as-101-dm-control-software-and-tasks-for-continuous-con.md) |
| 102 | robosuite: A Modular Simulation Framework and Benchmark for Robot Learning | [paper-as-2009-12293-robosuite-a-modular-simulation-framework-and-ben](../entities/paper-as-2009-12293-robosuite-a-modular-simulation-framework-and-ben.md) |

### Survey Papers

| # | 论文 | 详情节点 |
|---|------|----------|
| 103 | A Brief Survey of Sim2Real Methods for Robot Learning | [paper-as-103-a-brief-survey-of-sim2real-methods-for-robot-lea](../entities/paper-as-103-a-brief-survey-of-sim2real-methods-for-robot-lea.md) |
| 104 | A Survey on Sim-to-Real Transfer Methods for Robotic Manipulation | [paper-as-104-a-survey-on-sim-to-real-transfer-methods-for-rob](../entities/paper-as-104-a-survey-on-sim-to-real-transfer-methods-for-rob.md) |
| 105 | A survey of sim-to-real transfer techniques applied to reinforcement learning for bioinspi | [paper-as-105-a-survey-of-sim-to-real-transfer-techniques-appl](../entities/paper-as-105-a-survey-of-sim-to-real-transfer-techniques-appl.md) |
| 106 | Crossing the reality gap: A survey on sim-to-real transferability of robot controllers in  | [paper-as-106-crossing-the-reality-gap-a-survey-on-sim-to-real](../entities/paper-as-106-crossing-the-reality-gap-a-survey-on-sim-to-real.md) |
| 107 | How simulation helps autonomous driving: A survey of sim2real, digital twins, and parallel | [paper-as-107-how-simulation-helps-autonomous-driving-a-survey](../entities/paper-as-107-how-simulation-helps-autonomous-driving-a-survey.md) |
| 108 | Parallel learning: Overview and perspective for computational learning across Syn2Real and | [paper-as-108-parallel-learning-overview-and-perspective-for-c](../entities/paper-as-108-parallel-learning-overview-and-perspective-for-c.md) |
| 109 | Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey | [paper-as-109-sim-to-real-transfer-in-deep-reinforcement-learn](../entities/paper-as-109-sim-to-real-transfer-in-deep-reinforcement-learn.md) |

### Transition / Domain Adaptation

| # | 论文 | 详情节点 |
|---|------|----------|
| 110 | A brief review of domain adaptation | [paper-as-2010-03978-a-brief-review-of-domain-adaptation](../entities/paper-as-2010-03978-a-brief-review-of-domain-adaptation.md) |
| 111 | Adversarial discriminative domain adaptation | [paper-as-111-adversarial-discriminative-domain-adaptation](../entities/paper-as-111-adversarial-discriminative-domain-adaptation.md) |
| 112 | Adversarial-learned loss for domain adaptation | [paper-as-2001-01046-adversarial-learned-loss-for-domain-adaptation](../entities/paper-as-2001-01046-adversarial-learned-loss-for-domain-adaptation.md) |
| 113 | Conditional adversarial domain adaptation | [paper-as-1705-10667-conditional-adversarial-domain-adaptation](../entities/paper-as-1705-10667-conditional-adversarial-domain-adaptation.md) |
| 114 | Multi-adversarial domain adaptation | [paper-as-1809-02176-multi-adversarial-domain-adaptation](../entities/paper-as-1809-02176-multi-adversarial-domain-adaptation.md) |

### Transition / Domain Randomization

| # | 论文 | 详情节点 |
|---|------|----------|
| 115 | Active Domain Randomization | [paper-as-1904-04762-active-domain-randomization](../entities/paper-as-1904-04762-active-domain-randomization.md) |
| 116 | Crossing the gap: A deep dive into zero-shot sim-to-real transfer for dynamics | [paper-as-116-crossing-the-gap-a-deep-dive-into-zero-shot-sim](../entities/paper-as-116-crossing-the-gap-a-deep-dive-into-zero-shot-sim.md) |
| 117 | Domain randomization for transferring deep neural networks from simulation to the real wor | [paper-as-117-domain-randomization-for-transferring-deep-neura](../entities/paper-as-117-domain-randomization-for-transferring-deep-neura.md) |
| 118 | Sim-to-real learning for bipedal locomotion under unsensed dynamic loads | [paper-as-2204-04340-sim-to-real-learning-for-bipedal-locomotion-unde](../entities/paper-as-2204-04340-sim-to-real-learning-for-bipedal-locomotion-unde.md) |
| 119 | Understanding domain randomization for sim-to-real transfer | [paper-notebook-domain-randomization-understanding-sim-to-real-t](../entities/paper-notebook-domain-randomization-understanding-sim-to-real-t.md) |

### Transition / Grounding Methods

| # | 论文 | 详情节点 |
|---|------|----------|
| 120 | An imitation from observation approach to transfer learning with dynamics mismatch | [paper-as-2008-01594-an-imitation-from-observation-approach-to-transf](../entities/paper-as-2008-01594-an-imitation-from-observation-approach-to-transf.md) |
| 121 | Grounded action transformation for robot learning in simulation | [paper-as-121-grounded-action-transformation-for-robot-learnin](../entities/paper-as-121-grounded-action-transformation-for-robot-learnin.md) |
| 122 | Reinforced grounded action transformation for sim-to-real transfer | [paper-as-2008-01279-reinforced-grounded-action-transformation-for-si](../entities/paper-as-2008-01279-reinforced-grounded-action-transformation-for-si.md) |
| 123 | Stochastic grounded action transformation for robot learning in simulation | [paper-as-123-stochastic-grounded-action-transformation-for-ro](../entities/paper-as-123-stochastic-grounded-action-transformation-for-ro.md) |
| 124 | Uncertainty-aware Grounded Action Transformation towards Sim-to-Real Transfer for Traffic  | [paper-as-2307-12388-uncertainty-aware-grounded-action-transformation](../entities/paper-as-2307-12388-uncertainty-aware-grounded-action-transformation.md) |

### Transition / LLM-Enhanced Approaches

| # | 论文 | 详情节点 |
|---|------|----------|
| 125 | Prompt to Transfer: Sim-to-Real Transfer for Traffic Signal Control with Prompt Learning | [paper-as-2308-14284-prompt-to-transfer-sim-to-real-transfer-for-traf](../entities/paper-as-2308-14284-prompt-to-transfer-sim-to-real-transfer-for-traf.md) |

### Transportation Environments

| # | 论文 | 详情节点 |
|---|------|----------|
| 126 | AutoVRL: A High Fidelity Autonomous Ground Vehicle Simulator for Sim-to-Real Deep Reinforc | [paper-as-2304-11496-autovrl-a-high-fidelity-autonomous-ground-vehicl](../entities/paper-as-2304-11496-autovrl-a-high-fidelity-autonomous-ground-vehicl.md) |
| 127 | CARLA: An open urban driving simulator | [paper-as-1711-03938-carla-an-open-urban-driving-simulator](../entities/paper-as-1711-03938-carla-an-open-urban-driving-simulator.md) |
| 128 | CityFlow: A Multi-Agent Reinforcement Learning Environment for Large Scale City Traffic Sc | [paper-as-128-cityflow-a-multi-agent-reinforcement-learning-en](../entities/paper-as-128-cityflow-a-multi-agent-reinforcement-learning-en.md) |
| 129 | Deepdrive Zero | [paper-as-129-deepdrive-zero](../entities/paper-as-129-deepdrive-zero.md) |
| 130 | Duckietown: An open, inexpensive and flexible platform for autonomy education and research | [paper-as-130-duckietown-an-open-inexpensive-and-flexible-plat](../entities/paper-as-130-duckietown-an-open-inexpensive-and-flexible-plat.md) |
| 131 | Highway-Env: An Environment for Autonomous Driving Decision-Making | [paper-as-131-highway-env-an-environment-for-autonomous-drivin](../entities/paper-as-131-highway-env-an-environment-for-autonomous-drivin.md) |
| 132 | InterSim: Interactive Traffic Simulation via Explicit Relation Modeling | [paper-as-2210-14413-intersim-interactive-traffic-simulation-via-expl](../entities/paper-as-2210-14413-intersim-interactive-traffic-simulation-via-expl.md) |
| 133 | MetaDrive: Composing Diverse Driving Scenarios for Generalizable Reinforcement Learning | [paper-as-2109-12674-metadrive-composing-diverse-driving-scenarios-fo](../entities/paper-as-2109-12674-metadrive-composing-diverse-driving-scenarios-fo.md) |
| 134 | Microscopic Traffic Simulation using SUMO | [paper-as-134-microscopic-traffic-simulation-using-sumo](../entities/paper-as-134-microscopic-traffic-simulation-using-sumo.md) |
| 135 | SMARTS: Scalable Multi-Agent Reinforcement Learning Training School for Autonomous Driving | [paper-as-2010-09776-smarts-scalable-multi-agent-reinforcement-learni](../entities/paper-as-2010-09776-smarts-scalable-multi-agent-reinforcement-learni.md) |
| 136 | SUMMIT: A Simulator for Urban Driving in Massive Mixed Traffic | [paper-as-1911-04074-summit-a-simulator-for-urban-driving-in-massive](../entities/paper-as-1911-04074-summit-a-simulator-for-urban-driving-in-massive.md) |
| 137 | TorchDriveEnv: A Reinforcement Learning Benchmark for Autonomous Driving with Reactive, Re | [paper-as-2405-04491-torchdriveenv-a-reinforcement-learning-benchmark](../entities/paper-as-2405-04491-torchdriveenv-a-reinforcement-learning-benchmark.md) |
| 138 | TrafficSim: Learning to Simulate Realistic Multi-Agent Behaviors | [paper-as-2101-06557-trafficsim-learning-to-simulate-realistic-multi](../entities/paper-as-2101-06557-trafficsim-learning-to-simulate-realistic-multi.md) |
| 139 | Waymax: An Accelerated, Data-Driven Simulator for Large-Scale Autonomous Driving Research | [paper-as-2310-08710-waymax-an-accelerated-data-driven-simulator-for](../entities/paper-as-2310-08710-waymax-an-accelerated-data-driven-simulator-for.md) |


## 局限与风险

- 索引级节点保留清单分组，**不替代** 深度论文页；主线工作应继续升格。
- 清单含非 arXiv 链接（IEEE / ResearchGate）；无 arXiv 条目以标题 slug 建节点，后续若补 arXiv 需合并去重。
- 上游更新后需重跑 `python3 scripts/generate_longchao_awesome_sim2real_entities.py` 再 `make ci-preflight`。

## 关联页面

- [AwesomeSim2Real（列表实体）](../entities/awesome-sim2real.md)
- [Sim2Real RL Survey（2502.13187）](../entities/paper-survey-sim2real-rl-foundation-models.md)
- [reinforcement-learning.md](../methods/reinforcement-learning.md)
- [locomotion.md](../tasks/locomotion.md)

## 参考来源

- [lc_awesome_sim2real_catalog.md](../../sources/papers/lc_awesome_sim2real_catalog.md)
- [sources/repos/awesome-sim2real.md](../../sources/repos/awesome-sim2real.md)
- 上游：<https://github.com/LongchaoDa/AwesomeSim2Real>

## 推荐继续阅读

- [AwesomeSim2Real GitHub](https://github.com/LongchaoDa/AwesomeSim2Real)
- [A Survey of Sim-to-Real Methods in RL（arXiv:2502.13187v3）](https://arxiv.org/abs/2502.13187v3)
