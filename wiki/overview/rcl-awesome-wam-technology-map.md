---
type: overview
tags: [overview, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog, technology-map]
status: complete
updated: 2026-09-25
summary: "RCL Awesome World-Action Models 技术地图：把 PAPERS.md 里的 564 条文献逐条拆成站内可点开的一页，按八大类浏览。"
related:
  - ../entities/awesome-world-action-models-rcl.md
  - ../methods/generative-world-models.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# RCL Awesome World-Action Models 技术地图

> 本页把 [Awesome World-Action Models](https://github.com/rcl-robotics/Awesome-World-Action-Models) 的 [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) 清单里的论文逐条拆成站内可点开的一页，方便按 **Foundational / VLA / WAMs / Datasets / …** 分组浏览、搜索，并顺着链接读同方向的工作。

## 一句话定义

**RCL Awesome WAM 技术地图** = 外部 PAPERS.md 的站内可点开版本（564 条、按 major category 分组，一点即达论文页）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- 原清单每条只有标题 + 链接 + 子类标签；这里逐条给出一页，可检索、可顺着相关内容继续读。
- 站内已有深读页的条目直接链过去；其余给出 **清单摘要页**：Contribution、原文链接与它在清单里的位置一页可见。
- 清单共 **564** 条，每条都有独立 detail 节点（arXiv 去重后链到 canonical 页）。

## 覆盖范围

| 项 | 值 |
|----|-----|
| 上游仓库 | <https://github.com/rcl-robotics/Awesome-World-Action-Models> |
| PAPERS.md | <https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md> |
| 列表实体 | [Awesome World-Action Models（RCL）](../entities/awesome-world-action-models-rcl.md) |
| 目录 source | [rcl_awesome_wam_catalog.md](../../sources/papers/rcl_awesome_wam_catalog.md) |

## 分组索引

### Benchmarks & simulators

| # | 论文 | 详情节点 |
|---|------|----------|
| 001 | A Large-scale Study of Representation Learning with the Visual Task Adaptation Benchmark | [paper-rcl-1910-04867-a-large-scale-study-of-representation-learning-w](../entities/paper-rcl-1910-04867-a-large-scale-study-of-representation-learning-w.md) |
| 002 | ACT-Bench: Towards Action Controllable World Models for Autonomous Driving | [paper-rcl-2412-05337-act-bench-towards-action-controllable-world-mode](../entities/paper-rcl-2412-05337-act-bench-towards-action-controllable-world-mode.md) |
| 003 | Behaviour Suite for Reinforcement Learning | [paper-rcl-1908-03568-behaviour-suite-for-reinforcement-learning](../entities/paper-rcl-1908-03568-behaviour-suite-for-reinforcement-learning.md) |
| 004 | Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-To-End Autonomous Drivi | [paper-rcl-2406-03877-bench2drive-towards-multi-ability-benchmarking-o](../entities/paper-rcl-2406-03877-bench2drive-towards-multi-ability-benchmarking-o.md) |
| 005 | Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments | [paper-rcl-ref-f851fa79e6baee5a919e-beyond-the-nav-graph-vision-and-language-navigat](../entities/paper-rcl-ref-f851fa79e6baee5a919e-beyond-the-nav-graph-vision-and-language-navigat.md) |
| 006 | CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipu | [paper-as-2112-03227-calvin-a-benchmark-for-language-conditioned-poli](../entities/paper-as-2112-03227-calvin-a-benchmark-for-language-conditioned-poli.md) |
| 007 | CARLA: An Open Urban Driving Simulator | [paper-rcl-ref-75baf2ba00d451202231-carla-an-open-urban-driving-simulator](../entities/paper-rcl-ref-75baf2ba00d451202231-carla-an-open-urban-driving-simulator.md) |
| 008 | Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning | [paper-rcl-ref-3145193f87c43012f17a-craftax-a-lightning-fast-benchmark-for-open-ende](../entities/paper-rcl-ref-3145193f87c43012f17a-craftax-a-lightning-fast-benchmark-for-open-ende.md) |
| 009 | Ctrl-World: A Controllable Generative World Model for Robot Manipulation | [paper-ctrl-world](../entities/paper-ctrl-world.md) |
| 010 | DeepMind Control Suite | [paper-rcl-1801-00690-deepmind-control-suite](../entities/paper-rcl-1801-00690-deepmind-control-suite.md) |
| 011 | Diffusion Transformer World-Action Model for AV Scene Prediction | [paper-rcl-2606-12987-diffusion-transformer-world-action-model-for-av](../entities/paper-rcl-2606-12987-diffusion-transformer-world-action-model-for-av.md) |
| 012 | Do World Action Models Generalize Better than VLAs? A Robustness Study | [paper-rcl-2603-22078-do-world-action-models-generalize-better-than-vl](../entities/paper-rcl-2603-22078-do-world-action-models-generalize-better-than-vl.md) |
| 013 | DriveDreamer-2: LLM-Enhanced World Models for Diverse Driving Video Generation | [paper-rcl-ref-eb71134f4ab2c037bb71-drivedreamer-2-llm-enhanced-world-models-for-div](../entities/paper-rcl-ref-eb71134f4ab2c037bb71-drivedreamer-2-llm-enhanced-world-models-for-div.md) |
| 014 | DrivingGen: A Comprehensive Benchmark for Generative Video World Models in Autonomous Driv | [paper-rcl-2601-01528-drivinggen-a-comprehensive-benchmark-for-generat](../entities/paper-rcl-2601-01528-drivinggen-a-comprehensive-benchmark-for-generat.md) |
| 015 | EA-WM: Event-Aware Generative World Model with Structured Kinematic-to-Visual Action Field | [paper-rcl-2605-06192-ea-wm-event-aware-generative-world-model-with-st](../entities/paper-rcl-2605-06192-ea-wm-event-aware-generative-world-model-with-st.md) |
| 016 | EWMBench: Evaluating Scene, Motion, and Semantic Quality in Embodied World Models | [ewmbench](../entities/ewmbench.md) |
| 017 | EgoGenesis: Egocentric World-Action Modeling with Online Anchored Projective Memory and Ac | [paper-rcl-2607-28243-egogenesis-egocentric-world-action-modeling-with](../entities/paper-rcl-2607-28243-egogenesis-egocentric-world-action-modeling-with.md) |
| 018 | EnerVerse-AC: Envisioning Embodied Environments with Action Condition | [paper-rcl-2505-09723-enerverse-ac-envisioning-embodied-environments-w](../entities/paper-rcl-2505-09723-enerverse-ac-envisioning-embodied-environments-w.md) |
| 019 | Evaluating Real-World Robot Manipulation Policies in Simulation | [paper-rcl-ref-76e692b7c04cea08a682-evaluating-real-world-robot-manipulation-policie](../entities/paper-rcl-ref-76e692b7c04cea08a682-evaluating-real-world-robot-manipulation-policie.md) |
| 020 | FMB: A functional manipulation benchmark for generalizable robotic learning | [paper-rcl-2401-08553-fmb-a-functional-manipulation-benchmark-for-gene](../entities/paper-rcl-2401-08553-fmb-a-functional-manipulation-benchmark-for-gene.md) |
| 021 | FolDeX: A Physical-World Benchmark for Long-Horizon Robotic Manipulation of Deformable Obj | [paper-foldex-deformable-clothes-benchmark](../entities/paper-foldex-deformable-clothes-benchmark.md) |
| 022 | GAIA-1: A Generative World Model for Autonomous Driving | [paper-gaia1](../entities/paper-gaia1.md) |
| 023 | GAIA-2: A Controllable Multi-View Generative World Model for Autonomous Driving | [paper-sa-2503-20523-gaia-2-a-controllable-multi-view-generative-worl](../entities/paper-sa-2503-20523-gaia-2-a-controllable-multi-view-generative-worl.md) |
| 024 | Generative World Modelling for Humanoids: 1X World Model Challenge Technical Report | [paper-rcl-2510-07092-generative-world-modelling-for-humanoids-1x-worl](../entities/paper-rcl-2510-07092-generative-world-modelling-for-humanoids-1x-worl.md) |
| 025 | Genie: Generative Interactive Environments | [paper-sa-2402-15391-genie-generative-interactive-environments](../entities/paper-sa-2402-15391-genie-generative-interactive-environments.md) |
| 026 | Habitat: A Platform for Embodied AI Research | [paper-rcl-ref-b06d4e2caeb9c57607b9-habitat-a-platform-for-embodied-ai-research](../entities/paper-rcl-ref-b06d4e2caeb9c57607b9-habitat-a-platform-for-embodied-ai-research.md) |
| 027 | INSPATIO-WORLD: A Real-Time 4D World Simulator via Spatiotemporal Autoregressive Modeling | [paper-sa-2604-07209-inspatio-world-a-real-time-4d-world-simulator-vi](../entities/paper-sa-2604-07209-inspatio-world-a-real-time-4d-world-simulator-vi.md) |
| 028 | Interactive World Simulator for Robot Policy Training and Evaluation | [paper-rcl-2603-08546-interactive-world-simulator-for-robot-policy-tra](../entities/paper-rcl-2603-08546-interactive-world-simulator-for-robot-policy-tra.md) |
| 029 | JailWAM: Jailbreaking World Action Models in Robot Control | [paper-rcl-2604-05498-jailwam-jailbreaking-world-action-models-in-robo](../entities/paper-rcl-2604-05498-jailwam-jailbreaking-world-action-models-in-robo.md) |
| 030 | LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models | [paper-rcl-2510-13626-libero-plus-in-depth-robustness-analysis-of-visi](../entities/paper-rcl-2510-13626-libero-plus-in-depth-robustness-analysis-of-visi.md) |
| 031 | LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning | [paper-rcl-2306-03310-libero-benchmarking-knowledge-transfer-for-lifel](../entities/paper-rcl-2306-03310-libero-benchmarking-knowledge-transfer-for-lifel.md) |
| 032 | Leveraging Procedural Generation to Benchmark Reinforcement Learning | [paper-rcl-ref-5c37def65669693c1520-leveraging-procedural-generation-to-benchmark-re](../entities/paper-rcl-ref-5c37def65669693c1520-leveraging-procedural-generation-to-benchmark-re.md) |
| 033 | LongScape: Advancing Long-Horizon Embodied World Models with Context-Aware MoE | [paper-rcl-2509-21790-longscape-advancing-long-horizon-embodied-world](../entities/paper-rcl-2509-21790-longscape-advancing-long-horizon-embodied-world.md) |
| 034 | MIND-V: Hierarchical World Model for Long-Horizon Robotic Manipulation with RL-based Physi | [paper-rcl-2512-06628-mind-v-hierarchical-world-model-for-long-horizon](../entities/paper-rcl-2512-06628-mind-v-hierarchical-world-model-for-long-horizon.md) |
| 035 | ManiSkill: Generalizable Manipulation Skill Benchmark with Large-Scale Demonstrations | [paper-rcl-ref-18fe7d2bfb0e93f4c45e-maniskill-generalizable-manipulation-skill-bench](../entities/paper-rcl-ref-18fe7d2bfb0e93f4c45e-maniskill-generalizable-manipulation-skill-bench.md) |
| 036 | ManipArena: A Controlled Benchmark for Diagnosing Generalization in Real-Robot Manipulatio | [paper-rcl-2603-28545-maniparena-a-controlled-benchmark-for-diagnosing](../entities/paper-rcl-2603-28545-maniparena-a-controlled-benchmark-for-diagnosing.md) |
| 037 | Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning | [paper-rcl-ref-4c7d069bbfa0f1875587-meta-world-a-benchmark-and-evaluation-for-multi](../entities/paper-rcl-ref-4c7d069bbfa0f1875587-meta-world-a-benchmark-and-evaluation-for-multi.md) |
| 038 | MoVieDrive: Urban Scene Synthesis with Multi-Modal Multi-View Video Diffusion Transformer | [paper-rcl-2508-14327-moviedrive-urban-scene-synthesis-with-multi-moda](../entities/paper-rcl-2508-14327-moviedrive-urban-scene-synthesis-with-multi-moda.md) |
| 039 | NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking | [paper-rcl-ref-e25271fa6f028e5611cf-navsim-data-driven-non-reactive-autonomous-vehic](../entities/paper-rcl-ref-e25271fa6f028e5611cf-navsim-data-driven-non-reactive-autonomous-vehic.md) |
| 040 | NuPlan: A closed-loop ML-based planning benchmark for autonomous vehicles | [paper-rcl-2106-11810-nuplan-a-closed-loop-ml-based-planning-benchmark](../entities/paper-rcl-2106-11810-nuplan-a-closed-loop-ml-based-planning-benchmark.md) |
| 041 | PAIWorld: A 3D-Consistent World Foundation Model for Robotic Manipulation | [paper-sa-2606-18375-paiworld-a-3d-consistent-world-foundation-model](../entities/paper-sa-2606-18375-paiworld-a-3d-consistent-world-foundation-model.md) |
| 042 | PAN: A World Model for General, Interactable, and Long-Horizon World Simulation | [paper-sa-2511-09057-pan-a-world-model-for-general-interactable-and-l](../entities/paper-sa-2511-09057-pan-a-world-model-for-general-interactable-and-l.md) |
| 043 | Persistent Robot World Models: Stabilizing Multi-Step Rollouts via Reinforcement Learning | [paper-rcl-2603-25685-persistent-robot-world-models-stabilizing-multi](../entities/paper-rcl-2603-25685-persistent-robot-world-models-stabilizing-multi.md) |
| 044 | Programmable World Model | [paper-rcl-2609-10540-programmable-world-model](../entities/paper-rcl-2609-10540-programmable-world-model.md) |
| 045 | Pseudo-Simulation for Autonomous Driving | [paper-rcl-ref-361c60313b366d2109f0-pseudo-simulation-for-autonomous-driving](../entities/paper-rcl-ref-361c60313b366d2109f0-pseudo-simulation-for-autonomous-driving.md) |
| 046 | RLBench: The Robot Learning Benchmark & Learning Environment | [paper-as-1909-12271-rlbench-the-robot-learning-benchmark-learning-en](../entities/paper-as-1909-12271-rlbench-the-robot-learning-benchmark-learning-en.md) |
| 047 | RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots | [paper-rcl-ref-685493265d49d3ee589a-robocasa-large-scale-simulation-of-everyday-task](../entities/paper-rcl-ref-685493265d49d3ee589a-robocasa-large-scale-simulation-of-everyday-task.md) |
| 048 | RoboDojo: A Unified Sim-and-Real Benchmark for Comprehensive Evaluation of Generalist Robo | [paper-rcl-2607-04434-robodojo-a-unified-sim-and-real-benchmark-for-co](../entities/paper-rcl-2607-04434-robodojo-a-unified-sim-and-real-benchmark-for-co.md) |
| 049 | RoboScape: Physics-informed Embodied World Model | [paper-sa-2506-23135-roboscape-physics-informed-embodied-world-model](../entities/paper-sa-2506-23135-roboscape-physics-informed-embodied-world-model.md) |
| 050 | RoboSynChallenge: Mastering Real-World Dexterity via Generalizing Synthesized Manipulation | [paper-robosynchallenge](../entities/paper-robosynchallenge.md) |
| 051 | RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins | [paper-rcl-ref-4aab09abc95513cc8eb4-robotwin-dual-arm-robot-benchmark-with-generativ](../entities/paper-rcl-ref-4aab09abc95513cc8eb4-robotwin-dual-arm-robot-benchmark-with-generativ.md) |
| 052 | RoboWM-Bench: A Benchmark for Evaluating World Models in Robotic Manipulation | [paper-rcl-2604-19092-robowm-bench-a-benchmark-for-evaluating-world-mo](../entities/paper-rcl-2604-19092-robowm-bench-a-benchmark-for-evaluating-world-mo.md) |
| 053 | SIMPLE: Simulation-Based Policy Learning and Evaluation for Humanoid Loco-manipulation | [paper-loco-manip-161-075-simple](../entities/paper-loco-manip-161-075-simple.md) |
| 054 | Scalable Policy Evaluation with Video World Models | [paper-rcl-2511-11520-scalable-policy-evaluation-with-video-world-mode](../entities/paper-rcl-2511-11520-scalable-policy-evaluation-with-video-world-mode.md) |
| 055 | The Arcade Learning Environment: An Evaluation Platform for General Agents | [paper-as-1207-4708-the-arcade-learning-environment-an-evaluation-pl](../entities/paper-as-1207-4708-the-arcade-learning-environment-an-evaluation-pl.md) |
| 056 | TourPhysics: Bringing Physics to World Models for Exploration and Manipulation from a Sing | [paper-rcl-2609-04911-tourphysics-bringing-physics-to-world-models-for](../entities/paper-rcl-2609-04911-tourphysics-bringing-physics-to-world-models-for.md) |
| 057 | Toward Physically Consistent Driving Video World Models under Challenging Trajectories | [paper-sa-2603-24506-toward-physically-consistent-driving-video-world](../entities/paper-sa-2603-24506-toward-physically-consistent-driving-video-world.md) |
| 058 | WorldArena: A Unified Benchmark for Evaluating Perception and Functional Utility of Embodi | [paper-sa-2602-08971-worldarena-a-unified-benchmark-for-evaluating-pe](../entities/paper-sa-2602-08971-worldarena-a-unified-benchmark-for-evaluating-pe.md) |
| 059 | WorldEval: World Model as Real-World Robot Policies Evaluator | [paper-sa-2505-19017-worldeval-world-model-as-real-world-robot-polici](../entities/paper-sa-2505-19017-worldeval-world-model-as-real-world-robot-polici.md) |
| 060 | WorldGym: World Model as An Environment for Policy Evaluation | [paper-rcl-ref-75da67884f9b1e1a961e-worldgym-world-model-as-an-environment-for-polic](../entities/paper-rcl-ref-75da67884f9b1e1a961e-worldgym-world-model-as-an-environment-for-polic.md) |
| 061 | WorldLens: Full-Spectrum Evaluations of Driving World Models in Real World | [paper-sa-2512-10958-worldlens-full-spectrum-evaluations-of-driving-w](../entities/paper-sa-2512-10958-worldlens-full-spectrum-evaluations-of-driving-w.md) |
| 062 | Wow, wo, val! A Comprehensive Embodied World Model Evaluation Turing Test | [paper-rcl-2601-04137-wow-wo-val-a-comprehensive-embodied-world-model](../entities/paper-rcl-2601-04137-wow-wo-val-a-comprehensive-embodied-world-model.md) |
| 063 | X-World: Controllable Ego-Centric Multi-Camera World Models for Scalable End-to-End Drivin | [paper-x-world](../entities/paper-x-world.md) |

### Components of WAMs

| # | 论文 | 详情节点 |
|---|------|----------|
| 064 | 3D Gaussian Splatting for Real-Time Radiance Field Rendering | [paper-rcl-2308-04079-3d-gaussian-splatting-for-real-time-radiance-fie](../entities/paper-rcl-2308-04079-3d-gaussian-splatting-for-real-time-radiance-fie.md) |
| 065 | Auto-Encoding Variational Bayes | [paper-rcl-1312-6114-auto-encoding-variational-bayes](../entities/paper-rcl-1312-6114-auto-encoding-variational-bayes.md) |
| 066 | BEVDet: High-performance Multi-camera 3D Object Detection in Bird-Eye-View | [paper-rcl-2112-11790-bevdet-high-performance-multi-camera-3d-object-d](../entities/paper-rcl-2112-11790-bevdet-high-performance-multi-camera-3d-object-d.md) |
| 067 | CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer | [paper-rcl-2408-06072-cogvideox-text-to-video-diffusion-models-with-an](../entities/paper-rcl-2408-06072-cogvideox-text-to-video-diffusion-models-with-an.md) |
| 068 | Cosmos-Transfer1: Conditional World Generation with Adaptive Multimodal Control | [paper-cosmos-transfer1](../entities/paper-cosmos-transfer1.md) |
| 069 | DINOv2: Learning Robust Visual Features without Supervision | [paper-dinov2](../entities/paper-dinov2.md) |
| 070 | DINOv3 | [paper-rcl-2508-10104-dinov3](../entities/paper-rcl-2508-10104-dinov3.md) |
| 071 | Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer | [paper-rcl-ref-626d1fe7536ab14c8fdc-exploring-the-limits-of-transfer-learning-with-a](../entities/paper-rcl-ref-626d1fe7536ab14c8fdc-exploring-the-limits-of-transfer-learning-with-a.md) |
| 072 | FAST: Efficient Action Tokenization for Vision-Language-Action Models | [paper-rcl-2501-09747-fast-efficient-action-tokenization-for-vision-la](../entities/paper-rcl-2501-09747-fast-efficient-action-tokenization-for-vision-la.md) |
| 073 | Gemma 3 Technical Report | [paper-rcl-2503-19786-gemma-3-technical-report](../entities/paper-rcl-2503-19786-gemma-3-technical-report.md) |
| 074 | High-Resolution Image Synthesis With Latent Diffusion Models | [paper-rcl-2112-10752-high-resolution-image-synthesis-with-latent-diff](../entities/paper-rcl-2112-10752-high-resolution-image-synthesis-with-latent-diff.md) |
| 075 | LLaMA: Open and Efficient Foundation Language Models | [paper-rcl-2302-13971-llama-open-and-efficient-foundation-language-mod](../entities/paper-rcl-2302-13971-llama-open-and-efficient-foundation-language-mod.md) |
| 076 | Language Models are Few-Shot Learners | [paper-rcl-2005-14165-language-models-are-few-shot-learners](../entities/paper-rcl-2005-14165-language-models-are-few-shot-learners.md) |
| 077 | Latent Action Pretraining from Videos | [paper-shenlan-wm-03-lapa](../entities/paper-shenlan-wm-03-lapa.md) |
| 078 | Learning Transferable Visual Models From Natural Language Supervision | [paper-rcl-ref-6f620ba80d9567d982a6-learning-transferable-visual-models-from-natural](../entities/paper-rcl-ref-6f620ba80d9567d982a6-learning-transferable-visual-models-from-natural.md) |
| 079 | Mixture-of-Transformers: A Sparse and Scalable Architecture for Multi-Modal Foundation Mod | [paper-rcl-ref-c3cbdc6867eb310bf60b-mixture-of-transformers-a-sparse-and-scalable-ar](../entities/paper-rcl-ref-c3cbdc6867eb310bf60b-mixture-of-transformers-a-sparse-and-scalable-ar.md) |
| 080 | Neural Discrete Representation Learning | [paper-rcl-1711-00937-neural-discrete-representation-learning](../entities/paper-rcl-1711-00937-neural-discrete-representation-learning.md) |
| 081 | PaLM-E: An Embodied Multimodal Language Model | [paper-palm-e-embodied-language-model](../entities/paper-palm-e-embodied-language-model.md) |
| 082 | PaliGemma: A versatile 3B VLM for transfer | [paper-rcl-2407-07726-paligemma-a-versatile-3b-vlm-for-transfer](../entities/paper-rcl-2407-07726-paligemma-a-versatile-3b-vlm-for-transfer.md) |
| 083 | PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation | [paper-rcl-ref-a6e65f34d4c161a2ba59-pointnet-deep-learning-on-point-sets-for-3d-clas](../entities/paper-rcl-ref-a6e65f34d4c161a2ba59-pointnet-deep-learning-on-point-sets-for-3d-clas.md) |
| 084 | Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models | [paper-rcl-ref-02a3d941412699a67ab3-prismatic-vlms-investigating-the-design-space-of](../entities/paper-rcl-ref-02a3d941412699a67ab3-prismatic-vlms-investigating-the-design-space-of.md) |
| 085 | Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading,  | [paper-rcl-2308-12966-qwen-vl-a-versatile-vision-language-model-for-un](../entities/paper-rcl-2308-12966-qwen-vl-a-versatile-vision-language-model-for-un.md) |
| 086 | Scalable Diffusion Models with Transformers | [paper-rcl-ref-b4155f59c47b47bdc94d-scalable-diffusion-models-with-transformers](../entities/paper-rcl-ref-b4155f59c47b47bdc94d-scalable-diffusion-models-with-transformers.md) |
| 087 | Seedance 1.0: Exploring the Boundaries of Video Generation Models | [paper-rcl-2506-09113-seedance-1-0-exploring-the-boundaries-of-video-g](../entities/paper-rcl-2506-09113-seedance-1-0-exploring-the-boundaries-of-video-g.md) |
| 088 | Sigmoid Loss for Language Image Pre-Training | [paper-rcl-2303-15343-sigmoid-loss-for-language-image-pre-training](../entities/paper-rcl-2303-15343-sigmoid-loss-for-language-image-pre-training.md) |
| 089 | Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets | [paper-rcl-2311-15127-stable-video-diffusion-scaling-latent-video-diff](../entities/paper-rcl-2311-15127-stable-video-diffusion-scaling-latent-video-diff.md) |
| 090 | V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning | [paper-sa-2603-14482-v-jepa-2-1-unlocking-dense-features-in-video-sel](../entities/paper-sa-2603-14482-v-jepa-2-1-unlocking-dense-features-in-video-sel.md) |
| 091 | Video generation models as world simulators | [paper-rcl-ref-2e934302c61e88be910f-video-generation-models-as-world-simulators](../entities/paper-rcl-ref-2e934302c61e88be910f-video-generation-models-as-world-simulators.md) |
| 092 | Wan: Open and Advanced Large-Scale Video Generative Models | [paper-wan-video](../entities/paper-wan-video.md) |
| 093 | World Simulation with Video Foundation Models for Physical AI | [paper-sa-2511-00062-world-simulation-with-video-foundation-models-fo](../entities/paper-sa-2511-00062-world-simulation-with-video-foundation-models-fo.md) |

### Datasets

| # | 论文 | 详情节点 |
|---|------|----------|
| 094 | A benchmark for the evaluation of RGB-D SLAM systems | [paper-rcl-ref-d0e3b6e2a926a91ba6c0-a-benchmark-for-the-evaluation-of-rgb-d-slam-sys](../entities/paper-rcl-ref-d0e3b6e2a926a91ba6c0-a-benchmark-for-the-evaluation-of-rgb-d-slam-sys.md) |
| 095 | ActiveGlasses: Learning Manipulation with Active Vision from Ego-centric Human Demonstrati | [paper-rcl-2604-08534-activeglasses-learning-manipulation-with-active](../entities/paper-rcl-2604-08534-activeglasses-learning-manipulation-with-active.md) |
| 096 | AgiBot World Colosseo: A Large-scale Manipulation Platform for Scalable and Intelligent Em | [paper-sa-2503-06669-agibot-world-colosseo-a-large-scale-manipulation](../entities/paper-sa-2503-06669-agibot-world-colosseo-a-large-scale-manipulation.md) |
| 097 | Are we ready for autonomous driving? The KITTI vision benchmark suite | [paper-rcl-ref-46f2622e9aef7e722cb2-are-we-ready-for-autonomous-driving-the-kitti-vi](../entities/paper-rcl-ref-46f2622e9aef7e722cb2-are-we-ready-for-autonomous-driving-the-kitti-vi.md) |
| 098 | Argoverse: 3D Tracking and Forecasting With Rich Maps | [paper-rcl-ref-901dd4e018709bd1ca32-argoverse-3d-tracking-and-forecasting-with-rich](../entities/paper-rcl-ref-901dd4e018709bd1ca32-argoverse-3d-tracking-and-forecasting-with-rich.md) |
| 099 | BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning | [paper-rcl-ref-663c359ac9bfe090d027-bdd100k-a-diverse-driving-dataset-for-heterogene](../entities/paper-rcl-ref-663c359ac9bfe090d027-bdd100k-a-diverse-driving-dataset-for-heterogene.md) |
| 100 | BridgeData V2: A Dataset for Robot Learning at Scale | [paper-rcl-ref-d0a7d699e0efc759ae64-bridgedata-v2-a-dataset-for-robot-learning-at-sc](../entities/paper-rcl-ref-d0a7d699e0efc759ae64-bridgedata-v2-a-dataset-for-robot-learning-at-sc.md) |
| 101 | Building Pretraining Data for World Models: An Unreal Engine-Based Pipeline for Action-Con | [paper-rcl-2609-03557-building-pretraining-data-for-world-models-an-un](../entities/paper-rcl-2609-03557-building-pretraining-data-for-world-models-an-un.md) |
| 102 | CoPeD-Advancing Multi-Robot Collaborative Perception: A Comprehensive Dataset in Real-Worl | [paper-rcl-2405-14731-coped-advancing-multi-robot-collaborative-percep](../entities/paper-rcl-2405-14731-coped-advancing-multi-robot-collaborative-percep.md) |
| 103 | DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset | [paper-rcl-ref-aa9c28fbf4242ea03696-droid-a-large-scale-in-the-wild-robot-manipulati](../entities/paper-rcl-ref-aa9c28fbf4242ea03696-droid-a-large-scale-in-the-wild-robot-manipulati.md) |
| 104 | DexGraspNet: A Large-Scale Robotic Dexterous Grasp Dataset for General Objects Based on Si | [paper-rcl-2210-02697-dexgraspnet-a-large-scale-robotic-dexterous-gras](../entities/paper-rcl-2210-02697-dexgraspnet-a-large-scale-robotic-dexterous-gras.md) |
| 105 | DreamGen: Unlocking Generalization in Robot Learning through Video World Models | [paper-notebook-dreamgen-unlocking-generalization-in-robot-learn](../entities/paper-notebook-dreamgen-unlocking-generalization-in-robot-learn.md) |
| 106 | Ego4D: Around the World in 3,000 Hours of Egocentric Video | [paper-rcl-ref-51f5ebd773eec33b2c6b-ego4d-around-the-world-in-3-000-hours-of-egocent](../entities/paper-rcl-ref-51f5ebd773eec33b2c6b-ego4d-around-the-world-in-3-000-hours-of-egocent.md) |
| 107 | EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video | [paper-notebook-egodex-learning-dexterous-manipulation-from-larg](../entities/paper-notebook-egodex-learning-dexterous-manipulation-from-larg.md) |
| 108 | GigaWorld-0: World Models as Data Engine to Empower Embodied AI | [paper-rcl-2511-19861-gigaworld-0-world-models-as-data-engine-to-empow](../entities/paper-rcl-2511-19861-gigaworld-0-world-models-as-data-engine-to-empow.md) |
| 109 | HiFi-UMI: Learning Deployable Manipulation Policies from High-Fidelity UMI Data Alone | [paper-hifi-umi](../entities/paper-hifi-umi.md) |
| 110 | How to Instruct Your Robot: Dense Language Annotations Power Robot Policy Learning | [paper-rcl-2605-17077-how-to-instruct-your-robot-dense-language-annota](../entities/paper-rcl-2605-17077-how-to-instruct-your-robot-dense-language-annota.md) |
| 111 | HowTo100M: Learning a Text-Video Embedding by Watching Hundred Million Narrated Video Clip | [paper-rcl-ref-887c80067c952c4655d9-howto100m-learning-a-text-video-embedding-by-wat](../entities/paper-rcl-ref-887c80067c952c4655d9-howto100m-learning-a-text-video-embedding-by-wat.md) |
| 112 | HumanX: Toward Agile and Generalizable Humanoid Interaction Skills from Human Videos | [paper-hrl-stack-05-humanx](../entities/paper-hrl-stack-05-humanx.md) |
| 113 | Matterport3D: Learning from RGB-D Data in Indoor Environments | [paper-rcl-1709-06158-matterport3d-learning-from-rgb-d-data-in-indoor](../entities/paper-rcl-1709-06158-matterport3d-learning-from-rgb-d-data-in-indoor.md) |
| 114 | MimicDreamer: Aligning Human and Robot Demonstrations for Scalable VLA Training | [paper-rcl-2509-22199-mimicdreamer-aligning-human-and-robot-demonstrat](../entities/paper-rcl-2509-22199-mimicdreamer-aligning-human-and-robot-demonstrat.md) |
| 115 | MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations | [paper-rcl-ref-5677f2aa425315809584-mimicgen-a-data-generation-system-for-scalable-r](../entities/paper-rcl-ref-5677f2aa425315809584-mimicgen-a-data-generation-system-for-scalable-r.md) |
| 116 | Open X-Embodiment: Robotic Learning Datasets and RT-X Models | [paper-open-x-embodiment](../entities/paper-open-x-embodiment.md) |
| 117 | OpenScene: The Largest Up-to-Date 3D Occupancy Prediction Benchmark in Autonomous Driving | [paper-rcl-ref-236eff7a0d0d26ae758e-openscene-the-largest-up-to-date-3d-occupancy-pr](../entities/paper-rcl-ref-236eff7a0d0d26ae758e-openscene-the-largest-up-to-date-3d-occupancy-pr.md) |
| 118 | RoboNet: Large-Scale Multi-Robot Learning | [paper-rcl-1910-11215-robonet-large-scale-multi-robot-learning](../entities/paper-rcl-1910-11215-robonet-large-scale-multi-robot-learning.md) |
| 119 | RoboTransfer: Controllable Geometry-Consistent Video Diffusion for Manipulation Policy Tra | [paper-rcl-2505-23171-robotransfer-controllable-geometry-consistent-vi](../entities/paper-rcl-2505-23171-robotransfer-controllable-geometry-consistent-vi.md) |
| 120 | RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for | [paper-rcl-2506-18088-robotwin-2-0-a-scalable-data-generator-and-bench](../entities/paper-rcl-2506-18088-robotwin-2-0-a-scalable-data-generator-and-bench.md) |
| 121 | Room-Across-Room: Multilingual Vision-and-Language Navigation with Dense Spatiotemporal Gr | [paper-rcl-ref-ad5e94306036d2729d9f-room-across-room-multilingual-vision-and-languag](../entities/paper-rcl-ref-ad5e94306036d2729d9f-room-across-room-multilingual-vision-and-languag.md) |
| 122 | Scalability in Perception for Autonomous Driving: Waymo Open Dataset | [paper-rcl-ref-7147250a035b50dba3eb-scalability-in-perception-for-autonomous-driving](../entities/paper-rcl-ref-7147250a035b50dba3eb-scalability-in-perception-for-autonomous-driving.md) |
| 123 | Scaling Data Generation in Vision-and-Language Navigation | [paper-rcl-ref-3fa175f81e25ece6b23f-scaling-data-generation-in-vision-and-language-n](../entities/paper-rcl-ref-3fa175f81e25ece6b23f-scaling-data-generation-in-vision-and-language-n.md) |
| 124 | Socially Compliant Navigation Dataset (SCAND): A Large-Scale Dataset of Demonstrations for | [paper-rcl-2203-15041-socially-compliant-navigation-dataset-scand-a-la](../entities/paper-rcl-2203-15041-socially-compliant-navigation-dataset-scand-a-la.md) |
| 125 | TartanDrive: A Large-Scale Dataset for Learning Off-Road Dynamics Models | [paper-rcl-2205-01791-tartandrive-a-large-scale-dataset-for-learning-o](../entities/paper-rcl-2205-01791-tartandrive-a-large-scale-dataset-for-learning-o.md) |
| 126 | The "Something Something" Video Database for Learning and Evaluating Visual Common Sense | [paper-rcl-ref-c71dcf53130e8fdc61fa-the-something-something-video-database-for-learn](../entities/paper-rcl-ref-c71dcf53130e8fdc61fa-the-something-something-video-database-for-learn.md) |
| 127 | The Kinetics Human Action Video Dataset | [paper-rcl-1705-06950-the-kinetics-human-action-video-dataset](../entities/paper-rcl-1705-06950-the-kinetics-human-action-video-dataset.md) |
| 128 | Universal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild Robots | [paper-rcl-ref-e064f89fc62c8df5da9f-universal-manipulation-interface-in-the-wild-rob](../entities/paper-rcl-ref-e064f89fc62c8df5da9f-universal-manipulation-interface-in-the-wild-rob.md) |
| 129 | Vision-and-Language Navigation: Interpreting Visually-Grounded Navigation Instructions in  | [paper-rcl-ref-e0201a8da35e8b32ce42-vision-and-language-navigation-interpreting-visu](../entities/paper-rcl-ref-e0201a8da35e8b32ce42-vision-and-language-navigation-interpreting-visu.md) |
| 130 | Zenseact Open Dataset: A large-scale and diverse multimodal dataset for autonomous driving | [paper-rcl-ref-12b97dfcece775bcc0b8-zenseact-open-dataset-a-large-scale-and-diverse](../entities/paper-rcl-ref-12b97dfcece775bcc0b8-zenseact-open-dataset-a-large-scale-and-diverse.md) |
| 131 | nuScenes: A Multimodal Dataset for Autonomous Driving | [paper-rcl-ref-a5d6a00ebc235c8fdf08-nuscenes-a-multimodal-dataset-for-autonomous-dri](../entities/paper-rcl-ref-a5d6a00ebc235c8fdf08-nuscenes-a-multimodal-dataset-for-autonomous-dri.md) |

### Evaluation metrics

| # | 论文 | 详情节点 |
|---|------|----------|
| 132 | Beyond Task Success: Behavioral and Representational Diagnostics for WAM and VLA | [paper-rcl-2606-01095-beyond-task-success-behavioral-and-representatio](../entities/paper-rcl-2606-01095-beyond-task-success-behavioral-and-representatio.md) |
| 133 | Beyond Task Success: Stage-Wise Reliability of World Model Planning under Sensing Degradat | [paper-rcl-2609-07126-beyond-task-success-stage-wise-reliability-of-wo](../entities/paper-rcl-2609-07126-beyond-task-success-stage-wise-reliability-of-wo.md) |
| 134 | Bleu: a Method for Automatic Evaluation of Machine Translation | [paper-rcl-ref-268dde884b1840a3a100-bleu-a-method-for-automatic-evaluation-of-machin](../entities/paper-rcl-ref-268dde884b1840a3a100-bleu-a-method-for-automatic-evaluation-of-machin.md) |
| 135 | CIDEr: Consensus-Based Image Description Evaluation | [paper-rcl-ref-24ca74d7b1d59b2ae206-cider-consensus-based-image-description-evaluati](../entities/paper-rcl-ref-24ca74d7b1d59b2ae206-cider-consensus-based-image-description-evaluati.md) |
| 136 | Deep Reinforcement Learning at the Edge of the Statistical Precipice | [paper-rcl-ref-92f666cd9d15e83951cb-deep-reinforcement-learning-at-the-edge-of-the-s](../entities/paper-rcl-ref-92f666cd9d15e83951cb-deep-reinforcement-learning-at-the-edge-of-the-s.md) |
| 137 | GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium | [paper-rcl-ref-b3c87355da574d40991f-gans-trained-by-a-two-time-scale-update-rule-con](../entities/paper-rcl-ref-b3c87355da574d40991f-gans-trained-by-a-two-time-scale-update-rule-con.md) |
| 138 | General Evaluation for Instruction Conditioned Navigation using Dynamic Time Warping | [paper-rcl-1907-05446-general-evaluation-for-instruction-conditioned-n](../entities/paper-rcl-1907-05446-general-evaluation-for-instruction-conditioned-n.md) |
| 139 | Image Quality Assessment: From Error Visibility to Structural Similarity | [paper-rcl-ref-0276752be34087981228-image-quality-assessment-from-error-visibility-t](../entities/paper-rcl-ref-0276752be34087981228-image-quality-assessment-from-error-visibility-t.md) |
| 140 | Is the Future Compatible? Diagnosing Dynamic Consistency in World Action Models | [paper-sa-2605-07514-action-state-consistency-is-the-future-compatibl](../entities/paper-sa-2605-07514-action-state-consistency-is-the-future-compatibl.md) |
| 141 | METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgmen | [paper-rcl-ref-73844778b2bcfa68b4e9-meteor-an-automatic-metric-for-mt-evaluation-wit](../entities/paper-rcl-ref-73844778b2bcfa68b4e9-meteor-an-automatic-metric-for-mt-evaluation-wit.md) |
| 142 | On Evaluation of Embodied Navigation Agents | [paper-rcl-1807-06757-on-evaluation-of-embodied-navigation-agents](../entities/paper-rcl-1807-06757-on-evaluation-of-embodied-navigation-agents.md) |
| 143 | Paired Exact-Reset Evaluation of a Prediction-Derived Medium-to-Full World-Model Cascade | [paper-rcl-2608-14650-paired-exact-reset-evaluation-of-a-prediction-de](../entities/paper-rcl-2608-14650-paired-exact-reset-evaluation-of-a-prediction-de.md) |
| 144 | ROUGE: A Package for Automatic Evaluation of Summaries | [paper-rcl-ref-00a732613bb51f4fb578-rouge-a-package-for-automatic-evaluation-of-summ](../entities/paper-rcl-ref-00a732613bb51f4fb578-rouge-a-package-for-automatic-evaluation-of-summ.md) |
| 145 | The Unreasonable Effectiveness of Deep Features as a Perceptual Metric | [paper-rcl-ref-beec1282930b73cf1272-the-unreasonable-effectiveness-of-deep-features](../entities/paper-rcl-ref-beec1282930b73cf1272-the-unreasonable-effectiveness-of-deep-features.md) |
| 146 | Towards Accurate Generative Models of Video: A New Metric & Challenges | [paper-rcl-1812-01717-towards-accurate-generative-models-of-video-a-ne](../entities/paper-rcl-1812-01717-towards-accurate-generative-models-of-video-a-ne.md) |

### Foundational work

| # | 论文 | 详情节点 |
|---|------|----------|
| 147 | AdaWorld: Learning Adaptable World Models with Latent Actions | [paper-rcl-ref-ca883d875395dd7ff120-adaworld-learning-adaptable-world-models-with-la](../entities/paper-rcl-ref-ca883d875395dd7ff120-adaworld-learning-adaptable-world-models-with-la.md) |
| 148 | Consistency Models | [paper-rcl-ref-041a05059886890708fc-consistency-models](../entities/paper-rcl-ref-041a05059886890708fc-consistency-models.md) |
| 149 | Contrastive Learning as Goal-Conditioned Reinforcement Learning | [paper-rcl-ref-4db46d89f6231c67051e-contrastive-learning-as-goal-conditioned-reinfor](../entities/paper-rcl-ref-4db46d89f6231c67051e-contrastive-learning-as-goal-conditioned-reinfor.md) |
| 150 | Critique of World Model | [paper-sa-2507-05169-critique-of-world-model](../entities/paper-sa-2507-05169-critique-of-world-model.md) |
| 151 | DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning | [paper-sa-2411-04983-dino-wm-world-models-on-pre-trained-visual-featu](../entities/paper-sa-2411-04983-dino-wm-world-models-on-pre-trained-visual-featu.md) |
| 152 | DayDreamer: World Models for Physical Robot Learning | [paper-daydreamer-world-models-real-robots](../entities/paper-daydreamer-world-models-real-robots.md) |
| 153 | Deep Reinforcement Learning from Human Preferences | [paper-rcl-ref-2062ffd7f6397312bd01-deep-reinforcement-learning-from-human-preferenc](../entities/paper-rcl-ref-2062ffd7f6397312bd01-deep-reinforcement-learning-from-human-preferenc.md) |
| 154 | Diffusion for World Modeling: Visual Details Matter in Atari | [paper-rcl-ref-3b350556ce83b51f84c8-diffusion-for-world-modeling-visual-details-matt](../entities/paper-rcl-ref-3b350556ce83b51f84c8-diffusion-for-world-modeling-visual-details-matt.md) |
| 155 | Diffusion policy: Visuomotor policy learning via action diffusion | [paper-rcl-ref-393a36f38d60db8631f4-diffusion-policy-visuomotor-policy-learning-via](../entities/paper-rcl-ref-393a36f38d60db8631f4-diffusion-policy-visuomotor-policy-learning-via.md) |
| 156 | Diversity is all you need: Learning skills without a reward function | [paper-rcl-ref-8c0a34d0c6ab6b64e3ae-diversity-is-all-you-need-learning-skills-withou](../entities/paper-rcl-ref-8c0a34d0c6ab6b64e3ae-diversity-is-all-you-need-learning-skills-withou.md) |
| 157 | Dream to Control: Learning Behaviors by Latent Imagination | [paper-rcl-ref-23813a5cbed0b2b6d37e-dream-to-control-learning-behaviors-by-latent-im](../entities/paper-rcl-ref-23813a5cbed0b2b6d37e-dream-to-control-learning-behaviors-by-latent-im.md) |
| 158 | Factor Graphs for Robot Perception | [paper-rcl-ref-dcea01617aaebb6971e4-factor-graphs-for-robot-perception](../entities/paper-rcl-ref-dcea01617aaebb6971e4-factor-graphs-for-robot-perception.md) |
| 159 | Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow | [paper-rcl-ref-8c87d43c8563d21b1a7e-flow-straight-and-fast-learning-to-generate-and](../entities/paper-rcl-ref-8c87d43c8563d21b1a7e-flow-straight-and-fast-learning-to-generate-and.md) |
| 160 | Improved Distribution Matching Distillation for Fast Image Synthesis | [paper-rcl-ref-8be34f0cf0d2ab79e166-improved-distribution-matching-distillation-for](../entities/paper-rcl-ref-8be34f0cf0d2ab79e166-improved-distribution-matching-distillation-for.md) |
| 161 | Learning Latent Dynamics for Planning from Pixels | [paper-planet-latent-dynamics](../entities/paper-planet-latent-dynamics.md) |
| 162 | Learning to summarize with human feedback | [paper-rcl-ref-066817e565cc911bd5c1-learning-to-summarize-with-human-feedback](../entities/paper-rcl-ref-066817e565cc911bd5c1-learning-to-summarize-with-human-feedback.md) |
| 163 | LoRA: Low-Rank Adaptation of Large Language Models | [paper-rcl-ref-71811bf04d6371d0bd82-lora-low-rank-adaptation-of-large-language-model](../entities/paper-rcl-ref-71811bf04d6371d0bd82-lora-low-rank-adaptation-of-large-language-model.md) |
| 164 | Mastering Atari with Discrete World Models | [paper-rcl-ref-e1815cc67e6f9fc2bb7e-mastering-atari-with-discrete-world-models](../entities/paper-rcl-ref-e1815cc67e6f9fc2bb7e-mastering-atari-with-discrete-world-models.md) |
| 165 | Mastering diverse control tasks through world models | [paper-rcl-ref-8fa0ebc722d8d35eaf75-mastering-diverse-control-tasks-through-world-mo](../entities/paper-rcl-ref-8fa0ebc722d8d35eaf75-mastering-diverse-control-tasks-through-world-mo.md) |
| 166 | Mean Flows for One-step Generative Modeling | [paper-rcl-ref-ce6916f81594a9c4c73f-mean-flows-for-one-step-generative-modeling](../entities/paper-rcl-ref-ce6916f81594a9c4c73f-mean-flows-for-one-step-generative-modeling.md) |
| 167 | Model predictive control: Theory and practice—A survey | [paper-rcl-ref-36bafee9274250697060-model-predictive-control-theory-and-practicea-su](../entities/paper-rcl-ref-36bafee9274250697060-model-predictive-control-theory-and-practicea-su.md) |
| 168 | Model-Based Reinforcement Learning for Atari | [paper-pai-1903-00374-simple](../entities/paper-pai-1903-00374-simple.md) |
| 169 | Monte-Carlo Planning in Large POMDPs | [paper-rcl-ref-f81b18b1d9818e0e2585-monte-carlo-planning-in-large-pomdps](../entities/paper-rcl-ref-f81b18b1d9818e0e2585-monte-carlo-planning-in-large-pomdps.md) |
| 170 | On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes | [paper-rcl-ref-8432892bf821930fe230-on-policy-distillation-of-language-models-learni](../entities/paper-rcl-ref-8432892bf821930fe230-on-policy-distillation-of-language-models-learni.md) |
| 171 | One-step Diffusion with Distribution Matching Distillation | [paper-rcl-ref-5c86e0d92b645b2b135f-one-step-diffusion-with-distribution-matching-di](../entities/paper-rcl-ref-5c86e0d92b645b2b135f-one-step-diffusion-with-distribution-matching-di.md) |
| 172 | Planning and acting in partially observable stochastic domains | [paper-rcl-ref-55da7bc51e67237814d2-planning-and-acting-in-partially-observable-stoc](../entities/paper-rcl-ref-55da7bc51e67237814d2-planning-and-acting-in-partially-observable-stoc.md) |
| 173 | Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning | [paper-rcl-2410-21845-precise-and-dexterous-robotic-manipulation-via-h](../entities/paper-rcl-2410-21845-precise-and-dexterous-robotic-manipulation-via-h.md) |
| 174 | Probabilistic Robotics | [paper-rcl-ref-47f59ffa32a9f466d486-probabilistic-robotics](../entities/paper-rcl-ref-47f59ffa32a9f466d486-probabilistic-robotics.md) |
| 175 | Rapid Exploration for Open-World Navigation with Latent Goal Models | [paper-rcl-ref-7874d48534f3bf22f87d-rapid-exploration-for-open-world-navigation-with](../entities/paper-rcl-ref-7874d48534f3bf22f87d-rapid-exploration-for-open-world-navigation-with.md) |
| 176 | Real-time humanoid motion generation through ZMP manipulation based on inverted pendulum c | [paper-rcl-ref-58789fde10c5f80c4c89-real-time-humanoid-motion-generation-through-zmp](../entities/paper-rcl-ref-58789fde10c5f80c4c89-real-time-humanoid-motion-generation-through-zmp.md) |
| 177 | Reinforcement Learning: An Introduction | [paper-rcl-ref-206bb9b995e39760f7d0-reinforcement-learning-an-introduction](../entities/paper-rcl-ref-206bb9b995e39760f7d0-reinforcement-learning-an-introduction.md) |
| 178 | Revisiting Sparse Rewards for Goal-Reaching Reinforcement Learning | [paper-rcl-ref-80d04bd5d03ee6f72663-revisiting-sparse-rewards-for-goal-reaching-rein](../entities/paper-rcl-ref-80d04bd5d03ee6f72663-revisiting-sparse-rewards-for-goal-reaching-rein.md) |
| 179 | SERL: A Software Suite for Sample-Efficient Robotic Reinforcement Learning | [paper-rcl-2401-16013-serl-a-software-suite-for-sample-efficient-robot](../entities/paper-rcl-2401-16013-serl-a-software-suite-for-sample-efficient-robot.md) |
| 180 | Scaling Offline Model-Based RL via Jointly-Optimized World-Action Model Pretraining | [paper-rcl-2410-00564-scaling-offline-model-based-rl-via-jointly-optim](../entities/paper-rcl-2410-00564-scaling-offline-model-based-rl-via-jointly-optim.md) |
| 181 | Sim-to-Real Transfer of Robotic Control with Dynamics Randomization | [paper-peng-dynamics-randomization-sim2real](../entities/paper-peng-dynamics-randomization-sim2real.md) |
| 182 | State Estimation for Robotics | [paper-rcl-ref-b116da337363048621f5-state-estimation-for-robotics](../entities/paper-rcl-ref-b116da337363048621f5-state-estimation-for-robotics.md) |
| 183 | TD-MPC2: Scalable, Robust World Models for Continuous Control | [paper-td-mpc2](../entities/paper-td-mpc2.md) |
| 184 | Temporal Difference Learning for Model Predictive Control | [paper-rcl-ref-8a72842d518697e95077-temporal-difference-learning-for-model-predictiv](../entities/paper-rcl-ref-8a72842d518697e95077-temporal-difference-learning-for-model-predictiv.md) |
| 185 | Transformers are Sample-Efficient World Models | [paper-rcl-2209-00588-transformers-are-sample-efficient-world-models](../entities/paper-rcl-2209-00588-transformers-are-sample-efficient-world-models.md) |
| 186 | When to Trust Your Model: Model-Based Policy Optimization | [paper-rcl-ref-e6aba3e8a3ffe044339d-when-to-trust-your-model-model-based-policy-opti](../entities/paper-rcl-ref-e6aba3e8a3ffe044339d-when-to-trust-your-model-model-based-policy-opti.md) |
| 187 | World Models | [paper-ha-schmidhuber-world-models](../entities/paper-ha-schmidhuber-world-models.md) |
| 188 | World Models via Policy-Guided Trajectory Diffusion | [paper-rcl-2312-08533-world-models-via-policy-guided-trajectory-diffus](../entities/paper-rcl-2312-08533-world-models-via-policy-guided-trajectory-diffus.md) |

### Major category not recorded

| # | 论文 | 详情节点 |
|---|------|----------|
| 189 | Advances and applications of occupancy models | [paper-rcl-ref-b5a6c332a7beaf120245-advances-and-applications-of-occupancy-models](../entities/paper-rcl-ref-b5a6c332a7beaf120245-advances-and-applications-of-occupancy-models.md) |
| 190 | BERT: A Review of Applications in Natural Language Processing and Understanding | [paper-rcl-2103-11943-bert-a-review-of-applications-in-natural-languag](../entities/paper-rcl-2103-11943-bert-a-review-of-applications-in-natural-languag.md) |
| 191 | Diagnosing and Mitigating Perception-Decision Misalignment in Omni-LLMs via Modality Subsp | [paper-rcl-2608-14655-diagnosing-and-mitigating-perception-decision-mi](../entities/paper-rcl-2608-14655-diagnosing-and-mitigating-perception-decision-mi.md) |

### Related resources

| # | 论文 | 详情节点 |
|---|------|----------|
| 192 | A Survey on Vision-Language-Action Models for Autonomous Driving | [paper-rcl-ref-5d03f3be7549eb8453b0-a-survey-on-vision-language-action-models-for-au](../entities/paper-rcl-ref-5d03f3be7549eb8453b0-a-survey-on-vision-language-action-models-for-au.md) |
| 193 | A Watermark for Vision-Language-Action and World Action Models | [paper-rcl-2606-23574-a-watermark-for-vision-language-action-and-world](../entities/paper-rcl-2606-23574-a-watermark-for-vision-language-action-and-world.md) |
| 194 | A review of learning-based dynamics models for robotic manipulation | [paper-rcl-ref-ea14fce8991bbd0da18e-a-review-of-learning-based-dynamics-models-for-r](../entities/paper-rcl-ref-ea14fce8991bbd0da18e-a-review-of-learning-based-dynamics-models-for-r.md) |
| 195 | Attacking the Trusted Imagination: Oracle-Level Integrity Attacks on Imagine-then-Act Worl | [paper-rcl-2606-22966-attacking-the-trusted-imagination-oracle-level-i](../entities/paper-rcl-2606-22966-attacking-the-trusted-imagination-oracle-level-i.md) |
| 196 | BadWAM: When World-Action Models Dream Right but Act Wrong | [paper-rcl-2607-15207-badwam-when-world-action-models-dream-right-but](../entities/paper-rcl-2607-15207-badwam-when-world-action-models-dream-right-but.md) |
| 197 | CLAM: Continuous Latent Action Models for Robot Learning from Unlabeled Demonstrations | [paper-rcl-2505-04999-clam-continuous-latent-action-models-for-robot-l](../entities/paper-rcl-2505-04999-clam-continuous-latent-action-models-for-robot-l.md) |
| 198 | Critique of Agent Model | [paper-rcl-2606-23991-critique-of-agent-model](../entities/paper-rcl-2606-23991-critique-of-agent-model.md) |
| 199 | DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection | [paper-rcl-2203-03605-dino-detr-with-improved-denoising-anchor-boxes-f](../entities/paper-rcl-2203-03605-dino-detr-with-improved-denoising-anchor-boxes-f.md) |
| 200 | Data Pyramid for Embodied Manipulation: A Survey | [paper-data-pyramid-embodied-manipulation](../entities/paper-data-pyramid-embodied-manipulation.md) |
| 201 | Dexterity from Smart Lenses: Multi-Fingered Robot Manipulation with In-the-Wild Human Demo | [paper-notebook-dexterity-from-smart-lenses-multi-fingered-robot](../entities/paper-notebook-dexterity-from-smart-lenses-multi-fingered-robot.md) |
| 202 | DynaMo: In-Domain Dynamics Pretraining for Visuo-Motor Control | [paper-rcl-2409-12192-dynamo-in-domain-dynamics-pretraining-for-visuo](../entities/paper-rcl-2409-12192-dynamo-in-domain-dynamics-pretraining-for-visuo.md) |
| 203 | EgoBridge: Domain Adaptation for Generalizable Imitation from Egocentric Human Data | [paper-rcl-ref-646db90b75f1827a347d-egobridge-domain-adaptation-for-generalizable-im](../entities/paper-rcl-ref-646db90b75f1827a347d-egobridge-domain-adaptation-for-generalizable-im.md) |
| 204 | Embodied.cpp: A Portable Inference Runtime of Embodied AI Models on Heterogeneous Robots | [paper-rcl-2607-02501-embodied-cpp-a-portable-inference-runtime-of-emb](../entities/paper-rcl-2607-02501-embodied-cpp-a-portable-inference-runtime-of-emb.md) |
| 205 | FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry | [paper-rcl-2408-14035-fast-livo2-fast-direct-lidar-inertial-visual-odo](../entities/paper-rcl-2408-14035-fast-livo2-fast-direct-lidar-inertial-visual-odo.md) |
| 206 | Factored Latent Action World Models | [paper-rcl-2602-16229-factored-latent-action-world-models](../entities/paper-rcl-2602-16229-factored-latent-action-world-models.md) |
| 207 | From World Action Models to Embodied Brains: A Roadmap for Open-World Physical Intelligenc | [paper-rcl-2607-11689-from-world-action-models-to-embodied-brains-a-ro](../entities/paper-rcl-2607-11689-from-world-action-models-to-embodied-brains-a-ro.md) |
| 208 | From World Models to World Action Models: A Concise Tutorial for Robotics | [paper-sa-2607-00836-from-world-models-to-world-action-models-a-conci](../entities/paper-sa-2607-00836-from-world-models-to-world-action-models-a-conci.md) |
| 209 | LIC-Fusion: LiDAR-Inertial-Camera Odometry | [paper-rcl-1909-04102-lic-fusion-lidar-inertial-camera-odometry](../entities/paper-rcl-1909-04102-lic-fusion-lidar-inertial-camera-odometry.md) |
| 210 | Learning Generalizable Robot Policy with Human Demonstration Video as a Prompt | [paper-rcl-2505-20795-learning-generalizable-robot-policy-with-human-d](../entities/paper-rcl-2505-20795-learning-generalizable-robot-policy-with-human-d.md) |
| 211 | On the Capability Separation Between World-Model Policy Learning and Imitated World-Action | [paper-rcl-2608-22197-on-the-capability-separation-between-world-model](../entities/paper-rcl-2608-22197-on-the-capability-separation-between-world-model.md) |
| 212 | PhyAI: Real-Time Physical AI at the Edge, Scalable Rollouts in the Cloud | [paper-rcl-2608-03682-phyai-real-time-physical-ai-at-the-edge-scalable](../entities/paper-rcl-2608-03682-phyai-real-time-physical-ai-at-the-edge-scalable.md) |
| 213 | RoboHarness: Memory-Driven Orchestration of Heterogeneous Robot Policies for Long-Horizon  | [paper-robo-harness](../entities/paper-robo-harness.md) |
| 214 | Teach and Grow: An Agent-Centered Architecture for General Robot Learning | [paper-rcl-2608-17209-teach-and-grow-an-agent-centered-architecture-fo](../entities/paper-rcl-2608-17209-teach-and-grow-an-agent-centered-architecture-fo.md) |
| 215 | The Role of World Models in Shaping Autonomous Driving: A Comprehensive Survey | [paper-rcl-2502-10498-the-role-of-world-models-in-shaping-autonomous-d](../entities/paper-rcl-2502-10498-the-role-of-world-models-in-shaping-autonomous-d.md) |
| 216 | Toward Unified Robot Learning: Bridging Representation, Vision-Language-Action, and World  | [paper-unified-robot-learning-survey](../entities/paper-unified-robot-learning-survey.md) |
| 217 | Towards Generalist Embodied AI: A Survey on World Models for VLA Agents | [paper-rcl-ref-c36572e136b3aca5087e-towards-generalist-embodied-ai-a-survey-on-world](../entities/paper-rcl-ref-c36572e136b3aca5087e-towards-generalist-embodied-ai-a-survey-on-world.md) |
| 218 | Valerant: An Automatic Navigable Game Map Generator via Action-Conditioned World Model Exp | [paper-rcl-2609-09418-valerant-an-automatic-navigable-game-map-generat](../entities/paper-rcl-2609-09418-valerant-an-automatic-navigable-game-map-generat.md) |
| 219 | World Action Models in Real Time: An Empirical Study of Smooth Execution via Asynchronous  | [paper-wam-realtime-async](../entities/paper-wam-realtime-async.md) |
| 220 | World Action Models: A Survey | [paper-sa-2606-20781-world-action-models-a-survey](../entities/paper-sa-2606-20781-world-action-models-a-survey.md) |
| 221 | World Action Models: The Next Frontier in Embodied AI | [paper-rcl-2605-12090-world-action-models-the-next-frontier-in-embodie](../entities/paper-rcl-2605-12090-world-action-models-the-next-frontier-in-embodie.md) |
| 222 | World Model for Robot Learning: A Comprehensive Survey | [paper-sa-2605-00080-world-model-for-robot-learning-a-comprehensive-s](../entities/paper-sa-2605-00080-world-model-for-robot-learning-a-comprehensive-s.md) |

### VLA

| # | 论文 | 详情节点 |
|---|------|----------|
| 223 | A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation | [paper-rcl-2507-05331-a-careful-examination-of-large-behavior-models-f](../entities/paper-rcl-2507-05331-a-careful-examination-of-large-behavior-models-f.md) |
| 224 | Advancing AI for the physical world | [paper-rcl-ref-23e2ef710ce5722e25a2-advancing-ai-for-the-physical-world](../entities/paper-rcl-ref-23e2ef710ce5722e25a2-advancing-ai-for-the-physical-world.md) |
| 225 | Artificial Foveated Perception for Mitigating Shortcut Learning in Robotic Foundation Mode | [paper-rcl-2607-10655-artificial-foveated-perception-for-mitigating-sh](../entities/paper-rcl-2607-10655-artificial-foveated-perception-for-mitigating-sh.md) |
| 226 | AttenA+: Rectifying Action Inequality in Robotic Foundation Models | [paper-rcl-2605-13548-attena-rectifying-action-inequality-in-robotic-f](../entities/paper-rcl-2605-13548-attena-rectifying-action-inequality-in-robotic-f.md) |
| 227 | AutoMoT: A Unified Vision-Language-Action Model with Asynchronous Mixture-of-Transformers  | [paper-rcl-2603-14851-automot-a-unified-vision-language-action-model-w](../entities/paper-rcl-2603-14851-automot-a-unified-vision-language-action-model-w.md) |
| 228 | AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Re | [paper-rcl-2506-13757-autovla-a-vision-language-action-model-for-end-t](../entities/paper-rcl-2506-13757-autovla-a-vision-language-action-model-for-end-t.md) |
| 229 | Beyond Data Scaling: Representation-Centric Continued Pre-training for Vision-Language-Act | [paper-vlact](../entities/paper-vlact.md) |
| 230 | CLAP: Contrastive Latent Action Pretraining for Learning Vision-Language-Action Models fro | [paper-rcl-2601-04061-clap-contrastive-latent-action-pretraining-for-l](../entities/paper-rcl-2601-04061-clap-contrastive-latent-action-pretraining-for-l.md) |
| 231 | EgoScale: Scaling Dexterous Manipulation with Diverse Egocentric Human Data | [paper-sa-2602-16710-egoscale-scaling-dexterous-manipulation-with-div](../entities/paper-sa-2602-16710-egoscale-scaling-dexterous-manipulation-with-div.md) |
| 232 | EventVLA: Event-Driven Visual Evidence Memory for Long-Horizon Vision-Language-Action Poli | [paper-eventvla-visual-evidence-memory](../entities/paper-eventvla-visual-evidence-memory.md) |
| 233 | Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success | [paper-rcl-2502-19645-fine-tuning-vision-language-action-models-optimi](../entities/paper-rcl-2502-19645-fine-tuning-vision-language-action-models-optimi.md) |
| 234 | GR00T N1: An Open Foundation Model for Generalist Humanoid Robots | [paper-hrl-stack-34-gr00t_n1](../entities/paper-hrl-stack-34-gr00t_n1.md) |
| 235 | Geometry Guided Self-Consistency for Physical AI | [paper-rcl-2605-08638-geometry-guided-self-consistency-for-physical-ai](../entities/paper-rcl-2605-08638-geometry-guided-self-consistency-for-physical-ai.md) |
| 236 | GigaBrain-0: A World Model-Powered Vision-Language-Action Model | [paper-sa-2510-19430-gigabrain-0-a-world-model-powered-vision-languag](../entities/paper-sa-2510-19430-gigabrain-0-a-world-model-powered-vision-languag.md) |
| 237 | Ground Slow, Move Fast: A Dual-System Foundation Model for Generalizable Vision-and-Langua | [paper-rcl-2512-08186-ground-slow-move-fast-a-dual-system-foundation-m](../entities/paper-rcl-2512-08186-ground-slow-move-fast-a-dual-system-foundation-m.md) |
| 238 | Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives via Memory-Guided  | [paper-harness-vla](../entities/paper-harness-vla.md) |
| 239 | Helix: A Vision-Language-Action Model for Generalist Humanoid Control | [paper-rcl-ref-cb61c489d1333f433fc4-helix-a-vision-language-action-model-for-general](../entities/paper-rcl-ref-cb61c489d1333f433fc4-helix-a-vision-language-action-model-for-general.md) |
| 240 | HiMem-WAM: Hierarchical Memory-Gated World Action Models for Robotic Manipulation | [paper-rcl-2606-10363-himem-wam-hierarchical-memory-gated-world-action](../entities/paper-rcl-2606-10363-himem-wam-hierarchical-memory-gated-world-action.md) |
| 241 | Hierarchical Latent Action Model | [paper-rcl-2603-05815-hierarchical-latent-action-model](../entities/paper-rcl-2603-05815-hierarchical-latent-action-model.md) |
| 242 | Key-Gram: Extensible World Knowledge for Embodied Manipulation | [paper-rcl-2605-18556-key-gram-extensible-world-knowledge-for-embodied](../entities/paper-rcl-2605-18556-key-gram-extensible-world-knowledge-for-embodied.md) |
| 243 | LatBot: Distilling Universal Latent Actions for Vision-Language-Action Models | [paper-rcl-2511-23034-latbot-distilling-universal-latent-actions-for-v](../entities/paper-rcl-2511-23034-latbot-distilling-universal-latent-actions-for-v.md) |
| 244 | MEM: Multi-Scale Embodied Memory for Vision Language Action Models | [paper-pai-2603-03596-memmultiscaleembodiedmemory](../entities/paper-pai-2603-03596-memmultiscaleembodiedmemory.md) |
| 245 | OASIS: Observation-Action Space Alignment via SE(3) Trajectory Prediction for Robotic Mani | [paper-rcl-2605-25829-oasis-observation-action-space-alignment-via-se](../entities/paper-rcl-2605-25829-oasis-observation-action-space-alignment-via-se.md) |
| 246 | Octo: An Open-Source Generalist Robot Policy | [paper-rcl-ref-1c048eabe2faa444f31b-octo-an-open-source-generalist-robot-policy](../entities/paper-rcl-ref-1c048eabe2faa444f31b-octo-an-open-source-generalist-robot-policy.md) |
| 247 | OpenDriveVLA: Towards End-to-end Autonomous Driving with Large Vision Language Action Mode | [paper-rcl-ref-b5386c6f934f87f4cec4-opendrivevla-towards-end-to-end-autonomous-drivi](../entities/paper-rcl-ref-b5386c6f934f87f4cec4-opendrivevla-towards-end-to-end-autonomous-drivi.md) |
| 248 | OpenVLA: An Open-Source Vision-Language-Action Model | [paper-openvla](../entities/paper-openvla.md) |
| 249 | Percept-WAM: Perception-Enhanced World-Awareness-Action Model for Robust End-to-End Autono | [paper-rcl-ref-ced7109d62bb4514d467-percept-wam-perception-enhanced-world-awareness](../entities/paper-rcl-ref-ced7109d62bb4514d467-percept-wam-perception-enhanced-world-awareness.md) |
| 250 | RL Token: Bootstrapping Online RL with Vision-Language-Action Models | [paper-rcl-2604-23073-rl-token-bootstrapping-online-rl-with-vision-lan](../entities/paper-rcl-2604-23073-rl-token-bootstrapping-online-rl-with-vision-lan.md) |
| 251 | RT-1: Robotics Transformer for Real-World Control at Scale | [paper-rcl-ref-d73223ab358ff6f8ecd9-rt-1-robotics-transformer-for-real-world-control](../entities/paper-rcl-ref-d73223ab358ff6f8ecd9-rt-1-robotics-transformer-for-real-world-control.md) |
| 252 | RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control | [paper-rt-2](../entities/paper-rt-2.md) |
| 253 | SLIM-0.5B: Learning Action-Grounded Predictive Latents for Robot Manipulation | [paper-slim-05b](../entities/paper-slim-05b.md) |
| 254 | Spatial Forcing: Implicit Spatial Representation Alignment for Vision-Language-Action Mode | [paper-rcl-ref-0f2536c81a1992e3c3b8-spatial-forcing-implicit-spatial-representation](../entities/paper-rcl-ref-0f2536c81a1992e3c3b8-spatial-forcing-implicit-spatial-representation.md) |
| 255 | StreamVLN: Streaming Vision-and-Language Navigation via SlowFast Context Modeling | [paper-rcl-2507-05240-streamvln-streaming-vision-and-language-navigati](../entities/paper-rcl-2507-05240-streamvln-streaming-vision-and-language-navigati.md) |
| 256 | UniDriveVLA: Unifying Understanding, Perception, and Action Planning for Autonomous Drivin | [paper-rcl-2604-02190-unidrivevla-unifying-understanding-perception-an](../entities/paper-rcl-2604-02190-unidrivevla-unifying-understanding-perception-an.md) |
| 257 | UniVLA: Learning to Act Anywhere with Task-centric Latent Actions | [paper-rcl-2505-06111-univla-learning-to-act-anywhere-with-task-centri](../entities/paper-rcl-2505-06111-univla-learning-to-act-anywhere-with-task-centri.md) |
| 258 | Unifying Language-Action Understanding and Generation for Autonomous Driving | [paper-rcl-2603-01441-unifying-language-action-understanding-and-gener](../entities/paper-rcl-2603-01441-unifying-language-action-understanding-and-gener.md) |
| 259 | Video2Act: A Dual-System Video Diffusion Policy with Robotic Spatio-Motional Modeling | [paper-rcl-2512-03044-video2act-a-dual-system-video-diffusion-policy-w](../entities/paper-rcl-2512-03044-video2act-a-dual-system-video-diffusion-policy-w.md) |
| 260 | VideoWorld 2: Learning Transferable Knowledge from Real-world Videos | [paper-rcl-2602-10102-videoworld-2-learning-transferable-knowledge-fro](../entities/paper-rcl-2602-10102-videoworld-2-learning-transferable-knowledge-fro.md) |
| 261 | Vision-Language Foundation Models as Effective Robot Imitators | [paper-rcl-ref-06ccc4edca82d6457b34-vision-language-foundation-models-as-effective-r](../entities/paper-rcl-ref-06ccc4edca82d6457b34-vision-language-foundation-models-as-effective-r.md) |
| 262 | Vtla: Vision-tactile-language-action model with preference learning for insertion manipula | [paper-sa-2505-09577-vtla-vision-tactile-language-action-model-with-p](../entities/paper-sa-2505-09577-vtla-vision-tactile-language-action-model-with-p.md) |
| 263 | What Matters for Latent Actions in Robot Learning | [paper-latent-actions-matter](../entities/paper-latent-actions-matter.md) |
| 264 | dVLA-RL: Reinforcement Learning over Denoising Trajectories for Discrete Diffusion Vision- | [paper-rcl-2606-23623-dvla-rl-reinforcement-learning-over-denoising-tr](../entities/paper-rcl-2606-23623-dvla-rl-reinforcement-learning-over-denoising-tr.md) |
| 265 | π^*_0.6: a VLA That Learns From Experience | [paper-rcl-2511-14759-0-6-a-vla-that-learns-from-experience](../entities/paper-rcl-2511-14759-0-6-a-vla-that-learns-from-experience.md) |
| 266 | π_RL: Online RL Fine-tuning for Flow-based Vision-Language-Action Models | [paper-rcl-2510-25889-rl-online-rl-fine-tuning-for-flow-based-vision-l](../entities/paper-rcl-2510-25889-rl-online-rl-fine-tuning-for-flow-based-vision-l.md) |
| 267 | π₀.₅: a Vision-Language-Action Model with Open-World Generalization | [paper-rcl-ref-a351afa5504418f3e26f-0-5-a-vision-language-action-model-with-open-wor](../entities/paper-rcl-ref-a351afa5504418f3e26f-0-5-a-vision-language-action-model-with-open-wor.md) |
| 268 | π₀: A Vision-Language-Action Flow Model for General Robot Control | [paper-rcl-ref-502ccb43b26687d765d7-0-a-vision-language-action-flow-model-for-genera](../entities/paper-rcl-ref-502ccb43b26687d765d7-0-a-vision-language-action-flow-model-for-genera.md) |

### WAMs

| # | 论文 | 详情节点 |
|---|------|----------|
| 269 | 3DFlowAction: Learning Cross-Embodiment Manipulation from 3D Flow World Model | [paper-sa-2506-06199-3dflowaction-learning-cross-embodiment-manipulat](../entities/paper-sa-2506-06199-3dflowaction-learning-cross-embodiment-manipulat.md) |
| 270 | 4D-WAM: 4D Consistent World Modeling for Autonomous Driving | [paper-rcl-2608-10107-4d-wam-4d-consistent-world-modeling-for-autonomo](../entities/paper-rcl-2608-10107-4d-wam-4d-consistent-world-modeling-for-autonomo.md) |
| 271 | 4D-WAM: Infusing Spatiotemporal Awareness into World Action Models through Trajectory Fiel | [paper-4d-wam](../entities/paper-4d-wam.md) |
| 272 | 4DGS-WAM: Bridging Past and Future with an Object-Centric World Action Model based on 4D G | [paper-rcl-2608-25956-4dgs-wam-bridging-past-and-future-with-an-object](../entities/paper-rcl-2608-25956-4dgs-wam-bridging-past-and-future-with-an-object.md) |
| 273 | A0: An Affordance-Aware Hierarchical Model for General Robotic Manipulation | [paper-rcl-ref-4482db2c9a502c2e3bae-a0-an-affordance-aware-hierarchical-model-for-ge](../entities/paper-rcl-ref-4482db2c9a502c2e3bae-a0-an-affordance-aware-hierarchical-model-for-ge.md) |
| 274 | ABot-M0.5: Unified Mobility-and-Manipulation World Action Model | [paper-abot-m05-mobile-manipulation-wam](../entities/paper-abot-m05-mobile-manipulation-wam.md) |
| 275 | ADriver-I: A General World Model for Autonomous Driving | [paper-rcl-2311-13549-adriver-i-a-general-world-model-for-autonomous-d](../entities/paper-rcl-2311-13549-adriver-i-a-general-world-model-for-autonomous-d.md) |
| 276 | AHA-WAM:Asynchronous Horizon-Adaptive World-Action Modeling with Observation-Guided Contex | [paper-rcl-2606-09811-aha-wam-asynchronous-horizon-adaptive-world-acti](../entities/paper-rcl-2606-09811-aha-wam-asynchronous-horizon-adaptive-world-acti.md) |
| 277 | AIM: Intent-Aware Unified world action Modeling with Spatial Value Maps | [paper-rcl-2604-11135-aim-intent-aware-unified-world-action-modeling-w](../entities/paper-rcl-2604-11135-aim-intent-aware-unified-world-action-modeling-w.md) |
| 278 | AMPLIFY: Actionless Motion Priors for Robot Learning from Videos | [paper-rcl-2506-14198-amplify-actionless-motion-priors-for-robot-learn](../entities/paper-rcl-2506-14198-amplify-actionless-motion-priors-for-robot-learn.md) |
| 279 | AcrossWAM1.0:A Modular Latent World-Action Stack for Compact Robot Policies | [paper-rcl-2608-29937-acrosswam1-0-a-modular-latent-world-action-stack](../entities/paper-rcl-2608-29937-acrosswam1-0-a-modular-latent-world-action-stack.md) |
| 280 | Act2Goal: From World Model To General Goal-conditioned Policy | [paper-rcl-2512-23541-act2goal-from-world-model-to-general-goal-condit](../entities/paper-rcl-2512-23541-act2goal-from-world-model-to-general-goal-condit.md) |
| 281 | Action Images: End-to-End Policy Learning via Multiview Video Generation | [paper-rcl-2604-06168-action-images-end-to-end-policy-learning-via-mul](../entities/paper-rcl-2604-06168-action-images-end-to-end-policy-learning-via-mul.md) |
| 282 | Adaptive-WAM: Quality-Guided Early-Exit Planning from Intermediate Video-Diffusion Feature | [paper-rcl-2608-06008-adaptive-wam-quality-guided-early-exit-planning](../entities/paper-rcl-2608-06008-adaptive-wam-quality-guided-early-exit-planning.md) |
| 283 | AeroAct: Action-Centered World-Action Models for Language-Conditioned Quadrotor Flight | [paper-rcl-2607-14997-aeroact-action-centered-world-action-models-for](../entities/paper-rcl-2607-14997-aeroact-action-centered-world-action-models-for.md) |
| 284 | Being-H0.7: A Latent World-Action Model from Egocentric Videos | [paper-rcl-2605-00078-being-h0-7-a-latent-world-action-model-from-egoc](../entities/paper-rcl-2605-00078-being-h0-7-a-latent-world-action-model-from-egoc.md) |
| 285 | BrainWAM: Action-Space Coordination of Semantic Priors and Predictive Dynamics for Autonom | [paper-rcl-2608-12854-brainwam-action-space-coordination-of-semantic-p](../entities/paper-rcl-2608-12854-brainwam-action-space-coordination-of-semantic-p.md) |
| 286 | Bridge-WA: Predicting Where and How the World Changes for Robotic Action | [paper-rcl-2607-02195-bridge-wa-predicting-where-and-how-the-world-cha](../entities/paper-rcl-2607-02195-bridge-wa-predicting-where-and-how-the-world-cha.md) |
| 287 | Bridging Scene Generation and Planning: Driving with World Model via Unifying Vision and M | [paper-rcl-2603-14948-bridging-scene-generation-and-planning-driving-w](../entities/paper-rcl-2603-14948-bridging-scene-generation-and-planning-driving-w.md) |
| 288 | C3^3ache: Accelerating World Action Models with Cross Inference Chunk Cache | [paper-rcl-2606-08962-c3-3ache-accelerating-world-action-models-with-c](../entities/paper-rcl-2606-08962-c3-3ache-accelerating-world-action-models-with-c.md) |
| 289 | CKT-WAM: Parameter-Efficient Context Knowledge Transfer Between World Action Models | [paper-rcl-2605-06247-ckt-wam-parameter-efficient-context-knowledge-tr](../entities/paper-rcl-2605-06247-ckt-wam-parameter-efficient-context-knowledge-tr.md) |
| 290 | Causal World Modeling for Robot Control | [paper-sa-2601-21998-lingbot-va-causal-video-action-world-model-for-g](../entities/paper-sa-2601-21998-lingbot-va-causal-video-action-world-model-for-g.md) |
| 291 | CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models | [paper-sa-2503-22020-cot-vla-visual-chain-of-thought-reasoning-for-vi](../entities/paper-sa-2503-22020-cot-vla-visual-chain-of-thought-reasoning-for-vi.md) |
| 292 | CoVAR: Co-generation of Video and Action for Robotic Manipulation via Multi-Modal Diffusio | [paper-rcl-2512-16023-covar-co-generation-of-video-and-action-for-robo](../entities/paper-rcl-2512-16023-covar-co-generation-of-video-and-action-for-robo.md) |
| 293 | CoWAM: Coordination Contracts for Selective Policy Intervention with WAMs | [paper-rcl-2608-02578-cowam-coordination-contracts-for-selective-polic](../entities/paper-rcl-2608-02578-cowam-coordination-contracts-for-selective-polic.md) |
| 294 | Compact Visuotactile World Models for Lifting: Prediction, Reward Alignment, and Force Con | [paper-compact-visuotactile-wm-lifting](../entities/paper-compact-visuotactile-wm-lifting.md) |
| 295 | Cosmos 3: Omnimodal World Models for Physical AI | [cosmos-3](../entities/cosmos-3.md) |
| 296 | Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning | [paper-shenlan-wm-11-cosmos-policy](../entities/paper-shenlan-wm-11-cosmos-policy.md) |
| 297 | Cross-Embodiment Dexterous Manipulation through World Model Learning | [paper-rcl-ref-219f4f2dd6b959414001-cross-embodiment-dexterous-manipulation-through](../entities/paper-rcl-ref-219f4f2dd6b959414001-cross-embodiment-dexterous-manipulation-through.md) |
| 298 | DC-WAM: Dynamic-Centric Visual Supervision and Reasoning for World-Action Models | [paper-rcl-2607-25918-dc-wam-dynamic-centric-visual-supervision-and-re](../entities/paper-rcl-2607-25918-dc-wam-dynamic-centric-visual-supervision-and-re.md) |
| 299 | DECOWAM: Decoupled Whole-Body World-Action Model for Legged Mobile Manipulation | [paper-decowam](../entities/paper-decowam.md) |
| 300 | DELE-w0.5: Inferring Action from Future Latent State for Robotic Manipulation | [paper-rcl-2608-22067-dele-w0-5-inferring-action-from-future-latent-st](../entities/paper-rcl-2608-22067-dele-w0-5-inferring-action-from-future-latent-st.md) |
| 301 | DIM-WAM: World-Action Modeling with Diverse Historical Event Memory | [paper-sa-2606-27677-dim-wam-world-action-modeling-with-diverse-histo](../entities/paper-sa-2606-27677-dim-wam-world-action-modeling-with-diverse-histo.md) |
| 302 | DREAMWALKER: Mental Planning for Continuous Vision-Language Navigation | [paper-rcl-2308-07498-dreamwalker-mental-planning-for-continuous-visio](../entities/paper-rcl-2308-07498-dreamwalker-mental-planning-for-continuous-visio.md) |
| 303 | DSWAM: A Dual-System World Action Foundation Model for Fine-Grained Robot Manipulation | [paper-dswam-dual-system-wam](../entities/paper-dswam-dual-system-wam.md) |
| 304 | DUET-DINO: Simultaneous Cross-View World Modeling for Latent Planning in Robot Manipulatio | [paper-duet-dino](../entities/paper-duet-dino.md) |
| 305 | Decoupling Intention from Trajectory: A Representational Deduction Framework for World Act | [paper-rcl-2608-06994-decoupling-intention-from-trajectory-a-represent](../entities/paper-rcl-2608-06994-decoupling-intention-from-trajectory-a-represent.md) |
| 306 | DexWorldModel: Causal Latent World Modeling towards Automated Learning of Embodied Tasks | [paper-rcl-2604-16484-dexworldmodel-causal-latent-world-modeling-towar](../entities/paper-rcl-2604-16484-dexworldmodel-causal-latent-world-modeling-towar.md) |
| 307 | DiT4DiT: Jointly Modeling Video Dynamics and Actions for Generalizable Robot Control | [paper-dit4dit-video-action-model](../entities/paper-dit4dit-video-action-model.md) |
| 308 | Disentangling Visuo-Tactile Foresight: Oracle-Guided Interface Discovery for World Action  | [paper-rcl-2608-00547-disentangling-visuo-tactile-foresight-oracle-gui](../entities/paper-rcl-2608-00547-disentangling-visuo-tactile-foresight-oracle-gui.md) |
| 309 | Doe-1: Closed-Loop Autonomous Driving with Large World Model | [paper-sa-2412-09627-doe-1-closed-loop-autonomous-driving-with-large](../entities/paper-sa-2412-09627-doe-1-closed-loop-autonomous-driving-with-large.md) |
| 310 | Dream-Tac: A Unified Tactile World Action Model for Contact-Rich Robot Manipulation | [paper-sa-2606-08737-dream-tac-a-unified-tactile-world-action-model-f](../entities/paper-sa-2606-08737-dream-tac-a-unified-tactile-world-action-model-f.md) |
| 311 | Dream2Flow: Bridging Video Generation and Open-World Manipulation with 3D Object Flow | [paper-rcl-2512-24766-dream2flow-bridging-video-generation-and-open-wo](../entities/paper-rcl-2512-24766-dream2flow-bridging-video-generation-and-open-wo.md) |
| 312 | DreamDojo: A Real-Time Robot World Model from Large-Scale Human Videos | [paper-hrl-stack-35-dreamdojo](../entities/paper-hrl-stack-35-dreamdojo.md) |
| 313 | DreamPlan: Efficient Reinforcement Fine-Tuning of Vision-Language Planners via Video World | [paper-rcl-2603-16860-dreamplan-efficient-reinforcement-fine-tuning-of](../entities/paper-rcl-2603-16860-dreamplan-efficient-reinforcement-fine-tuning-of.md) |
| 314 | DreamVLA: A Vision-Language-Action Model Dreamed with Comprehensive World Knowledge | [paper-sa-2507-04447-dreamvla-a-vision-language-action-model-dreamed](../entities/paper-sa-2507-04447-dreamvla-a-vision-language-action-model-dreamed.md) |
| 315 | DreamWAM: Beyond RGB Future Prediction for World Action Models | [paper-dreamwam](../entities/paper-dreamwam.md) |
| 316 | DreamerAD: Efficient Reinforcement Learning via Latent World Model for Autonomous Driving | [paper-rcl-2603-24587-dreamerad-efficient-reinforcement-learning-via-l](../entities/paper-rcl-2603-24587-dreamerad-efficient-reinforcement-learning-via-l.md) |
| 317 | Dreaming when Necessary: Advancing World Action Models with Adaptive Multi-Modal Reasoning | [paper-rcl-2606-07089-dreaming-when-necessary-advancing-world-action-m](../entities/paper-rcl-2606-07089-dreaming-when-necessary-advancing-world-action-m.md) |
| 318 | Dreamitate: Real-World Visuomotor Policy Learning via Video Generation | [paper-rcl-2406-16862-dreamitate-real-world-visuomotor-policy-learning](../entities/paper-rcl-2406-16862-dreamitate-real-world-visuomotor-policy-learning.md) |
| 319 | Drive-HWM: Hierarchical World Models for Dynamic-Latent Guided Autonomous Driving | [paper-rcl-2609-03572-drive-hwm-hierarchical-world-models-for-dynamic](../entities/paper-rcl-2609-03572-drive-hwm-hierarchical-world-models-for-dynamic.md) |
| 320 | DriveDreamer-Policy: A Geometry-Grounded World-Action Model for Unified Generation and Pla | [paper-rcl-2604-01765-drivedreamer-policy-a-geometry-grounded-world-ac](../entities/paper-rcl-2604-01765-drivedreamer-policy-a-geometry-grounded-world-ac.md) |
| 321 | DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving | [paper-rcl-ref-604316201f6ab37b2f9a-drivedreamer-towards-real-world-driven-world-mod](../entities/paper-rcl-ref-604316201f6ab37b2f9a-drivedreamer-towards-real-world-driven-world-mod.md) |
| 322 | DriveLaW: Unifying Planning and Video Generation in a Latent Driving World | [paper-rcl-ref-b21a967bfc27f43d29b2-drivelaw-unifying-planning-and-video-generation](../entities/paper-rcl-ref-b21a967bfc27f43d29b2-drivelaw-unifying-planning-and-video-generation.md) |
| 323 | DriveVA: Video Action Models are Zero-Shot Drivers | [paper-rcl-2604-04198-driveva-video-action-models-are-zero-shot-driver](../entities/paper-rcl-2604-04198-driveva-video-action-models-are-zero-shot-driver.md) |
| 324 | DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving | [paper-rcl-ref-3a25df8af09d3c9d3d7a-drivevla-w0-world-models-amplify-data-scaling-la](../entities/paper-rcl-ref-3a25df8af09d3c9d3d7a-drivevla-w0-world-models-amplify-data-scaling-la.md) |
| 325 | DriveWAM: Video Generative Priors Enable Scalable World-Action Modeling for Autonomous Dri | [paper-rcl-2605-28544-drivewam-video-generative-priors-enable-scalable](../entities/paper-rcl-2605-28544-drivewam-video-generative-priors-enable-scalable.md) |
| 326 | DriveWorld-VLA: Unified Latent-Space World Modeling with Vision-Language-Action for Autono | [paper-rcl-2602-06521-driveworld-vla-unified-latent-space-world-modeli](../entities/paper-rcl-2602-06521-driveworld-vla-unified-latent-space-world-modeli.md) |
| 327 | Driving in the Occupancy World: Vision-Centric 4D Occupancy Forecasting and Planning via W | [paper-rcl-ref-07d6fe5781f45177707e-driving-in-the-occupancy-world-vision-centric-4d](../entities/paper-rcl-ref-07d6fe5781f45177707e-driving-in-the-occupancy-world-vision-centric-4d.md) |
| 328 | DrivingGPT: Unifying Driving World Modeling and Planning with Multi-modal Autoregressive T | [paper-sa-2412-18607-drivinggpt-unifying-driving-world-modeling-and-p](../entities/paper-sa-2412-18607-drivinggpt-unifying-driving-world-modeling-and-p.md) |
| 329 | DrivingWorld: Constructing World Model for Autonomous Driving via Video GPT | [paper-sa-2412-19505-drivingworld-constructing-world-model-for-autono](../entities/paper-sa-2412-19505-drivingworld-constructing-world-model-for-autono.md) |
| 330 | Dual-Stream Diffusion for World-Model Augmented Vision-Language-Action Model | [paper-rcl-2510-27607-dual-stream-diffusion-for-world-model-augmented](../entities/paper-rcl-2510-27607-dual-stream-diffusion-for-world-model-augmented.md) |
| 331 | DyWA: Dynamics-adaptive World Action Model for Generalizable Non-prehensile Manipulation | [paper-rcl-ref-283b7da95c4145cf56d0-dywa-dynamics-adaptive-world-action-model-for-ge](../entities/paper-rcl-ref-283b7da95c4145cf56d0-dywa-dynamics-adaptive-world-action-model-for-ge.md) |
| 332 | DynVLA: Learning World Dynamics for Action Reasoning in Autonomous Driving | [paper-rcl-2603-11041-dynvla-learning-world-dynamics-for-action-reason](../entities/paper-rcl-2603-11041-dynvla-learning-world-dynamics-for-action-reason.md) |
| 333 | Dyna-2: A 1-million-hour scaling law for world-action models | [paper-rcl-ref-ed0e9bb8027f431c1f20-dyna-2-a-1-million-hour-scaling-law-for-world-ac](../entities/paper-rcl-ref-ed0e9bb8027f431c1f20-dyna-2-a-1-million-hour-scaling-law-for-world-ac.md) |
| 334 | DynamicWAM: Dual-Path Motion Conditioning for World-Action Models in Dynamic Manipulation | [paper-rcl-2608-00793-dynamicwam-dual-path-motion-conditioning-for-wor](../entities/paper-rcl-2608-00793-dynamicwam-dual-path-motion-conditioning-for-wor.md) |
| 335 | EVA: Aligning Video World Models with Executable Robot Actions via Inverse Dynamics Reward | [paper-rcl-2603-17808-eva-aligning-video-world-models-with-executable](../entities/paper-rcl-2603-17808-eva-aligning-video-world-models-with-executable.md) |
| 336 | EWAM: An Enhanced World Action Model for Closed-Loop Online Adaptation in Embodied Intelli | [paper-rcl-2606-12690-ewam-an-enhanced-world-action-model-for-closed-l](../entities/paper-rcl-2606-12690-ewam-an-enhanced-world-action-model-for-closed-l.md) |
| 337 | Efficient Sim-to-Real Transfer of World-Action Models from Synthetic Priors | [paper-sa-2606-31101-efficient-sim-to-real-transfer-of-world-action-m](../entities/paper-sa-2606-31101-efficient-sim-to-real-transfer-of-world-action-m.md) |
| 338 | Efficient-WAM: A 1B-Parameter World-Action Model with Low-Cost Future Imagination | [paper-rcl-2606-10040-efficient-wam-a-1b-parameter-world-action-model](../entities/paper-rcl-2606-10040-efficient-wam-a-1b-parameter-world-action-model.md) |
| 339 | Ego-Vision World Model for Humanoid Contact Planning | [paper-hrl-stack-33-ego_vision_world_model_for_humanoid](../entities/paper-hrl-stack-33-ego_vision_world_model_for_humanoid.md) |
| 340 | EgoWAM: World Action Models Beyond Pixels with In-the-Wild Egocentric Human Data | [paper-sa-2607-08436-egowam-world-action-models-beyond-pixels-with-in](../entities/paper-sa-2607-08436-egowam-world-action-models-beyond-pixels-with-in.md) |
| 341 | EndoWAM: A Grounded World-Action Model for Generalizable Endoscopic Navigation | [paper-rcl-2608-01221-endowam-a-grounded-world-action-model-for-genera](../entities/paper-rcl-2608-01221-endowam-a-grounded-world-action-model-for-genera.md) |
| 342 | EnerVerse: Envisioning Embodied Future Space for Robotics Manipulation | [paper-sa-2501-01895-enerverse-envisioning-embodied-future-space-for](../entities/paper-sa-2501-01895-enerverse-envisioning-embodied-future-space-for.md) |
| 343 | Enhancing End-to-End Autonomous Driving with Latent World Model | [paper-sa-2406-08481-law-enhancing-end-to-end-autonomous-driving-with](../entities/paper-sa-2406-08481-law-enhancing-end-to-end-autonomous-driving-with.md) |
| 344 | Enhancing Policy Learning with World-Action Model | [paper-rcl-2603-28955-enhancing-policy-learning-with-world-action-mode](../entities/paper-rcl-2603-28955-enhancing-policy-learning-with-world-action-mode.md) |
| 345 | Epona: Autoregressive Diffusion World Model for Autonomous Driving | [paper-sa-2506-24113-epona-autoregressive-diffusion-world-model-for-a](../entities/paper-sa-2506-24113-epona-autoregressive-diffusion-world-model-for-a.md) |
| 346 | FACT: Failure-Aware Causal Training for World-Action Models | [paper-fact](../entities/paper-fact.md) |
| 347 | FAWAM: Force-Aware World Action Models for Closed-Loop Contact-Rich Manipulation | [paper-rcl-2606-08555-fawam-force-aware-world-action-models-for-closed](../entities/paper-rcl-2606-08555-fawam-force-aware-world-action-models-for-closed.md) |
| 348 | FBFM: A Training-Free Asynchronous Feedback Mechanism for Flow-Matching in World-Action Mo | [paper-rcl-2607-29235-fbfm-a-training-free-asynchronous-feedback-mecha](../entities/paper-rcl-2607-29235-fbfm-a-training-free-asynchronous-feedback-mecha.md) |
| 349 | FLARE: Robot Learning with Implicit World Modeling | [paper-sa-2505-15659-flare-robot-learning-with-implicit-world-modelin](../entities/paper-sa-2505-15659-flare-robot-learning-with-implicit-world-modelin.md) |
| 350 | Fast-WAM: Do World Action Models Need Test-time Future Imagination? | [paper-fast-wam](../entities/paper-fast-wam.md) |
| 351 | Faster-WAM: Do World Action Models Need Deep Action Modules? | [paper-rcl-2608-02365-faster-wam-do-world-action-models-need-deep-acti](../entities/paper-rcl-2608-02365-faster-wam-do-world-action-models-need-deep-acti.md) |
| 352 | Faster-WAM: Efficient Inference-Time Future Conditioning for Robust World Action Models | [paper-rcl-2608-04404-faster-wam-efficient-inference-time-future-condi](../entities/paper-rcl-2608-04404-faster-wam-efficient-inference-time-future-condi.md) |
| 353 | Flash-WAM: Modality-Aware Distillation for World Action Models | [paper-sa-2606-05254-flash-wam-modality-aware-distillation-for-world](../entities/paper-sa-2606-05254-flash-wam-modality-aware-distillation-for-world.md) |
| 354 | Flex-ππ: A Multi-Stream World-Action Model with Compute Flexibility | [paper-flex-pi](../entities/paper-flex-pi.md) |
| 355 | FlowDAgger: Human-in-the-Loop Adaptation of Generative Robot Policies in Latent Space | [paper-rcl-2607-08877-flowdagger-human-in-the-loop-adaptation-of-gener](../entities/paper-rcl-2607-08877-flowdagger-human-in-the-loop-adaptation-of-gener.md) |
| 356 | FlowPilot: Real-Time World-Action Modeling for Agile UAV Navigation | [paper-rcl-2608-00635-flowpilot-real-time-world-action-modeling-for-ag](../entities/paper-rcl-2608-00635-flowpilot-real-time-world-action-modeling-for-ag.md) |
| 357 | FlowWAM: Optical Flow as a Unified Action Representation for World Action Models | [paper-sa-2607-13017-flowwam-optical-flow-as-a-unified-action-represe](../entities/paper-sa-2607-13017-flowwam-optical-flow-as-a-unified-action-represe.md) |
| 358 | ForeTime-VLA: Causal Future-Token Distillation from a World Action Model for Conveyor-Belt | [paper-foretime-vla](../entities/paper-foretime-vla.md) |
| 359 | Foresight Without Seeing: Latent Futures for World Action Models | [paper-rcl-2608-11605-foresight-without-seeing-latent-futures-for-worl](../entities/paper-rcl-2608-11605-foresight-without-seeing-latent-futures-for-worl.md) |
| 360 | FutureNav: Unified World-Action Modeling for Vision-and-Language Navigation | [paper-rcl-2606-30367-futurenav-unified-world-action-modeling-for-visi](../entities/paper-rcl-2606-30367-futurenav-unified-world-action-modeling-for-visi.md) |
| 361 | GE-Act 2.0: Pretraining and Scaling a World-Action Model for Robotic Manipulation | [paper-ge-act-2](../entities/paper-ge-act-2.md) |
| 362 | GIFT: Guided Intermediate Feature Training via Action-Oriented Structural Supervision for  | [paper-gift-intermediate-feature-training](../entities/paper-gift-intermediate-feature-training.md) |
| 363 | GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge for Robot Manipula | [paper-rcl-2410-06158-gr-2-a-generative-video-language-action-model-wi](../entities/paper-rcl-2410-06158-gr-2-a-generative-video-language-action-model-wi.md) |
| 364 | GWM: Towards Scalable Gaussian World Models for Robotic Manipulation | [paper-sa-2508-17600-gwm-towards-scalable-gaussian-world-models-for-r](../entities/paper-sa-2508-17600-gwm-towards-scalable-gaussian-world-models-for-r.md) |
| 365 | GameWAM: A World Action Model for Video Games | [paper-rcl-2608-26200-gamewam-a-world-action-model-for-video-games](../entities/paper-rcl-2608-26200-gamewam-a-world-action-model-for-video-games.md) |
| 366 | GaussianWAM: Distilling Geometry and Semantics from 3D Gaussian Fields into World-Action M | [paper-rcl-2608-24714-gaussianwam-distilling-geometry-and-semantics-fr](../entities/paper-rcl-2608-24714-gaussianwam-distilling-geometry-and-semantics-fr.md) |
| 367 | Gen2Act: Human Video Generation in Novel Scenarios enables Generalizable Robot Manipulatio | [paper-rcl-2409-16283-gen2act-human-video-generation-in-novel-scenario](../entities/paper-rcl-2409-16283-gen2act-human-video-generation-in-novel-scenario.md) |
| 368 | Generalized Predictive Model for Autonomous Driving | [paper-rcl-ref-baf06f1a599804c753fb-generalized-predictive-model-for-autonomous-driv](../entities/paper-rcl-ref-baf06f1a599804c753fb-generalized-predictive-model-for-autonomous-driv.md) |
| 369 | Genie Envisioner: A Unified World Foundation Platform for Robotic Manipulation | [paper-sa-2508-05635-genie-envisioner-a-unified-world-foundation-plat](../entities/paper-sa-2508-05635-genie-envisioner-a-unified-world-foundation-plat.md) |
| 370 | GeoSem-WAM: Geometry- and Semantic-Aware World Action Models | [paper-rcl-2606-03188-geosem-wam-geometry-and-semantic-aware-world-act](../entities/paper-rcl-2606-03188-geosem-wam-geometry-and-semantic-aware-world-act.md) |
| 371 | GeoWAM: Visual Geometry World Action Models for Autonomous Driving | [paper-rcl-2608-23486-geowam-visual-geometry-world-action-models-for-a](../entities/paper-rcl-2608-23486-geowam-visual-geometry-world-action-models-for-a.md) |
| 372 | GeoWorldAD: Geometry World Action Model for Autonomous Driving | [paper-sa-2607-17521-geoworldad-geometry-world-action-model-for-auton](../entities/paper-sa-2607-17521-geoworldad-geometry-world-action-model-for-auton.md) |
| 373 | Geometric Action Model for Robot Policy Learning | [paper-rcl-2606-17046-geometric-action-model-for-robot-policy-learning](../entities/paper-rcl-2606-17046-geometric-action-model-for-robot-policy-learning.md) |
| 374 | GigaBrain-0.5M*: a VLA That Learns From World Model-Based Reinforcement Learning | [paper-sa-2602-12099-gigabrain-0-5m-a-vla-that-learns-from-world-mode](../entities/paper-sa-2602-12099-gigabrain-0-5m-a-vla-that-learns-from-world-mode.md) |
| 375 | GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch | [paper-sa-2607-13960-gigaworld-policy-0-5-a-faster-and-stronger-wam-e](../entities/paper-sa-2607-13960-gigaworld-policy-0-5-a-faster-and-stronger-wam-e.md) |
| 376 | GigaWorld-Policy: An Efficient Action-Centered World–Action Model | [paper-rcl-2603-17240-gigaworld-policy-an-efficient-action-centered-wo](../entities/paper-rcl-2603-17240-gigaworld-policy-an-efficient-action-centered-wo.md) |
| 377 | GlanceWAM: Sparse Test-Time Imagination for World-Action Models | [paper-glancewam](../entities/paper-glancewam.md) |
| 378 | Grounding Generated Video Plans in Simulation Towards Versatile Dexterous Controllers | [paper-galatea](../entities/paper-galatea.md) |
| 379 | HALO-WA: Hybrid-Attention Latent-Guided Online Reinforcement Learning for World-Action Mod | [paper-rcl-2607-04265-halo-wa-hybrid-attention-latent-guided-online-re](../entities/paper-rcl-2607-04265-halo-wa-hybrid-attention-latent-guided-online-re.md) |
| 380 | HaWMPO: Hallucination-Aware World Model-based Policy Optimization for Generalist Robot Pol | [paper-rcl-2609-09941-hawmpo-hallucination-aware-world-model-based-pol](../entities/paper-rcl-2609-09941-hawmpo-hallucination-aware-world-model-based-pol.md) |
| 381 | HarmoWAM: Harmonizing Generalizable and Precise Manipulation via Adaptive World Action Mod | [paper-rcl-2605-10942-harmowam-harmonizing-generalizable-and-precise-m](../entities/paper-rcl-2605-10942-harmowam-harmonizing-generalizable-and-precise-m.md) |
| 382 | HarnessWAM: Bridging Prediction and Deliberation in World Action Models | [paper-rcl-2608-09516-harnesswam-bridging-prediction-and-deliberation](../entities/paper-rcl-2608-09516-harnesswam-bridging-prediction-and-deliberation.md) |
| 383 | HiTac-WAM: A Hierarchical Tactile World Action Model for Contact-Rich Robot Manipulation | [paper-hitac-wam](../entities/paper-hitac-wam.md) |
| 384 | How to Learn from What a Human Would Avoid? Intervention-Aware World Models with Real-Worl | [paper-rcl-2609-06009-how-to-learn-from-what-a-human-would-avoid-inter](../entities/paper-rcl-2609-06009-how-to-learn-from-what-a-human-would-avoid-inter.md) |
| 385 | Hydra-0: Action Flow for Generalist World Modeling and Control | [paper-hydra-0](../entities/paper-hydra-0.md) |
| 386 | Hydra: A Navigation World Action Model with Discrete Latent Planning and Continuous Flow-M | [paper-rcl-2608-28995-hydra-a-navigation-world-action-model-with-discr](../entities/paper-rcl-2608-28995-hydra-a-navigation-world-action-model-with-discr.md) |
| 387 | IRL-VLA: Training an Vision-Language-Action Policy via Reward World Model | [paper-sa-2508-06571-irl-vla-training-an-vision-language-action-polic](../entities/paper-sa-2508-06571-irl-vla-training-an-vision-language-action-polic.md) |
| 388 | ImageWAM: Do World Action Models Really Need Video Generation, or Just Image Editing? | [paper-rcl-2606-19531-imagewam-do-world-action-models-really-need-vide](../entities/paper-rcl-2606-19531-imagewam-do-world-action-models-really-need-vide.md) |
| 389 | ImagineUAV: Aerial Vision-Language Navigation via World-Action Modeling and Kinodynamic Pl | [paper-rcl-2606-01205-imagineuav-aerial-vision-language-navigation-via](../entities/paper-rcl-2606-01205-imagineuav-aerial-vision-language-navigation-via.md) |
| 390 | Inference-Time Enhancement of Generative Robot Policies via Predictive World Modeling | [paper-sa-2502-00622-strengthening-generative-robot-policies-through](../entities/paper-sa-2502-00622-strengthening-generative-robot-policies-through.md) |
| 391 | JEPA Policy: Diffusion-Free Imitation Learning via Paired Action and Future Representation | [paper-jepa-policy](../entities/paper-jepa-policy.md) |
| 392 | JEPA-WAM: Learning Vision-Language-Action Policies with Joint-Embedding World Modeling | [paper-jepa-wam](../entities/paper-jepa-wam.md) |
| 393 | Kairos: A Regret-Aware Native World-Action Model Stack for Physical AI | [paper-kairos-native-world-model-stack](../entities/paper-kairos-native-world-model-stack.md) |
| 394 | Keep the Future, Drop the Rollout: RIFT for World Action Models | [paper-rift-wam](../entities/paper-rift-wam.md) |
| 395 | LD4WAM: Learning Latent Dynamics from Human Videos for World Action Models | [paper-ld4wam](../entities/paper-ld4wam.md) |
| 396 | LDA-1B: Scaling Latent Dynamics Action Model via Universal Embodied Data Ingestion | [paper-sa-2602-12215-lda-1b-scaling-latent-dynamics-action-model-via](../entities/paper-sa-2602-12215-lda-1b-scaling-latent-dynamics-action-model-via.md) |
| 397 | LMGenDrive: Bridging Multimodal Understanding and Generative World Modeling for End-to-End | [paper-rcl-2604-08719-lmgendrive-bridging-multimodal-understanding-and](../entities/paper-rcl-2604-08719-lmgendrive-bridging-multimodal-understanding-and.md) |
| 398 | LUMOS: Language-Conditioned Imitation Learning with World Models | [paper-sa-2503-10370-lumos-language-conditioned-imitation-learning-wi](../entities/paper-sa-2503-10370-lumos-language-conditioned-imitation-learning-wi.md) |
| 399 | LaWAM: Latent World Action Models for Efficient Dynamics-Aware Robot Policies | [paper-lawam](../entities/paper-lawam.md) |
| 400 | Latent Action Pretraining Through World Modeling | [paper-sa-2509-18428-lawm-latent-action-pretraining-through-world-mod](../entities/paper-sa-2509-18428-lawm-latent-action-pretraining-through-world-mod.md) |
| 401 | Latent Action as Intention Enables Efficient Future Imagination for World Action Models | [paper-lawa](../entities/paper-lawa.md) |
| 402 | Latent Energy Action Planning with World Models | [paper-rcl-2609-03294-latent-energy-action-planning-with-world-models](../entities/paper-rcl-2609-03294-latent-energy-action-planning-with-world-models.md) |
| 403 | Latent-WAM: Latent World Action Modeling for End-to-End Autonomous Driving | [paper-sa-2603-24581-latent-wam-latent-world-action-modeling-for-end](../entities/paper-sa-2603-24581-latent-wam-latent-world-action-modeling-for-end.md) |
| 404 | LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels | [paper-lewm](../entities/paper-lewm.md) |
| 405 | LeapBot-WA: World-Anchor Action Models via Predictive Latent Alignments | [paper-rcl-2607-23969-leapbot-wa-world-anchor-action-models-via-predic](../entities/paper-rcl-2607-23969-leapbot-wa-world-anchor-action-models-via-predic.md) |
| 406 | Learned Perceptive Forward Dynamics Model for Safe and Platform-aware Robotic Navigation | [paper-rcl-2504-19322-learned-perceptive-forward-dynamics-model-for-sa](../entities/paper-rcl-2504-19322-learned-perceptive-forward-dynamics-model-for-sa.md) |
| 407 | Learning 4D Geometric Priors for Inference-Efficient World Action Models | [paper-meco-wam-4d-geometry-cotraining](../entities/paper-meco-wam-4d-geometry-cotraining.md) |
| 408 | Learning Counterfactual World Models for Embodied Reasoning under Partial Observability | [paper-rcl-2609-05834-learning-counterfactual-world-models-for-embodie](../entities/paper-rcl-2609-05834-learning-counterfactual-world-models-for-embodie.md) |
| 409 | Learning Latent Action World Models In The Wild | [paper-sa-2601-05230-learning-latent-action-world-models-in-the-wild](../entities/paper-sa-2601-05230-learning-latent-action-world-models-in-the-wild.md) |
| 410 | Learning Massively Multitask World Models for Continuous Control | [paper-rcl-2511-19584-learning-massively-multitask-world-models-for-co](../entities/paper-rcl-2511-19584-learning-massively-multitask-world-models-for-co.md) |
| 411 | Learning Physics from Pretrained Video Models: A Multimodal Continuous and Sequential Worl | [paper-rcl-2603-00110-learning-physics-from-pretrained-video-models-a](../entities/paper-rcl-2603-00110-learning-physics-from-pretrained-video-models-a.md) |
| 412 | Learning Robot Manipulation from Audio World Models | [paper-rcl-2512-08405-learning-robot-manipulation-from-audio-world-mod](../entities/paper-rcl-2512-08405-learning-robot-manipulation-from-audio-world-mod.md) |
| 413 | Learning Universal Policies via Text-Guided Video Generation | [paper-rcl-2302-00111-learning-universal-policies-via-text-guided-vide](../entities/paper-rcl-2302-00111-learning-universal-policies-via-text-guided-vide.md) |
| 414 | Learning Vision-Language-Action World Models for Autonomous Driving | [paper-rcl-2604-09059-learning-vision-language-action-world-models-for](../entities/paper-rcl-2604-09059-learning-vision-language-action-world-models-for.md) |
| 415 | Learning Visual Feature-Based World Models via Residual Latent Action | [paper-rcl-2605-07079-learning-visual-feature-based-world-models-via-r](../entities/paper-rcl-2605-07079-learning-visual-feature-based-world-models-via-r.md) |
| 416 | Learning to Use Imagination: Progress-Conditioned Future Utilization for World Action Mode | [paper-rcl-2609-06578-learning-to-use-imagination-progress-conditioned](../entities/paper-rcl-2609-06578-learning-to-use-imagination-progress-conditioned.md) |
| 417 | Learning to unfold cloth: Scaling up world models to deformable object manipulation | [paper-rcl-2602-16675-learning-to-unfold-cloth-scaling-up-world-models](../entities/paper-rcl-2602-16675-learning-to-unfold-cloth-scaling-up-world-models.md) |
| 418 | LiLa-WAM: Lightweight Latent Reasoning World-Action Model for Robotic Manipulation | [paper-rcl-2608-03701-lila-wam-lightweight-latent-reasoning-world-acti](../entities/paper-rcl-2608-03701-lila-wam-lightweight-latent-reasoning-world-acti.md) |
| 419 | Light-WAM: Efficient World Action Models with State-Fusion Action Decoding | [paper-rcl-2606-08242-light-wam-efficient-world-action-models-with-sta](../entities/paper-rcl-2606-08242-light-wam-efficient-world-action-models-with-sta.md) |
| 420 | MV-WAM: Manifold-Aware World Action Model with Value Augmentation | [paper-rcl-2606-21088-mv-wam-manifold-aware-world-action-model-with-va](../entities/paper-rcl-2606-21088-mv-wam-manifold-aware-world-action-model-with-va.md) |
| 421 | Making Foresight Actionable: Repurposing Representation Alignment in World Action Models | [paper-rcl-2606-12217-making-foresight-actionable-repurposing-represen](../entities/paper-rcl-2606-12217-making-foresight-actionable-repurposing-represen.md) |
| 422 | Making Latent Evolution Explicit: Operator-Structured Transitions for World Action Models | [paper-rcl-2608-27259-making-latent-evolution-explicit-operator-struct](../entities/paper-rcl-2608-27259-making-latent-evolution-explicit-operator-struct.md) |
| 423 | ManiGaussian++: General Robotic Bimanual Manipulation with Hierarchical Gaussian World Mod | [paper-sa-2506-19842-manigaussian-general-robotic-bimanual-manipulati](../entities/paper-sa-2506-19842-manigaussian-general-robotic-bimanual-manipulati.md) |
| 424 | MaskWAM: Unifying Mask Prompting and Prediction for World-Action Models | [paper-rcl-2606-13515-maskwam-unifying-mask-prompting-and-prediction-f](../entities/paper-rcl-2606-13515-maskwam-unifying-mask-prompting-and-prediction-f.md) |
| 425 | MemoryVAM: Integrating Memory into Video Action Model for Robot Manipulation | [paper-rcl-2606-20679-memoryvam-integrating-memory-into-video-action-m](../entities/paper-rcl-2606-20679-memoryvam-integrating-memory-into-video-action-m.md) |
| 426 | MemoryWAM: Efficient World Action Modeling with Persistent Memory | [paper-memorywam](../entities/paper-memorywam.md) |
| 427 | Metis: A Generalizable and Efficient World-Action Model for Autonomous Driving and Urban N | [paper-rcl-2606-15869-metis-a-generalizable-and-efficient-world-action](../entities/paper-rcl-2606-15869-metis-a-generalizable-and-efficient-world-action.md) |
| 428 | MindDrive: An All-in-One Framework Bridging World Models and Vision-Language Model for End | [paper-rcl-2512-04441-minddrive-an-all-in-one-framework-bridging-world](../entities/paper-rcl-2512-04441-minddrive-an-all-in-one-framework-bridging-world.md) |
| 429 | MoWM: Mixture-of-World-Models for Embodied Planning via Latent-to-Pixel Feature Modulation | [paper-rcl-2509-21797-mowm-mixture-of-world-models-for-embodied-planni](../entities/paper-rcl-2509-21797-mowm-mixture-of-world-models-for-embodied-planni.md) |
| 430 | MobileWAM: Bridging World Action Models to Mobile Manipulation with Chain-of-Foresight | [paper-rcl-2608-04657-mobilewam-bridging-world-action-models-to-mobile](../entities/paper-rcl-2608-04657-mobilewam-bridging-world-action-models-to-mobile.md) |
| 431 | MonoDream: Monocular Vision-Language Navigation with Panoramic Dreaming | [paper-rcl-ref-c268d974279d36619757-monodream-monocular-vision-language-navigation-w](../entities/paper-rcl-ref-c268d974279d36619757-monodream-monocular-vision-language-navigation-w.md) |
| 432 | MotionWAM: Towards Foundation World Action Models for Real-Time Humanoid Loco-Manipulation | [paper-motionwam-humanoid-loco-manipulation-wam](../entities/paper-motionwam-humanoid-loco-manipulation-wam.md) |
| 433 | Motubrain: An Advanced World Action Model for Robot Control | [paper-motubrain](../entities/paper-motubrain.md) |
| 434 | Motus2: A Self-Evolving General World Model for Dexterous Manipulation | [paper-motus2](../entities/paper-motus2.md) |
| 435 | Motus: A Unified Latent Action World Model | [paper-sa-2512-13030-motus-a-unified-latent-action-world-model](../entities/paper-sa-2512-13030-motus-a-unified-latent-action-world-model.md) |
| 436 | NVIDIA OmniDreams: Real-Time Generative World Model for Closed-Loop Autonomous Vehicle Sim | [paper-sa-2606-03159-nvidia-omnidreams-real-time-generative-world-mod](../entities/paper-sa-2606-03159-nvidia-omnidreams-real-time-generative-world-mod.md) |
| 437 | NavWAM: A Navigation World Action Model for Goal-Conditioned Visual Navigation | [paper-navwam-goal-conditioned-visual-navigation-wam](../entities/paper-navwam-goal-conditioned-visual-navigation-wam.md) |
| 438 | Navigation World Models | [paper-rcl-ref-2624494a0f9cbb4a61d2-navigation-world-models](../entities/paper-rcl-ref-2624494a0f9cbb4a61d2-navigation-world-models.md) |
| 439 | Next Forcing: Causal World Modeling with Multi-Chunk Prediction | [paper-sa-2606-11187-next-forcing-causal-world-modeling-with-multi-ch](../entities/paper-sa-2606-11187-next-forcing-causal-world-modeling-with-multi-ch.md) |
| 440 | NoiseGate: Learning Per-Latent Timestep Schedules as Information Gating in World Action Mo | [paper-rcl-2605-07794-noisegate-learning-per-latent-timestep-schedules](../entities/paper-rcl-2605-07794-noisegate-learning-per-latent-timestep-schedules.md) |
| 441 | N₀-TWAM: Scaling Tactile-Native World-Action Model for Contact-Rich Manipulation | [paper-sa-2607-23783-n0-twam-scaling-tactile-native-world-action-mode](../entities/paper-sa-2607-23783-n0-twam-scaling-tactile-native-world-action-mode.md) |
| 442 | OA-WAM: Object-Addressable World Action Model for Robust Robot Manipulation | [paper-rcl-2605-06481-oa-wam-object-addressable-world-action-model-for](../entities/paper-rcl-2605-06481-oa-wam-object-addressable-world-action-model-for.md) |
| 443 | OccLLaMA: An Occupancy-Language-Action Generative World Model for Autonomous Driving | [paper-sa-2409-03272-occllama-an-occupancy-language-action-generative](../entities/paper-sa-2409-03272-occllama-an-occupancy-language-action-generative.md) |
| 444 | OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving | [paper-sa-2311-16038-occworld-learning-a-3d-occupancy-world-model-for](../entities/paper-sa-2311-16038-occworld-learning-a-3d-occupancy-world-model-for.md) |
| 445 | OpenWAM: An Open, Modular Exploration Towards Systematic World-Action Model Pretraining | [paper-openwam](../entities/paper-openwam.md) |
| 446 | PIN-WM: Learning Physics-INformed World Models for Non-Prehensile Manipulation | [paper-sa-2504-16693-pin-wm-learning-physics-informed-world-models-fo](../entities/paper-sa-2504-16693-pin-wm-learning-physics-informed-world-models-fo.md) |
| 447 | ParticleFormer: A 3D Point Cloud World Model for Multi-Object, Multi-Material Robotic Mani | [paper-rcl-ref-a4279cce45a54f73d7ba-particleformer-a-3d-point-cloud-world-model-for](../entities/paper-rcl-ref-a4279cce45a54f73d7ba-particleformer-a-3d-point-cloud-world-model-for.md) |
| 448 | Pathdreamer: A World Model for Indoor Navigation | [paper-rcl-2105-08756-pathdreamer-a-world-model-for-indoor-navigation](../entities/paper-rcl-2105-08756-pathdreamer-a-world-model-for-indoor-navigation.md) |
| 449 | PerceptDrive: Perception Prior World-Action Modeling with Adaptive Expert Routing for End- | [paper-rcl-2607-20175-perceptdrive-perception-prior-world-action-model](../entities/paper-rcl-2607-20175-perceptdrive-perception-prior-world-action-model.md) |
| 450 | Point Tracking Improves World Action Models | [paper-rcl-2605-23856-point-tracking-improves-world-action-models](../entities/paper-rcl-2605-23856-point-tracking-improves-world-action-models.md) |
| 451 | PointWorld: Scaling 3D World Models for In-The-Wild Robotic Manipulation | [paper-sa-2601-03782-pointworld](../entities/paper-sa-2601-03782-pointworld.md) |
| 452 | Pondering the Way: Spatial-perceiving World Action Model for Embodied Navigation | [paper-rcl-2606-29908-pondering-the-way-spatial-perceiving-world-actio](../entities/paper-rcl-2606-29908-pondering-the-way-spatial-perceiving-world-actio.md) |
| 453 | Prediction with Action: Visual Policy Learning via Joint Denoising Process | [paper-rcl-2411-18179-prediction-with-action-visual-policy-learning-vi](../entities/paper-rcl-2411-18179-prediction-with-action-visual-policy-learning-vi.md) |
| 454 | Predictive Inverse Dynamics Models are Scalable Learners for Robotic Manipulation | [paper-rcl-2412-15109-predictive-inverse-dynamics-models-are-scalable](../entities/paper-rcl-2412-15109-predictive-inverse-dynamics-models-are-scalable.md) |
| 455 | Privileged Foresight Distillation: Zero-Cost Future Correction for World Action Models | [paper-rcl-2604-25859-privileged-foresight-distillation-zero-cost-futu](../entities/paper-rcl-2604-25859-privileged-foresight-distillation-zero-cost-futu.md) |
| 456 | ProphetDWM: A Driving World Model for Rolling Out Future Actions and Videos | [paper-rcl-2505-18650-prophetdwm-a-driving-world-model-for-rolling-out](../entities/paper-rcl-2505-18650-prophetdwm-a-driving-world-model-for-rolling-out.md) |
| 457 | QuantWAMs: Calibrating at the Right Granularity for World Action Models | [paper-rcl-2607-28405-quantwams-calibrating-at-the-right-granularity-f](../entities/paper-rcl-2607-28405-quantwams-calibrating-at-the-right-granularity-f.md) |
| 458 | RISE: Adaptive Imagination for World Action Models | [paper-rise-adaptive-imagination-wam](../entities/paper-rise-adaptive-imagination-wam.md) |
| 459 | RISE: Self-Improving Robot Policy with Compositional World Model | [paper-sa-2602-11075-rise-self-improving-robot-policy-with-compositio](../entities/paper-sa-2602-11075-rise-self-improving-robot-policy-with-compositio.md) |
| 460 | RLVR-World: Training World Models with Reinforcement Learning | [paper-rcl-ref-a8d6b1d31a7e72a424da-rlvr-world-training-world-models-with-reinforcem](../entities/paper-rcl-ref-a8d6b1d31a7e72a424da-rlvr-world-training-world-models-with-reinforcem.md) |
| 461 | ReWorld: Representation Learning for World Action Models | [paper-rcl-2606-27504-reworld-representation-learning-for-world-action](../entities/paper-rcl-2606-27504-reworld-representation-learning-for-world-action.md) |
| 462 | RepWAM: World Action Modeling with Representation Visual-Action Tokenizers | [paper-rcl-2606-13674-repwam-world-action-modeling-with-representation](../entities/paper-rcl-2606-13674-repwam-world-action-modeling-with-representation.md) |
| 463 | Rethink Before You Execute: Adaptive Execution for World Action Models | [paper-tempowam](../entities/paper-tempowam.md) |
| 464 | Retrieve, Don't Retrain: Extending Vision Language Action Models to New Tasks at Test Time | [paper-rcl-2606-15631-retrieve-don-t-retrain-extending-vision-language](../entities/paper-rcl-2606-15631-retrieve-don-t-retrain-extending-vision-language.md) |
| 465 | Riemann-1.0: An Embodied World Action Model for Physical AI | [paper-rcl-2608-27033-riemann-1-0-an-embodied-world-action-model-for-p](../entities/paper-rcl-2608-27033-riemann-1-0-an-embodied-world-action-model-for-p.md) |
| 466 | RoboDreamer: Learning Compositional World Models for Robot Imagination | [paper-sa-2404-12377-robodreamer-learning-compositional-world-models](../entities/paper-sa-2404-12377-robodreamer-learning-compositional-world-models.md) |
| 467 | RoboHorizon: An LLM-Assisted Multi-View World Model for Long-Horizon Robotic Manipulation | [paper-sa-2501-06605-robohorizon-an-llm-assisted-multi-view-world-mod](../entities/paper-sa-2501-06605-robohorizon-an-llm-assisted-multi-view-world-mod.md) |
| 468 | Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in Robotics | [paper-sa-2501-10100-robotic-world-model-a-neural-network-simulator-f](../entities/paper-sa-2501-10100-robotic-world-model-a-neural-network-simulator-f.md) |
| 469 | Robust-WAM: Bridging Generative Pretraining and Semantic Foresight in World-Action Models | [paper-rcl-2608-05903-robust-wam-bridging-generative-pretraining-and-s](../entities/paper-rcl-2608-05903-robust-wam-bridging-generative-pretraining-and-s.md) |
| 470 | RynnVLA-002: A Unified Vision-Language-Action and World Model | [paper-rcl-2511-17502-rynnvla-002-a-unified-vision-language-action-and](../entities/paper-rcl-2511-17502-rynnvla-002-a-unified-vision-language-action-and.md) |
| 471 | S-VAM: Shortcut Video-Action Model by Self-Distilling Geometric and Semantic Foresight | [paper-rcl-2603-16195-s-vam-shortcut-video-action-model-by-self-distil](../entities/paper-rcl-2603-16195-s-vam-shortcut-video-action-model-by-self-distil.md) |
| 472 | SAMoE-VLA: A Scene Adaptive Mixture-of-Experts Vision-Language-Action Model for Autonomous | [paper-rcl-2603-08113-samoe-vla-a-scene-adaptive-mixture-of-experts-vi](../entities/paper-rcl-2603-08113-samoe-vla-a-scene-adaptive-mixture-of-experts-vi.md) |
| 473 | SANTS: A State-Adaptive Scheduler for World Action Models | [paper-rcl-2605-27947-sants-a-state-adaptive-scheduler-for-world-actio](../entities/paper-rcl-2605-27947-sants-a-state-adaptive-scheduler-for-world-actio.md) |
| 474 | SG-WAM: Self-Guided World Modeling in Geometry-Aware Policy Space | [paper-rcl-2608-01397-sg-wam-self-guided-world-modeling-in-geometry-aw](../entities/paper-rcl-2608-01397-sg-wam-self-guided-world-modeling-in-geometry-aw.md) |
| 475 | SG-WAM: Text-Grounded and Spatial-aware Semantic Guidance for World-Action Models | [paper-sg-wam-semantic-guidance](../entities/paper-sg-wam-semantic-guidance.md) |
| 476 | ST-WAM: Semantic-Temporal World Action Model for Robust Manipulation under Visual Distribu | [paper-rcl-2607-28993-st-wam-semantic-temporal-world-action-model-for](../entities/paper-rcl-2607-28993-st-wam-semantic-temporal-world-action-model-for.md) |
| 477 | SV-WAM: An Efficient Surround-View World-Action Model for End-to-End Autonomous Driving | [paper-rcl-2609-03602-sv-wam-an-efficient-surround-view-world-action-m](../entities/paper-rcl-2609-03602-sv-wam-an-efficient-surround-view-world-action-m.md) |
| 478 | Selective Cross-View Consistency for World Action Models: Held-Out Viewpoint Robustness Wi | [paper-rcl-2608-21402-selective-cross-view-consistency-for-world-actio](../entities/paper-rcl-2608-21402-selective-cross-view-consistency-for-world-actio.md) |
| 479 | Self-Correcting VLA: Online Action Refinement via Sparse World Imagination | [paper-rcl-2602-21633-self-correcting-vla-online-action-refinement-via](../entities/paper-rcl-2602-21633-self-correcting-vla-online-action-refinement-via.md) |
| 480 | SelfWAM: A Self-Grounded Unified World Action Model for Fast Robot Control | [paper-rcl-2608-00725-selfwam-a-self-grounded-unified-world-action-mod](../entities/paper-rcl-2608-00725-selfwam-a-self-grounded-unified-world-action-mod.md) |
| 481 | SimWAM: A Simple World Action Model for End-to-End Autonomous Driving | [paper-rcl-2608-07468-simwam-a-simple-world-action-model-for-end-to-en](../entities/paper-rcl-2608-07468-simwam-a-simple-world-action-model-for-end-to-en.md) |
| 482 | Spatial Policy: Guiding Visuomotor Robotic Manipulation with Spatial-Aware Modeling and Re | [paper-rcl-2508-15874-spatial-policy-guiding-visuomotor-robotic-manipu](../entities/paper-rcl-2508-15874-spatial-policy-guiding-visuomotor-robotic-manipu.md) |
| 483 | SpatialVAM:Spatial-Aware Multi-View Video Diffusion as a Data-Efficient Robot Policy | [paper-rcl-2604-03181-spatialvam-spatial-aware-multi-view-video-diffus](../entities/paper-rcl-2604-03181-spatialvam-spatial-aware-multi-view-video-diffus.md) |
| 484 | Spatially Aware World Action Model via Geometric Latent Diffusion | [paper-sa-wam](../entities/paper-sa-wam.md) |
| 485 | StageWAM: Joint-Embedding Stage Prediction for World-Action Models in Robot Manipulation | [paper-rcl-2608-10780-stagewam-joint-embedding-stage-prediction-for-wo](../entities/paper-rcl-2608-10780-stagewam-joint-embedding-stage-prediction-for-wo.md) |
| 486 | Steering Robustness into World Action Models via Mechanistic Interpretability and Optimal  | [paper-rcl-2607-14943-steering-robustness-into-world-action-models-via](../entities/paper-rcl-2607-14943-steering-robustness-into-world-action-models-via.md) |
| 487 | Surgical WAM: A World-Action Model for Data-Efficient Surgical Robot Learning | [paper-rcl-2608-11204-surgical-wam-a-world-action-model-for-data-effic](../entities/paper-rcl-2608-11204-surgical-wam-a-world-action-model-for-data-effic.md) |
| 488 | SyncWorld: Visual Calibration Enables World Models as Zero-Shot Simulators | [paper-rcl-2609-09155-syncworld-visual-calibration-enables-world-model](../entities/paper-rcl-2609-09155-syncworld-visual-calibration-enables-world-model.md) |
| 489 | TacPAC: Tactile Prediction and Real-Time Action Correction in World-Action Models for Cont | [paper-tacpac](../entities/paper-tacpac.md) |
| 490 | TacWAM: Anchor-Guided World Action Model with Mechanics-Aware Tactile Prediction | [paper-rcl-2607-28391-tacwam-anchor-guided-world-action-model-with-mec](../entities/paper-rcl-2607-28391-tacwam-anchor-guided-world-action-model-with-mec.md) |
| 491 | Tactile-WAM: Touch-Aware World Action Model with Tactile Asymmetric Attention | [paper-sa-2606-26663-tactile-wam-touch-aware-world-action-model-with](../entities/paper-sa-2606-26663-tactile-wam-touch-aware-world-action-model-with.md) |
| 492 | Test-Time Scaling for World Action Models via Zero-Shot Geometric Evaluation | [paper-rcl-2607-17454-test-time-scaling-for-world-action-models-via-ze](../entities/paper-rcl-2607-17454-test-time-scaling-for-world-action-models-via-ze.md) |
| 493 | The DAWN of World-Action Interactive Models | [paper-rcl-2605-11550-the-dawn-of-world-action-interactive-models](../entities/paper-rcl-2605-11550-the-dawn-of-world-action-interactive-models.md) |
| 494 | This&That: Language-Gesture Controlled Video Generation for Robot Planning | [paper-rcl-2407-05530-this-that-language-gesture-controlled-video-gene](../entities/paper-rcl-2407-05530-this-that-language-gesture-controlled-video-gene.md) |
| 495 | Toward Physically Grounded JEPA World Models for Goal-Conditioned Robotic Planning | [paper-rcl-2609-03565-toward-physically-grounded-jepa-world-models-for](../entities/paper-rcl-2609-03565-toward-physically-grounded-jepa-world-models-for.md) |
| 496 | Towards Practical World Model-based Reinforcement Learning for Vision-Language-Action Mode | [paper-rcl-2603-20607-towards-practical-world-model-based-reinforcemen](../entities/paper-rcl-2603-20607-towards-practical-world-model-based-reinforcemen.md) |
| 497 | Towards Predictive, Aligned, and Scalable Robot Learning | [lumo-2](../entities/lumo-2.md) |
| 498 | Towards Surgical World-Action Modeling: A Preliminary Joint Visual-Trajectory Forecasting  | [paper-rcl-2608-20284-towards-surgical-world-action-modeling-a-prelimi](../entities/paper-rcl-2608-20284-towards-surgical-world-action-modeling-a-prelimi.md) |
| 499 | Towards a Generalizable Bimanual Foundation Policy via Flow-based Video Prediction | [paper-rcl-2505-24156-towards-a-generalizable-bimanual-foundation-poli](../entities/paper-rcl-2505-24156-towards-a-generalizable-bimanual-foundation-poli.md) |
| 500 | UNIVERSE: Unified Video Action Models for Autonomous Driving with Flexible Mask-Modulated  | [paper-rcl-2607-05133-universe-unified-video-action-models-for-autonom](../entities/paper-rcl-2607-05133-universe-unified-video-action-models-for-autonom.md) |
| 501 | Uncertainty-Aware Robotic World Model Makes Offline Model-Based Reinforcement Learning Wor | [paper-rcl-ref-024007cef3657e49b45b-uncertainty-aware-robotic-world-model-makes-offl](../entities/paper-rcl-ref-024007cef3657e49b45b-uncertainty-aware-robotic-world-model-makes-offl.md) |
| 502 | Understanding and Mitigating the Video-Action Generalization Gap via Temporal Ratio | [paper-rcl-2607-08127-understanding-and-mitigating-the-video-action-ge](../entities/paper-rcl-2607-08127-understanding-and-mitigating-the-video-action-ge.md) |
| 503 | Uni-World VLA: Interleaved World Modeling and Planning for Autonomous Driving | [paper-rcl-2603-27287-uni-world-vla-interleaved-world-modeling-and-pla](../entities/paper-rcl-2603-27287-uni-world-vla-interleaved-world-modeling-and-pla.md) |
| 504 | UniDrive-WM: Unified Understanding, Planning and Generation World Model for Autonomous Dri | [paper-rcl-2601-04453-unidrive-wm-unified-understanding-planning-and-g](../entities/paper-rcl-2601-04453-unidrive-wm-unified-understanding-planning-and-g.md) |
| 505 | UniNav: A Unified World-Action Diffusion Model for Visual Navigation | [paper-rcl-2608-03244-uninav-a-unified-world-action-diffusion-model-fo](../entities/paper-rcl-2608-03244-uninav-a-unified-world-action-diffusion-model-fo.md) |
| 506 | Unified 4D World Action Modeling from Video Priors with Asynchronous Denoising | [paper-rcl-2604-26694-unified-4d-world-action-modeling-from-video-prio](../entities/paper-rcl-2604-26694-unified-4d-world-action-modeling-from-video-prio.md) |
| 507 | Unified Video Action Model | [paper-shenlan-wm-10-uva](../entities/paper-shenlan-wm-10-uva.md) |
| 508 | Unified Video-Action Joint Denoising for Dexterous Action and Data Generation | [paper-rcl-2606-03868-unified-video-action-joint-denoising-for-dextero](../entities/paper-rcl-2606-03868-unified-video-action-joint-denoising-for-dextero.md) |
| 509 | Unified World Models: Coupling Video and Action Diffusion for Pretraining on Large Robotic | [paper-shenlan-wm-08-uwm](../entities/paper-shenlan-wm-08-uwm.md) |
| 510 | Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation | [paper-rcl-ref-e1a2abbaffcea1e2e971-unleashing-large-scale-video-generative-pre-trai](../entities/paper-rcl-ref-e1a2abbaffcea1e2e971-unleashing-large-scale-video-generative-pre-trai.md) |
| 511 | V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning | [paper-vjepa2](../entities/paper-vjepa2.md) |
| 512 | VAG: Dual-Stream Video-Action Generation for Embodied Data Synthesis | [paper-rcl-2604-09330-vag-dual-stream-video-action-generation-for-embo](../entities/paper-rcl-2604-09330-vag-dual-stream-video-action-generation-for-embo.md) |
| 513 | VAMPO: Policy Optimization for Improving Visual Dynamics in Video Action Models | [paper-rcl-2603-19370-vampo-policy-optimization-for-improving-visual-d](../entities/paper-rcl-2603-19370-vampo-policy-optimization-for-improving-visual-d.md) |
| 514 | VILP: Imitation Learning with Latent Video Planning | [paper-rcl-2502-01784-vilp-imitation-learning-with-latent-video-planni](../entities/paper-rcl-2502-01784-vilp-imitation-learning-with-latent-video-planni.md) |
| 515 | VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model | [paper-sa-2602-10098-vla-jepa-enhancing-vision-language-action-model](../entities/paper-sa-2602-10098-vla-jepa-enhancing-vision-language-action-model.md) |
| 516 | VLAW: Iterative Co-Improvement of Vision-Language-Action Policy and World Model | [paper-rcl-2602-12063-vlaw-iterative-co-improvement-of-vision-language](../entities/paper-rcl-2602-12063-vlaw-iterative-co-improvement-of-vision-language.md) |
| 517 | VT-WAM: Visual-Tactile World Action Model for Contact-Rich Manipulation | [paper-vt-wam-visuotactile-contact-rich](../entities/paper-vt-wam-visuotactile-contact-rich.md) |
| 518 | VTAM: Video-Tactile-Action Models for Complex Physical Interaction Beyond VLAs | [paper-sa-2603-23481-vtam-video-tactile-action-models-for-complex-phy](../entities/paper-sa-2603-23481-vtam-video-tactile-action-models-for-complex-phy.md) |
| 519 | VaViM and VaVAM: Autonomous Driving through Video Generative Modeling | [paper-rcl-2502-15672-vavim-and-vavam-autonomous-driving-through-video](../entities/paper-rcl-2502-15672-vavim-and-vavam-autonomous-driving-through-video.md) |
| 520 | Vega: Learning to Drive with Natural Language Instructions | [paper-rcl-2603-25741-vega-learning-to-drive-with-natural-language-ins](../entities/paper-rcl-2603-25741-vega-learning-to-drive-with-natural-language-ins.md) |
| 521 | ViPRA: Video Prediction for Robot Actions | [paper-sa-2511-07732-vipra-video-prediction-for-robot-actions](../entities/paper-sa-2511-07732-vipra-video-prediction-for-robot-actions.md) |
| 522 | Vid2WAM: Distilling Video Diffusion Priors into World Action Models | [paper-rcl-2608-08558-vid2wam-distilling-video-diffusion-priors-into-w](../entities/paper-rcl-2608-08558-vid2wam-distilling-video-diffusion-priors-into-w.md) |
| 523 | VidMan: Exploiting Implicit Dynamics from Video Diffusion Model for Effective Robot Manipu | [paper-rcl-2411-09153-vidman-exploiting-implicit-dynamics-from-video-d](../entities/paper-rcl-2411-09153-vidman-exploiting-implicit-dynamics-from-video-d.md) |
| 524 | Vidar: Embodied Video Diffusion Model for Generalist Manipulation | [paper-sa-2507-12898-vidar-embodied-video-diffusion-model-for-general](../entities/paper-sa-2507-12898-vidar-embodied-video-diffusion-model-for-general.md) |
| 525 | Video Generators are Robot Policies | [paper-shenlan-wm-06-video-gen-robot-policies](../entities/paper-shenlan-wm-06-video-gen-robot-policies.md) |
| 526 | Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations | [paper-rcl-ref-2b3f47a14556997eb476-video-prediction-policy-a-generalist-robot-polic](../entities/paper-rcl-ref-2b3f47a14556997eb476-video-prediction-policy-a-generalist-robot-polic.md) |
| 527 | VideoVLA: Video Generators Can Be Generalizable Robot Manipulators | [paper-sa-2512-06963-videovla-video-generators-can-be-generalizable-r](../entities/paper-sa-2512-06963-videovla-video-generators-can-be-generalizable-r.md) |
| 528 | VideoWorld: Exploring Knowledge Learning from Unlabeled Videos | [paper-rcl-2501-09781-videoworld-exploring-knowledge-learning-from-unl](../entities/paper-rcl-2501-09781-videoworld-exploring-knowledge-learning-from-unl.md) |
| 529 | Visuo-Tactile World Models | [paper-sa-2602-06001-visuo-tactile-world-models-vt-wm](../entities/paper-sa-2602-06001-visuo-tactile-world-models-vt-wm.md) |
| 530 | WA-JEPA: Rethinking the Video JEPA Paradigm for World-Action Modeling in Autonomous Drivin | [paper-rcl-2608-20974-wa-jepa-rethinking-the-video-jepa-paradigm-for-w](../entities/paper-rcl-2608-20974-wa-jepa-rethinking-the-video-jepa-paradigm-for-w.md) |
| 531 | WALL-WM: Carving World Action Modeling at the Event Joints | [paper-rcl-2606-01955-wall-wm-carving-world-action-modeling-at-the-eve](../entities/paper-rcl-2606-01955-wall-wm-carving-world-action-modeling-at-the-eve.md) |
| 532 | WAM-Nav: Asymmetric Latent World-Action Modeling for Unified Visual Navigation | [paper-rcl-2606-04907-wam-nav-asymmetric-latent-world-action-modeling](../entities/paper-rcl-2606-04907-wam-nav-asymmetric-latent-world-action-modeling.md) |
| 533 | WAM-OPD: On-Policy Distillation for World Action Models | [paper-rcl-2608-22364-wam-opd-on-policy-distillation-for-world-action](../entities/paper-rcl-2608-22364-wam-opd-on-policy-distillation-for-world-action.md) |
| 534 | WAM-RL: World-Action Model Reinforcement Learning with Reconstruction Rewards and Online V | [paper-sa-2606-17906-wam-rl-world-action-model-reinforcement-learning](../entities/paper-sa-2606-17906-wam-rl-world-action-model-reinforcement-learning.md) |
| 535 | WAM-TTT: Steering World-Action Models by Watching Human Play at Test Time | [paper-wam-ttt-human-video-test-time-steering](../entities/paper-wam-ttt-human-video-test-time-steering.md) |
| 536 | WAM4D: Fast 4D World Action Model via Spatial Register Tokens | [paper-rcl-2606-14048-wam4d-fast-4d-world-action-model-via-spatial-reg](../entities/paper-rcl-2606-14048-wam4d-fast-4d-world-action-model-via-spatial-reg.md) |
| 537 | WISE: World-model-guided Imagination Scheduling for Efficient Post-training of Vision-Lang | [paper-rcl-2609-03681-wise-world-model-guided-imagination-scheduling-f](../entities/paper-rcl-2609-03681-wise-world-model-guided-imagination-scheduling-f.md) |
| 538 | WM-Craftnet: World Synesthesia Model for Generalizable and Robust Dexterous In-Hand Manipu | [paper-wm-craftnet](../entities/paper-wm-craftnet.md) |
| 539 | WMPO: World Model-based Policy Optimization for Vision-Language-Action Models | [paper-sa-2511-09515-wmpo-world-model-based-policy-optimization-for-v](../entities/paper-sa-2511-09515-wmpo-world-model-based-policy-optimization-for-v.md) |
| 540 | WNM-3D: A World Navigation Model with 3D Scene Conditioning for Closed-Loop VLN | [paper-wnm-3d-vln](../entities/paper-wnm-3d-vln.md) |
| 541 | When to Trust Imagination: Adaptive Action Execution for World Action Models | [paper-rcl-2605-06222-when-to-trust-imagination-adaptive-action-execut](../entities/paper-rcl-2605-06222-when-to-trust-imagination-adaptive-action-execut.md) |
| 542 | WoVR: World Models as Reliable Simulators for Post-Training VLA Policies with RL | [paper-rcl-2602-13977-wovr-world-models-as-reliable-simulators-for-pos](../entities/paper-rcl-2602-13977-wovr-world-models-as-reliable-simulators-for-pos.md) |
| 543 | World Action Models Enable Continual Imitation Learning with Recurrent Generative Replays | [paper-rcl-2606-27374-world-action-models-enable-continual-imitation-l](../entities/paper-rcl-2606-27374-world-action-models-enable-continual-imitation-l.md) |
| 544 | World Action Models are Zero-shot Policies | [paper-notebook-dreamzero-world-action-models-are-zero-shot-poli](../entities/paper-notebook-dreamzero-world-action-models-are-zero-shot-poli.md) |
| 545 | World Model-based Perception for Visual Legged Locomotion | [paper-sa-2409-16784-wmp-world-model-based-perception-for-visual-legg](../entities/paper-sa-2409-16784-wmp-world-model-based-perception-for-visual-legg.md) |
| 546 | World Models for Learning Dexterous Hand-Object Interactions from Human Videos | [paper-sa-2512-13644-dex-wm](../entities/paper-sa-2512-13644-dex-wm.md) |
| 547 | World Pilot: Steering Vision-Language-Action Models with World-Action Priors | [paper-rcl-2606-12403-world-pilot-steering-vision-language-action-mode](../entities/paper-rcl-2606-12403-world-pilot-steering-vision-language-action-mode.md) |
| 548 | World Tokens: Enhancing Embodied Policies with Training-Time World Modeling | [paper-world-tokens-inference-trimmed-wam](../entities/paper-world-tokens-inference-trimmed-wam.md) |
| 549 | World-Coherent Decoding: Self-Verifying Test-Time Planning for World Action Models | [paper-rcl-2609-02159-world-coherent-decoding-self-verifying-test-time](../entities/paper-rcl-2609-02159-world-coherent-decoding-self-verifying-test-time.md) |
| 550 | World-Language-Action Model for Unified World Modeling, Language Reasoning, and Action Syn | [paper-sa-2606-05979-world-language-action-model-for-unified-world-mo](../entities/paper-sa-2606-05979-world-language-action-model-for-unified-world-mo.md) |
| 551 | World-VLA-Loop: Closed-Loop Learning of Video World Model and VLA Policy | [paper-rcl-2602-06508-world-vla-loop-closed-loop-learning-of-video-wor](../entities/paper-rcl-2602-06508-world-vla-loop-closed-loop-learning-of-video-wor.md) |
| 552 | World2Act: Latent Action Post-Training from World Model Dynamics | [paper-rcl-2603-10422-world2act-latent-action-post-training-from-world](../entities/paper-rcl-2603-10422-world2act-latent-action-post-training-from-world.md) |
| 553 | World4Drive: End-to-End Autonomous Driving via Intention-aware Physical Latent World Model | [paper-sa-2507-00603-world4drive-end-to-end-autonomous-driving-via-in](../entities/paper-sa-2507-00603-world4drive-end-to-end-autonomous-driving-via-in.md) |
| 554 | World4RL: Diffusion World Models for Policy Refinement with Reinforcement Learning for Rob | [paper-sa-2509-19080-world4rl-diffusion-world-models-for-policy-refin](../entities/paper-sa-2509-19080-world4rl-diffusion-world-models-for-policy-refin.md) |
| 555 | WorldAgen: Unified State-Action Prediction with Test-Time World Model Training | [paper-rcl-2609-08162-worldagen-unified-state-action-prediction-with-t](../entities/paper-rcl-2609-08162-worldagen-unified-state-action-prediction-with-t.md) |
| 556 | WorldScape Policy 2.0: Empowering Steerable World Action Modeling with Reasoning-Augmented | [paper-worldscape-policy-2](../entities/paper-worldscape-policy-2.md) |
| 557 | WorldVLA: Towards Autoregressive Action World Model | [paper-shenlan-wm-07-worldvla](../entities/paper-shenlan-wm-07-worldvla.md) |
| 558 | WorldVLN: Autoregressive World Action Model for Aerial Vision-Language Navigation | [paper-worldvln-aerial-vln-wam](../entities/paper-worldvln-aerial-vln-wam.md) |
| 559 | X-MOBILITY: End-To-End Generalizable Navigation via World Modeling | [paper-sa-2410-17491-x-mobility-end-to-end-generalizable-navigation-v](../entities/paper-sa-2410-17491-x-mobility-end-to-end-generalizable-navigation-v.md) |
| 560 | Zero-WAM: In-Context World-Action Modeling from Human Videos for Open-Ended Task Generaliz | [paper-zero-wam](../entities/paper-zero-wam.md) |
| 561 | ZimaBlue: Evolving Generalizable World Action Models through Scalable Video Pre-training | [paper-rcl-2609-00188-zimablue-evolving-generalizable-world-action-mod](../entities/paper-rcl-2609-00188-zimablue-evolving-generalizable-world-action-mod.md) |
| 562 | mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs | [paper-sa-2512-15692-mimic-video-video-action-models-for-generalizabl](../entities/paper-sa-2512-15692-mimic-video-video-action-models-for-generalizabl.md) |
| 563 | π_0.7: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities | [paper-rcl-2604-15483-0-7-a-steerable-generalist-robotic-foundation-mo](../entities/paper-rcl-2604-15483-0-7-a-steerable-generalist-robotic-foundation-mo.md) |
| 564 | ω-0: A Latent Predictive World Action Model for Concurrent Humanoid Loco-Manipulation | [paper-omega-0](../entities/paper-omega-0.md) |


## 局限与风险

- 清单摘要页只给 Contribution 要点，**不替代** 原文；要深读请从论文链接进。
- 清单里混有非 arXiv 链接（OpenReview / IEEE / DOI），这类条目按标题收录；若同一工作另有 arXiv 深度页，catalog 会链到 canonical 节点。
- 上游清单仍在更新，本页是 2026-09-25 的快照；最新条目以上游 `docs/PAPERS.md` 为准。

## 关联页面

- [Awesome World-Action Models（RCL）](../entities/awesome-world-action-models-rcl.md)
- [World Action Models（WAM）](../concepts/world-action-models.md)
- [VLA](../methods/vla.md)

## 参考来源

- [rcl_awesome_wam_catalog.md](../../sources/papers/rcl_awesome_wam_catalog.md)
- [sources/repos/awesome-world-action-models-rcl.md](../../sources/repos/awesome-world-action-models-rcl.md)
- 上游：[docs/PAPERS.md](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md)

## 推荐继续阅读

- [Awesome World-Action Models GitHub](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [Paper library（站点）](https://rcl-robotics.github.io/Awesome-World-Action-Models/papers/)
