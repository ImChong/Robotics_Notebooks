---
type: overview
tags: [overview, curated-index, physical-ai, awesome-physical-ai, technology-map]
status: complete
updated: 2026-09-20
summary: "Physical AI 双清单技术地图：natnew ∪ aichr 去重后 384 条，新建 241、复用 143。"
related:
  - ../entities/awesome-physical-ai-natnew.md
  - ../entities/awesome-physical-ai-aichr.md
  - ../comparisons/awesome-physical-ai-curated-lists.md
  - ../methods/vla.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/repos/awesome-physical-ai-union-catalog.md
  - ../../sources/repos/awesome-physical-ai-natnew.md
  - ../../sources/repos/awesome-physical-ai-aichr.md
---

# Awesome Physical AI 技术地图

> 本页把 [natnew/awesome-physical-ai](https://github.com/natnew/awesome-physical-ai) 与 [aichr/awesome-physical-ai](https://github.com/aichr/awesome-physical-ai) 的清单条目映射为站内 **独立详情节点**（新建 `pai-*` / `paper-pai-*` 或复用已有 canonical 页）。

## 一句话定义

**Physical AI 双清单技术地图** = 两份同名 Awesome 列表的并集节点化索引（按清单分组浏览，一点即达详情页）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| PAI | Physical AI | 感知–推理–行动闭环的物理智能 |
| VLA | Vision-Language-Action | 两清单共同主线 |
| RFM | Robotics Foundation Model | natnew canonical 类 |
| Sim2Real | Simulation to Real | 迁移与评测独立类 |

## 为什么重要

- Awesome 列表本身不是知识图谱节点；若不升格子条目，图谱只能停在清单 hub。
- 本地图 **优先复用** 库内已有 arXiv / GitHub / 标题 canonical 页，仅对缺失条目新建索引级节点。
- 统计：去重后 **384** 条（新建 **241**，复用 **143**）。

## 覆盖范围

| 项 | 值 |
|----|-----|
| 上游 | <https://github.com/natnew/awesome-physical-ai> · <https://github.com/aichr/awesome-physical-ai> |
| 目录 source | [awesome-physical-ai-union-catalog.md](../../sources/repos/awesome-physical-ai-union-catalog.md) |
| 对照页 | [natnew vs aichr](../comparisons/awesome-physical-ai-curated-lists.md) |

## 分组索引

### 3D Computer Vision

| # | 条目 | 详情节点 | 来源 |
|---|------|----------|------|
| 001 | 2025 GS Paper List | [painode-001-2025gspaperlist](../entities/painode-001-2025gspaperlist.md) | aichr |
| 002 | Awesome 3D Gaussian Splatting | [painode-002-awesome3dgaussiansplatting](../entities/painode-002-awesome3dgaussiansplatting.md) | aichr |
| 003 | Depth Anything | [painode-003-depthanything](../entities/painode-003-depthanything.md) | aichr |
| 004 | Grounded SAM 2 | [painode-004-groundedsam2](../entities/painode-004-groundedsam2.md) | aichr |
| 005 | MiDaS | [painode-005-midas](../entities/painode-005-midas.md) | aichr |
| 006 | NeRF + GS for Robotics | [painode-006-nerfgsforrobotics](../entities/painode-006-nerfgsforrobotics.md) | aichr |
| 007 | SAM 3 | [paper-sam3](../entities/paper-sam3.md) | aichr |
| 008 | SAM 3D | [paper-pai-2306-03908-sam3d](../entities/paper-pai-2306-03908-sam3d.md) | aichr |

### Benchmarks

| # | 条目 | 详情节点 | 来源 |
|---|------|----------|------|
| 009 | ALFRED | [painode-009-alfred](../entities/painode-009-alfred.md) | natnew |
| 010 | ARNOLD | [painode-010-arnold](../entities/painode-010-arnold.md) | natnew |
| 011 | CARLA Leaderboard | [painode-011-carlaleaderboard](../entities/painode-011-carlaleaderboard.md) | natnew |
| 012 | Colosseum | [painode-012-colosseum](../entities/painode-012-colosseum.md) | natnew |
| 013 | FurnitureBench | [painode-013-furniturebench](../entities/painode-013-furniturebench.md) | natnew |
| 014 | HumanoidBench | [humanoid-bench](../entities/humanoid-bench.md) | natnew |
| 015 | LIBERO | [libero-benchmark](../entities/libero-benchmark.md) | natnew+aichr |
| 016 | ManiSkill Benchmark | [maniskill2](../entities/maniskill2.md) | natnew+aichr |
| 017 | MetaWorld | [paper-hrl-stack-32-metaworld](../entities/paper-hrl-stack-32-metaworld.md) | natnew |
| 018 | MineDojo | [painode-018-minedojo](../entities/painode-018-minedojo.md) | natnew |
| 019 | OpenEQA | [painode-019-openeqa](../entities/painode-019-openeqa.md) | natnew |
| 020 | RLBench | [rlbench](../entities/rlbench.md) | natnew |
| 021 | RoboTHOR | [painode-021-robothor](../entities/painode-021-robothor.md) | natnew |
| 022 | RoboTwin | [robotwin](../entities/robotwin.md) | aichr |
| 023 | TEACh | [painode-023-teach](../entities/painode-023-teach.md) | natnew |
| 024 | VLABench | [painode-024-vlabench](../entities/painode-024-vlabench.md) | aichr |

### Books

| # | 条目 | 详情节点 | 来源 |
|---|------|----------|------|
| 025 | A Mathematical Introduction to Robotic Manipulation | [painode-025-amathematicalintroductiontorobot](../entities/painode-025-amathematicalintroductiontorobot.md) | natnew |
| 026 | Introduction to Autonomous Robots | [painode-026-introductiontoautonomousrobots](../entities/painode-026-introductiontoautonomousrobots.md) | natnew |
| 027 | Modern Robotics | [modern-robotics-book](../entities/modern-robotics-book.md) | natnew |
| 028 | Planning Algorithms | [painode-028-planningalgorithms](../entities/painode-028-planningalgorithms.md) | natnew |
| 029 | Probabilistic Robotics | [painode-029-probabilisticrobotics](../entities/painode-029-probabilisticrobotics.md) | natnew |
| 030 | Reinforcement Learning: An Introduction | [painode-030-reinforcementlearninganintroducti](../entities/painode-030-reinforcementlearninganintroducti.md) | natnew |
| 031 | Robotics, Vision and Control | [painode-031-roboticsvisionandcontrol](../entities/painode-031-roboticsvisionandcontrol.md) | natnew |

### Community

| # | 条目 | 详情节点 | 来源 |
|---|------|----------|------|
| 032 | Hugging Face Discord | [painode-032-huggingfacediscord](../entities/painode-032-huggingfacediscord.md) | natnew |
| 033 | Pollen Robotics Discord | [painode-033-pollenroboticsdiscord](../entities/painode-033-pollenroboticsdiscord.md) | natnew |
| 034 | r/MachineLearning | [painode-034-rmachinelearning](../entities/painode-034-rmachinelearning.md) | aichr |
| 035 | r/robotics | [painode-035-rrobotics](../entities/painode-035-rrobotics.md) | natnew+aichr |
| 036 | r/ROS | [painode-036-rros](../entities/painode-036-rros.md) | aichr |
| 037 | Robot Learning Discord | [painode-037-robotlearningdiscord](../entities/painode-037-robotlearningdiscord.md) | natnew |
| 038 | Robotics Stack Exchange | [painode-038-roboticsstackexchange](../entities/painode-038-roboticsstackexchange.md) | natnew |
| 039 | ROS Discourse | [painode-039-xrosdiscourse](../entities/painode-039-xrosdiscourse.md) | natnew+aichr |

### Companies

| # | 条目 | 详情节点 | 来源 |
|---|------|----------|------|
| 040 | 1X Technologies | [1x-technologies](../entities/1x-technologies.md) | natnew |
| 041 | Agility Robotics | [painode-041-agilityrobotics](../entities/painode-041-agilityrobotics.md) | natnew |
| 042 | Apptronik | [painode-042-apptronik](../entities/painode-042-apptronik.md) | natnew |
| 043 | Boston Dynamics | [boston-dynamics](../entities/boston-dynamics.md) | natnew |
| 044 | Covariant | [painode-044-covariant](../entities/painode-044-covariant.md) | natnew |
| 045 | Dexterity | [painode-045-dexterity](../entities/painode-045-dexterity.md) | natnew |
| 046 | Intrinsic | [painode-046-intrinsic](../entities/painode-046-intrinsic.md) | natnew |
| 047 | Pollen Robotics (Hugging Face) | [pollen-reachy2](../entities/pollen-reachy2.md) | natnew |
| 048 | Sanctuary AI | [painode-048-sanctuaryai](../entities/painode-048-sanctuaryai.md) | natnew |
| 049 | Skild AI | [skild-ai](../entities/skild-ai.md) | natnew |
| 050 | Sunday Robotics | [painode-050-sundayrobotics](../entities/painode-050-sundayrobotics.md) | natnew |
| 051 | Tesla Optimus | [tesla-optimus](../entities/tesla-optimus.md) | natnew |
| 052 | Unitree Robotics | [unitree](../entities/unitree.md) | natnew+aichr |
| 053 | Wayve | [painode-053-wayve](../entities/painode-053-wayve.md) | natnew |

### Conferences

| # | 条目 | 详情节点 | 来源 |
|---|------|----------|------|
| 054 | CoRL | [painode-054-corl](../entities/painode-054-corl.md) | natnew+aichr |
| 055 | HRI | [painode-055-hri](../entities/painode-055-hri.md) | natnew |
| 056 | Humanoids | [painode-056-humanoids](../entities/painode-056-humanoids.md) | natnew |
| 057 | ICLR | [painode-057-iclr](../entities/painode-057-iclr.md) | natnew |
| 058 | ICML | [painode-058-icml](../entities/painode-058-icml.md) | natnew |
| 059 | ICRA | [painode-059-icra](../entities/painode-059-icra.md) | natnew+aichr |
| 060 | IROS | [painode-060-iros](../entities/painode-060-iros.md) | natnew+aichr |
| 061 | NeurIPS | [painode-061-neurips](../entities/painode-061-neurips.md) | natnew |
| 062 | RSS | [painode-062-rss](../entities/painode-062-rss.md) | natnew+aichr |

### Courses

| # | 条目 | 详情节点 | 来源 |
|---|------|----------|------|
| 063 | 16-745 — Optimal Control and Reinforcement Learning (CMU) | [painode-063-16745optimalcontrolandreinforce](../entities/painode-063-16745optimalcontrolandreinforce.md) | natnew |
| 064 | 16-831 — Introduction to Robot Learning (CMU) | [painode-064-16831introductiontorobotlearnin](../entities/painode-064-16831introductiontorobotlearnin.md) | natnew |
| 065 | CS 224R — Deep RL for Robotics (Stanford) | [painode-065-cs224rdeeprlforroboticsstanfor](../entities/painode-065-cs224rdeeprlforroboticsstanfor.md) | natnew+aichr |
| 066 | CS 234 — Reinforcement Learning (Stanford) | [painode-066-cs234reinforcementlearningstanfo](../entities/painode-066-cs234reinforcementlearningstanfo.md) | natnew |
| 067 | CS 285 — Deep Reinforcement Learning (Berkeley) | [painode-067-cs285deepreinforcementlearningb](../entities/painode-067-cs285deepreinforcementlearningb.md) | natnew |
| 068 | CS 287 — Advanced Robotics (Berkeley) | [painode-068-cs287advancedroboticsberkeley](../entities/painode-068-cs287advancedroboticsberkeley.md) | natnew |
| 069 | CS 336 — Robot Learning (Stanford) | [painode-069-cs336robotlearningstanford](../entities/painode-069-cs336robotlearningstanford.md) | natnew |
| 070 | Deep RL Bootcamp | [painode-070-deeprlbootcamp](../entities/painode-070-deeprlbootcamp.md) | natnew |
| 071 | DeepMind x UCL RL Lecture Series | [painode-071-deepmindxuclrllectureseries](../entities/painode-071-deepmindxuclrllectureseries.md) | natnew |
| 072 | Fast.ai Practical Deep Learning | [painode-072-fastaipracticaldeeplearning](../entities/painode-072-fastaipracticaldeeplearning.md) | natnew |
| 073 | Hugging Face Deep RL Course | [painode-073-huggingfacedeeprlcourse](../entities/painode-073-huggingfacedeeprlcourse.md) | natnew |
| 074 | MIT Underactuated Robotics | [painode-074-mitunderactuatedrobotics](../entities/painode-074-mitunderactuatedrobotics.md) | natnew |
| 075 | NVIDIA DLI Robotics | [painode-075-nvidiadlirobotics](../entities/painode-075-nvidiadlirobotics.md) | natnew |
| 076 | Spinning Up in Deep RL (OpenAI) | [painode-076-spinningupindeeprlopenai](../entities/painode-076-spinningupindeeprlopenai.md) | natnew |
| 077 | zero2robot | [painode-077-zero2robot](../entities/painode-077-zero2robot.md) | natnew |

### Courses & Tutorials

| # | 条目 | 详情节点 | 来源 |
|---|------|----------|------|
| 078 | Isaac Lab Documentation | [isaac-lab](../entities/isaac-lab.md) | aichr |
| 079 | LeRobot Tutorial | [lerobot](../entities/lerobot.md) | aichr |
| 080 | MIT Foundation Models & AI | [painode-080-mitfoundationmodelsai](../entities/painode-080-mitfoundationmodelsai.md) | aichr |
| 081 | Physical AI for Science | [painode-081-physicalaiforscience](../entities/painode-081-physicalaiforscience.md) | aichr |

### Datasets

| # | 条目 | 详情节点 | 来源 |
|---|------|----------|------|
| 082 | AgiBot World | [paper-sa-2503-06669-agibot-world-colosseo-a-large-scale-manipulation](../entities/paper-sa-2503-06669-agibot-world-colosseo-a-large-scale-manipulation.md) | natnew |
| 083 | Argoverse 2 | [painode-083-argoverse2](../entities/painode-083-argoverse2.md) | natnew |
| 084 | BEHAVIOR-1K | [behavior-1k](../entities/behavior-1k.md) | natnew |
| 085 | BridgeData V2 | [painode-085-bridgedatav2](../entities/painode-085-bridgedatav2.md) | natnew+aichr |
| 086 | CALVIN ABC-D | [calvin-benchmark](../entities/calvin-benchmark.md) | natnew+aichr |
| 087 | DROID | [droid-policy-learning](../entities/droid-policy-learning.md) | natnew |
| 088 | Ego4D | [paper-ego4d](../entities/paper-ego4d.md) | natnew |
| 089 | EPIC-KITCHENS-100 | [painode-089-epickitchens100](../entities/painode-089-epickitchens100.md) | natnew |
| 090 | nuScenes | [painode-090-nuscenes](../entities/painode-090-nuscenes.md) | natnew |
| 091 | Open X-Embodiment | [paper-open-x-embodiment](../entities/paper-open-x-embodiment.md) | natnew+aichr |
| 092 | RH20T | [painode-092-rh20t](../entities/painode-092-rh20t.md) | natnew |
| 093 | RLDS | [painode-093-rlds](../entities/painode-093-rlds.md) | natnew |
| 094 | RoboMIND | [painode-094-robomind](../entities/painode-094-robomind.md) | natnew |
| 095 | RT-1 Dataset | [paper-rt-1](../entities/paper-rt-1.md) | aichr |
| 096 | Something-Something V2 | [painode-096-somethingsomethingv2](../entities/painode-096-somethingsomethingv2.md) | natnew |
| 097 | Waymo Open Dataset | [painode-097-waymoopendataset](../entities/painode-097-waymoopendataset.md) | natnew |

### Edge AI & Inference

| # | 条目 | 详情节点 | 来源 |
|---|------|----------|------|
| 098 | AutoGPTQ | [painode-098-autogptq](../entities/painode-098-autogptq.md) | aichr |
| 099 | llama.cpp | [painode-099-llamacpp](../entities/painode-099-llamacpp.md) | aichr |
| 100 | NVIDIA Jetson | [nvidia-jetson](../entities/nvidia-jetson.md) | aichr |
| 101 | TensorRT | [tensorrt](../entities/tensorrt.md) | aichr |
| 102 | TensorRT-LLM for Edge | [painode-102-tensorrtllmforedge](../entities/painode-102-tensorrtllmforedge.md) | aichr |
| 103 | vLLM | [painode-103-vllm](../entities/painode-103-vllm.md) | aichr |

### Evaluation Methodology

| # | 条目 | 详情节点 | 来源 |
|---|------|----------|------|
| 104 | Bench2Drive | [painode-104-bench2drive](../entities/painode-104-bench2drive.md) | natnew |
| 105 | CARLA ScenarioRunner | [painode-105-carlascenariorunner](../entities/painode-105-carlascenariorunner.md) | natnew |
| 106 | CodaLab Competitions | [painode-106-codalabcompetitions](../entities/painode-106-codalabcompetitions.md) | natnew |
| 107 | Deep RL That Matters | [paper-pai-1709-06560-deeprlthatmatters](../entities/paper-pai-1709-06560-deeprlthatmatters.md) | natnew |
| 108 | Empirical Design in Reinforcement Learning | [paper-pai-2102-03479-empiricaldesigninreinforcementle](../entities/paper-pai-2102-03479-empiricaldesigninreinforcementle.md) | natnew |
| 109 | Eval-vs-Train Mismatch (Kumar et al.) | [paper-pai-2306-13085-evalvstrainmismatchkumaretal](../entities/paper-pai-2306-13085-evalvstrainmismatchkumaretal.md) | natnew |
| 110 | EvalAI | [painode-110-evalai](../entities/painode-110-evalai.md) | natnew |
| 111 | LeRobot Evaluation Scripts | [lerobot](../entities/lerobot.md) | natnew+aichr |
| 112 | nuPlan Devkit | [painode-112-nuplandevkit](../entities/painode-112-nuplandevkit.md) | natnew |
| 113 | RoboArena | [roboarena](../methods/roboarena.md) | natnew |
| 114 | RoboHive | [painode-114-robohive](../entities/painode-114-robohive.md) | natnew |
| 115 | robomimic | [robomimic](../entities/robomimic.md) | natnew |
| 116 | SimplerEnv | [painode-116-xsimplerenv](../entities/painode-116-xsimplerenv.md) | natnew |
| 117 | Statistical Reliability of RL Evaluations | [painode-117-statisticalreliabilityofrlevalua](../entities/painode-117-statisticalreliabilityofrlevalua.md) | natnew |
| 118 | Waymo Open Challenges | [painode-118-waymoopenchallenges](../entities/painode-118-waymoopenchallenges.md) | natnew |

### Foundation Models (VLA)

| # | 条目 | 详情节点 | 来源 |
|---|------|----------|------|
| 119 | LingBot-VLA | [lingbot-vla](../entities/lingbot-vla.md) | aichr |
| 120 | Nvidia GR00T | [isaac-gr00t](../entities/isaac-gr00t.md) | aichr |
| 121 | Physical Intelligence π0 | [paper-pi0](../entities/paper-pi0.md) | aichr |
| 122 | Physical Intelligence π0.5 | [paper-pi05-open-world-vla](../entities/paper-pi05-open-world-vla.md) | aichr |
| 123 | Recursive Belief VLA | [paper-pai-2602-20659-recursivebeliefvla](../entities/paper-pai-2602-20659-recursivebeliefvla.md) | aichr |

### Frameworks & Libraries

| # | 条目 | 详情节点 | 来源 |
|---|------|----------|------|
| 124 | copper-rs | [painode-124-copperrs](../entities/painode-124-copperrs.md) | aichr |
| 125 | LangChain | [painode-125-langchain](../entities/painode-125-langchain.md) | aichr |
| 126 | LlamaFactory | [painode-126-llamafactory](../entities/painode-126-llamafactory.md) | aichr |
| 127 | OpenClaw | [openclaw](../entities/openclaw.md) | aichr |
| 128 | OpenHands | [painode-128-openhands](../entities/painode-128-openhands.md) | aichr |
| 129 | RAI | [painode-129-rai](../entities/painode-129-rai.md) | aichr |
| 130 | ROS 2 AI | [painode-130-xros2ai](../entities/painode-130-xros2ai.md) | aichr |

### Governance & Policy

| # | 条目 | 详情节点 | 来源 |
|---|------|----------|------|
| 131 | EU AI Act | [painode-131-euaiact](../entities/painode-131-euaiact.md) | natnew |
| 132 | EU Machinery Regulation (EU 2023/1230) | [painode-132-eumachineryregulationeu20231230](../entities/painode-132-eumachineryregulationeu20231230.md) | natnew |
| 133 | IEEE 7000 Series | [painode-133-ieee7000series](../entities/painode-133-ieee7000series.md) | natnew |
| 134 | ISO 10218 / ISO/TS 15066 | [painode-134-iso10218isots15066](../entities/painode-134-iso10218isots15066.md) | natnew |
| 135 | ISO 13482 | [painode-135-iso13482](../entities/painode-135-iso13482.md) | natnew |
| 136 | ISO 26262 | [painode-136-iso26262](../entities/painode-136-iso26262.md) | natnew |
| 137 | ISO/IEC 42001 | [painode-137-isoiec42001](../entities/painode-137-isoiec42001.md) | natnew |
| 138 | NIST AI Risk Management Framework | [painode-138-nistairiskmanagementframework](../entities/painode-138-nistairiskmanagementframework.md) | natnew |
| 139 | NIST AI RMF Generative AI Profile | [painode-139-nistairmfgenerativeaiprofile](../entities/painode-139-nistairmfgenerativeaiprofile.md) | natnew |
| 140 | OECD AI Principles | [painode-140-oecdaiprinciples](../entities/painode-140-oecdaiprinciples.md) | natnew |
| 141 | UK AI Safety Institute | [painode-141-ukaisafetyinstitute](../entities/painode-141-ukaisafetyinstitute.md) | natnew |
| 142 | UL 4600 | [painode-142-ul4600](../entities/painode-142-ul4600.md) | natnew |
| 143 | UNECE R155 | [painode-143-unecer155](../entities/painode-143-unecer155.md) | natnew |
| 144 | UNECE R156 | [painode-144-unecer156](../entities/painode-144-unecer156.md) | natnew |
| 145 | White House Executive Order on AI (14110) | [painode-145-whitehouseexecutiveorderonai14](../entities/painode-145-whitehouseexecutiveorderonai14.md) | natnew |

### Hardware & Actuation

| # | 条目 | 详情节点 | 来源 |
|---|------|----------|------|
| 146 | AnySkin | [painode-146-anyskin](../entities/painode-146-anyskin.md) | aichr |
| 147 | Awesome-Touch | [awesome-touch](../entities/awesome-touch.md) | aichr |
| 148 | DexSkin | [painode-148-dexskin](../entities/painode-148-dexskin.md) | aichr |
| 149 | Dynamixel | [painode-149-dynamixel](../entities/painode-149-dynamixel.md) | aichr |
| 150 | EtherCAT | [ethercat-protocol](../concepts/ethercat-protocol.md) | aichr |
| 151 | GelSight | [painode-151-gelsight](../entities/painode-151-gelsight.md) | aichr |
| 152 | Google Coral | [painode-152-googlecoral](../entities/painode-152-googlecoral.md) | aichr |
| 153 | Intel Neural Compute Stick | [painode-153-intelneuralcomputestick](../entities/painode-153-intelneuralcomputestick.md) | aichr |

### Hardware Platforms

| # | 条目 | 详情节点 | 来源 |
|---|------|----------|------|
| 154 | Boston Dynamics Atlas | [boston-dynamics](../entities/boston-dynamics.md) | natnew |
| 155 | Boston Dynamics Spot | [boston-dynamics](../entities/boston-dynamics.md) | natnew |
| 156 | Clearpath Robotics | [painode-156-clearpathrobotics](../entities/painode-156-clearpathrobotics.md) | natnew |
| 157 | Dex-UMI | [paper-notebook-dexumi-using-human-hand-as-the-universal-manipul](../entities/paper-notebook-dexumi-using-human-hand-as-the-universal-manipul.md) | natnew |
| 158 | DexUMI Code & Data | [paper-notebook-dexumi-using-human-hand-as-the-universal-manipul](../entities/paper-notebook-dexumi-using-human-hand-as-the-universal-manipul.md) | natnew |
| 159 | Fourier Intelligence GR-1 | [fourier-grx-n1](../entities/fourier-grx-n1.md) | natnew |
| 160 | Franka Emika | [franka-research-3](../entities/franka-research-3.md) | natnew |
| 161 | Gello | [painode-161-gello](../entities/painode-161-gello.md) | natnew |
| 162 | Hello Robot Stretch | [painode-162-hellorobotstretch](../entities/painode-162-hellorobotstretch.md) | natnew |
| 163 | Kinova | [painode-163-kinova](../entities/painode-163-kinova.md) | natnew |
| 164 | Kuka iiwa | [painode-164-kukaiiwa](../entities/painode-164-kukaiiwa.md) | natnew |
| 165 | Open Dynamic Robot Initiative | [painode-165-opendynamicrobotinitiative](../entities/painode-165-opendynamicrobotinitiative.md) | natnew |
| 166 | Open Manipulator | [robotis-open-manipulator-line](../entities/robotis-open-manipulator-line.md) | natnew |
| 167 | PAL Robotics TIAGo | [painode-167-palroboticstiago](../entities/painode-167-palroboticstiago.md) | natnew |
| 168 | Reachy 2 (Pollen Robotics / Hugging Face) | [pollen-reachy2](../entities/pollen-reachy2.md) | natnew |
| 169 | Reachy Mini | [painode-169-reachymini](../entities/painode-169-reachymini.md) | natnew |
| 170 | SO-ARM100 | [painode-170-soarm100](../entities/painode-170-soarm100.md) | natnew |
| 171 | Stanford Pupper | [stanford-doggo-and-pupper](../entities/stanford-doggo-and-pupper.md) | natnew |
| 172 | UMI Gripper | [painode-172-umigripper](../entities/painode-172-umigripper.md) | natnew |
| 173 | Universal Robots | [painode-173-universalrobots](../entities/painode-173-universalrobots.md) | natnew |
| 174 | xArm | [painode-174-xarm](../entities/painode-174-xarm.md) | natnew |

### Key Papers

| # | 条目 | 详情节点 | 来源 |
|---|------|----------|------|
| 175 | ASAP: Aligning Simulation and Real-World Physics | [paper-notebook-asap-aligning-simulation-and-real-world-physics](../entities/paper-notebook-asap-aligning-simulation-and-real-world-physics.md) | natnew |
| 176 | Attention-Based Map Encoding for Generalized Legged Locomotion | [painode-176-attentionbasedmapencodingforgen](../entities/painode-176-attentionbasedmapencodingforgen.md) | natnew |
| 177 | Denoising World Model Learning for Humanoid Locomotion | [paper-notebook-advancing-humanoid-locomotion-mastering-challeng](../entities/paper-notebook-advancing-humanoid-locomotion-mastering-challeng.md) | natnew |
| 178 | Dex-UMI: A Benchmark for Generalizable Dexterous Manipulation | [paper-sa-2602-16710-egoscale-scaling-dexterous-manipulation-with-div](../entities/paper-sa-2602-16710-egoscale-scaling-dexterous-manipulation-with-div.md) | natnew |
| 179 | Expressive Whole-Body Control for Humanoid Robots | [paper-exbody-expressive-humanoid](../entities/paper-exbody-expressive-humanoid.md) | natnew |
| 180 | H2O: Human-to-Humanoid Real-Time Whole-Body Teleoperation | [paper-hrl-stack-07-learning_human_to_humanoid_real_time](../entities/paper-hrl-stack-07-learning_human_to_humanoid_real_time.md) | natnew |
| 181 | HOVER: Versatile Neural Whole-Body Controller | [paper-bfm-14-hover](../entities/paper-bfm-14-hover.md) | natnew |
| 182 | HugWBC: Unified Humanoid Whole-Body Controller | [paper-pai-2502-03206-hugwbcunifiedhumanoidwholebodyc](../entities/paper-pai-2502-03206-hugwbcunifiedhumanoidwholebodyc.md) | natnew |
| 183 | UMI: An Open-Source Underactuated Manipulator for Dexterous Grasping | [paper-notebook-dreamzero-world-action-models-are-zero-shot-poli](../entities/paper-notebook-dreamzero-world-action-models-are-zero-shot-poli.md) | natnew |

### Locomotion

| # | 条目 | 详情节点 | 来源 |
|---|------|----------|------|
| 184 | Agile and Dynamic Motor Skills | [paper-rma-rapid-motor-adaptation](../entities/paper-rma-rapid-motor-adaptation.md) | natnew |
| 185 | ANYmal Parkour (RSL) | [anymal](../entities/anymal.md) | natnew+aichr |
| 186 | ASAP — Sim-to-Real for Humanoid Whole-Body Skills | [paper-notebook-asap-aligning-simulation-and-real-world-physics](../entities/paper-notebook-asap-aligning-simulation-and-real-world-physics.md) | natnew |
| 187 | Bipedal Soccer (DeepMind OP3) | [paper-pai-2304-13653-bipedalsoccerdeepmindop3](../entities/paper-pai-2304-13653-bipedalsoccerdeepmindop3.md) | natnew |
| 188 | CASSI (Max Planck / Martius Lab) | [painode-188-cassimaxplanckmartiuslab](../entities/painode-188-cassimaxplanckmartiuslab.md) | natnew |
| 189 | Cassie Bipedal Locomotion | [paper-pai-2105-08328-cassiebipedallocomotion](../entities/paper-pai-2105-08328-cassiebipedallocomotion.md) | natnew |
| 190 | DeepMimic | [deepmimic](../methods/deepmimic.md) | natnew |
| 191 | Expressive Whole-Body Control | [paper-exbody-expressive-humanoid](../entities/paper-exbody-expressive-humanoid.md) | natnew |
| 192 | FLD — Fourier Latent Dynamics (MIT Biomimetics) | [painode-192-fldfourierlatentdynamicsmitbiom](../entities/painode-192-fldfourierlatentdynamicsmitbiom.md) | natnew |
| 193 | HOVER — Versatile Humanoid Whole-Body Controller | [paper-bfm-14-hover](../entities/paper-bfm-14-hover.md) | natnew |
| 194 | Humanoid Parkour Learning | [paper-notebook-humanoid-parkour-learning](../entities/paper-notebook-humanoid-parkour-learning.md) | natnew |
| 195 | HumanPlus | [paper-loco-manip-161-012-humanplus](../entities/paper-loco-manip-161-012-humanplus.md) | natnew |
| 196 | Isaac Gym | [isaac-gym](../entities/isaac-gym.md) | natnew |
| 197 | Learning Quadrupedal Locomotion over Challenging Terrain | [paper-notebook-learning-agile-and-dynamic-motor-skills-for-legg](../entities/paper-notebook-learning-agile-and-dynamic-motor-skills-for-legg.md) | natnew |
| 198 | Learning to Walk in Minutes (ETH/RSL) | [legged-gym](../entities/legged-gym.md) | natnew |
| 199 | MuJoCo Menagerie | [mujoco-menagerie](../entities/mujoco-menagerie.md) | natnew |
| 200 | OmniH2O | [paper-hrl-stack-08-omnih2o](../entities/paper-hrl-stack-08-omnih2o.md) | natnew |
| 201 | Periodic Reward Composition for Bipedal Gaits | [paper-pai-2011-01387-periodicrewardcompositionforbipe](../entities/paper-pai-2011-01387-periodicrewardcompositionforbipe.md) | natnew |
| 202 | Rapid Locomotion via RL | [paper-rapid-locomotion-rl](../entities/paper-rapid-locomotion-rl.md) | natnew |
| 203 | Real-World Humanoid Locomotion with RL | [paper-digit-humanoid-locomotion-rl](../entities/paper-digit-humanoid-locomotion-rl.md) | natnew |
| 204 | RMA — Rapid Motor Adaptation | [paper-rma-rapid-motor-adaptation](../entities/paper-rma-rapid-motor-adaptation.md) | natnew |
| 205 | Robust Parameterized Bipedal Locomotion (Cassie) | [paper-pai-2103-14295-robustparameterizedbipedallocomot](../entities/paper-pai-2103-14295-robustparameterizedbipedallocomot.md) | natnew |
| 206 | RSL-RL | [rsl-rl](../entities/rsl-rl.md) | natnew |
| 207 | Walk These Ways | [paper-walk-these-ways-quadruped-mob](../entities/paper-walk-these-ways-quadruped-mob.md) | natnew |
| 208 | WASABI (Max Planck / Martius Lab) | [painode-208-wasabimaxplanckmartiuslab](../entities/painode-208-wasabimaxplanckmartiuslab.md) | natnew |

### Manipulation

| # | 条目 | 详情节点 | 来源 |
|---|------|----------|------|
| 209 | 3D Diffusion Policy (DP3) | [painode-209-3ddiffusionpolicydp3](../entities/painode-209-3ddiffusionpolicydp3.md) | natnew |
| 210 | ACT (Action Chunking Transformers) | [aloha](../entities/aloha.md) | natnew |
| 211 | ALOHA Unleashed | [painode-211-alohaunleashed](../entities/painode-211-alohaunleashed.md) | natnew |
| 212 | AnyGrasp | [anygrasp](../entities/anygrasp.md) | natnew |
| 213 | CLIPort | [painode-213-cliport](../entities/painode-213-cliport.md) | natnew |
| 214 | Contact-GraspNet | [painode-214-contactgraspnet](../entities/painode-214-contactgraspnet.md) | natnew |
| 215 | Dex-Net | [painode-215-dexnet](../entities/painode-215-dexnet.md) | natnew |
| 216 | Diffusion Policy | [paper-diffusion-policy](../entities/paper-diffusion-policy.md) | natnew |
| 217 | DreamZero | [paper-notebook-dreamzero-world-action-models-are-zero-shot-poli](../entities/paper-notebook-dreamzero-world-action-models-are-zero-shot-poli.md) | natnew |
| 218 | GraspNet-1Billion | [painode-218-graspnet1billion](../entities/painode-218-graspnet1billion.md) | natnew |
| 219 | MIT 6.4210 — Robotic Manipulation | [painode-219-mit64210roboticmanipulation](../entities/painode-219-mit64210roboticmanipulation.md) | natnew |
| 220 | Mobile ALOHA | [aloha](../entities/aloha.md) | natnew+aichr |
| 221 | PerAct | [painode-221-peract](../entities/painode-221-peract.md) | natnew |
| 222 | RoboCasa | [robocasa](../entities/robocasa.md) | natnew |
| 223 | robosuite | [robosuite](../entities/robosuite.md) | natnew |
| 224 | T-Rex | [paper-trex-tactile-reactive-dexterous-manipulation](../entities/paper-trex-tactile-reactive-dexterous-manipulation.md) | natnew |
| 225 | Transporter Networks | [painode-225-transporternetworks](../entities/painode-225-transporternetworks.md) | natnew |

### Newsletters & Blogs

| # | 条目 | 详情节点 | 来源 |
|---|------|----------|------|
| 226 | Ahead of AI | [painode-226-aheadofai](../entities/painode-226-aheadofai.md) | natnew |
| 227 | Boston Dynamics Blog | [boston-dynamics](../entities/boston-dynamics.md) | natnew |
| 228 | Chipstrat | [painode-228-chipstrat](../entities/painode-228-chipstrat.md) | natnew |
| 229 | Google DeepMind Blog | [painode-229-googledeepmindblog](../entities/painode-229-googledeepmindblog.md) | natnew |
| 230 | Hugging Face Blog | [painode-230-huggingfaceblog](../entities/painode-230-huggingfaceblog.md) | natnew |
| 231 | IEEE Spectrum Robotics | [painode-231-ieeespectrumrobotics](../entities/painode-231-ieeespectrumrobotics.md) | natnew+aichr |
| 232 | Import AI | [painode-232-importai](../entities/painode-232-importai.md) | natnew |
| 233 | Interconnects | [painode-233-interconnects](../entities/painode-233-interconnects.md) | natnew |
| 234 | Meta AI Blog | [painode-234-metaaiblog](../entities/painode-234-metaaiblog.md) | natnew |
| 235 | NVIDIA Developer Blog | [painode-235-nvidiadeveloperblog](../entities/painode-235-nvidiadeveloperblog.md) | natnew |
| 236 | Robotics 24/7 | [painode-236-robotics247](../entities/painode-236-robotics247.md) | natnew |
| 237 | Robots & Startups | [painode-237-robotsstartups](../entities/painode-237-robotsstartups.md) | natnew |
| 238 | The Batch | [painode-238-thebatch](../entities/painode-238-thebatch.md) | natnew |
| 239 | The Robot Report | [painode-239-therobotreport](../entities/painode-239-therobotreport.md) | natnew+aichr |

### People to Follow

| # | 条目 | 详情节点 | 来源 |
|---|------|----------|------|
| 240 | Andra Keay | [painode-240-andrakeay](../entities/painode-240-andrakeay.md) | natnew |
| 241 | Andrej Karpathy | [andrej-karpathy](../entities/andrej-karpathy.md) | natnew |
| 242 | Angelica Lim | [painode-242-angelicalim](../entities/painode-242-angelicalim.md) | natnew |
| 243 | Austin Lyons | [painode-243-austinlyons](../entities/painode-243-austinlyons.md) | natnew |
| 244 | Brett Adcock | [painode-244-brettadcock](../entities/painode-244-brettadcock.md) | natnew |
| 245 | Chelsea Finn | [painode-245-chelseafinn](../entities/painode-245-chelseafinn.md) | natnew |
| 246 | Dieter Fox | [painode-246-dieterfox](../entities/painode-246-dieterfox.md) | natnew |
| 247 | Fei-Fei Li | [painode-247-feifeili](../entities/painode-247-feifeili.md) | natnew |
| 248 | Kate Darling | [painode-248-katedarling](../entities/painode-248-katedarling.md) | natnew |
| 249 | Pieter Abbeel | [painode-249-pieterabbeel](../entities/painode-249-pieterabbeel.md) | natnew |
| 250 | Rodney Brooks | [painode-250-rodneybrooks](../entities/painode-250-rodneybrooks.md) | natnew |
| 251 | Russ Tedrake | [painode-251-russtedrake](../entities/painode-251-russtedrake.md) | natnew |
| 252 | Sergey Levine | [sergey-levine-diffusion-expressive-policies](../overview/sergey-levine-diffusion-expressive-policies.md) | natnew |
| 253 | Soumith Chintala | [painode-253-soumithchintala](../entities/painode-253-soumithchintala.md) | natnew |
| 254 | Yann LeCun | [painode-254-yannlecun](../entities/painode-254-yannlecun.md) | natnew |

### Production Patterns / Reference Architectures

| # | 条目 | 详情节点 | 来源 |
|---|------|----------|------|
| 255 | ABot-AgentOS | [paper-pai-2607-10350-abotagentos](../entities/paper-pai-2607-10350-abotagentos.md) | natnew |
| 256 | BehaviorTree.CPP | [painode-256-behaviortreecpp](../entities/painode-256-behaviortreecpp.md) | natnew |
| 257 | DDS Security (OMG Spec) | [painode-257-ddssecurityomgspec](../entities/painode-257-ddssecurityomgspec.md) | natnew |
| 258 | Eclipse Cyclone DDS | [cyclone-dds](../entities/cyclone-dds.md) | natnew |
| 259 | Fast DDS | [fast-dds](../entities/fast-dds.md) | natnew |
| 260 | Foxglove | [foxglove-studio](../entities/foxglove-studio.md) | natnew |
| 261 | MCAP | [mcap-log-format](../entities/mcap-log-format.md) | natnew |
| 262 | micro-ROS | [painode-262-microros](../entities/painode-262-microros.md) | natnew |
| 263 | MoveIt 2 | [moveit2](../entities/moveit2.md) | natnew |
| 264 | Nav2 | [navigation2](../entities/navigation2.md) | natnew |
| 265 | NVIDIA Isaac ROS | [isaac-ros-nvblox](../entities/isaac-ros-nvblox.md) | natnew |
| 266 | Open-RMF | [painode-266-openrmf](../entities/painode-266-openrmf.md) | natnew |
| 267 | ROS 2 | [ros2-basics](../concepts/ros2-basics.md) | natnew+aichr |
| 268 | ros2_control | [ros2-control](../entities/ros2-control.md) | natnew+aichr |
| 269 | rosbag2 | [painode-269-xrosbag2](../entities/painode-269-xrosbag2.md) | natnew |
| 270 | Zenoh | [painode-270-zenoh](../entities/painode-270-zenoh.md) | natnew |

### Related Awesome Lists

| # | 条目 | 详情节点 | 来源 |
|---|------|----------|------|
| 271 | Awesome Agentic AI Security | [painode-271-awesomeagenticaisecurity](../entities/painode-271-awesomeagenticaisecurity.md) | natnew |
| 272 | Awesome Agentic Engineering | [painode-272-awesomeagenticengineering](../entities/painode-272-awesomeagenticengineering.md) | natnew |
| 273 | Awesome AI Scientists | [painode-273-awesomeaiscientists](../entities/painode-273-awesomeaiscientists.md) | natnew |
| 274 | Awesome Deep RL | [painode-274-awesomedeeprl](../entities/painode-274-awesomedeeprl.md) | natnew |
| 275 | Awesome Embodied Agent | [painode-275-awesomeembodiedagent](../entities/painode-275-awesomeembodiedagent.md) | natnew |
| 276 | Awesome Generative AI | [painode-276-awesomegenerativeai](../entities/painode-276-awesomegenerativeai.md) | natnew |
| 277 | Awesome Imitation Learning | [painode-277-awesomeimitationlearning](../entities/painode-277-awesomeimitationlearning.md) | natnew |
| 278 | Awesome LLM Robotics | [painode-278-awesomellmrobotics](../entities/painode-278-awesomellmrobotics.md) | natnew |
| 279 | Awesome Robotics | [painode-279-awesomerobotics](../entities/painode-279-awesomerobotics.md) | natnew |
| 280 | Awesome Robotics 3D | [painode-280-awesomerobotics3d](../entities/painode-280-awesomerobotics3d.md) | natnew |
| 281 | Awesome Robotics Libraries | [painode-281-awesomeroboticslibraries](../entities/painode-281-awesomeroboticslibraries.md) | natnew |
| 282 | Awesome World Models | [awesome-world-models](../entities/awesome-world-models.md) | natnew |
| 283 | Bipedal Robot Learning Collection | [painode-283-bipedalrobotlearningcollection](../entities/painode-283-bipedalrobotlearningcollection.md) | natnew |

### Research Labs

| # | 条目 | 详情节点 | 来源 |
|---|------|----------|------|
| 284 | Berkeley RAIL Lab | [painode-284-berkeleyraillab](../entities/painode-284-berkeleyraillab.md) | aichr |
| 285 | Google DeepMind | [painode-285-googledeepmind](../entities/painode-285-googledeepmind.md) | aichr |
| 286 | MIT Distributed Robotics Lab | [painode-286-mitdistributedroboticslab](../entities/painode-286-mitdistributedroboticslab.md) | aichr |
| 287 | MIT Media Lab Personal Robots | [painode-287-mitmedialabpersonalrobots](../entities/painode-287-mitmedialabpersonalrobots.md) | aichr |
| 288 | MIT Robotics | [painode-288-mitrobotics](../entities/painode-288-mitrobotics.md) | aichr |
| 289 | Nvidia GEAR Lab | [nvidia-gear-lab](../entities/nvidia-gear-lab.md) | aichr |
| 290 | Physical Intelligence | [painode-290-physicalintelligence](../entities/painode-290-physicalintelligence.md) | aichr |
| 291 | Stanford RISELab | [painode-291-stanfordriselab](../entities/painode-291-stanfordriselab.md) | aichr |
| 292 | Stanford VL | [painode-292-stanfordvl](../entities/painode-292-stanfordvl.md) | aichr |

### Robot Platforms

| # | 条目 | 详情节点 | 来源 |
|---|------|----------|------|
| 293 | Agility Robotics - Digit | [painode-293-agilityroboticsdigit](../entities/painode-293-agilityroboticsdigit.md) | aichr |
| 294 | Amazon Robotics | [painode-294-amazonrobotics](../entities/painode-294-amazonrobotics.md) | aichr |
| 295 | Apollo Robot | [painode-295-apollorobot](../entities/painode-295-apollorobot.md) | aichr |

### Robotics Foundation Models

| # | 条目 | 详情节点 | 来源 |
|---|------|----------|------|
| 296 | Code as Policies | [paper-pai-2209-07753-codeaspolicies](../entities/paper-pai-2209-07753-codeaspolicies.md) | natnew |
| 297 | Gato | [paper-pai-2205-06175-gato](../entities/paper-pai-2205-06175-gato.md) | natnew |
| 298 | Gemini Robotics | [gemini-robotics](../entities/gemini-robotics.md) | natnew+aichr |
| 299 | GR00T N1 (NVIDIA) | [isaac-gr00t](../entities/isaac-gr00t.md) | natnew |
| 300 | Helix (Figure) | [helix-25](../entities/helix-25.md) | natnew |
| 301 | MEM — Multi-Scale Embodied Memory | [paper-pai-2603-03596-memmultiscaleembodiedmemory](../entities/paper-pai-2603-03596-memmultiscaleembodiedmemory.md) | natnew |
| 302 | Octo | [paper-octo](../entities/paper-octo.md) | natnew+aichr |
| 303 | OpenVLA | [openvla](../entities/openvla.md) | natnew |
| 304 | PaLM-E | [paper-palm-e-embodied-language-model](../entities/paper-palm-e-embodied-language-model.md) | natnew+aichr |
| 305 | R&B-EnCoRe | [paper-pai-2602-08167-rbencore](../entities/paper-pai-2602-08167-rbencore.md) | natnew |
| 306 | RoboFlamingo | [paper-pai-2311-01378-roboflamingo](../entities/paper-pai-2311-01378-roboflamingo.md) | natnew |
| 307 | RT-1 | [paper-rt-1](../entities/paper-rt-1.md) | natnew+aichr |
| 308 | RT-2 | [paper-rt-2](../entities/paper-rt-2.md) | natnew+aichr |
| 309 | SayCan | [saycan](../methods/saycan.md) | natnew |
| 310 | VIMA | [paper-pai-2210-03094-vima](../entities/paper-pai-2210-03094-vima.md) | natnew |
| 311 | π0 (Physical Intelligence) | [paper-pi0](../entities/paper-pi0.md) | natnew |

### Safety & Robustness

| # | 条目 | 详情节点 | 来源 |
|---|------|----------|------|
| 312 | Constrained Policy Optimization (Achiam et al.) | [paper-pai-1705-10528-constrainedpolicyoptimizationachi](../entities/paper-pai-1705-10528-constrainedpolicyoptimizationachi.md) | natnew |
| 313 | Control Barrier Functions | [control-barrier-function](../concepts/control-barrier-function.md) | natnew |
| 314 | OmniSafe | [painode-314-omnisafe](../entities/painode-314-omnisafe.md) | natnew |
| 315 | Realistic Adversarial Driving (Wang et al.) | [paper-pai-2003-01197-realisticadversarialdrivingwange](../entities/paper-pai-2003-01197-realisticadversarialdrivingwange.md) | natnew |
| 316 | Responsibility-Sensitive Safety (RSS) | [painode-316-responsibilitysensitivesafetyrss](../entities/painode-316-responsibilitysensitivesafetyrss.md) | natnew |
| 317 | Robot Trust & Safety (Stanford CRFM) | [painode-317-robottrustsafetystanfordcrfm](../entities/painode-317-robottrustsafetystanfordcrfm.md) | natnew |
| 318 | Robust Policy Optimization | [paper-pai-1906-03710-robustpolicyoptimization](../entities/paper-pai-1906-03710-robustpolicyoptimization.md) | natnew |
| 319 | S-TaLiRo | [painode-319-staliro](../entities/painode-319-staliro.md) | natnew |
| 320 | Safe Control Gym | [painode-320-safecontrolgym](../entities/painode-320-safecontrolgym.md) | natnew |
| 321 | Safe Reinforcement Learning Survey | [paper-pai-2205-10330-safereinforcementlearningsurvey](../entities/paper-pai-2205-10330-safereinforcementlearningsurvey.md) | natnew |
| 322 | Safety Gym (OpenAI) | [painode-322-safetygymopenai](../entities/painode-322-safetygymopenai.md) | natnew |
| 323 | Safety-Gymnasium | [painode-323-safetygymnasium](../entities/painode-323-safetygymnasium.md) | natnew |
| 324 | Scenic | [painode-324-scenic](../entities/painode-324-scenic.md) | natnew |
| 325 | VerifAI | [painode-325-verifai](../entities/painode-325-verifai.md) | natnew |
| 326 | Verifiable Reinforcement Learning (DeepMind) | [paper-pai-2308-13247-verifiablereinforcementlearningde](../entities/paper-pai-2308-13247-verifiablereinforcementlearningde.md) | natnew |

### Sim-to-Real

| # | 条目 | 详情节点 | 来源 |
|---|------|----------|------|
| 327 | Automatic Domain Randomization | [paper-as-1910-07113-solving-rubik-s-cube-with-a-robot-hand](../entities/paper-as-1910-07113-solving-rubik-s-cube-with-a-robot-hand.md) | natnew |
| 328 | BayesSim | [paper-pai-1906-01728-bayessim](../entities/paper-pai-1906-01728-bayessim.md) | natnew |
| 329 | DextrAH-G | [painode-329-dextrahg](../entities/painode-329-dextrahg.md) | natnew |
| 330 | DeXtreme (NVIDIA) | [painode-330-dextremenvidia](../entities/painode-330-dextremenvidia.md) | natnew |
| 331 | Domain Randomization (Tobin et al.) | [paper-notebook-domain-randomization-for-transferring-deep-neura](../entities/paper-notebook-domain-randomization-for-transferring-deep-neura.md) | natnew |
| 332 | Eureka (NVIDIA) | [painode-332-eurekanvidia](../entities/painode-332-eurekanvidia.md) | natnew |
| 333 | Learning Agile Flight in the Wild | [paper-pai-1909-11652-learningagileflightinthewild](../entities/paper-pai-1909-11652-learningagileflightinthewild.md) | natnew |
| 334 | Learning Dexterous In-Hand Manipulation (OpenAI) | [paper-pai-1808-00177-learningdexterousinhandmanipulat](../entities/paper-pai-1808-00177-learningdexterousinhandmanipulat.md) | natnew |
| 335 | Learning Robust Perceptive Locomotion (Miki et al.) | [paper-robust-perceptive-locomotion-wild](../entities/paper-robust-perceptive-locomotion-wild.md) | natnew |
| 336 | Privileged Learning for Rapid Motor Adaptation | [paper-anymal-walk-minutes-parallel-drl](../entities/paper-anymal-walk-minutes-parallel-drl.md) | natnew |
| 337 | Residual Reinforcement Learning for Robot Control | [paper-residual-rl-robot-control](../entities/paper-residual-rl-robot-control.md) | natnew |
| 338 | Sim-to-Real via Sim-to-Sim (Koos et al. line) | [paper-pai-1812-07252-simtorealviasimtosimkooseta](../entities/paper-pai-1812-07252-simtorealviasimtosimkooseta.md) | natnew |
| 339 | SimGAN | [paper-pai-1612-07828-simgan](../entities/paper-pai-1612-07828-simgan.md) | natnew |
| 340 | SimOpt | [paper-pai-1910-13325-simopt](../entities/paper-pai-1910-13325-simopt.md) | natnew |
| 341 | SimToolReal | [paper-sa-2602-16863-simtoolreal-an-object-centric-policy-for-zero-sh](../entities/paper-sa-2602-16863-simtoolreal-an-object-centric-policy-for-zero-sh.md) | natnew |

### Simulators

| # | 条目 | 详情节点 | 来源 |
|---|------|----------|------|
| 342 | AirSim | [airsim](../entities/airsim.md) | natnew |
| 343 | Brax | [brax](../entities/brax.md) | natnew |
| 344 | CARLA | [carla](../entities/carla.md) | natnew |
| 345 | CoppeliaSim | [coppeliasim](../entities/coppeliasim.md) | natnew |
| 346 | Drake | [drake](../entities/drake.md) | natnew |
| 347 | Gazebo | [gazebo-sim](../entities/gazebo-sim.md) | natnew |
| 348 | Genesis | [genesis-sim](../entities/genesis-sim.md) | natnew+aichr |
| 349 | Gymnasium | [gymnasium](../entities/gymnasium.md) | aichr |
| 350 | Habitat | [habitat-sim](../entities/habitat-sim.md) | natnew+aichr |
| 351 | iGibson | [igibson](../entities/igibson.md) | aichr |
| 352 | Isaac Lab | [isaac-lab](../entities/isaac-lab.md) | natnew+aichr |
| 353 | MuJoCo | [mujoco](../entities/mujoco.md) | natnew+aichr |
| 354 | NVIDIA Isaac Sim | [isaac-sim](../entities/isaac-sim.md) | natnew |
| 355 | PyBullet | [pybullet](../entities/pybullet.md) | natnew+aichr |
| 356 | RaiSim | [raisim](../entities/raisim.md) | natnew |
| 357 | SAPIEN | [sapien](../entities/sapien.md) | natnew+aichr |
| 358 | Webots | [webots](../entities/webots.md) | natnew+aichr |

### Survey Papers

| # | 条目 | 详情节点 | 来源 |
|---|------|----------|------|
| 359 | 3D Gaussian Splatting in Robotics | [paper-pai-2410-12262-3dgaussiansplattinginrobotics](../entities/paper-pai-2410-12262-3dgaussiansplattinginrobotics.md) | natnew |
| 360 | A Survey on LLM-based Autonomous Agents | [paper-pai-2308-11432-asurveyonllmbasedautonomousage](../entities/paper-pai-2308-11432-asurveyonllmbasedautonomousage.md) | natnew |
| 361 | Foundation Models in Robotics | [paper-pai-2312-07843-foundationmodelsinrobotics](../entities/paper-pai-2312-07843-foundationmodelsinrobotics.md) | natnew |
| 362 | Neural Fields in Robotics | [paper-pai-2410-20220-neuralfieldsinrobotics](../entities/paper-pai-2410-20220-neuralfieldsinrobotics.md) | natnew |
| 363 | Robot Learning Survey | [paper-pai-2312-08591-robotlearningsurvey](../entities/paper-pai-2312-08591-robotlearningsurvey.md) | natnew |
| 364 | World Models Survey | [paper-pai-2403-02622-worldmodelssurvey](../entities/paper-pai-2403-02622-worldmodelssurvey.md) | natnew |

### Tutorials & Guides

| # | 条目 | 详情节点 | 来源 |
|---|------|----------|------|
| 365 | MuJoCo Documentation | [mujoco](../entities/mujoco.md) | natnew |
| 366 | Open X-Embodiment Tutorial | [open-x-embodiment](../concepts/open-x-embodiment.md) | natnew |
| 367 | PyBullet Quickstart | [pybullet](../entities/pybullet.md) | natnew |
| 368 | ROS 2 Tutorials | [ros2-basics](../concepts/ros2-basics.md) | natnew |

### World Models

| # | 条目 | 详情节点 | 来源 |
|---|------|----------|------|
| 369 | DayDreamer | [paper-daydreamer-world-models-real-robots](../entities/paper-daydreamer-world-models-real-robots.md) | natnew+aichr |
| 370 | Dream to Control | [paper-dreamer-latent-imagination](../entities/paper-dreamer-latent-imagination.md) | natnew |
| 371 | DreamerV2 | [paper-sa-2010-02193-dreamerv2-mastering-atari-with-discrete-world-mo](../entities/paper-sa-2010-02193-dreamerv2-mastering-atari-with-discrete-world-mo.md) | natnew |
| 372 | DreamerV3 | [paper-shenlan-wm-13-dreamerv3](../entities/paper-shenlan-wm-13-dreamerv3.md) | natnew |
| 373 | GAIA-1 (Wayve) | [paper-gaia1](../entities/paper-gaia1.md) | natnew |
| 374 | Genie 2 (DeepMind) | [painode-374-genie2deepmind](../entities/painode-374-genie2deepmind.md) | natnew |
| 375 | I-JEPA | [paper-sa-2301-08243-i-jepa-self-supervised-learning-from-images-with](../entities/paper-sa-2301-08243-i-jepa-self-supervised-learning-from-images-with.md) | natnew |
| 376 | MuZero | [paper-muzero-planning-latent-dynamics](../entities/paper-muzero-planning-latent-dynamics.md) | natnew |
| 377 | NVIDIA Cosmos | [cosmos-3](../entities/cosmos-3.md) | natnew |
| 378 | PlaNet | [paper-planet-latent-dynamics](../entities/paper-planet-latent-dynamics.md) | natnew |
| 379 | Robotic World Model (ETH RSL) | [robotic-world-model-eth-rsl](../entities/robotic-world-model-eth-rsl.md) | natnew |
| 380 | SimPLe | [paper-pai-1903-00374-simple](../entities/paper-pai-1903-00374-simple.md) | natnew |
| 381 | TD-MPC2 | [paper-td-mpc2](../entities/paper-td-mpc2.md) | natnew |
| 382 | UniSim | [paper-unisim](../entities/paper-unisim.md) | natnew |
| 383 | V-JEPA 2 (Meta FAIR) | [paper-vjepa2](../entities/paper-vjepa2.md) | natnew |
| 384 | World Models (Ha & Schmidhuber) | [paper-ha-schmidhuber-world-models](../entities/paper-ha-schmidhuber-world-models.md) | natnew |


## 局限与风险

- 索引级节点保留清单摘要，**不替代** 深度论文/工具页。
- 人物、新闻通讯与部分实验室条目只有社交媒体或新闻链接，节点用于图谱覆盖而非复现。
- 上游更新后需重跑 `python3 scripts/generate_physical_ai_awesome_entities.py` 再 `make ci-preflight`。

## 关联页面

- [awesome-physical-ai（natnew）](../entities/awesome-physical-ai-natnew.md)
- [awesome-physical-ai（aichr）](../entities/awesome-physical-ai-aichr.md)
- [Physical AI 策展清单对比](../comparisons/awesome-physical-ai-curated-lists.md)
- [VLA](../methods/vla.md)
- [Sim2Real](../concepts/sim2real.md)

## 参考来源

- [awesome-physical-ai-union-catalog.md](../../sources/repos/awesome-physical-ai-union-catalog.md)
- [sources/repos/awesome-physical-ai-natnew.md](../../sources/repos/awesome-physical-ai-natnew.md)
- [sources/repos/awesome-physical-ai-aichr.md](../../sources/repos/awesome-physical-ai-aichr.md)

## 推荐继续阅读

- [natnew GitHub](https://github.com/natnew/awesome-physical-ai)
- [aichr GitHub](https://github.com/aichr/awesome-physical-ai)
- [natnew docs overview](https://natnew.github.io/awesome-physical-ai/docs/overview)
