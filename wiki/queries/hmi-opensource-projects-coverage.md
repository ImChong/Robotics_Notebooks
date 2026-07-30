---
title: HMI 开源项目主表 · 本库导读
type: query
status: complete
created: 2026-07-30
updated: 2026-07-30
summary: 把具身智能研究室开源项目主表（166 项）逐条接到本库对应详情页：左侧是上游策展入口，右侧是可继续深读的 wiki 节点；同主题多入口会共用同一页。
sources:
  - ../../sources/repos/humanoid-motion-intelligence.md
---

> **Query 产物**：本页由以下问题触发：「开源项目主表里的项目，在本知识库里分别对应哪一页、该怎么读？」
> 综合来源：[Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)、[开源运动控制项目结构化摘要](./open-source-motion-control-projects.md)、上游 [开源项目主表](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md)

# HMI 开源项目主表 · 本库导读

## 一句话定义

这是一张**导读表**：把 [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 上游策展的 [开源项目主表](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md)（166 项）接到本库已有的方法 / 论文 / 工程详情页，方便从「听说过这个项目」走到「在本库里读懂它」。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| HMI | Humanoid Motion Intelligence | 具身智能研究室的人形运动智能公开知识库 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| WBC | Whole-Body Control | 全身控制 |
| Sim2Real | Simulation to Real | 仿真到真机 |
| LocoManip | Loco-Manipulation | 边走边操作 / 移动操作 |
| WBT | Whole-Body Tracking | 全身动作跟踪 |

## 为什么重要

- **上游擅长策展，本库擅长深读。** 主表按技术路线把代码与项目页收拢成一张地图；本库则把其中大量条目沉淀成可交叉引用的知识页。两者不是互相替代，而是「目录 → 正文」。
- **避免同名项目迷路。** 主表里一个名字可能对应仓库、论文或整条产品线；右侧链接指向本库认定的**主阅读页**，减少在相似标题间来回跳。
- **同主题可以共用一页。** 例如官方实现与社区复现、框架与官方示例环境，若知识对象相同，会指向同一详情页，并在该页说明各入口差异。

## 怎么用

1. 在上游主表或本页分组里找到感兴趣的项目名。
2. 点左侧链接去官方仓库 / 项目页看代码与许可证。
3. 点右侧「本库详情」进入方法机制、评测、局限与交叉引用；需要对照仿真 / 控制 / VLA 主线时，再从详情页的关联区继续跳。
4. 若右侧多个主表条目指向同一页，先读该页开头与「开源入口」类小节，再判断自己关心的是算法、复现仓还是部署仓。

| 规模 | 说明 |
|------|------|
| 主表条目 | 166 |
| 均可在本库点开详情 | 166（当前无缺口） |
| 分组 | 与上游主表一致：数据与重定向、Locomotion、全身跟踪、LocoManip、VLA/世界模型、工程部署 |

## 导读总表（按主表分组）

### 动作数据与重定向（12）

| 上游项目 | 本库详情 |
| --- | --- |
| [BifrostUMI](https://baai-aether.github.io/BifrostUMI/) | [BifrostUMI](../entities/paper-bifrost-umi.md) |
| [DDR](https://arxiv.org/abs/2605.23762) | [DDR](../entities/ddr-direct-dynamics-retargeting.md) |
| [DynaRetarget](https://atarilab.github.io/dynaretarget.io/) | [DynaRetarget](../entities/paper-notebook-dynaretarget-dynamically-feasible-retargeting-us.md) |
| [GMR](https://github.com/YanjieZe/GMR) | [GMR: 通用动作重定向](../methods/motion-retargeting-gmr.md) |
| [GRAIL](https://github.com/NVlabs/GRAIL) | [GRAIL](../entities/paper-grail.md) |
| [GVHMR](https://github.com/zju3dv/GVHMR) | [GVHMR](../entities/gvhmr.md) |
| [HumanoidMimicGen](https://humanoidmimicgen.github.io/) | [HumanoidMimicGen](../entities/paper-humanoidmimicgen.md) |
| [NMR / MakeTrackingEasy](https://github.com/NJU3DV-HumanoidGroup/MakeTrackingEasy) | [NMR（神经运动重定向与人形全身控制）](../methods/neural-motion-retargeting-nmr.md) |
| [OmniRetarget](https://github.com/amazon-far/holosoma) | [OmniRetarget](../entities/paper-hrl-stack-03-omniretarget.md) |
| [PHC](https://github.com/ZhengyiLuo/PHC) | [PHC（Perpetual Humanoid Control）](../entities/phc.md) |
| [TRAM](https://github.com/yufu-wang/tram) | [TRAM](../entities/paper-motion-cerebellum-tram.md) |
| [WHAM](https://github.com/yohanshin/WHAM) | [WHAM](../entities/wham-world-human-motion.md) |

### Locomotion与运动先验（24）

| 上游项目 | 本库详情 |
| --- | --- |
| [AMP_mjlab](https://github.com/ccrpRepo/AMP_mjlab) | [AMP_mjlab (G1 统一 AMP 策略)](../entities/amp-mjlab.md) |
| [BFM-Zero](https://github.com/LeCAR-Lab/BFM-Zero) | [BFM-Zero](../entities/paper-bfm-zero.md) |
| [Booster Gym](https://github.com/BoosterRobotics/booster_gym) | [Booster Gym](../entities/paper-notebook-booster-gym-an-end-to-end-rl-framework-for-human.md) |
| [DBHL窄地形全身运动](https://whole-body-loco.github.io/) | [DBHL窄地形全身运动](../entities/dbhl-whole-body-loco.md) |
| [DreamWaQ（社区实现）](https://github.com/Manaro-Alpha/DreamWaQ) | [DreamWaQ：盲走一阶段鲁棒行走](../methods/dreamwaq.md) · 同主题共用 |
| [Generative Motion Prior](https://sites.google.com/view/humanoid-gmp) | [T-GMP](../entities/paper-motion-cerebellum-t-gmp.md) |
| [Hiking in the Wild](https://project-instinct.github.io/hiking-in-the-wild/) | [Hiking in the Wild：可扩展感知跑酷框架](../entities/paper-hiking-in-the-wild.md) |
| [Humanoid Parkour Learning](https://humanoid4parkour.github.io/) | [Humanoid Parkour Learning：无动作先验的视觉全身跑酷](../entities/paper-notebook-humanoid-parkour-learning.md) |
| [Humanoid-Gym](https://github.com/roboterax/humanoid-gym) | [Humanoid-Gym（人形零样本 Sim2Real 训练框架）](../entities/humanoid-gym.md) |
| [InternRobotics运动控制开源生态](https://github.com/InternRobotics) | [InternRobotics运动控制开源生态](../entities/internrobotics.md) |
| [Legged Lab DWAQ（Unitree G1）](https://gitee.com/chaomingsanhua/legged_lab) | [DreamWaQ：盲走一阶段鲁棒行走](../methods/dreamwaq.md) · 同主题共用 |
| [legged_gym](https://github.com/leggedrobotics/legged_gym) | [legged_gym](../entities/legged-gym.md) |
| [MoRE](https://github.com/TeleHuman/MoRE) | [MoRE：复杂地形上的人形多步态残差专家混合](../entities/paper-amp-survey-08-more.md) |
| [Perceptive Humanoid Parkour](https://php-parkour.github.io/) | [Perceptive Humanoid Parkour（PHP）](../entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md) |
| [Project Instinct](https://project-instinct.github.io/) | [Project Instinct](../entities/project-instinct.md) · 同主题共用 |
| [PULSE](https://github.com/ZhengyiLuo/PULSE) | [PULSE](../entities/pulse-physics.md) |
| [roboparty_train](https://github.com/Roboparty/roboparty_train) | [RoboParty（萝博派对）](../entities/roboparty.md) |
| [Robot Parkour Learning](https://robot-parkour.github.io/) | [Extreme Parkour（端到端四足感知跑酷）](../entities/extreme-parkour.md) |
| [SafeFall](https://safefall.github.io/) | [SafeFall](../entities/paper-hrl-stack-41-safefall.md) |
| [UFO](https://github.com/Roboparty/UFO) | [UFO（Roboparty 无监督 RL 控制框架）](../entities/roboparty-ufo.md) |
| [Unitree RL Gym](https://github.com/unitreerobotics/unitree_rl_gym) | [unitree_rl_gym](../entities/unitree-rl-gym.md) |
| [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab) | [unitree_rl_lab](../entities/unitree-rl-lab.md) |
| [Unitree RL Mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) | [unitree_rl_mjlab (Unitree 官方 RL 框架)](../entities/unitree-rl-mjlab.md) |
| [X-Loco](https://x-loco-humanoid.github.io/) | [X-Loco](../entities/x-loco-humanoid.md) |

### 动作跟踪与全身控制（24）

| 上游项目 | 本库详情 |
| --- | --- |
| [ALMI-Open](https://github.com/TeleHuman/ALMI-Open) | [ALMI：对抗 locomotion 与运动模仿的人形策略学习](../entities/paper-amp-survey-07-adversarial_locomotion_and_motion_im.md) |
| [BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking) | [BeyondMimic](../methods/beyondmimic.md) · 同主题共用 |
| [BeyondMimic-Reproduction](https://github.com/hunter20041220/BeyondMimic-Reproduction) | [BeyondMimic](../methods/beyondmimic.md) · 同主题共用 |
| [Deep Whole-Body Parkour](https://project-instinct.github.io/deep-whole-body-parkour/) | [Deep Whole-body Parkour：感知式全身跑酷](../entities/paper-deep-whole-body-parkour.md) |
| [DeepMimic](https://github.com/xbpeng/DeepMimic) | [DeepMimic: 示例引导的技能学习](../methods/deepmimic.md) |
| [Embrace Collisions](https://project-instinct.github.io/embrace-collisions/) | [Embrace Collisions：可部署的接触无关人形 Shadowing](../entities/paper-amp-survey-19-embrace_collisions.md) |
| [engineai_rl_lab](https://github.com/engineai-robotics/engineai_rl_lab) | [engineai_rl_lab](../entities/engineai-rl-lab.md) |
| [GenMimic](https://genmimic.github.io/) | [GenMimic：生成视频到物理可执行轨迹](../entities/paper-hrl-stack-04-from_generated_human_videos_to_physi.md) |
| [GMT](https://github.com/zixuan417/humanoid-general-motion-tracking) | [GMT（General Motion Tracking for Humanoid Whole-Body Control）](../entities/paper-gmt.md) |
| [H2O / human2humanoid](https://github.com/LeCAR-Lab/human2humanoid) | [H2O：人体到人形的实时全身遥操作](../entities/paper-hrl-stack-07-learning_human_to_humanoid_real_time.md) |
| [Heracles](https://heracles-humanoid-control.github.io/) | [Heracles：跟踪精度与生成式恢复的扩散中间件](../entities/paper-heracles-humanoid-diffusion.md) |
| [HoloMotion](https://github.com/HorizonRobotics/HoloMotion) | [HoloMotion（HoloMotion-1）](../entities/holomotion.md) |
| [HumanPlus](https://github.com/MarkFzp/humanplus) | [HumanPlus](../entities/paper-loco-manip-161-012-humanplus.md) |
| [LIMMT / GQS](https://github.com/GalaxyGeneralRobotics/Humanoid-GPT) | [Humanoid-GPT（Scaling Data and Structure for Zero-Shot Motion Tracking）](../entities/paper-humanoid-gpt.md) |
| [MaskedMimic / ProtoMotions](https://github.com/NVlabs/ProtoMotions) | [ProtoMotions: 大规模人形机器人仿真框架](../entities/protomotions.md) |
| [MimicKit](https://github.com/xbpeng/MimicKit) | [MimicKit: 运动模仿与控制研究套件](../entities/mimickit.md) |
| [Motion-Between BFM-2](https://www.agibot.com.cn/article/315/detail/161.html) | [BFM-2（智元运控基座）](../entities/agibot-bfm-2.md) |
| [OmniH2O](https://omni.human2humanoid.com/) | [OmniH2O](../entities/paper-hrl-stack-08-omnih2o.md) |
| [OmniTrack](https://omnitrack-humanoid.github.io/) | [OmniTrack](../entities/paper-hrl-stack-12-omnitrack.md) |
| [OmniXtreme](https://github.com/Perkins729/OmniXtreme) | [OmniXtreme](../entities/paper-hrl-stack-16-omnixtreme.md) |
| [OpenTrack / Any2Track](https://github.com/GalaxyGeneralRobotics/OpenTrack) | [Track Any Motions under Any Disturbances](../entities/paper-opentrack.md) |
| [TrackerLab](https://github.com/Renforce-Dynamics/trackerLab) | [TrackerLab](../entities/trackerlab.md) |
| [TWIST](https://github.com/YanjieZe/TWIST) | [TWIST](../entities/paper-twist.md) |
| [TWIST2](https://github.com/amazon-far/TWIST2) | [TWIST2](../entities/paper-twist2.md) |

### LocoManip（21）

| 上游项目 | 本库详情 |
| --- | --- |
| [CEER](https://robotproject8.github.io/ceer_page/) | [CEER](../entities/paper-motion-cerebellum-ceer.md) |
| [CHIP](https://nvlabs.github.io/CHIP/) | [CHIP](../entities/paper-hrl-stack-36-chip.md) |
| [CoorDex](https://github.com/Skevinci/coordex) | [CoorDex（Coordinating Body and Hand Priors for Continuous Dexterous Hu…](../entities/paper-coordex-dexterous-humanoid-loco-manipulation.md) |
| [DoorMan](https://doorman-humanoid.github.io/) | [DoorMan（Opening the Sim-to-Real Door for Humanoid Pixel-to-Action Pol…](../entities/paper-doorman-opening-sim2real-door.md) |
| [FACET](https://facet.pages.dev/) | [FACET](../entities/facet-impedance.md) |
| [GentleHumanoid](https://gentle-humanoid.axell.top/) | [GentleHumanoid](../entities/paper-gentlehumanoid.md) |
| [HANDOFF](https://github.com/lzyang2000/HANDOFF) | [HANDOFF](../entities/paper-motion-cerebellum-handoff.md) |
| [HDMI](https://github.com/LeCAR-Lab/HDMI) | [HDMI](../entities/paper-hrl-stack-06-hdmi.md) |
| [HumanX](https://wyhuai.github.io/human-x/) | [HumanX](../entities/paper-hrl-stack-05-humanx.md) |
| [OASIS](https://github.com/TeleHuman/OASIS) | [OASIS（From Simulation Data Collection to Real-World Humanoid Loco-Man…](../entities/paper-loco-manip-04-oasis.md) |
| [OmniContact](https://github.com/Ingrid789/OmniContact_sim2sim) | [OmniContact sim2sim](../entities/omnicontact-sim2sim.md) |
| [OpenHLM](https://huggingface.co/OpenHLM) | [OpenHLM](../entities/paper-loco-manip-161-154-openhlm.md) |
| [SceneBot](https://ericcsr.github.io/scenebot/) | [SceneBot（Contact-Prompted Whole-Body Tracking with Scene-Interaction）](../entities/paper-scenebot.md) |
| [SimToolReal](https://github.com/tylerlum/simtoolreal) | [SimToolReal](../entities/simtoolreal.md) |
| [SkillBlender](https://github.com/Humanoid-SkillBlender/SkillBlender) | [SkillBlender](../entities/paper-loco-manip-161-077-skillblender.md) |
| [SoFTA / Hold My Beer](https://github.com/LeCAR-Lab/SoFTA) | [Hold](../entities/paper-loco-manip-161-042-hold.md) |
| [SoftMimic](https://gmargo11.github.io/softmimic/) | [SoftMimic](../entities/paper-notebook-softmimic-learning-compliant-whole-body-control.md) |
| [SplitAdapter](https://splitadapter.github.io/) | [SplitAdapter](../entities/paper-splitadapter-load-aware-loco-manipulation.md) |
| [Thor](https://baai-aether.github.io/baai-thor/) | [Thor](../entities/paper-hrl-stack-42-thor.md) |
| [VIRAL](https://viral-humanoid.github.io/) | [VIRAL（Visual Sim-to-Real at Scale for Humanoid Loco-Manipulation）](../entities/paper-viral-humanoid-visual-sim2real.md) |
| [WT-UMI](https://wt-umi.github.io/WTUMI/) | [WT-UMI](../entities/paper-loco-manip-07-wt-umi.md) |

### 世界模型、VLA与Agent（15）

| 上游项目 | 本库详情 |
| --- | --- |
| [ACT](https://github.com/tonyzhaozh/act) | [Action Chunking（动作块输出）](../methods/action-chunking.md) |
| [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) | [Diffusion Policy](../methods/diffusion-policy.md) |
| [DreamDojo](https://github.com/NVIDIA/DreamDojo) | [DreamDojo](../entities/paper-hrl-stack-35-dreamdojo.md) |
| [DreamZero](https://github.com/dreamzero0/dreamzero) | [DreamZero](../entities/paper-notebook-dreamzero-world-action-models-are-zero-shot-poli.md) |
| [DROID Policy Learning](https://github.com/droid-dataset/droid_policy_learning) | [DROID Policy Learning](../entities/droid-policy-learning.md) |
| [GE-2 / GE-Sim 2.0](https://github.com/AgibotTech/GE-Sim-V2) | [GE-Sim 2.0（Genie Envisioner World Simulator 2.0）](../entities/ge-sim-2.md) |
| [GigaWorld-0](https://giga-world-0.github.io/) | [GigaWorld-0](../entities/gigaworld-0.md) |
| [GO-2](https://www.agibot.com/article/231/detail/56.html) | [GO-2（智元执行基座）](../entities/go-2.md) |
| [HoloAgent](https://github.com/HorizonRobotics/HoloAgent) | [HoloAgent](../entities/holoagent.md) |
| [Isaac-GR00T / GR00T N1.7](https://github.com/NVIDIA/Isaac-GR00T) | [GR00T N1](../entities/paper-hrl-stack-34-gr00t_n1.md) |
| [Octo](https://github.com/octo-models/octo) | [Octo（开源 Generalist Policy）](../methods/octo-model.md) |
| [openpi](https://github.com/Physical-Intelligence/openpi) | [π₀ (Pi-zero) 策略模型](../methods/π0-policy.md) |
| [OpenVLA](https://github.com/openvla/openvla) | [OpenVLA](../entities/openvla.md) |
| [WholeBodyVLA](https://github.com/OpenDriveLab/WholebodyVLA) | [WholeBodyVLA](../entities/paper-hrl-stack-30-wholebodyvla.md) |
| [WorldArena](https://github.com/tsinghua-fib-lab/WorldArena) | [WorldArena](../entities/worldarena.md) |

### 工程与实机部署（70）

| 上游项目 | 本库详情 |
| --- | --- |
| [ASAP](https://github.com/LeCAR-Lab/ASAP) | [ASAP Aligning Simulation and Real-World Physics for Agile Humanoid Sk…](../entities/paper-notebook-asap-aligning-simulation-and-real-world-physics.md) |
| [BEHAVIOR / OmniGibson](https://github.com/StanfordVL/BEHAVIOR-1K) | [BEHAVIOR-1K](../entities/behavior-1k.md) · 同主题共用 |
| [Brax](https://github.com/google/brax) | [Brax（JAX 可微物理与 RL 训练）](../entities/brax.md) |
| [CALVIN](https://github.com/mees/calvin) | [CALVIN](../entities/calvin-benchmark.md) |
| [CleanRL](https://github.com/vwxyzjn/cleanrl) | [CleanRL](../entities/cleanrl.md) |
| [CoppeliaSim](https://github.com/CoppeliaRobotics/coppeliaSimLib) | [CoppeliaSim](../entities/coppeliasim.md) |
| [Crocoddyl](https://github.com/loco-3d/crocoddyl) | [Crocoddyl](../entities/crocoddyl.md) |
| [DexMimicGen](https://github.com/NVlabs/dexmimicgen) | [DexMimicGen](../entities/paper-notebook-dexmimicgen-automated-data-generation-for-bimanu.md) |
| [Drake](https://github.com/RobotLocomotion/drake) | [Drake (机器人工具箱)](../entities/drake.md) |
| [EmbodiedGen V2](https://github.com/HorizonRobotics/EmbodiedGen) | [EmbodiedGen V2（Simulation-Ready 3D World Engine · arXiv:2607.07459）](../entities/paper-embodiedgen-v2-sim-ready-world-engine.md) |
| [EngineAI Native SDK](https://github.com/engineai-robotics/engineai_robotics_native_sdk) | [EngineAI Native SDK](../entities/engineai-native-sdk.md) |
| [Foxglove](https://github.com/foxglove/studio) | [Foxglove](../entities/foxglove-studio.md) |
| [Gazebo Sim](https://github.com/gazebosim/gz-sim) | [Gazebo Sim](../entities/gazebo-sim.md) |
| [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) | [Genesis (仿真器)](../entities/genesis-sim.md) |
| [Genie Sim 3.0](https://github.com/AgibotTech/genie_sim) | [Genie Sim 3.0](../entities/genie-sim-3.md) |
| [Genie Studio Agent](https://www.agibot.com/article/231/detail/59.html) | [Genie Studio Agent](../entities/genie-studio-agent.md) |
| [Humanoid Everyday](https://github.com/physical-superintelligence-lab/Humanoid-Everyday) | [Humanoid Everyday](../entities/humanoid-everyday-dataset.md) |
| [HumanoidBench](https://github.com/carlosferrazza/humanoid-bench) | [HumanoidBench](../entities/humanoid-bench.md) |
| [HumanoidVerse](https://github.com/LeCAR-Lab/HumanoidVerse) | [HumanoidVerse](../entities/paper-notebook-humanoidverse.md) |
| [Hydra](https://github.com/facebookresearch/hydra) | [Hydra](../entities/hydra-config.md) |
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab) | [Isaac Lab](../entities/isaac-lab.md) |
| [Isaac Sim](https://github.com/isaac-sim/IsaacSim) | [Isaac Sim](../entities/isaac-sim.md) |
| [IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs) | [Isaac Gym](../entities/isaac-gym.md) |
| [LeRobot](https://github.com/huggingface/lerobot) | [LeRobot (Hugging Face)](../entities/lerobot.md) |
| [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | [LIBERO](../entities/libero-benchmark.md) |
| [LocoMuJoCo](https://github.com/robfiras/loco-mujoco) | [LocoMuJoCo](../entities/loco-mujoco.md) |
| [ManiSkill](https://github.com/haosulab/ManiSkill) | [ManiSkill3](../entities/paper-notebook-maniskill3-gpu-parallelized-robotics-simulation.md) |
| [mc_rtc](https://github.com/jrl-umi3218/mc_rtc) | [mc_rtc](../entities/mc-rtc.md) |
| [MCAP](https://github.com/foxglove/mcap) | [MCAP](../entities/mcap-log-format.md) |
| [MetaWorld](https://github.com/Farama-Foundation/Metaworld) | [MetaWorld](../entities/paper-hrl-stack-32-metaworld.md) |
| [MimicGen](https://github.com/NVlabs/mimicgen) | [MimicGen](../entities/mimicgen.md) |
| [Mink](https://github.com/kevinzakka/mink) | [Mink](../entities/mink-ik.md) |
| [MJX](https://github.com/google-deepmind/mujoco/tree/main/mjx) | [MuJoCo vs Isaac Lab：仿真器选型对比](../comparisons/mujoco-vs-isaac-lab.md) |
| [MLflow](https://github.com/mlflow/mlflow) | [MLflow](../entities/mlflow.md) |
| [MOS9 开源人形机器人](https://github.com/THMOS2025/MOS-9-Open-Source-Humanoid-Robot) | [MOS9 开源人形机器人](../entities/mos9-open-source-humanoid.md) |
| [MoveIt 2](https://github.com/moveit/moveit2) | [MoveIt 2](../entities/moveit2.md) |
| [MuJoCo](https://github.com/google-deepmind/mujoco) | [MuJoCo (物理引擎)](../entities/mujoco.md) · 同主题共用 |
| [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) | [MuJoCo (物理引擎)](../entities/mujoco.md) · 同主题共用 |
| [OCS2](https://github.com/leggedrobotics/ocs2) | [OCS2](../entities/ocs2.md) |
| [OmniGibson](https://github.com/StanfordVL/OmniGibson) | [BEHAVIOR-1K](../entities/behavior-1k.md) · 同主题共用 |
| [OSQP](https://github.com/osqp/osqp) | [OSQP](../entities/osqp.md) |
| [Pink](https://github.com/stephane-caron/pink) | [Pink](../entities/pink-ik.md) |
| [Pinocchio](https://github.com/stack-of-tasks/pinocchio) | [Pinocchio (刚体动力学库)](../entities/pinocchio.md) |
| [PlotJuggler](https://github.com/facontidavide/PlotJuggler) | [PlotJuggler](../entities/plotjuggler.md) |
| [PRIME](https://github.com/well-robotics/PRIME) | [PRIME](../entities/prime-system-id.md) |
| [Project Instinct InstinctLab](https://github.com/project-instinct/instinctlab) | [Project Instinct](../entities/project-instinct.md) · 同主题共用 |
| [Project Instinct Robot Motion Editor](https://github.com/project-instinct/robot-motion-editor) | [机器人关键帧与运动编辑工具（选型入口）](../entities/robot-motion-keyframe-editors.md) |
| [project-instinct/instinct_onboard](https://github.com/project-instinct/instinct_onboard) | [Project Instinct](../entities/project-instinct.md) · 同主题共用 |
| [project-instinct/instinct_rl](https://github.com/project-instinct/instinct_rl) | [Project Instinct](../entities/project-instinct.md) · 同主题共用 |
| [ProxSuite](https://github.com/Simple-Robotics/proxsuite) | [ProxSuite](../entities/proxsuite.md) |
| [PyBullet](https://github.com/bulletphysics/bullet3) | [PyBullet](../entities/pybullet.md) |
| [RaiSim](https://github.com/raisimTech/raisimLib) | [RaiSim](../entities/raisim.md) |
| [rerun](https://github.com/rerun-io/rerun) | [rerun](../entities/rerun-io.md) |
| [rl_games](https://github.com/Denys88/rl_games) | [rl_games](../entities/rl-games.md) |
| [RLBench](https://github.com/stepjam/RLBench) | [RLBench](../entities/rlbench.md) |
| [RoboCasa](https://github.com/robocasa/robocasa) | [RoboCasa](../entities/paper-notebook-robocasa-large-scale-simulation-of-everyday-task.md) |
| [robomimic](https://github.com/ARISE-Initiative/robomimic) | [robomimic](../entities/robomimic.md) |
| [robosuite](https://github.com/ARISE-Initiative/robosuite) | [robosuite](../entities/robosuite.md) |
| [robot_descriptions.py](https://github.com/robot-descriptions/robot_descriptions.py) | [robot_descriptions.py](../entities/robot-descriptions-py.md) |
| [ROS 2](https://github.com/ros2/ros2) | [Unitree ROS 2（本库 ROS 2 相关工程入口）](../entities/unitree-ros2.md) |
| [ros2_control](https://github.com/ros-controls/ros2_control) | [ros2_control](../entities/ros2-control.md) |
| [rsl_rl](https://github.com/leggedrobotics/rsl_rl) | [AMP-RSL-RL](../entities/amp-rsl-rl.md) |
| [SafeWBC](https://kwlee365.github.io/SafeWBC-Website/) | [SafeWBC](../entities/paper-motion-cerebellum-safewbc.md) |
| [SAPIEN](https://github.com/haosulab/SAPIEN) | [SAPIEN (仿真引擎)](../entities/sapien.md) |
| [skrl](https://github.com/Toni-SM/skrl) | [skrl](../entities/skrl.md) |
| [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) | [Stable-Baselines3](../entities/stable-baselines3.md) |
| [ToddlerBot](https://github.com/hshi74/toddlerbot) | [ToddlerBot](../entities/paper-loco-manip-161-141-toddlerbot.md) |
| [TSID](https://github.com/stack-of-tasks/tsid) | [TSID](../concepts/tsid.md) |
| [Webots](https://github.com/cyberbotics/webots) | [Webots](../entities/webots.md) |
| [Weights & Biases](https://github.com/wandb/wandb) | [Weights & Biases vs TensorBoard（训练实验监控选型）](../comparisons/wandb-vs-tensorboard.md) |

## 同主题共用一页（阅读提示）

主表按「可点开的入口」分列；本库按「知识对象」收页。下列组合会落到同一详情页——读的时候把多个入口当作同一主题的不同侧面即可：

| 主表上常见的多个入口 | 本库主阅读页 | 怎么理解 |
|------|------|------|
| ACT | [Action Chunking / ACT](../methods/action-chunking.md) | 方法页承载 ACT 与 action chunk 主线 |
| openpi（π₀ 开源栈） | [π₀ Policy](../methods/π0-policy.md) | 官方开源实现挂在 π₀ 方法页 |
| IsaacGymEnvs | [Isaac Gym](../entities/isaac-gym.md) | 示例环境归入 Gym 世代说明 |
| BEHAVIOR-1K / OmniGibson | [BEHAVIOR-1K](../entities/behavior-1k.md) | 基准与仿真栈同页 |
| InstinctLab / instinct_rl / instinct_onboard / 站群 | [Project Instinct](../entities/project-instinct.md) | 站群 + 训练/仿真/板载三仓 |
| DreamWaQ 社区实现、Legged Lab DWAQ | [DreamWaQ](../methods/dreamwaq.md) | 盲走方法页；勿与 DreamWaQ++ 混淆 |
| roboparty_train | [RoboParty](../entities/roboparty.md) | 组织/训练栈入口 |
| BeyondMimic 与社区复现仓 | [BeyondMimic](../methods/beyondmimic.md) | 方法 + 复现入口 |
| Motion-Between BFM-2 | [AgiBot BFM-2](../entities/agibot-bfm-2.md) | 产品线实体页 |
| MuJoCo Menagerie | [MuJoCo](../entities/mujoco.md) | 官方模型库挂在仿真引擎页 |
| ALMI-Open | [ALMI](../entities/paper-amp-survey-07-adversarial_locomotion_and_motion_im.md) | 开源仓归入对应论文页 |
| GenMimic 项目页 | [GenMimic（生成视频→可执行轨迹）](../entities/paper-hrl-stack-04-from_generated_human_videos_to_physi.md) | 项目页与论文深读同页 |

```mermaid
flowchart LR
  A["上游开源项目主表"] --> B["本页导读表"]
  B --> C["官方仓库 / 项目页"]
  B --> D["本库详情页"]
  D --> E["方法 / 论文 / 工程交叉阅读"]
```

## 局限与使用边界

- **主表会更新，本页是快照式导读。** 上游新增或改名条目后，以主表原文为准，并回本页核对应链接是否仍成立。
- **右侧是「主阅读页」，不是唯一相关页。** 同一项目常还会出现在路线图、对比页或姊妹论文里；详情页的「关联页面」更完整。
- **开源状态以官方 README / 项目页为准。** 本库会摘要训练/推理/数据是否开放，但许可证与发布节奏可能变化。

## 一句话记忆

主表帮你**找到入口**；本库帮你**读懂机制**——先点左链看代码，再点右链读知识页。

## 关联页面

- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) — 上游知识库总览
- [开源运动控制项目结构化摘要](./open-source-motion-control-projects.md) — 另一份开源运动控制策展的方法地图
- [人形 RL 运动控制身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md)
- [运动小脑技术地图](../overview/humanoid-motion-cerebellum-technology-map.md)

## 参考来源

- [sources/repos/humanoid-motion-intelligence.md](../../sources/repos/humanoid-motion-intelligence.md)
- [开源项目主表（上游）](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md)

## 推荐继续阅读

- [上游开源项目主表](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md)
- [Humanoid Motion Intelligence 仓库](https://github.com/RealXiaoze/humanoid-motion-intelligence)
