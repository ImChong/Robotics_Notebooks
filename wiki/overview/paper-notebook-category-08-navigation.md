---
type: overview
tags: [humanoid-paper-notebooks, paper-index, overview]
status: complete
updated: 2026-11-21
related:
  - ./humanoid-paper-notebooks-index.md
  - ../../roadmap/depth-navigation.md
  - ../tasks/vision-language-navigation.md
sources:
  - ../../sources/repos/robot_learning_paper_notebooks_readme_reading_cat08.md
  - ../../sources/sites/humanoid-robot-learning-paper-notebooks.md
original:
  - ../../sources/repos/robot_learning_paper_notebooks_readme_reading_cat08.md
summary: "Paper Notebooks 分类 08：人形与足式导航论文深读索引，涵盖局部分层避障、扩散/流策略、人类第一视角数据与 VLN。"
---

# Paper Notebooks · Navigation

## 一句话定义

**Paper Notebooks · Navigation（`08_Navigation`）** 是 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 中专门索引 **人形与足式导航** 的主题分类枢纽，串联底层局部控制、扩散轨迹先验、第一视角记忆与视觉-语言导航（VLN）。

**Navigation**（`08_Navigation`）是 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 主页面的第 **08** 类，与上游仓库 `papers/08_Navigation` 目录（当前 28 篇专题笔记）保持对应。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- **导航与运控的桥梁：** 不同于纯平面小车导航，人形与四足导航必须同时应对窄障姿态调整、三维地形落脚稳定以及长程视觉语义推理。
- **技术脉络清晰对照：** 同一索引页横向收纳了规则/RL 分层控制（[FocusNav](../entities/paper-notebook-focusnav.md) / [Gallant](../entities/paper-loco-manip-161-137-gallant.md)）、多模态轨迹扩散（[NoMaD](../entities/paper-notebook-nomad-goal-masked-diffusion-policies-for-navigat.md) / [NavDP](../entities/paper-notebook-navdp-learning-sim-to-real-navigation-diffusion.md)）和全身 VLA（[NaVILA](../entities/paper-notebook-navila-legged-robot-vision-language-action-model.md) / [HumanoidVLN](../entities/paper-humanoidvln.md)）。

## 核心机制与本类论文索引

```mermaid
flowchart LR
    local["局部地形与姿态控制<br/>FocusNav / Gallant / GuideWalk"] --> diff["扩散与第一视角轨迹先验<br/>NoMaD / NavDP / EgoNav / LookOut"]
    diff --> vln["语义占用与视觉-语言导航<br/>NaVILA / HumanoidVLN / LOVON"]
```

## 本类论文索引

- [ARMOR](../entities/paper-notebook-armor-egocentric-perception-for-humanoid-robot-c.md) — 待深读
- [EgoActor](../entities/paper-notebook-egoactor.md) — [深读笔记](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/08_Navigation/EgoActor__Grounding_Task_Planning_into_Spatial-aware_Egocentric_Actions_for_Hum/EgoActor__Grounding_Task_Planning_into_Spatial-aware_Egocentric_Actions_for_Hum.html)
- [EgoNav](../entities/paper-notebook-egonav.md) — [深读笔记](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/08_Navigation/EgoNav__Learning_Humanoid_Navigation_from_Human_Data/EgoNav__Learning_Humanoid_Navigation_from_Human_Data.html)
- [FocusNav](../entities/paper-notebook-focusnav.md) — [深读笔记](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/08_Navigation/FocusNav__Spatial_Selective_Attention_with_Waypoint_Guidance_for_Humanoid_Local/FocusNav__Spatial_Selective_Attention_with_Waypoint_Guidance_for_Humanoid_Local.html)
- [Gallant](../entities/paper-loco-manip-161-137-gallant.md) — 见 wiki 实体页
- [GuideWalk](../entities/paper-motion-cerebellum-guidewalk.md) — 见 wiki 实体页
- [Hand-Eye Autonomous Delivery](../entities/paper-notebook-hand-eye-autonomous-delivery-learning-humanoid-n.md) — [深读笔记](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/08_Navigation/HEAD__Hand-Eye_Autonomous_Delivery_Humanoid_Navigation_Locomotion_and_Reaching/HEAD__Hand-Eye_Autonomous_Delivery_Humanoid_Navigation_Locomotion_and_Reaching.html)
- [Humanoid Occupancy](../entities/paper-notebook-humanoid-occupancy-enabling-a-generalized-multim.md) — [深读笔记](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/08_Navigation/Humanoid_Occupancy__Generalized_Multimodal_Occupancy_Perception_System/Humanoid_Occupancy__Generalized_Multimodal_Occupancy_Perception_System.html)
- [HumanoidPano](../entities/paper-notebook-humanoidpano-hybrid-spherical-panoramic-lidar-cr.md) — 待深读
- [INTENTION](../entities/paper-notebook-intention-inferring-tendencies-of-humanoid-robot.md) — [深读笔记](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/08_Navigation/INTENTION__Inferring_Tendencies_of_Humanoid_Robot_Motion_Through_Interactive_Int/INTENTION__Inferring_Tendencies_of_Humanoid_Robot_Motion_Through_Interactive_Int.html)
- [LOVON](../entities/paper-notebook-lovon-legged-open-vocabulary-object-navigator.md) — [深读笔记](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/08_Navigation/LOVON__Legged_Open-Vocabulary_Object_Navigator/LOVON__Legged_Open-Vocabulary_Object_Navigator.html)
- [Learning Social Navigation from Positive and Negative Demonstrations and Rule-Based Specifications](../entities/paper-notebook-learning-social-navigation-from-positive-and-neg.md) — [深读笔记](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/08_Navigation/Learning_Social_Navigation_from_Positive_and_Negative_Demonstrations_and_Rule-Based_Specifications/Learning_Social_Navigation_from_Positive_and_Negative_Demonstrations_and_Rule-Based_Specifications.html)
- [LookOut](../entities/paper-notebook-lookout.md) — [深读笔记](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/08_Navigation/LookOut__Real-World_Humanoid_Egocentric_Navigation/LookOut__Real-World_Humanoid_Egocentric_Navigation.html)
- [NaVILA](../entities/paper-notebook-navila-legged-robot-vision-language-action-model.md) — [深读笔记](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/08_Navigation/NaVILA_Legged_Robot_Vision-Language-Action_Model_for_Navigation/NaVILA_Legged_Robot_Vision-Language-Action_Model_for_Navigation.html)
- [NavDP](../entities/paper-notebook-navdp-learning-sim-to-real-navigation-diffusion.md) — [深读笔记](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/08_Navigation/NavDP__Learning_Sim-to-Real_Navigation_Diffusion_Policy/NavDP__Learning_Sim-to-Real_Navigation_Diffusion_Policy.html)
- [NoMaD](../entities/paper-notebook-nomad-goal-masked-diffusion-policies-for-navigat.md) — wiki 详细实体（Paper Notebooks 深读待撰）
- [Quantum deep reinforcement learning for humanoid robot navigation task](../entities/paper-notebook-quantum-deep-rl-for-humanoid-robot-navigation.md) — [深读笔记](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/08_Navigation/Quantum_Deep_RL_for_Humanoid_Robot_Navigation/Quantum_Deep_RL_for_Humanoid_Robot_Navigation.html)
- [Reinforcement Learning with Data Bootstrapping for Dynamic Subgoal Pursuit in Humanoid Robot Navigation](../entities/paper-notebook-rl-with-data-bootstrapping-for-dynamic-subgoal-p.md) — [深读笔记](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/08_Navigation/RL_with_Data_Bootstrapping_for_Dynamic_Subgoal_Pursuit_in_Humanoid_Navigation/RL_with_Data_Bootstrapping_for_Dynamic_Subgoal_Pursuit_in_Humanoid_Navigation.html)
- [STATE-NAV](../tasks/vision-language-navigation.md) — 见 wiki 实体页

## 工程实践

- **按技术代际跳转阅读：** 关注传统/RL 分层落脚稳定优先看 `FocusNav` / `GuideWalk`；关注跨本体可复现局部策略优先看已开源的 `NoMaD` 与 `NavDP`；关注自然语言长程任务看 `NaVILA` 与 [导航纵深路线](../../roadmap/depth-navigation.md)。
- **口径核对：** 本站首页按大类展示已公开发表精读数，上游仓库 `papers/08_Navigation` 目前共收录 28 篇子目录；部分条目在本 wiki 内尚标“待深读”，具体以各实体页正文及 `sources/` 核对为准。

## 局限与常见误区

- **误区：把导航分类当成底层步态全集。** 底层全身跟踪（HOVER / SONIC / AMP）归于 `04_Loco-Manipulation` 与相关运控章节，本分类聚焦“去往何处与如何避障规划”的高/中层接口。
- **局限：** 作为导航枢纽索引，具体模型公式、超参数与评测细节仍分布在各篇 `wiki/entities/paper-*.md` 节点中。

## 关联页面

- 总索引：[humanoid-paper-notebooks-index.md](./humanoid-paper-notebooks-index.md)
- 纵深学习路线：[导航纵深路线](../../roadmap/depth-navigation.md)
- 任务语境：[视觉–语言导航（VLN）](../tasks/vision-language-navigation.md)
- 笔记主页：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html>

## 参考来源

- [上游 `08_Navigation` 清单通读归档](../../sources/repos/robot_learning_paper_notebooks_readme_reading_cat08.md)
- [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html)
- 站点来源归档：[humanoid-robot-learning-paper-notebooks](../../sources/sites/humanoid-robot-learning-paper-notebooks.md)
- 本类论文来源笔记（节选）：[EgoNav：从人类数据学习人形导航](../../sources/papers/humanoid_pnb_egonav.md)、[ARMOR：人形避障与运动规划的第一人称感知](../../sources/papers/humanoid_pnb_armor-egocentric-perception-for-humanoid-robot-c.md)、[Gallant](../../sources/papers/loco_manip_161_survey_137_gallant.md)

## 推荐继续阅读

- [机器人论文阅读笔记总站](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html)
