---
type: overview
tags: [loco-manipulation, humanoid, category-hub, survey]
status: complete
updated: 2026-11-14
original:
  - ../../sources/blogs/wechat_humanoid_loco_manip_161_survey_reading_hub161.md
  - ../../sources/papers/gr00t_n1_arxiv_2503_14734.md
summary: "人形 Loco-Manip 161 篇 · 09 人形 VLA、世界模型与通用操作（15 篇）— 视觉、语言、动作和世界建模接到执行层。"
related:
  - ./humanoid-loco-manip-161-papers-technology-map.md
  - ../methods/vla.md
  - ../concepts/world-action-models.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
  - ../../sources/blogs/wechat_humanoid_loco_manip_161_survey_reading_hub161.md
  - ../../sources/papers/humanoid_loco_manip_161_catalog.md
  - ../../sources/papers/gr00t_n1_arxiv_2503_14734.md
---

# Loco-Manip 161 分类 09：人形 VLA、世界模型与通用操作

> **分类页**：**09 人形 VLA、世界模型与通用操作**；总地图见 [人形 Loco-Manip 161 篇技术地图](./humanoid-loco-manip-161-papers-technology-map.md)。

## 一句话定义

把高维视觉–语言语义推理与世界状态预测（VLA / WM）接入人形双臂/全身物理控制器的 15 篇代表工作分类枢纽页。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |
| WM | World Model | 学习环境动态以供想象/规划的世界模型 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |

## 核心问题

视觉、语言、动作和世界建模接到执行层

## 本组论文（15 篇）

| # | 工作 | Wiki 实体 |
|---|------|-----------|
| 146 | AgiBot World Colosseo | [paper-loco-manip-161-146-agibot-world-colosseo](../entities/agibot-world-2026.md) |
| 147 | DIAL | [paper-loco-manip-161-147-dial](../methods/dial-instruction-augmentation.md) |
| 148 | GR00T N1 | [paper-hrl-stack-34-gr00t_n1](../entities/paper-hrl-stack-34-gr00t_n1.md) |
| 149 | Gemini Robotics | [paper-loco-manip-161-149-gemini-robotics](../entities/gemini-robotics.md) |
| 150 | Genie Envisioner | [paper-loco-manip-161-150-genie-envisioner](../entities/ge-sim-2.md) |
| 151 | LEGS | [paper-loco-manip-161-151-legs](../entities/paper-legs-embodied-gaussian-splatting-vla.md) |
| 152 | LingBot-VLA | [LingBot-VLA](../entities/lingbot-vla.md)（1.0；[2.0](../entities/lingbot-vla-v2.md)） |
| 153 | MetaWorld-X | [paper-loco-manip-161-153-metaworld-x](../entities/paper-hrl-stack-32-metaworld.md) |
| 154 | OpenHLM | [paper-loco-manip-161-154-openhlm](../entities/paper-loco-manip-161-154-openhlm.md) |
| 155 | PhysiFlow | [paper-loco-manip-161-155-physiflow](../entities/paper-loco-manip-161-155-physiflow.md) |
| 156 | Psi0 | [paper-loco-manip-161-156-psi0](../entities/paper-loco-manip-161-156-psi0.md) |
| 157 | REFINE-DP | [paper-loco-manip-161-157-refine-dp](../entities/paper-loco-manip-161-157-refine-dp.md) |
| 158 | ROVE | [paper-loco-manip-161-158-rove](../entities/paper-rove-humanoid-vla-intervention.md) |
| 159 | 世界行动模型是零样本策略 | [paper-loco-manip-161-159-n159](../entities/paper-loco-manip-161-159-n159.md) |
| 160 | 通过主动空间大脑和通用动作小脑进行人形全身操作 | [paper-loco-manip-161-160-n160](../entities/paper-loco-manip-161-160-n160.md) |

## 为什么重要

- **语义高层与物理底层合流**：早期的全身运动控制解决“站稳并到达”，但缺乏对开放场景的语义理解和长程任务规划；本组工作补齐视觉–语言–动作和未来推演接口。
- **跨本体与数据供给主阵地**：覆盖 AgiBot World 等百万真机轨、Genie Envisioner 等视频世界模型及 GR00T N1 / LingBot-VLA / Gemini Robotics 等通用双臂–全身策略。

## 核心原理

- **System 2 + System 1 快慢解耦**：如 GR00T N1（arXiv:2503.14734）将预训练 VLM（低频语义/状态表征）与 flow-matching / diffusion 动作头（高频动作块生成）耦合成双系统闭环。
- **世界模型与执行耦合**：如 Genie Envisioner、PhysiFlow 等引入视频/潜空间未来状态预测或隐式动力学先验，提升复杂交互与长程规划鲁棒性。

## 工程实践

- **带宽与接口匹配**：高层 VLA / WM 输出典型为 5–30 Hz 的末端位姿或子目标轨迹，必须经底层全身控制（WBC）或高频运动小脑重采样至 50–500 Hz 关节指令，才能避免人形失稳。
- **按需求检索而非按榜单全读**：需要双臂–灵巧手真机数据集入口先查 AgiBot World Colosseo 与 DIAL；需要双系统架构与开源代码先查 GR00T N1 与 LingBot-VLA。

## 局限与风险

- **榜单/分类口径局限**：15 篇分组来源于 161 篇社区综述快照，部分闭源工作（如 Gemini Robotics）仅公开定性演示或有限指标，不可与开源权重直接等效复现。
- **全身耦合欠验证**：多数 VLA 模型仍以桌面或半身固定底座操作验证为主，行走–操作强动力学耦合下的安全性仍高度依赖底层 WBC 安全约束。

## 关联页面

- [人形 Loco-Manip 161 篇技术地图](./humanoid-loco-manip-161-papers-technology-map.md)
- [Loco-Manipulation 任务页](../tasks/loco-manipulation.md)
- [VLA 方法页](../methods/vla.md)
- [World Action Models](../concepts/world-action-models.md)

## 参考来源

- [wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- [wechat_humanoid_loco_manip_161_survey_reading_hub161.md](../../sources/blogs/wechat_humanoid_loco_manip_161_survey_reading_hub161.md) — 161 篇综述原文框架与分类边界核对
- [humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md)
- [gr00t_n1_arxiv_2503_14734.md](../../sources/papers/gr00t_n1_arxiv_2503_14734.md) — 人形双系统 VLA（System 2 VLM + System 1 DiT）机制核对

## 推荐继续阅读

- [运动小脑 64 篇技术地图](./humanoid-motion-cerebellum-technology-map.md)
