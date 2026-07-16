---
type: overview
tags: [humanoid, rl, motion-control, survey, body-system-stack, vla, world-model]
status: complete
updated: 2026-07-16
related:
  - ./robot-training-stack-layers-technology-map.md
  - ./robot-world-models-training-loop-taxonomy.md
  - ./humanoid-motion-control-know-how.md
  - ../tasks/humanoid-locomotion.md
  - ../tasks/loco-manipulation.md
  - ../tasks/ultra-survey.md
  - ../methods/deepmimic.md
  - ../methods/sonic-motion-tracking.md
  - ../methods/beyondmimic.md
  - ../methods/any2track.md
  - ../methods/ams.md
  - ../methods/motion-retargeting-gmr.md
  - ../methods/neural-motion-retargeting-nmr.md
  - ../entities/paper-doorman-opening-sim2real-door.md
  - ../entities/paper-viral-humanoid-visual-sim2real.md
  - ../entities/paper-behavior-foundation-model-humanoid.md
  - ../concepts/behavior-foundation-model.md
  - ./bfm-41-papers-technology-map.md
  - ./humanoid-motion-cerebellum-technology-map.md
  - ./ego-9-papers-technology-map.md
  - ../entities/gr00t-wholebodycontrol.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
  - ../../sources/blogs/wechat_embodied_ai_lab_robot_training_stack_layers_2026.md
  - ../../sources/blogs/wechat_human_five_jason_peng_flexible_motion_skills.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/sites/wechat-embodied-ai-lab-humanoid-rl-motion-survey-2026-05-18.md
summary: "把 42 篇 humanoid RL 运动控制 / 移动操作论文整理成一套八层身体系统栈（数据 → 跟踪 → 控制 → 感知 → 接触 → 安全 → 任务接口 → 世界模型），并据此给出 6 个研究判断；核心主张：动作不是能力，动作在真实世界精细交互闭环里才是能力，VLA 调用是这层成熟后的结果，不是起点。"
---

# 人形机器人 RL 运动控制：身体系统栈视角

> **本页定位**：把一篇按论文堆栈整理的综述（42 篇 humanoid RL 控制工作）压缩成「按系统层组织」的检索表，便于跨页面查阅；不复述每篇论文细节，只保留**层间分工**与**研究判断**。

## 一句话观点

人形机器人真正难的不是「让动作做出来」，而是让动作进入真实世界的**精细交互闭环**——视觉、接触、力、负载、失败恢复都参与控制；VLA / 世界模型对身体的稳定调用，是这层能力成熟之后的下一阶段，不是起点。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 本栈多数运控论文的数据驱动训练范式 |
| VLA | Vision-Language-Action | 语言+视觉+动作统一策略，在身体 API 成熟后被调用 |
| BFM | Behavior Foundation Model | 身体级基础模型，封装走、平衡、接触、恢复等能力 |
| WBT | Whole-Body Tracking | 参考动作进入仿真并稳定跟踪的控制层 |
| GMR | General Motion Retargeting | 人体/视频动作重定向为机器人可执行参考 |
| AMP | Adversarial Motion Prior | 运动先验线，约束策略分布接近人类运动 |
| Sim2Real | Simulation to Real | 视觉/动力学从仿真到真机的迁移瓶颈 |

## 为什么按「系统栈」而不是「论文榜单」

单看 demo，人形机器人会跑、会跳、会踢球、会开门、会搬东西。把多篇论文摆在一起看，会发现真正缺的不是某一个动作，而是把动作稳定接入真实任务的**系统能力**。因此可以把问题分四个抽象层：

- **动作层**：物理可执行、能随状态调整，而不是只看「像不像参考」。
- **感知层**：视觉 / 深度 / 本体感知进入闭环，而不是「执行前看一眼目标」。
- **接触层**：能控制力 / 柔顺 / 负载 / 反作用，而不是「刚性追踪」。
- **任务层**：能组合成长任务，而后再谈被语言或 VLA 调用。

这四层进一步细化为下面的八层系统栈。

## 八层身体系统栈（与已有 wiki 页面的挂接）

> 同一篇工作可能横跨两层（例如 BeyondMimic 兼跨「跟踪控制」与「动作生成」）；下表只列其**主要站位**。已建独立 wiki 页面的工作以链接标出，其余仅列名字，作为后续 ingest 的候选。

| 层 | 关注 | 已有 wiki 链接 | 仅在源文中提及 |
|---|---|---|---|
| **1. 数据** | 人类动作 / 视频 / 遥操作 → 机器人可执行参考 | [GMR](../methods/motion-retargeting-gmr.md)、[NMR](../methods/neural-motion-retargeting-nmr.md) | OmniRetarget, GenMimic, HumanX, HDMI, H2O, OmniH2O, TWIST, TWIST2 |
| **2. 参考 / 跟踪控制** | 参考动作进入物理仿真、稳定跟踪、在线修正 | [DeepMimic](../methods/deepmimic.md)、[BeyondMimic](../methods/beyondmimic.md) | OmniTrack, Motion Gen+Tracking, Heracles |
| **3. 控制（通用 tracker / 身体基础模型）** | 多动作跟踪、抗扰、负载适应、恢复；身体级 foundation model | [SONIC](../methods/sonic-motion-tracking.md)、[Any2Track](../methods/any2track.md)、[AMS](../methods/ams.md)、[BFM 论文](../entities/paper-behavior-foundation-model-humanoid.md) | RGMT, OmniXtreme, HALO, PvP, Adaptive Humanoid Control |
| **4. 感知（视觉闭环）** | RGB / 深度进入动作闭环、sim-to-real | [VIRAL 论文](../entities/paper-viral-humanoid-visual-sim2real.md)、[DoorMan 论文](../entities/paper-doorman-opening-sim2real-door.md) | PHP, Deep Whole-body Parkour, Hiking in the Wild, ASAP, 视觉足球 |
| **5. 接触 / 柔顺** | 力 / 柔顺 / 对象动力学进入控制；接触安全 | [GentleHumanoid](../methods/gentlehumanoid-motion-tracking.md) | CHIP, HAIC, Thor |
| **6. 安全 / 失败恢复** | 跌倒、异常姿态恢复、冲击力管理 | — | SafeFall（强相关：[balance-recovery](../tasks/balance-recovery.md)） |
| **7. 任务接口 / VLA 调用** | 身体能力封装成 skill token / latent action / action chunk，供语言或 VLA 调用 | [GR00T-WholeBodyControl](../entities/gr00t-wholebodycontrol.md)、[ULTRA Survey](../tasks/ultra-survey.md)、[DAJI 预期关节意图](../entities/paper-daji-anticipatory-joint-intent.md) | SENTINEL, WholeBodyVLA, MetaWorld, BFM-Zero |
| **8. 世界模型** | 执行前预测接触后果、评估策略可行性 | [机器人世界模型：训练闭环与三线 taxonomy](./robot-world-models-training-loop-taxonomy.md) | Ego-Vision World Model, DreamDojo |

## 六个研究判断（沿用原文，但只保留可执行结论）

1. **动作库 ≠ 能力**：未来比拼的是「可重定向 / 物理一致 / 含交互信息」（视觉 + 接触 + 力 + 失败恢复）的高质量数据，不是动捕条数。
2. **视觉从识别变控制闭环**：身体动作 ↔ 视觉输入双向耦合 → **视觉 sim-to-real** 是核心瓶颈，参考 [VIRAL 论文](../entities/paper-viral-humanoid-visual-sim2real.md)、[DoorMan 论文](../entities/paper-doorman-opening-sim2real-door.md)。
3. **柔顺与力控决定数据质量**：控制器太硬 → 遥操采集擦白板 / 推车 / 抱人 / 搬箱子都不稳 → 上层 VLA 模型再大也学不好。
4. **失败恢复会成为严肃指标**：论文不能只展示成功视频，应量化「失败位置 / 冲击力 / 保护 / 异常姿态恢复 / 避免伤人」；与 [balance-recovery](../tasks/balance-recovery.md) 对应。
5. **VLA 调用是结果，不是起点**：稳定移动 → 精细全身交互 → 身体 API（skill token / latent action / 接触模式 / 短时 action chunk / 柔顺系数 / 视觉闭环目标 / 恢复策略），上层模型才能稳定调用。
6. **世界模型 = 上线前试运行**：价值在 action-conditioned rollout（预测接触后果、失败概率、长时程漂移），不在生成好看的未来视频。

## 与现有 wiki 的位置

- 本页是**总框架**，回答「这批论文整体在搭什么」；具体工程经验沉淀在 [humanoid-motion-control-know-how](./humanoid-motion-control-know-how.md)（传感器 / 电机 / 热管理 / 接触估计）。
- 单篇方法的方法学细节在 `wiki/methods/`（[DeepMimic](../methods/deepmimic.md)、[SONIC](../methods/sonic-motion-tracking.md)、[BeyondMimic](../methods/beyondmimic.md)、[Any2Track](../methods/any2track.md)、[AMS](../methods/ams.md)、[GMR](../methods/motion-retargeting-gmr.md)、[NMR](../methods/neural-motion-retargeting-nmr.md) 等）；跟踪局限与超越路径见 [Jason Peng 灵活运动技能学习](./jason-peng-flexible-motion-skill-learning.md)。
- 任务侧的统一控制讨论见 [ULTRA Survey](../tasks/ultra-survey.md)（统一多模态全身 loco-manipulation 控制器）。
- 单篇论文页：[DoorMan](../entities/paper-doorman-opening-sim2real-door.md)、[VIRAL](../entities/paper-viral-humanoid-visual-sim2real.md)、[BFM](../entities/paper-behavior-foundation-model-humanoid.md) 等。

## 关联页面

- [训练栈分层技术地图](./robot-training-stack-layers-technology-map.md) — 工具链六层（与本文「身体能力八层」互补）

- [人形 AMP 运动先验综述](./humanoid-amp-motion-prior-survey.md) — 19 篇 AMP / 运动先验论文的姊妹篇导航（与本文 42 篇栈互补）
- [运动小脑 64 篇技术地图](./humanoid-motion-cerebellum-technology-map.md) — 同一公众号 2026-06 姊妹篇：论文高度重叠，按「动作小脑」A–I 九组重框（**复用** 本文 `paper-hrl-stack-*` 等节点，不重复建页）
- [人形机器人运动控制 Know-How](./humanoid-motion-control-know-how.md) — 真实部署中的硬核工程经验（传感器 / 电机 / 热管理）
- [ULTRA Survey](../tasks/ultra-survey.md) — 统一多模态全身 loco-manipulation 控制器的综述视角
- [humanoid-locomotion](../tasks/humanoid-locomotion.md)、[loco-manipulation](../tasks/loco-manipulation.md)、[balance-recovery](../tasks/balance-recovery.md) — 任务侧入口
- [Query：人形运动跟踪方法选型](../queries/humanoid-motion-tracking-method-selection.md) — DeepMimic / AMP 家族 / 通用 tracker 的工程选型
- [DeepMimic](../methods/deepmimic.md)、[SONIC](../methods/sonic-motion-tracking.md)、[GentleHumanoid](../methods/gentlehumanoid-motion-tracking.md)、[BeyondMimic](../methods/beyondmimic.md)、[Any2Track](../methods/any2track.md)、[AMS](../methods/ams.md)、[GMR](../methods/motion-retargeting-gmr.md)、[NMR](../methods/neural-motion-retargeting-nmr.md) — 跟踪 / 控制层方法页
- [DoorMan 论文](../entities/paper-doorman-opening-sim2real-door.md)、[VIRAL 论文](../entities/paper-viral-humanoid-visual-sim2real.md)、[BFM 论文](../entities/paper-behavior-foundation-model-humanoid.md)、[GR00T-WholeBodyControl](../entities/gr00t-wholebodycontrol.md) — 视觉闭环 / 身体基础模型 / VLA 调用相关单篇

## Wiki 实体索引（站内详情页）

> 以下 42 篇均已升格为 `wiki/entities/` 详情页（可搜索、进图谱）。

| # | 论文 | 实体页 |
|---|------|--------|
| 01 | Retargeting Matters | [paper-hrl-stack-01-retargeting_matters.md](../entities/paper-hrl-stack-01-retargeting_matters.md) |
| 02 | Make Tracking Easy | [paper-hrl-stack-02-make_tracking_easy.md](../entities/paper-hrl-stack-02-make_tracking_easy.md) |
| 03 | OmniRetarget | [paper-hrl-stack-03-omniretarget.md](../entities/paper-hrl-stack-03-omniretarget.md) |
| 04 | From Generated Human Videos to Physically Plausible Robot Trajectories | [paper-hrl-stack-04-from_generated_human_videos_to_physi.md](../entities/paper-hrl-stack-04-from_generated_human_videos_to_physi.md) |
| 05 | HumanX | [paper-hrl-stack-05-humanx.md](../entities/paper-hrl-stack-05-humanx.md) |
| 06 | HDMI | [paper-hrl-stack-06-hdmi.md](../entities/paper-hrl-stack-06-hdmi.md) |
| 07 | Learning Human-to-Humanoid Real-Time Whole-Body Teleoperation | [paper-hrl-stack-07-learning_human_to_humanoid_real_time.md](../entities/paper-hrl-stack-07-learning_human_to_humanoid_real_time.md) |
| 08 | OmniH2O | [paper-hrl-stack-08-omnih2o.md](../entities/paper-hrl-stack-08-omnih2o.md) |
| 09 | TWIST | [paper-twist.md](../entities/paper-twist.md) |
| 10 | TWIST2 | [paper-twist2.md](../entities/paper-twist2.md) |
| 11 | DeepMimic | [deepmimic.md](../methods/deepmimic.md) |
| 12 | OmniTrack | [paper-hrl-stack-12-omnitrack.md](../entities/paper-hrl-stack-12-omnitrack.md) |
| 13 | Track Any Motions under Any Disturbances | [paper-opentrack.md](../entities/paper-opentrack.md) |
| 14 | Robust and Generalized Humanoid Motion Tracking | [paper-hrl-stack-14-robust_and_generalized_humanoid_moti.md](../entities/paper-hrl-stack-14-robust_and_generalized_humanoid_moti.md) |
| 15 | BeyondMimic | [beyondmimic.md](../methods/beyondmimic.md) |
| 16 | OmniXtreme | [paper-hrl-stack-16-omnixtreme.md](../entities/paper-hrl-stack-16-omnixtreme.md) |
| 17 | SONIC | [sonic-motion-tracking.md](../methods/sonic-motion-tracking.md) |
| 18 | Agility Meets Stability | [ams.md](../methods/ams.md) |
| 19 | BFM-Zero | [paper-bfm-zero.md](../entities/paper-bfm-zero.md) |
| 20 | PvP | [paper-hrl-stack-20-pvp.md](../entities/paper-hrl-stack-20-pvp.md) |
| 21 | Towards Adaptive Humanoid Control via Multi-Behavior Distillation and Reinforced Fine-Tuning | [paper-adaptive-humanoid-control.md](../entities/paper-adaptive-humanoid-control.md) |
| 22 | Perceptive Humanoid Parkour | [paper-hrl-stack-22-perceptive_humanoid_parkour.md](../entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md) |
| 23 | Deep Whole-body Parkour | [paper-deep-whole-body-parkour.md](../entities/paper-deep-whole-body-parkour.md) |
| 24 | Hiking in the Wild | [paper-hiking-in-the-wild.md](../entities/paper-hiking-in-the-wild.md) |
| 25 | ASAP | [paper-hrl-stack-25-asap.md](../entities/paper-hrl-stack-25-asap.md) |
| 26 | Learning Vision-Driven Reactive Soccer Skills for Humanoid Robots | [paper-hrl-stack-26-learning_vision_driven_reactive_socc.md](../entities/paper-hrl-stack-26-learning_vision_driven_reactive_socc.md) |
| 27 | Learning Whole-Body Humanoid Locomotion via Motion Generation and Motion Tracking（**全文消化** · arXiv:2604.17335） | [paper-hrl-stack-27-learning_whole_body_humanoid_locomot.md](../entities/paper-hrl-stack-27-learning_whole_body_humanoid_locomot.md) |
| 28 | VIRAL | [paper-viral-humanoid-visual-sim2real.md](../entities/paper-viral-humanoid-visual-sim2real.md) |
| 29 | Opening the Sim-to-Real Door for Humanoid Pixel-to-Action Policy Transfer | [paper-doorman-opening-sim2real-door.md](../entities/paper-doorman-opening-sim2real-door.md) |
| 30 | WholeBodyVLA | [paper-hrl-stack-30-wholebodyvla.md](../entities/paper-hrl-stack-30-wholebodyvla.md) |
| 31 | SENTINEL | [paper-sentinel.md](../entities/paper-sentinel.md) |
| 32 | MetaWorld | [paper-hrl-stack-32-metaworld.md](../entities/paper-hrl-stack-32-metaworld.md) |
| 33 | Ego-Vision World Model for Humanoid Contact Planning | [paper-hrl-stack-33-ego_vision_world_model_for_humanoid.md](../entities/paper-hrl-stack-33-ego_vision_world_model_for_humanoid.md) |
| 34 | GR00T N1 | [paper-hrl-stack-34-gr00t_n1.md](../entities/paper-hrl-stack-34-gr00t_n1.md) |
| 35 | DreamDojo | [paper-hrl-stack-35-dreamdojo.md](../entities/paper-hrl-stack-35-dreamdojo.md) |
| 36 | CHIP | [paper-hrl-stack-36-chip.md](../entities/paper-hrl-stack-36-chip.md) |
| 37 | GentleHumanoid | [gentlehumanoid-motion-tracking.md](../methods/gentlehumanoid-motion-tracking.md) |
| 38 | HAIC | [haic.md](../methods/haic.md) |
| 39 | Closing Sim-to-Real Gap for Heavy-loaded Humanoid Agile Motion Skills via Differentiable Simulation | [paper-hrl-stack-39-closing_sim_to_real_gap_for_heavy_lo.md](../entities/paper-hrl-stack-39-closing_sim_to_real_gap_for_heavy_lo.md) |
| 40 | Heracles | [paper-heracles-humanoid-diffusion.md](../entities/paper-heracles-humanoid-diffusion.md) |
| 41 | SafeFall | [paper-hrl-stack-41-safefall.md](../entities/paper-hrl-stack-41-safefall.md) |
| 42 | Thor | [paper-hrl-stack-42-thor.md](../entities/paper-hrl-stack-42-thor.md) |

## 局限

- 上表 42 篇均已各有 `wiki/entities/` **编译实体页**（含 `paper-hrl-stack-*` 与 TWIST/SONIC 等别名节点）；部分工作另有 `wiki/methods/` 或独立实体 **深读页**（如 GMR、DeepMimic、SONIC）。实体页已补充一句话定义、核心机制与常见误区；方法细节仍以 arXiv / 项目页为准。
- 原文「八层栈」是作者基于 42 篇论文归纳出的**叙述框架**，不是已有共识；不同综述可能划分不同（例如把「跟踪控制」并入「控制」，或把「任务接口」拆为「skill abstraction」+「VLA 接口」）。把本表当作**导航图**而非分类公理。

## 参考来源

- [两万字长文，读懂人形机器人强化学习运动控制：42 篇论文搭起的算法圣经（微信公众号原文）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [具身智能研究室 · 人形机器人 RL 运动控制 42 篇综述（仓库内归档）](../../sources/sites/wechat-embodied-ai-lab-humanoid-rl-motion-survey-2026-05-18.md)
