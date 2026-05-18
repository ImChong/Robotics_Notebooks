---
type: overview
tags: [humanoid, rl, motion-control, survey, body-system-stack, vla, world-model]
status: complete
updated: 2026-05-18
related:
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
  - ../entities/gr00t-wholebodycontrol.md
sources:
  - ../../sources/sites/wechat-embodied-ai-lab-humanoid-rl-motion-survey-2026-05-18.md
summary: "把 42 篇 humanoid RL 运动控制 / 移动操作论文整理成一套八层身体系统栈（数据 → 跟踪 → 控制 → 感知 → 接触 → 安全 → 任务接口 → 世界模型），并据此给出 6 个研究判断；核心主张：动作不是能力，动作在真实世界精细交互闭环里才是能力，VLA 调用是这层成熟后的结果，不是起点。"
---

# 人形机器人 RL 运动控制：身体系统栈视角

> **本页定位**：把一篇按论文堆栈整理的综述（42 篇 humanoid RL 控制工作）压缩成「按系统层组织」的检索表，便于跨页面查阅；不复述每篇论文细节，只保留**层间分工**与**研究判断**。

## 一句话观点

人形机器人真正难的不是「让动作做出来」，而是让动作进入真实世界的**精细交互闭环**——视觉、接触、力、负载、失败恢复都参与控制；VLA / 世界模型对身体的稳定调用，是这层能力成熟之后的下一阶段，不是起点。

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
| **5. 接触 / 柔顺** | 力 / 柔顺 / 对象动力学进入控制；接触安全 | — | CHIP, GentleHumanoid, HAIC, Thor |
| **6. 安全 / 失败恢复** | 跌倒、异常姿态恢复、冲击力管理 | — | SafeFall（强相关：[balance-recovery](../tasks/balance-recovery.md)） |
| **7. 任务接口 / VLA 调用** | 身体能力封装成 skill token / latent action / action chunk，供语言或 VLA 调用 | [GR00T-WholeBodyControl](../entities/gr00t-wholebodycontrol.md)、[ULTRA Survey](../tasks/ultra-survey.md) | SENTINEL, WholeBodyVLA, MetaWorld, BFM-Zero |
| **8. 世界模型** | 执行前预测接触后果、评估策略可行性 | — | Ego-Vision World Model, DreamDojo |

## 六个研究判断（沿用原文，但只保留可执行结论）

1. **动作库 ≠ 能力**：未来比拼的是「可重定向 / 物理一致 / 含交互信息」（视觉 + 接触 + 力 + 失败恢复）的高质量数据，不是动捕条数。
2. **视觉从识别变控制闭环**：身体动作 ↔ 视觉输入双向耦合 → **视觉 sim-to-real** 是核心瓶颈，参考 [VIRAL 论文](../entities/paper-viral-humanoid-visual-sim2real.md)、[DoorMan 论文](../entities/paper-doorman-opening-sim2real-door.md)。
3. **柔顺与力控决定数据质量**：控制器太硬 → 遥操采集擦白板 / 推车 / 抱人 / 搬箱子都不稳 → 上层 VLA 模型再大也学不好。
4. **失败恢复会成为严肃指标**：论文不能只展示成功视频，应量化「失败位置 / 冲击力 / 保护 / 异常姿态恢复 / 避免伤人」；与 [balance-recovery](../tasks/balance-recovery.md) 对应。
5. **VLA 调用是结果，不是起点**：稳定移动 → 精细全身交互 → 身体 API（skill token / latent action / 接触模式 / 短时 action chunk / 柔顺系数 / 视觉闭环目标 / 恢复策略），上层模型才能稳定调用。
6. **世界模型 = 上线前试运行**：价值在 action-conditioned rollout（预测接触后果、失败概率、长时程漂移），不在生成好看的未来视频。

## 与现有 wiki 的位置

- 本页是**总框架**，回答「这批论文整体在搭什么」；具体工程经验沉淀在 [humanoid-motion-control-know-how](./humanoid-motion-control-know-how.md)（传感器 / 电机 / 热管理 / 接触估计）。
- 单篇方法的方法学细节在 `wiki/methods/`（[DeepMimic](../methods/deepmimic.md)、[SONIC](../methods/sonic-motion-tracking.md)、[BeyondMimic](../methods/beyondmimic.md)、[Any2Track](../methods/any2track.md)、[AMS](../methods/ams.md)、[GMR](../methods/motion-retargeting-gmr.md)、[NMR](../methods/neural-motion-retargeting-nmr.md) 等）。
- 任务侧的统一控制讨论见 [ULTRA Survey](../tasks/ultra-survey.md)（统一多模态全身 loco-manipulation 控制器）。
- 单篇论文页：[DoorMan](../entities/paper-doorman-opening-sim2real-door.md)、[VIRAL](../entities/paper-viral-humanoid-visual-sim2real.md)、[BFM](../entities/paper-behavior-foundation-model-humanoid.md) 等。

## 关联页面

- [人形机器人运动控制 Know-How](./humanoid-motion-control-know-how.md) — 真实部署中的硬核工程经验（传感器 / 电机 / 热管理）
- [ULTRA Survey](../tasks/ultra-survey.md) — 统一多模态全身 loco-manipulation 控制器的综述视角
- [humanoid-locomotion](../tasks/humanoid-locomotion.md)、[loco-manipulation](../tasks/loco-manipulation.md)、[balance-recovery](../tasks/balance-recovery.md) — 任务侧入口
- [DeepMimic](../methods/deepmimic.md)、[SONIC](../methods/sonic-motion-tracking.md)、[BeyondMimic](../methods/beyondmimic.md)、[Any2Track](../methods/any2track.md)、[AMS](../methods/ams.md)、[GMR](../methods/motion-retargeting-gmr.md)、[NMR](../methods/neural-motion-retargeting-nmr.md) — 跟踪 / 控制层方法页
- [DoorMan 论文](../entities/paper-doorman-opening-sim2real-door.md)、[VIRAL 论文](../entities/paper-viral-humanoid-visual-sim2real.md)、[BFM 论文](../entities/paper-behavior-foundation-model-humanoid.md)、[GR00T-WholeBodyControl](../entities/gr00t-wholebodycontrol.md) — 视觉闭环 / 身体基础模型 / VLA 调用相关单篇

## 局限

- 表格中「仅在源文中提及」一栏的工作（OmniRetarget, RGMT, OmniXtreme, HALO, Thor, SafeFall, SENTINEL, WholeBodyVLA, DreamDojo 等）**尚未在本仓库建立单页**；本 overview 只引用其分类位置，不复述其方法细节。后续如要为某篇升格 entity / method，应回到该论文 arXiv / 项目主页另起 sources 条目，不以本公众号为唯一来源。
- 原文「八层栈」是作者基于 42 篇论文归纳出的**叙述框架**，不是已有共识；不同综述可能划分不同（例如把「跟踪控制」并入「控制」，或把「任务接口」拆为「skill abstraction」+「VLA 接口」）。把本表当作**导航图**而非分类公理。

## 参考来源

- [两万字长文，读懂人形机器人强化学习运动控制：42 篇论文搭起的算法圣经（微信公众号原文）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [具身智能研究室 · 人形机器人 RL 运动控制 42 篇综述（仓库内归档）](../../sources/sites/wechat-embodied-ai-lab-humanoid-rl-motion-survey-2026-05-18.md)
