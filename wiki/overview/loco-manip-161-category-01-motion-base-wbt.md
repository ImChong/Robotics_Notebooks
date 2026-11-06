---
type: overview
tags: [loco-manipulation, humanoid, category-hub, survey]
status: complete
updated: 2026-09-18
summary: "人形 Loco-Manip 161 篇 · 01 运控基座与通用全身跟踪（31 篇）— 底层身体控制、运动跟踪、抗扰动与通用动作执行。"
original:
  - ../../sources/papers/beyondmimic_reading_arxiv_2508_08241.md
related:
  - ./humanoid-loco-manip-161-papers-technology-map.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
  - ../../sources/papers/humanoid_loco_manip_161_catalog.md
---

# Loco-Manip 161 分类 01：运控基座与通用全身跟踪

> **分类页**：**01 运控基座与通用全身跟踪**；总地图见 [人形 Loco-Manip 161 篇技术地图](./humanoid-loco-manip-161-papers-technology-map.md)。

## 一句话定义

161 篇人形 Loco-Manip 地图中的第 01 类分类页：收录 31 篇关于运控基座、运动跟踪、抗扰动与通用全身动作执行的工作，是上层操作与任务策略依赖的“身体底座”。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |
| WM | World Model | 学习环境动态以供想象/规划的世界模型 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |

## 为什么重要

上层 VLA、遥操作与任务规划最终都要落到全身关节指令上；底座的跟踪精度、抗扰动能力与动作覆盖面决定了上层能调用多少“动作原语”。本组以全身运动跟踪为主线（如 BeyondMimic、GMT、HOVER、TWIST 系列），是其它 Loco-Manip 分类的共同依赖。

## 核心问题

底层身体控制、运动跟踪、抗扰动与通用动作执行

## 本组论文（31 篇）

| # | 工作 | Wiki 实体 |
|---|------|-----------|
| 001 | Agility Meets Stability | [paper-loco-manip-161-001-agility-meets-stability](../methods/ams.md) |
| 002 | Any2Any | [paper-loco-manip-161-002-any2any](../entities/paper-any2any-cross-embodiment-wbt.md) |
| 003 | BFM-Zero | [paper-loco-manip-161-003-bfm-zero](../entities/paper-bfm-zero.md) |
| 004 | BeyondMimic | [paper-loco-manip-161-004-beyondmimic](../methods/beyondmimic.md) |
| 005 | CHIP | [paper-loco-manip-161-005-chip](../entities/paper-hrl-stack-36-chip.md) |
| 006 | CLONE | [paper-loco-manip-161-006-clone](../entities/paper-bfm-12-clone.md) |
| 007 | ExBody2 | [paper-loco-manip-161-007-exbody2](../entities/paper-loco-manip-161-007-exbody2.md) |
| 008 | FRoM-W1 | [paper-loco-manip-161-008-from-w1](../entities/paper-loco-manip-161-096-from-w1.md) |
| 009 | GMT | [paper-gmt](../entities/paper-gmt.md) · [161 索引](../entities/paper-gmt.md) |
| 010 | HOVER | [paper-loco-manip-161-010-hover](../entities/paper-bfm-14-hover.md) |
| 011 | HoloMotion | [paper-loco-manip-161-011-holomotion](../entities/holomotion.md) |
| 012 | HumanPlus | [paper-loco-manip-161-012-humanplus](../entities/paper-loco-manip-161-012-humanplus.md) |
| 013 | KungfuBot2 | [paper-loco-manip-161-013-kungfubot2](../entities/paper-notebook-kungfubot-2.md) |
| 014 | MOSAIC | [paper-loco-manip-161-014-mosaic](../entities/paper-loco-manip-161-014-mosaic.md) |
| 015 | Make Tracking Easy | [paper-loco-manip-161-015-make-tracking-easy](../entities/paper-hrl-stack-02-make_tracking_easy.md) |
| 016 | OmniH2O | [paper-loco-manip-161-016-omnih2o](../entities/paper-hrl-stack-08-omnih2o.md) |
| 017 | OmniRetarget | [paper-loco-manip-161-017-omniretarget](../entities/paper-hrl-stack-03-omniretarget.md) |
| 018 | Retargeting | [paper-loco-manip-161-018-retargeting](../entities/paper-hrl-stack-01-retargeting_matters.md) |
| 019 | SONIC | [sonic-motion-tracking](../methods/sonic-motion-tracking.md) |
| 020 | TWIST2 | [paper-loco-manip-161-020-twist2](../entities/paper-twist2.md) |
| 021 | TWIST | [paper-loco-manip-161-021-twist](../entities/paper-twist.md) |
| 022 | TextOp | [paper-loco-manip-161-022-textop](../entities/paper-loco-manip-161-022-textop.md) |
| 023 | UniAct | [paper-loco-manip-161-023-uniact](../entities/paper-loco-manip-161-023-uniact.md) |
| 024 | UniTracker | [paper-loco-manip-161-024-unitracker](../entities/paper-loco-manip-161-024-unitracker.md) |
| 025 | 人形操作接口 | [paper-loco-manip-161-025-n025](../entities/paper-loco-manip-161-025-n025.md) |
| 026 | 人形机器人富有表现力的全身控制 | [paper-loco-manip-161-026-n026](../entities/paper-loco-manip-161-026-n026.md) |
| 027 | 人形机器人行为基础模型 | [paper-loco-manip-161-027-n027](../entities/paper-loco-manip-161-027-n027.md) |
| 028 | 从专家到通用：走向人形机器人的通用全身控制 | [paper-loco-manip-161-028-n028](../entities/paper-loco-manip-161-028-n028.md) |
| 029 | 学习人对人的实时全身远程操作 | [paper-loco-manip-161-029-n029](../entities/paper-hrl-stack-07-learning_human_to_humanoid_real_time.md) |
| 030 | 跟踪任何干扰下的任何动作 | [paper-loco-manip-161-030-n030](../methods/any2track.md) |
| 031 | 鲁棒且通用的人形运动跟踪 | [paper-loco-manip-161-031-n031](../entities/paper-hrl-stack-14-robust_and_generalized_humanoid_moti.md) |

## 方法栈与代表思路

- **参考运动重定向 → 跟踪策略（RL）**：把人类动作重定向到人形，再用 RL 训练跟踪策略（见 018 Retargeting、017 OmniRetarget、004 BeyondMimic）。
- **以 BeyondMimic 为例（据原文通读归档）**：锚点相对的紧凑跟踪 MDP，奖励仅含位置/朝向/速度项与 3 项正则，用共享超参 PPO 训练；再蒸馏为潜空间扩散模型，测试时用 guidance 完成摇杆、航点等任务。
- **通用/统一跟踪与遥操作**：HOVER、GMT、TWIST/TWIST2、OmniH2O 等强调单一策略覆盖多种输入与动作（推断：具体设置以各论文页为准）。

## 工程实践

- 跟踪类策略的 PD 增益与动作尺度应与电机反射惯量（armature）一致，BeyondMimic 消融显示 armature 置零会导致过摆与自碰。
- 对难动作段做自适应起始采样，可显著加速训练收敛（同上原文消融）。
- 观测中加入历史帧并非总有益（BeyondMimic 消融中变差），需逐任务验证。

## 局限

- 本页是分类索引，未逐篇通读 31 篇原文；原文通读仅覆盖 BeyondMimic（见 frontmatter `original`）。
- 论文间设定（机型、数据、评测）不统一，条目之间不宜直接比较数值。
- BeyondMimic 自述局限：状态估计误差会传入生成轨迹，预测视野仅 0.64 s，不足以长程规划。

## 关联页面

- [人形 Loco-Manip 161 篇技术地图](./humanoid-loco-manip-161-papers-technology-map.md)
- [Loco-Manipulation 任务页](../tasks/loco-manipulation.md)

## 参考来源

- [wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- [humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md)

## 推荐继续阅读

- [运动小脑 64 篇技术地图](./humanoid-motion-cerebellum-technology-map.md)
