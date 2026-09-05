---
type: method
title: ZEST (Zero-shot Embodied Skill Transfer)
tags: [robot-learning, humanoid, locomotion, atlas, sim2real, multi-contact, rai-institute, boston-dynamics]
status: complete
summary: "ZEST（Science Robotics 2026）用自适应采样与辅助扳手课程，把 MoCap / 视频 / 动画变成 Atlas、G1、Spot 上的零样本高动态技能。论文实体见 paper-zest；确认未开源。"
updated: 2026-09-05
related:
  - ../entities/paper-zest.md
  - ./mtrg-reference-goal-driven-rl.md
  - ./hil-hybrid-imitation-learning.md
  - ../entities/paper-hil-hybrid-imitation-learning.md
  - ../comparisons/hil-vs-mtrg-vs-zest-parkour-imitation.md
  - ../comparisons/zest-vs-sonic-vs-vision-soccer.md
  - ../concepts/curriculum-learning.md
sources:
  - ../../sources/papers/zest.md
  - ../../sources/blogs/wechat_embodied_ai_lab_scirobotics_three_humanoid_papers_2026.md
---

# ZEST (Zero-shot Embodied Skill Transfer)

**ZEST** 是 RAI Institute 与 Boston Dynamics 的统一具身技能迁移配方：用强化学习把 MoCap、单目视频（ViCap）和关键帧动画变成高动态、多接触全身技能，并在全电 Atlas、Unitree G1、Spot 上**零样本**部署。期刊标题为 *Embodied skill transfer for locomotion control*（[*Science Robotics* 11(117)](https://doi.org/10.1126/scirobotics.aec7695)）；机制、表格与开源结论见 [论文实体](../entities/paper-zest.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ZEST | Zero-shot Embodied Skill Transfer | 本方法：异构参考 → 单阶段 RL → 硬件零样本 |
| ViCap | Video-Captured motion | 手持视频重建的参考，相对 MoCap 更脏 |
| RSI | Reference State Initialization | 按参考相位 reset；本文按失败率自适应 |
| PLA | Parallel-Linkage Actuator | 闭链驱动；名义电枢决定 PD 增益 |
| MDP | Markov Decision Process | 部署接口被压到本体 + 下一步参考 |
| G1 | Unitree G1 Humanoid | RAI 侧完成的小型人形验证平台 |

## 为什么重要

工业侧第一次把「膝/肘/躯干贴地」的多接触技能稳定做到全尺寸 Atlas，同时证明同一套极简接口可以跨到 G1 和 Spot。它把运动数据当物理正则，而不是再叠一层判别器或接触计划。

## 主要技术路线

```text
异构运动数据 (MoCap / ViCap / Animation)
          ↓
  运动学重定向（无接触标签）
          ↓
  物理正则化 RL + 自适应 RSI
          ↓
  辅助扳手自动课程（β → 0）
          ↓
  极简部署（无历史窗 / 无估计器）
          ↓
     零样本（Atlas / G1 / Spot）
```

## 关键技术

### 1. 自适应采样

长轨迹切 bin，用失败率 EMA 衡量难度，categorical 采样偏向难点，并留地板概率防止灾难性遗忘。乒乓球等长时程技能点名依赖它。

### 2. 虚拟辅助扳手

高动态（空翻、侧手翻）早期容易立刻终止。训练时在基座上施加模型基辅助力，幅度由同一套失败水平调制，跟踪上来后衰减到零。简单步态可以弱化或关掉。

### 3. 极简部署 MDP

- **输入：** 当前本体（IMU 角速度、投影重力、关节、上一动作）+ **下一步**参考。
- **动作：** 残差关节目标叠到参考，再进与仿真相同的 PD。
- **不要：** 接触标签、状态估计器、观测/参考长窗、任务专用奖励。

### 4. 闭链执行器与增益

膝/踝/腰 PLA 用名义构型电枢近似，再按临界阻尼选 \(K_p,K_d\)。这是 Sim2Real 的主工程，不是网络结构。

## 性能表现

- **Atlas MoCap：** 战术爬行、地板舞、前滚、侧手翻；走/跑出现跟脚滚转与近满伸膝。
- **ViCap：** Atlas 舞段/踢球；G1 芭蕾与爬箱（不给箱子位姿，5/5 重复）。
- **Spot 动画：** 连续后空翻、滚桶（IMU 饱和仍完成）。
- **vs BD MPC：** 干净步行接近；慢跑/侧手翻与多接触技能 RL 明显更能做完。

## 与 HIL / MTRG 的关系（同作者脉络）

| 工作 | 焦点 | 参考在部署时的角色 |
|------|------|-------------------|
| [HIL](./hil-hybrid-imitation-learning.md) | 物理角色跑酷动画 | tracking + AMP 并行；仿真角色 |
| **ZEST** | 多源异构模仿 → 硬件零样本 | **下一步参考**作为策略输入 |
| [MTRG](./mtrg-reference-goal-driven-rl.md) | G1 箱式跑酷 OOD 泛化 | **仅 goal**；参考只参与训练奖励 |

MTRG 复用 ZEST 的 assistive-wrench \(\lambda\) 课程；beyond-nominal 下相对「ZEST mocap + 参考输入跟踪」成功率更高（MTRG Table I：walk-jump 0.62 vs 0.17）。

## 关联页面

- [ZEST 论文实体](../entities/paper-zest.md) — Science Robotics 评测、消融与开源结论
- [MTRG](./mtrg-reference-goal-driven-rl.md) — 参考塑形 + goal 部署
- [HIL](./hil-hybrid-imitation-learning.md) — tracking + AMP（角色动画）
- [HIL 论文实体](../entities/paper-hil-hybrid-imitation-learning.md) — TOG 2026 评测与开源边界
- [HIL vs MTRG vs ZEST](../comparisons/hil-vs-mtrg-vs-zest-parkour-imitation.md)
- [ZEST vs SONIC vs 视觉足球](../comparisons/zest-vs-sonic-vs-vision-soccer.md) — SciRob 同期三层对照
- [Curriculum Learning](../concepts/curriculum-learning.md)
- [EFGCL](./efgcl.md) — 学术侧辅助力课程
- [Sim2Real](../concepts/sim2real.md)
- [DeepMimic](./deepmimic.md)
- [VideoMimic](../entities/videomimic.md)
- [Boston Dynamics](../entities/boston-dynamics.md) / [G1](../entities/unitree-g1.md)
- [X-Morph](../entities/paper-xmorph.md) — 另一条人体运动作非人形先验的管线

## 参考来源

- [ZEST / Embodied skill transfer 归档](../../sources/papers/zest.md)
- [同期三篇层级读法](../../sources/blogs/wechat_embodied_ai_lab_scirobotics_three_humanoid_papers_2026.md)
- [论文实体](../entities/paper-zest.md)
- [arXiv:2602.00401](https://arxiv.org/abs/2602.00401)
- [Science Robotics DOI](https://doi.org/10.1126/scirobotics.aec7695)
- [机器人论文阅读笔记：ZEST](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/ZEST__Zero-shot_Embodied_Skill_Transfer_for_Athletic_Robot_Control/ZEST__Zero-shot_Embodied_Skill_Transfer_for_Athletic_Robot_Control.html)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2602.00401)
- [MTRG 论文](https://arxiv.org/abs/2602.20375)
