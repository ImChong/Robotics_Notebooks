---
type: overview
tags: [overview, survey, failure-recovery, navigation, humanoid, tactile, benchmark, technology-map]
status: complete
updated: 2026-11-10
related:
  - embodied-station-2026-11-09-29-papers-technology-map.md
  - embodied-station-2026-11-06-20-papers-technology-map.md
  - ../methods/vla.md
sources:
  - ../../sources/blogs/wechat_embodied_station_22_papers_a2meas_tpsastar_2026-11-10.md
original:
  - ../../sources/blogs/wechat_embodied_station_22_papers_a2meas_tpsastar_2026-11-10.md
summary: "具身智能小站 2026-11-10 的 22 篇论文阅读坐标：失败诊断与纠错接口、导航规划与安全控制、跨形态与人形、触觉与工具使用、数据集与人机交互；每条指向唯一详情节点（新建 20 / 复用 2），本库逐篇通读新建节点并标出小站说法与论文的出入。"
---

# 失败恢复、安全规划与接触交互：22 篇论文阅读坐标

> **本页定位**：为 [具身智能小站 · 22 篇](https://mp.weixin.qq.com/s/Ugm4meP1cQm-R1omnGtpyQ)（2026-11-10）提供阅读坐标，并把 22 条论文逐条指向站内唯一详情节点。

## 一句话定义

具身智能小站 11-10 的 22 篇论文更新（1 篇深读、2 篇跟进、19 篇扫读）的阅读坐标页。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉语言动作模型 |
| RL | Reinforcement Learning | 强化学习 |
| BC | Behavior Cloning | 行为克隆 |
| CBF | Control Barrier Function | 控制屏障函数，用于安全滤波 |
| NMPC | Nonlinear Model Predictive Control | 非线性模型预测控制 |
| HRI | Human-Robot Interaction | 人机交互 |

## 为什么重要

- 小站每条只有一句话判断；本页让每条落到可追溯的详情节点，节点里有逐篇通读论文后的核对结论。
- **22 篇里真正用强化学习训练策略的是少数**：新建节点读过的有 DexWriting（BC 加 PPO）、Grasp-TAGS（PPO 加评论家蒸馏）、SUGARv2（教师蒸馏加 PPO）、SimDuoReal（RL 专家蒸馏）；其余多是监督或模仿学习（A²-MEAS、BATCH、Tiresias）、经典规划与滤波（TPS-A*、LACART、CPNAV）、黑盒优化加 NMPC（被动折叠四旋翼）、基准与数据集。归类时不能只看「具身」二字。

## 核心原理

按小站分栏与本库新建节点的读后观察（以下归纳是本库判断）：

- **失败诊断与纠错：** A²-MEAS（深读）把恢复指令、空间证据和执行历史编码成纠错码，加到冻结动作专家的残差上；它不是 RL，是对演示上挖掘出的偏差做监督。
- **导航规划与安全：** TPS-A*（经典搜索式规划，窄门优势不在成功率）、CPNAV（保形预测校准，保证对象是 oracle 路径不被剪）、LACART（CBF 安全滤波）、被动折叠四旋翼（黑盒优化加固定 NMPC）、DexDICE（离线模仿学习的安全示范支撑）、EmbodiedIntentBench（基准，自适应策略并不优于最强固定配置）、DobotTrack（行人跟随 VLA）。
- **跨形态与人形：** BATCH（人类视频与机器人数据的阶段对齐，真机每任务仅 20 / 30 次滚动）、SUGARv2、统一惯性编码器、SimDuoReal（真机 61/80）、DKWBC（复用）。
- **触觉、工具与抓取：** TouchAnnotator（离线标注，不等于校准的真实力）、DexWriting（仿真）、Grasp-TAGS（只在训练用物体上评估）、ToolDex（复用）。
- **世界模型与硬件：** Tiresias（离线评测，无真机）、WMPA（概念设计，无综合）。
- **数据集与人机交互：** 柔性条带四视角数据集、投入度评测挑战赛概述、主动开场白数据集（合成数据）。
- **证据强度：** 本期新建的节点里，带真机数字的只有少数，且多为每任务 10–30 次试验；不少结果全在仿真或离线评测。

## 工程实践

- 查证某篇论文的细节时，以详情节点和其引用的 arXiv 原文为准，不以小站一句话判断为准。
- 新增小站条目时先按 arXiv ID 去重（同一 ID 只允许一个 canonical 节点）。

## 局限

- 小站只给一句话判断；新建的 20 个节点逐篇通读了论文，复用的 2 个节点沿用既有页面、**没有重新读论文**。
- 小站对若干论文的说法与论文不符，已在各详情节点的核对表里逐条列出（例如 TPS-A* 的 277 个结果来自开阔场景而非窄门、A²-MEAS 的 45.40% 是公开集自报而官方隐藏集只有 32.05%、DexDICE 的真机只有单场景 10 次）。
- 上面的归类是本库判断，只基于新建节点读到的内容。

## 论文索引

| # | 档 | 论文节点 | arXiv | 状态 |
|---|----|----------|-------|------|
| 01 | 深读 | [抓空之后，让纠错真正改变下一步动作](../entities/paper-a2-meas-agentic-embodied-intelligence.md) | [2611.09124](https://arxiv.org/abs/2611.09124) | 新建 |
| 02 | 跟进 | [异形机器人过窄门，按真实轮廓检查碰撞](../entities/paper-tps-astar.md) | [2611.09210](https://arxiv.org/abs/2611.09210) | 新建 |
| 03 | 跟进 | [让四旋翼接触折臂，穿过比机身窄的缝](../entities/paper-soft-morphing-quadrotor-gap-traversal.md) | [2611.09272](https://arxiv.org/abs/2611.09272) | 新建 |
| 04 | 扫读 | [给导航模型的自信分数加一道校准](../entities/paper-cpnav.md) | [2611.10366](https://arxiv.org/abs/2611.10366) | 新建 |
| 05 | 扫读 | [从第一视角视频补出接触与相对压力](../entities/paper-touchannotator.md) | [2611.09129](https://arxiv.org/abs/2611.09129) | 新建 |
| 06 | 扫读 | [把安全示范与高收益示范分开学](../entities/paper-dexdice.md) | [2611.09433](https://arxiv.org/abs/2611.09433) | 新建 |
| 07 | 扫读 | [给物体目标，让双手自己学会交接](../entities/paper-simduoreal-object-centric-bimanual.md) | [2611.10410](https://arxiv.org/abs/2611.10410) | 新建 |
| 08 | 扫读 | [自由握笔写字，同时学会稳笔与控力](../entities/paper-dexwriting.md) | [2611.10451](https://arxiv.org/abs/2611.10451) | 新建 |
| 09 | 扫读 | [感知更准，也可能错过整场任务](../entities/paper-embodiedintentbench.md) | [2611.10498](https://arxiv.org/abs/2611.10498) | 新建 |
| 10 | 扫读 | [跟随指定行人，先保住目标身份](../entities/paper-dobottrack.md) | [2611.09041](https://arxiv.org/abs/2611.09041) | 新建 |
| 11 | 扫读 | [学人类示范，先对齐动作发生的阶段](../entities/paper-batch-behavior-alignment.md) | [2611.10148](https://arxiv.org/abs/2611.10148) | 新建 |
| 12 | 扫读 | [先筛候选动作，再想象执行后果](../entities/paper-tiresias.md) | [2611.10392](https://arxiv.org/abs/2611.10392) | 新建 |
| 13 | 扫读 | [动作跟踪交给动力学，平衡交给残差](../entities/paper-dkwbc.md) | [2611.09605](https://arxiv.org/abs/2611.09605) | 复用 |
| 14 | 扫读 | [用可复用交互模板训练人形搬运](../entities/paper-sugarv2-humanoid-interaction-reference.md) | [2611.09266](https://arxiv.org/abs/2611.09266) | 新建 |
| 15 | 扫读 | [没有标准工具时，记录人如何借物完成任务](../entities/paper-tooldex.md) | [2611.09036](https://arxiv.org/abs/2611.09036) | 复用 |
| 16 | 扫读 | [把人体运动知识迁移到人形惯性估计](../entities/paper-unified-inertial-encoder.md) | [2611.10137](https://arxiv.org/abs/2611.10137) | 新建 |
| 17 | 扫读 | [互动投入度评测，把儿童与成人一起纳入](../entities/paper-multimediate-26-engagement-estimation.md) | [2611.09913](https://arxiv.org/abs/2611.09913) | 新建 |
| 18 | 扫读 | [在路径跟踪外加一层轻量避障过滤](../entities/paper-lacart.md) | [2611.09640](https://arxiv.org/abs/2611.09640) | 新建 |
| 19 | 扫读 | [搬椅子前，先选有利于最终摆放的抓点](../entities/paper-grasp-tags.md) | [2611.09864](https://arxiv.org/abs/2611.09864) | 新建 |
| 20 | 扫读 | [用四部手机记录柔性物体的受控运动](../entities/paper-lagsplat-mpov-dataset.md) | [2611.09959](https://arxiv.org/abs/2611.09959) | 新建 |
| 21 | 扫读 | [世界模型上机运行，先看数据搬运的代价](../entities/paper-wmpa-world-model-planning-accelerator.md) | [2611.10316](https://arxiv.org/abs/2611.10316) | 新建 |
| 22 | 扫读 | [让社交机器人按情境说好第一句话](../entities/paper-proactive-conversation-opener-cco.md) | [2611.09906](https://arxiv.org/abs/2611.09906) | 新建 |

## 关联页面

- [上一期（11-09）29 篇阅读坐标](./embodied-station-2026-11-09-29-papers-technology-map.md)
- [前期（11-06）20 篇阅读坐标](./embodied-station-2026-11-06-20-papers-technology-map.md)

## 参考来源

- [小站 11-10 来源归档](../../sources/blogs/wechat_embodied_station_22_papers_a2meas_tpsastar_2026-11-10.md)

## 继续阅读

- 各详情节点引用的 arXiv 原文；复用节点需要时再逐篇重读。
