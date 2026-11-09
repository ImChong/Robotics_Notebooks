---
type: overview
tags: [survey, humanoid, quadruped, locomotion, whole-body-control, senlanke]
status: complete
updated: 2026-11-09
related:
  - senlanke-weekly-2026-10-26-30-humanoid-quadruped-technology-map.md
  - senlanke-weekly-2026-11-02-06-manipulation-technology-map.md
sources:
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-11-02_06.md
original:
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-11-02_06.md
summary: "senlanke 具身运控lab 11-02–11-06 人形 / 四足周报：30 条论文条目，每条对应唯一 arXiv 详情节点（新建 18 / 复用 12），本库逐篇通读新建节点并核对周报说法。"
---

# senlanke 周报索引：11-02–11-06 人形 / 四足（30 篇）

> **本页定位**：为 [【11.02-11.06前沿论文动态】人形／四足30篇](https://mp.weixin.qq.com/s/eiBI2o3Psa9JOATL1XTtTg) 提供 30 个条目到唯一 arXiv 详情节点的索引。

## 一句话定义

senlanke 具身运控lab 11-02–11-06 一周 人形 / 四足论文周报（30 篇）的分栏索引页，把每条映射到唯一 arXiv 详情节点。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| MPC | Model Predictive Control | 模型预测控制 |
| WBC | Whole-Body Control | 全身控制 |
| VLA | Vision-Language-Action | 视觉语言动作模型 |
| MARL | Multi-Agent RL | 多智能体强化学习 |

## 为什么重要

- 周报只给摘要级描述；本页让每条都落到可追溯的详情节点，节点里有逐篇通读后的核对结论。
- 周报条目里不少是模型驱动或规划类工作，并不都是强化学习；索引页把这类差别写出来，避免只看 ★ 标注就归类。

## 核心原理

索引遵循「周报分栏 → 唯一 arXiv 节点」的映射：同一 arXiv 论文跨「人形 / 四足」与「Manipulation」两篇周报时复用同一节点（HITCH、CRISP 即如此）；新建节点已逐篇通读论文。

周报把 30 篇分成人形 25、四足 3、轮足与通用腿式 2。本库逐篇通读后，按**方法性质**重新看这一周（以下归纳是本库判断，不是周报分栏）：

- **学习类（RL / 模仿 / 蒸馏）：** 本库新建节点读过的有 ReGATE、H²MARL、PhySA、STEP、Terrain-Indexed 跑酷（对抗式运动先验加几何奖励）；MAPO、ReContact、HLO-AMP、Ctrl-F、ReflexArc 等是复用节点，沿用既有页面的归类，本库这次没有重读。GRASP 是基于采样的物理重定向（不是 RL）。
- **模型驱动、不含强化学习：** iLQR-MPC 加全身控制、CRISP（NMPC）、接触隐式轨迹优化、ActOR（LLM 编排预定义技能）、Caging VLN（MPPI 加屏障函数）、SCRUB（卡尔曼滤波加 MPC）、四杆腿运动学、腰部弹性几何标定、Sequential Recovery（CEM 搜索加线性分类器，无新的梯度更新）。另有「学习加优化」的 Dogger 协同设计：先用 PPO 训练一个以设计参数为输入的策略，再用优化器选设计。周报的 ★ 标注与「RL」并不等价。
- **仿真与真机证据差别很大：** STEP 与 Terrain-Indexed 跑酷在 Noetix E1 上有真机数字；ReGATE 定量结果全在仿真、真机只有 10 次命令序列；iLQR-MPC、CRISP 的结果只在 MuJoCo 或站立实验里；接触隐式轨迹优化只有低动态动作上机。
- **VLM / VLA 与人形的接口：** HITCH 用全身接口让 VLM 操作 G1（零任务轨迹 51.1% / 42.5%，两个基准都换了相机和指令设置）；AffectiveVLA 用意图门控降低误触发（没有消融）；ActOR 实机只有 20 次单瓶丢弃。

## 工程实践

- 查证某篇论文的技术细节时，以详情节点和其引用的 arXiv 原文为准，不以周报摘要为准。
- 新增周报时按分栏补表，并先按 arXiv ID 去重（同一 ID 只允许一个 canonical 节点）。

## 局限

- 周报只给摘要级描述；本页的新建节点逐篇通读了论文，复用节点沿用既有页面、**没有重新读论文**。
- 周报机构多处写「待核实」，新建节点已按论文首页补全；论文首页也没有机构的会如实写明。
- 周报对若干论文的说法与论文不符，已在各详情节点的核对表里逐条列出（例如 STEP 的「丢帧下仍稳定」只在一步延迟下成立、CRISP 的「三个子系统」其实只有躯干与双臂在 NMPC 中、四杆腿论文只推导了地面姿态的运动学）。
- 上面按方法性质的归类是本库判断，只基于新建节点读到的内容；复用节点没有按此重新核对。

## 论文索引

### 人形机器人（25）

| # | 论文节点 | arXiv | 状态 |
|---|----------|-------|------|
| 01 | ★ [ReGATE：面向人形机器人反应式物体交互的循环门控视觉融合](../entities/paper-regate.md) | [2611.02871](https://arxiv.org/abs/2611.02871) | 新建 |
| 02 | ★ [MAPO：用于通用人形动作跟踪的混合专家交替策略优化](../entities/paper-mapo-motion-tracking.md) | [2611.02613](https://arxiv.org/abs/2611.02613) | 复用 |
| 03 | [基于能量模型的策略编排](../entities/paper-energy-based-policy-orchestration.md) | [2611.02546](https://arxiv.org/abs/2611.02546) | 复用 |
| 04 | [保持成功轨迹的学习控制器序贯恢复更新](../entities/paper-sequential-recovery-updates.md) | [2611.02555](https://arxiv.org/abs/2611.02555) | 新建 |
| 05 | [ReContact：从视频重建并重定向人体接触以实现接触丰富的人形操作](../entities/paper-recontact.md) | [2611.02968](https://arxiv.org/abs/2611.02968) | 复用 |
| 06 | [结合全身控制的在线 iLQR-MPC 鲁棒人形运动](../entities/paper-ilqr-mpc-wbc-humanoid.md) | [2611.04204](https://arxiv.org/abs/2611.04204) | 新建 |
| 07 | [ActOR：基于环境智能体编排的手术室人形机器人自主系统](../entities/paper-actor-humanoid-or.md) | [2611.04255](https://arxiv.org/abs/2611.04255) | 新建 |
| 08 | [ReflexArc：将瞬时触觉反射自底向上集成到人形全身控制](../entities/paper-reflexarc.md) | [2611.04424](https://arxiv.org/abs/2611.04424) | 复用 |
| 09 | [通过动作条件潜在前瞻协调人形移动操作](../entities/paper-action-conditioned-latent-foresight.md) | [2611.04839](https://arxiv.org/abs/2611.04839) | 复用 |
| 10 | [HLO-AMP：基于物体感知动作先验的分层移动操作](../entities/paper-hlo-amp.md) | [2611.05113](https://arxiv.org/abs/2611.05113) | 复用 |
| 11 | [Ctrl-F：以力作为人形移动操作的统一接口](../entities/paper-ctrl-f.md) | [2611.05507](https://arxiv.org/abs/2611.05507) | 复用 |
| 12 | ★ [面向参数化不平地形动态人形运动的接触隐式全身轨迹优化](../entities/paper-contact-implicit-wbto-uneven-terrain.md) | [2611.05585](https://arxiv.org/abs/2611.05585) | 新建 |
| 13 | [GRASP：用于接触丰富物理动作重定向的梯度引导采样](../entities/paper-grasp.md) | [2611.05754](https://arxiv.org/abs/2611.05754) | 复用 |
| 14 | ★ [H²MARL：面向人形机器人控制的分层异构多智能体强化学习](../entities/paper-h2marl-hierarchical-heterogeneous-marl-humanoid.md) | [2611.06523](https://arxiv.org/abs/2611.06523) | 新建 |
| 15 | ★ [像人类一样踢足球：基于潜在动作先验学习多样敏捷的人形足球技能](../entities/paper-soccerbot-latent-motion-priors.md) | [2611.06564](https://arxiv.org/abs/2611.06564) | 复用 |
| 16 | ★ [从仿真学习感知式人形运动](../entities/paper-halo-perceptive-humanoid-motion-sim.md) | [2611.06908](https://arxiv.org/abs/2611.06908) | 复用 |
| 17 | [CRISP：通过约束解析的子系统交互实现预测式全身协同](../entities/paper-crisp-whole-body-nmpc.md) | [2611.07268](https://arxiv.org/abs/2611.07268) | 新建 |
| 18 | [PhySA：人形运动中审美奖励与物理奖励的解耦](../entities/paper-physa-humanoid-locomotion.md) | [2611.07679](https://arxiv.org/abs/2611.07679) | 新建 |
| 19 | [AffectiveVLA：面向人—人形社交交互的安全意图感知 VLA 控制](../entities/paper-affectivevla.md) | [2611.07712](https://arxiv.org/abs/2611.07712) | 新建 |
| 20 | [HITCH：通过全身控制接口教视觉语言模型操作人形机器人](../entities/paper-hitch.md) | [2611.08039](https://arxiv.org/abs/2611.08039) | 新建 |
| 21 | ★ [面向极限人形跑酷的地形索引动作先验与几何支撑](../entities/paper-terrain-indexed-parkour.md) | [2611.08056](https://arxiv.org/abs/2611.08056) | 新建 |
| 22 | ★ [STEP：用于感知式人形运动的显式预测时空训练](../entities/paper-step-spatiotemporal-humanoid-locomotion.md) | [2611.08097](https://arxiv.org/abs/2611.08097) | 新建 |
| 23 | [Social Humanoid：具备社会响应能力的人—人形机器人交互](../entities/paper-social-humanoid.md) | [2611.08407](https://arxiv.org/abs/2611.08407) | 复用 |
| 24 | [人形拥抱机器人手指运动效果研究](../entities/paper-hugging-robot-finger-movements.md) | [2611.02517](https://arxiv.org/abs/2611.02517) | 新建 |
| 25 | [RAIPAL：面向人类任务的高负载敏捷人形机械臂](../entities/paper-raipal.md) | [2611.04334](https://arxiv.org/abs/2611.04334) | 复用 |

### 四足机器人（3）

| # | 论文节点 | arXiv | 状态 |
|---|----------|-------|------|
| 01 | [SCRUB：面向无机械臂四足机器人力—运动控制的接触式表面坐标系估计](../entities/paper-scrub-surface-frame-estimation-armless-quadruped.md) | [2611.06649](https://arxiv.org/abs/2611.06649) | 新建 |
| 02 | ★ [学习面向任务的仿生四足机器人协同设计](../entities/paper-dogger-purposeful-codesign-quadruped.md) | [2611.02462](https://arxiv.org/abs/2611.02462) | 新建 |
| 03 | [面向四足—五旋翼多模态机器人的双闭环四杆腿运动学分析](../entities/paper-quadruped-pentarotor-four-bar-leg.md) | [2611.07470](https://arxiv.org/abs/2611.07470) | 新建 |

### 轮足与通用腿式机器人（2）

| # | 论文节点 | arXiv | 状态 |
|---|----------|-------|------|
| 01 | ★ [通过屏障认证安全架构约束腿式机器人视觉—语言导航](../entities/paper-legged-vln-barrier-caging.md) | [2611.04972](https://arxiv.org/abs/2611.04972) | 新建 |
| 02 | [基于共享腰部双链模型的轮式人形机器人全身弹性—几何标定](../entities/paper-shared-waist-dual-chain-wheeled-humanoid-calibration.md) | [2611.07560](https://arxiv.org/abs/2611.07560) | 新建 |

## 关联页面

- [Manipulation 周报索引（同一周）](./senlanke-weekly-2026-11-02-06-manipulation-technology-map.md)
- [上一周人形 / 四足周报索引（10-26–10-30）](./senlanke-weekly-2026-10-26-30-humanoid-quadruped-technology-map.md)

## 参考来源

- [周报来源归档](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-11-02_06.md)

## 继续阅读

- 各详情节点引用的 arXiv 原文；复用节点沿用既有页面，需要时再逐篇重读。
