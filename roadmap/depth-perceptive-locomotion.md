# 路线（纵深）：如果目标是感知越障（Perceptive Locomotion）

**摘要**：面向"让机器人看着地形上楼梯、跨障碍、跑酷"的纵深路线，从本体感知盲走基线到地形表征、感知策略训练，再到楼梯/跑酷进阶与导航栈整合，按 Stage 0–4 串通核心方法；本路线是 [运动控制主路线](motion-control.md) 的一条分支。

## 路线一览

```mermaid
flowchart LR
  S0["**Stage 0**<br/>盲走基线<br/><em>本体感知 RL locomotion</em>"]
  S1["**Stage 1**<br/>感知与地形表征<br/><em>深度相机 / 高程图 / latent</em>"]
  S2["**Stage 2**<br/>感知策略训练<br/><em>Teacher-Student / 地形课程</em>"]
  S3["**Stage 3**<br/>越障进阶<br/><em>楼梯 / 离散地形 / 跑酷</em>"]
  S4["**Stage 4**<br/>系统整合<br/><em>导航栈 / 真机部署</em>"]

  S0 --> S1 --> S2 --> S3 --> S4

  classDef stage fill:#142a3a,stroke:#e74c3c,stroke-width:2px,color:#fff
  class S0,S1,S2,S3,S4 stage
```

## 这条路径怎么用

- 目标读者是已经能在平地上训出 RL locomotion 策略、想让机器人"带眼睛"过复杂地形的人
- 前置是 [RL 纵深路线](depth-rl-locomotion.md) Stage 0–2 的水平：能在仿真里训练一个盲走策略
- 每个阶段有前置知识、核心问题、推荐做什么、学完输出什么

**和主路线的关系：**
- 感知越障是主路线 L5（RL locomotion）之后最主流的进阶方向之一，对应 L7.1 感知层的实战化
- 训练侧大量复用 [RL 纵深](depth-rl-locomotion.md) 的 sim2real 经验（DR、延迟、状态估计）
- 楼梯/离散地形的落脚点逻辑与 [传统模型控制纵深](depth-classical-control.md) 的 footstep planning 一脉相承

---

## Stage 0 盲走基线（本体感知 locomotion）

**"盲走"（blind locomotion）指只用关节编码器 + IMU、不看地形的策略。感知策略几乎都以它为基线和退路。**

### 前置知识
- [RL 纵深路线](depth-rl-locomotion.md) Stage 0–2：PPO、reward 设计、能在仿真训练 locomotion
- 了解 [legged_gym](../wiki/entities/legged-gym.md) 或 Isaac Lab 训练管线

### 核心问题
- 盲走策略靠什么"感知"地形（本体感知历史隐式估计接触与坡度）
- 盲走的能力边界在哪：为什么楼梯、大台阶、缝隙必须上外感知
- Privileged training（特权学习）如何让策略从仿真真值学到可部署的估计

### 推荐读什么
- [Terrain Adaptation](../wiki/concepts/terrain-adaptation.md)
- [Privileged Training](../wiki/concepts/privileged-training.md)
- [Curriculum Learning](../wiki/concepts/curriculum-learning.md)
- [ANYmal](../wiki/entities/anymal.md) — RSL 盲走与野外行走工作线

### 推荐做什么
- 在 legged_gym / Isaac Lab 里用课程学习训一个能过粗糙地形的盲走策略
- 给策略输入砍掉地形高度真值，对比有/无特权信息的性能差距

### 学完输出什么
- 一个能在随机粗糙地形上稳定行走的盲走策略
- 能说清盲走在哪类地形上必然失败、为什么

---

## Stage 1 感知与地形表征

### 核心问题
- 深度相机和 LiDAR 各适合什么场景（视场、精度、功耗、纱窗/反光失效模式）
- 高程图（elevation map）和端到端深度输入各有什么取舍
- 地形 latent 表征为什么比原始点云/深度图更好训练

### 推荐读什么
- [Terrain Latent Representation](../wiki/concepts/terrain-latent-representation.md)
- [Sensor Fusion](../wiki/concepts/sensor-fusion.md)
- [State Estimation](../wiki/concepts/state-estimation.md)
- [LiDAR SLAM / LIO / VIO 选型](../wiki/comparisons/lidar-slam-lio-vio-selection.md)
- [Query：感知 backbone 选型](../wiki/queries/perception-backbone-selection.md)

### 推荐做什么
- 在仿真里给机器人挂一个深度相机，把深度图投成机器人系高程图
- 对比"高程图输入"和"深度图直接输入"两种观测的训练收敛速度

### 学完输出什么
- 能解释高程图管线（点云 → 栅格 → 补洞 → 机器人系变换）每一步的失效来源
- 能为给定平台（人形/四足、室内/野外）选出合理的感知配置

---

## Stage 2 感知策略训练（Teacher-Student / 地形课程）

### 核心问题
- 为什么感知策略主流用 teacher-student：teacher 看仿真真值地形，student 只看噪声化的相机/高程图
- 地形课程怎么设计（难度递进、地形种类配比、reset 策略）
- 感知噪声怎么在仿真里建模（深度噪声、遮挡、延迟、自遮挡）

### 推荐读什么
- [Privileged Training](../wiki/concepts/privileged-training.md)
- [Procedural Terrain Generation](../wiki/concepts/procedural-terrain-generation.md)
- [Domain Randomization](../wiki/concepts/domain-randomization.md)
- [DreamWaQ++](../wiki/entities/dreamwaq-plus.md)
- Miki et al., *Learning Robust Perceptive Locomotion for Quadrupedal Robots in the Wild* (2022)

### 推荐做什么
- 把 Stage 0 的盲走策略升级为 teacher-student：teacher 用地形真值高度采样点，student 用带噪高程图蒸馏
- 用程序化地形生成一套楼梯 + 台阶 + 缝隙课程，观察课程顺序对成功率的影响

### 学完输出什么
- 一个在仿真里能看着高程图上楼梯的 student 策略
- 对感知噪声建模和课程设计的第一手直觉

---

## Stage 3 越障进阶（楼梯 / 离散地形 / 跑酷）

### 核心问题
- 楼梯与离散地形（梅花桩、缝隙）为什么比坡地难一个量级（落脚点是硬约束）
- 跑酷类工作怎么把"选落脚点"隐式塞进端到端策略（Extreme Parkour 的内积 reward、waypoint 引导）
- 人形和四足在感知越障上的差异（质心高、视野盲区、双足支撑域小）

### 推荐读什么
- [楼梯与障碍感知 locomotion 任务枢纽](../wiki/tasks/stair-obstacle-perceptive-locomotion.md) — 本方向论文全景入口
- [Extreme Parkour](../wiki/entities/extreme-parkour.md)
- [ANYmal Parkour 深读笔记](../wiki/entities/paper-notebook-anymal-parkour-robust-perceptive-locomotion.md)
- [Humanoid Parkour Learning 深读笔记](../wiki/entities/paper-notebook-humanoid-parkour-learning.md)
- [Footstep Planning](../wiki/concepts/footstep-planning.md) — 与 model-based 落脚点方法对照

### 推荐做什么
- 复现一个开源 parkour 工作（Extreme Parkour / humanoid parkour）在仿真里的训练
- 逐项消融：去掉深度输入、去掉课程、去掉 waypoint 引导，记录哪个环节掉点最狠

### 学完输出什么
- 一个能在仿真里过楼梯 + 至少一类离散障碍的感知策略
- 能对一篇新的感知越障论文快速定位它的贡献属于感知、课程还是 reward 设计

---

## Stage 4 系统整合（导航栈 / 真机部署）

### 核心问题
- 局部越障策略怎么接上全局导航（waypoint 从哪来、语义目标怎么下发）
- 真机上感知延迟、里程漂移、外参标定误差怎么处理
- 何时该退回盲走模式（感知失效检测与降级策略）

### 推荐读什么
- [分层四足导航栈](../wiki/concepts/hierarchical-quadruped-navigation-stack.md)
- [平滑导航路径生成](../wiki/methods/smooth-navigation-path-generation.md)
- [Sim2Real](../wiki/concepts/sim2real.md)
- [Query：locomotion 失效模式](../wiki/queries/locomotion-failure-modes.md)

### 推荐做什么
- 把 Stage 3 策略接入一个"目标点 → 路径 → 局部越障"的分层栈，在仿真里跑长距离任务
- 给感知链路注入延迟与漂移，验证降级到盲走的切换逻辑

### 学完输出什么
- 一条"感知 → 越障策略 → 导航栈"的完整仿真系统链
- 真机部署前的检查清单：标定、延迟、失效降级

---

## 快速入口汇总

| 阶段 | 核心问题 | 本仓库入口 |
|------|---------|-----------|
| Stage 0 | 盲走基线 | [Terrain Adaptation](../wiki/concepts/terrain-adaptation.md) |
| Stage 1 | 地形表征 | [Terrain Latent Representation](../wiki/concepts/terrain-latent-representation.md) |
| Stage 2 | 感知策略训练 | [Privileged Training](../wiki/concepts/privileged-training.md) |
| Stage 3 | 楼梯 / 跑酷 | [楼梯与障碍感知 locomotion 枢纽](../wiki/tasks/stair-obstacle-perceptive-locomotion.md) |
| Stage 4 | 导航栈整合 | [分层四足导航栈](../wiki/concepts/hierarchical-quadruped-navigation-stack.md) |

## 和其他页面的关系

- 完整成长路线参考：[主路线：运动控制算法工程师成长路线](motion-control.md)
- 其它纵深路径：
  - [人形 RL 运动控制](depth-rl-locomotion.md) — 本路线的直接前置
  - [传统模型控制（LIP/ZMP → MPC → WBC）](depth-classical-control.md)
  - [模仿学习与技能迁移](depth-imitation-learning.md)
  - [安全控制（CLF/CBF）](depth-safe-control.md)
  - [接触丰富的操作任务](depth-contact-manipulation.md)
  - [VLA 与 BFM（具身基础模型）](depth-vla-bfm.md)
- 关联知识页：
  - [楼梯与障碍感知 locomotion 任务枢纽](../wiki/tasks/stair-obstacle-perceptive-locomotion.md)
  - [Terrain Adaptation](../wiki/concepts/terrain-adaptation.md)
  - [Terrain Latent Representation](../wiki/concepts/terrain-latent-representation.md)
  - [Privileged Training](../wiki/concepts/privileged-training.md)
  - [Procedural Terrain Generation](../wiki/concepts/procedural-terrain-generation.md)
  - [Extreme Parkour](../wiki/entities/extreme-parkour.md)

## 参考来源

本路线基于以下原始资料的归纳：

- [楼梯与障碍感知 locomotion 任务枢纽](../wiki/tasks/stair-obstacle-perceptive-locomotion.md)
- [Terrain Adaptation](../wiki/concepts/terrain-adaptation.md)
- [Privileged Training](../wiki/concepts/privileged-training.md)
- [ANYmal Parkour 深读笔记](../wiki/entities/paper-notebook-anymal-parkour-robust-perceptive-locomotion.md)
- Miki et al., *Learning Robust Perceptive Locomotion for Quadrupedal Robots in the Wild* (2022)
- Cheng et al., *Extreme Parkour with Legged Robots* (2023)
