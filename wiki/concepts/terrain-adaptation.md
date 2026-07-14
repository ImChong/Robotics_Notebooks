---
type: concept
tags: [locomotion, terrain, perception, footstep-planning, sim2real]
status: complete
updated: 2026-07-14
summary: "Terrain Adaptation 指机器人根据地形感知结果调整步位、身体姿态和接触策略，以在不平整环境中保持稳定移动。"
related:
  - ../entities/paper-ame-attention-based-map-encoding.md
  - ../entities/paper-notebook-ame-2-agile-and-generalized-legged-locomotion-vi.md
  - ../entities/paper-discrete-terrain-minimal-proximity-sensing.md
  - ../tasks/locomotion.md
  - ../tasks/stair-obstacle-perceptive-locomotion.md
  - ./footstep-planning.md
  - ./terrain-latent-representation.md
  - ./sim2real.md
  - ./privileged-training.md
  - ../tasks/balance-recovery.md
  - ../entities/paper-e-sds-environment-aware-humanoid-locomotion-rl.md
  - ../entities/paper-faststair-humanoid-stair-ascent.md
  - ../entities/paper-explicit-stair-geometry-humanoid-locomotion.md
  - ../entities/paper-amp-survey-08-more.md
  - ../entities/dreamwaq-plus.md
  - ../entities/paper-rpl-robust-humanoid-perceptive-locomotion.md
  - ../entities/paper-ladderman-humanoid-perceptive-ladder-climbing.md
sources:
  - ../../sources/papers/footstep_and_balance.md
  - ../../sources/papers/privileged_training.md
  - ../../sources/papers/dreamwaq_plus_arxiv_2409_19709.md
  - ../../sources/papers/contact_planning.md
  - ../../sources/papers/e_sds_arxiv_2512_16446.md
  - ../../sources/papers/rpl_arxiv_2602_03002.md
  - ../../sources/papers/discrete_terrain_minimal_proximity_sensing_arxiv_2606_31912.md
  - ../../sources/papers/ame_arxiv_2506_09588.md
  - ../../sources/papers/humanoid_pnb_ame-2-agile-and-generalized-legged-locomotion-vi.md
---

# Terrain Adaptation（地形适应）

**Terrain Adaptation**：让腿式或人形机器人根据地形感知结果，动态调整落脚点、身体姿态、接触时序和控制参数，从而在楼梯、碎石、草地、台阶和坡面上稳定行走。

## 一句话定义

地形适应不是“看见地面”，而是把看见的地形真正转成可执行的接触决策。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| MPC | Model Predictive Control | 滚动时域内优化控制序列的预测控制 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| OOD | Out-of-Distribution | 分布外样本/未见场景，泛化评测关注点 |
| VLM | Vision-Language Model | 视觉-语言多模态理解模型，VLA 的上游 |
| LiDAR | Light Detection and Ranging | 激光雷达，地形感知与建图主传感器 |
| PPO | Proximal Policy Optimization | 人形/足式 locomotion 中最常用的 on-policy 策略梯度算法 |
| Isaac Lab | NVIDIA Isaac Lab | 基于 Omniverse 的机器人学习训练框架 |
| G1 | Unitree G1 Humanoid | 宇树入门级教育科研人形平台 |
| CNN | Convolutional Neural Network | 卷积神经网络，处理图像/深度感知 |
| DCM | Divergent Component of Motion | 质心发散分量，用于落脚点与平衡调节 |
| Locomotion | Robot Locomotion | 足式/人形等无轮移动能力的总称 |
| ANYmal | ANYbotics Quadruped | ANYbotics 的四足机器人研究平台 |

## 为什么重要

平地 locomotion 可以依赖固定步态和简化模型，但一旦进入真实世界：
- 可落脚区域离散且不规则
- 摩擦和高度变化引入失稳风险
- 感知误差会直接转化为踩空或绊倒

因此地形适应是 locomotion 从实验室走向真实环境的关键门槛。

## 典型闭环

```text
高度图 / 点云 / 触觉估计
        ↓
   可行落脚区域提取
        ↓
   步位 / 接触序列规划
        ↓
  MPC / RL / WBC 执行调整
```

## 常见信息来源

### 1. 高度图（Height Map）
最常见的结构化表示。把局部地形编码成栅格高度，可直接供 teacher policy、MPC 或步位规划器使用。

### 2. 点云 / 深度图
信息更丰富，但噪声更大、实时处理代价更高，常需先转成局部可行接触区域，或经 Encoder 压成 **低维 terrain latent**（见 [地形 Latent 表征](./terrain-latent-representation.md)），而非显式 64×64 高度栅格。

### 3. 接触反馈
足端是否滑动、接触是否建立、法向力是否异常，都是补救感知误差的重要信号。

### 4. 足底接近传感（Proximity / ToF）
在 **摆动相** 用足端 **短距 ToF** 获取 **接触前** 局部几何，绕开机身深度/LiDAR 的 **自遮挡与地图延迟**；ETH RSL 在 ANYmal-D 上验证踏石/沟/平衡木场景（见 [离散地形最小感知](../entities/paper-discrete-terrain-minimal-proximity-sensing.md)）。

## 主要策略路线

| 路线 | 做法 | 优点 | 局限 |
|------|------|------|------|
| 传统规划 | 感知 + footstep planning + MPC/WBC | 可解释、约束清晰 | 感知和规划耦合复杂 |
| 特权训练 | teacher 用高度图，student 蒸馏到本体感知 | sim2real 友好 | teacher/student 设计复杂 |
| 端到端 RL | 直接输入高度图/点云预测动作 | 反应式强 | 对训练分布依赖高 |
| 多模态点云 RL（DreamWaQ++） | 分层 $SE(3)$ 点云记忆 + PointNet + 本体 Mixer，单阶段非对称 AC | 障碍前瞻、传感器无关、OOD 本体回退 | 训练重、公开页暂无同署官方代码仓 |
| 环境感知自动奖励（E-SDS） | VLM 读地形统计 + 行为分解生成调用高度图/LiDAR 的 Python 奖励 | 跨四类地形少手工调参；楼梯下降为分水岭 | 每地形专用策略、仅仿真、首轮仍依赖 prompt 工程 |
| 足底最小感知（Foot-ToF） | 四足足底 4×4 ToF 直接进 LSTM-PPO，无高程图栈 | 低算力/低延迟、抗腿自遮挡、仿真 ray cast 易建模 | 近场落脚专用；材质/泥污敏感；不等价全局导航 |

### 近期案例：E-SDS 的环境感知奖励合成

[E-SDS（arXiv:2512.16446）](../entities/paper-e-sds-environment-aware-humanoid-locomotion-rl.md) 把 **地形适应** 从「控制器读传感器」前移到 **奖励设计阶段**：Environment Analysis Agent 在目标地形上跑千机短 rollout，统计 **缺口率、障碍密度、崎岖度**，与 SUS 行为分解一并喂给 VLM，生成显式调用 **27×21 高度栅格 + 144 线 LiDAR** 的奖励代码；再经双候选 PPO + 反馈迭代精炼。在 Isaac Lab + Unitree G1 上，相对手工 13 项感知基线 **速度跟踪误差降 51.9–82.6%**，且 **仅该方法完成 12 cm 台阶下降**。

### 近期案例：四足足底 ToF 最小感知（arXiv:2606.31912）

[离散地形最小感知](../entities/paper-discrete-terrain-minimal-proximity-sensing.md) 把 **地形适应** 的传感重心下移到 **足端接触点**：ANYmal-D 每足 **VL53L5CX** 4×4 网格（60 Hz）在 **触地前** 扫描落脚区；策略 **不建图**，仅用 proximity + 本体在 Isaac Gym 两阶段课程学 **统一离散地形策略**。仿真对照表明，相对从 **机载理想高程图/六相机融合图** 运动学投影到足端的读数，足端直连传感对 **关节偏置、连杆误差、里程计噪声** 更稳健；实机完成 **60 cm 沟** 与错落踏石，平均约 **0.52 m/s**。

### 近期案例：显式楼梯几何 token（arXiv:2605.09944）

[显式楼梯几何条件化](../entities/paper-explicit-stair-geometry-humanoid-locomotion.md) 把 **楼梯** 从「高维高程图 / 视觉 latent」收束为 **踢面高度、踏面深度、航向与楼梯状态** 四维 token，由 **BEV 点云 CNN** 估计后直接条件化 **PPO**；相对 **11×17 高程图** 与 **盲走** 在 Isaac Lab 楼梯任务上成功率 **96% / 88% / 52%**，在 **未见踢面高度** 上优于 **MoRE** 视觉基线；**Unitree G1** 实机含 **户外 33 级连续上楼**。与 [FastStair](../entities/paper-faststair-humanoid-stair-ascent.md) 的 **DCM 落点规划监督** 形成对照：本文强调 **可解释几何接口 + teacher–student 感知**，而非高速规划引导。

## 与其他页面的关系

- [Footstep Planning](./footstep-planning.md) 决定“下一步踩哪里”。
- [Locomotion](../tasks/locomotion.md) 关注更完整的移动任务。
- [楼梯与障碍 Locomotion（中心节点）](../tasks/stair-obstacle-perceptive-locomotion.md) 汇总楼梯/越障上的感知 vs 盲走文献索引。
- [Privileged Training](./privileged-training.md) 展示了复杂地形上 teacher 用高度图、student 用本体感知的经典方案。
- [DreamWaQ++](../entities/dreamwaq-plus.md) 把 **3D 点云** 与 **本体历史** 在单阶段 RL 中融合，是四足 **点云地形适应** 的代表实现。
- [离散地形最小感知](../entities/paper-discrete-terrain-minimal-proximity-sensing.md) 展示四足 **足底 ToF** 在垫脚石/沟上的 **任务对齐最小传感** 路线。
- [RPL](../entities/paper-rpl-robust-humanoid-perceptive-locomotion.md) 用 **多视角深度** 做 **双向/多向** 地形适应，并以 **DFSV/RSM** 处理非对称视野与未见窄地形宽度。
- [LadderMan](../entities/paper-ladderman-humanoid-perceptive-ladder-climbing.md) 把 **稀疏踏棍梯子** 当作极端薄结构地形：端到端 **深度 + VFM** 适应，配合 **RFM** 聚焦踏棍几何。
- [Sim2Real](./sim2real.md) 强调地形感知和真实传感器偏差是迁移痛点。

## 常见误区

- **误区 1：腿足机器人不需要地形感知。**
  在结构化平地上也许可勉强成立，但在户外和障碍环境里通常不现实。
- **误区 2：有高度图就自动等于稳定行走。**
  还需要把感知结果转成步位、姿态和接触约束。
- **误区 3：地形适应只属于高层规划。**
  实际上它影响从 perception 到控制执行的全链路。

## 参考来源

- [sources/papers/footstep_and_balance.md](../../sources/papers/footstep_and_balance.md) — 步位规划、DCM 与不平地形步行基础
- [sources/papers/privileged_training.md](../../sources/papers/privileged_training.md) — ANYmal 高度图 teacher / proprioception student 经典路线
- [sources/papers/contact_planning.md](../../sources/papers/contact_planning.md) — 不平整地形接触区域与多步接触规划
- [sources/papers/e_sds_arxiv_2512_16446.md](../../sources/papers/e_sds_arxiv_2512_16446.md) — E-SDS 环境感知 VLM 奖励合成
- [sources/papers/dreamwaq_plus_arxiv_2409_19709.md](../../sources/papers/dreamwaq_plus_arxiv_2409_19709.md) — DreamWaQ++ 多模态点云四足 loco
- [sources/papers/explicit_stair_geometry_arxiv_2605_09944.md](../../sources/papers/explicit_stair_geometry_arxiv_2605_09944.md) — 显式楼梯几何条件化人形爬梯
- [sources/papers/discrete_terrain_minimal_proximity_sensing_arxiv_2606_31912.md](../../sources/papers/discrete_terrain_minimal_proximity_sensing_arxiv_2606_31912.md) — ETH RSL 四足足底 ToF 离散地形

## 关联页面

- [Locomotion](../tasks/locomotion.md)
- [Footstep Planning](./footstep-planning.md)
- [野外机器人排障指南](../queries/field-robotics-troubleshooting.md) — 应对非结构化地形下的感知与平衡失效
- [Privileged Training](./privileged-training.md)
- [Sim2Real](./sim2real.md)
- [Balance Recovery](../tasks/balance-recovery.md)
- [MoRE（复杂地形多步态 AMP）](../entities/paper-amp-survey-08-more.md) — 深度外感知 + 多判别器先验下的 gait 切换范例

## 推荐继续阅读

- Lee et al., *Learning to Walk in Difficult Terrain*
- Deits & Tedrake, *Footstep Planning on Uneven Terrain with MICP*
- [Query：从零训练人形机器人 RL 策略的完整 checklist？](../queries/humanoid-rl-cookbook.md)
