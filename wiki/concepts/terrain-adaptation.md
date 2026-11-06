---
type: concept
tags: [locomotion, terrain, perception, footstep-planning, sim2real]
status: complete
updated: 2026-10-21
summary: "Terrain Adaptation 指机器人根据地形感知结果调整步位、身体姿态和接触策略，以在不平整环境中保持稳定移动。"
related:
  - ../queries/robot-perception-stack-selection-loop.md
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
  - ../entities/paper-humoslope-physics-guided-slope-locomotion.md
  - ../entities/paper-e-sds-environment-aware-humanoid-locomotion-rl.md
  - ../entities/paper-faststair-humanoid-stair-ascent.md
  - ../entities/paper-explicit-stair-geometry-humanoid-locomotion.md
  - ../entities/paper-amp-survey-08-more.md
  - ../entities/dreamwaq-plus.md
  - ../entities/paper-rpl-robust-humanoid-perceptive-locomotion.md
  - ../entities/paper-notebook-dpl-depth-only-perceptive-humanoid-locomotion-vi.md
  - ../entities/paper-cref.md
  - ../entities/paper-ladderman-humanoid-perceptive-ladder-climbing.md
  - ../entities/paper-egohtr.md
  - ../entities/paper-legged-load-adapt-unknown-dynamic-load.md
  - ../entities/paper-terrain-aware-ugv-dem-local-planning.md
  - ../entities/paper-notebook-vb-com-learning-vision-blind-composite-humanoid.md
original:
  - ../../sources/papers/learning_quadrupedal_locomotion_challenging_terrain_reading_arxiv_2010_11251.md
  - ../../sources/papers/learning_to_walk_in_minutes_reading_arxiv_2109_11978.md
sources:
  - ../../sources/papers/footstep_and_balance.md
  - ../../sources/papers/privileged_training.md
  - ../../sources/papers/dreamwaq_plus_arxiv_2409_19709.md
  - ../../sources/papers/contact_planning.md
  - ../../sources/papers/e_sds_arxiv_2512_16446.md
  - ../../sources/papers/rpl_arxiv_2602_03002.md
  - ../../sources/papers/dpl_arxiv_2510_07152.md
  - ../../sources/papers/cref_arxiv_2603_29452.md
  - ../../sources/papers/discrete_terrain_minimal_proximity_sensing_arxiv_2606_31912.md
  - ../../sources/papers/vb_com_arxiv_2502_14814.md
  - ../../sources/papers/ame_arxiv_2506_09588.md
  - ../../sources/papers/humanoid_pnb_ame-2-agile-and-generalized-legged-locomotion-vi.md
  - ../../sources/papers/egohtr_arxiv_2607_13472.md
  - ../../sources/papers/humoslope_arxiv_2607_07830.md
  - ../../sources/papers/terrain_aware_ugv_navigation_arxiv_2608_17038.md
  - ../../sources/papers/legged_load_adapt_arxiv_2507_07825.md
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
| 单阶段 raw 深度（CReF） | 本体查询交叉注意 + GRU highway，**不建 2.5D 图** | 无建图延迟；下楼落脚可塑形 | 无官方代码；纯深度无纹理 |
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
- [Legged Load Adapt](../entities/paper-legged-load-adapt-unknown-dynamic-load.md) 在楼梯/rough/斜坡课程上叠加 **箱载动态载荷** 适应（盲本体 + load characteristics 估计）。
- [DreamWaQ++](../entities/dreamwaq-plus.md) 把 **3D 点云** 与 **本体历史** 在单阶段 RL 中融合，是四足 **点云地形适应** 的代表实现。
- [离散地形最小感知](../entities/paper-discrete-terrain-minimal-proximity-sensing.md) 展示四足 **足底 ToF** 在垫脚石/沟上的 **任务对齐最小传感** 路线。
- [RPL](../entities/paper-rpl-robust-humanoid-perceptive-locomotion.md) 用 **多视角深度** 做 **双向/多向** 地形适应，并以 **DFSV/RSM** 处理非对称视野与未见窄地形宽度。
- [DPL](../entities/paper-notebook-dpl-depth-only-perceptive-humanoid-locomotion-vi.md) 用 **单深度 → 交叉注意力高程重建**（自遮挡射线合成进 RL 环）在无外定位下做前向楼梯/缝隙适应；与 elevation map 多传感器路线对照。
- [SOLO](../entities/paper-solo.md) 用 **逐格查询高程** 保住踏石/台阶沿，并把下一步师生分歧写入 PPO，支撑连续 **1.5 km** 户外。
- [LadderMan](../entities/paper-ladderman-humanoid-perceptive-ladder-climbing.md) 把 **稀疏踏棍梯子** 当作极端薄结构地形：端到端 **深度 + VFM** 适应，配合 **RFM** 聚焦踏棍几何。
- [Sim2Real](./sim2real.md) 强调地形感知和真实传感器偏差是迁移痛点。

## 核心原理

地形适应的共同骨架是「地形信息 → 接触决策 → 低层执行」，差别在于地形信息从哪来、由谁转成动作：

- **特权 teacher + 本体 student（Lee et al., arXiv:2010.11251）**：先在仿真里用地形轮廓、接触状态与力、摩擦系数等特权信息训 teacher，再蒸馏为只用关节编码器 + IMU 历史的 TCN student，由历史隐式推断接触与打滑，不设显式估计模块；动作经 PMTG 足端轨迹发生器（策略 50 Hz、关节 PD 400 Hz）变成关节目标。
- **并行 RL + 自动地形课程（Rudin et al., arXiv:2109.11978）**：数千台机器人并行 + PPO，地形按「走出边界升级、走不够目标距离一半降级」自动调难度；观测含 108 个高度采样点，台阶高度 5 cm 增至 20 cm，平地约 4 分钟、不平地形约 20 分钟训出可上真机的策略。
- 两者互补：前者解决「感知不可靠时如何靠本体隐式适应」，后者解决「如何便宜地覆盖足够多的地形分布」；上文各案例是在此基础上换感知接口（点云、深度、ToF、几何 token）。

## 工程实践

- **课程难度**：Lee 等取可通过性在 [0.5, 0.9] 的地形作为「合适难度」，仅用刚性程序化地形（hills / steps / stairs）训练，仍零样本泛化到泥、雪、植被、碎石。
- **指令设计**：崎岖地形上可行速度不明确，Lee 等只给方向与转向指令，不给目标速度；Rudin 等真机把速度指令上限降到 0.6 m/s。
- **sim2real 随机化**：摩擦系数 [0.5, 1.25]、观测噪声、周期性随机推；ANYmal 用学习得到的执行器模型缩小动力学差距。
- **感知质量要单独评估**：真机高度图来自 LiDAR 高程图，质量不完美，鲁棒性低于仿真（Rudin 等 §4.3）。
- **超时 vs 失败终止**：并行训练中超时需用 critic bootstrap，文中报告总奖励约提升 10%–20%。

## 局限与边界

- 纯本体感知（盲走）步态保守，且无法预判悬崖：被指令走向悬崖也会走下去（Lee 等 §3），需要外感知补充。
- 仿真课程覆盖不到的地形（如 Rudin 等中超过 25° 的斜坡）无法爬升；高度图不完美会把感知误差直接带进策略。
- 两篇均以 ANYmal 四足 trot 为主要验证对象；向人形迁移的结论属推断，需以上文人形案例页为准。
- 本页上方的近期案例多为仿真或单平台结果，证据边界见各实体页。

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
- [sources/papers/egohtr_arxiv_2607_13472.md](../../sources/papers/egohtr_arxiv_2607_13472.md) — EgoHTR：rough-terrain 人–场景 4D 演示与 G1 感知 mimic
- [sources/papers/humoslope_arxiv_2607_07830.md](../../sources/papers/humoslope_arxiv_2607_07830.md) — HumoSlope：盲策略下连续陡坡姿态/步态适应（局部平面 ZMP + BSGA）
- [sources/papers/legged_load_adapt_arxiv_2507_07825.md](../../sources/papers/legged_load_adapt_arxiv_2507_07825.md) — Legged Load Adapt：崎岖地形 + 未知动态载荷
- [sources/papers/dpl_arxiv_2510_07152.md](../../sources/papers/dpl_arxiv_2510_07152.md) — DPL：单深度交叉注意力高程重建 + 盲骨干多教师
- [sources/papers/solo_arxiv_2608_26583.md](../../sources/papers/solo_arxiv_2608_26583.md) — SOLO：逐格查询重建 + TA-MSE 长程感知行走
- [sources/papers/vb_com_arxiv_2502_14814.md](../../sources/papers/vb_com_arxiv_2502_14814.md) — VB-Com：高程图失效时视觉/盲策略切换

## 关联页面

- [Locomotion](../tasks/locomotion.md)
- [Footstep Planning](./footstep-planning.md)
- [野外机器人排障指南](../queries/field-robotics-troubleshooting.md) — 应对非结构化地形下的感知与平衡失效
- [Privileged Training](./privileged-training.md)
- [Sim2Real](./sim2real.md)
- [Balance Recovery](../tasks/balance-recovery.md)
- [MoRE（复杂地形多步态 AMP）](../entities/paper-amp-survey-08-more.md) — 深度外感知 + 多判别器先验下的 gait 切换范例
- [EgoHTR](../entities/paper-egohtr.md) — 场景对齐人演示支撑 foothold-critical 感知穿越
- [HumoSlope](../entities/paper-humoslope-physics-guided-slope-locomotion.md) — 无在线外感知的连续陡坡适应：局部平面 ZMP + 生物力学门控
- [CReF](../entities/paper-cref.md) — 前向深度不经高程图的单阶段适应 + 足端可支撑落脚奖励
- [VB-Com](../entities/paper-notebook-vb-com-learning-vision-blind-composite-humanoid.md) — 高程图失效时切盲策略做缺口/动态障碍恢复，而不是把噪声硬塞进单策略
- [DEM + LiDAR UGV 导航论文](../entities/paper-terrain-aware-ugv-dem-local-planning.md) — 轮式 UGV 的区域级坡度规划与近场感知案例，仿真证据边界见实体页

## 推荐继续阅读

- Lee et al., *Learning to Walk in Difficult Terrain*
- Deits & Tedrake, *Footstep Planning on Uneven Terrain with MICP*
- [Query：从零训练人形机器人 RL 策略的完整 checklist？](../queries/humanoid-rl-cookbook.md)
