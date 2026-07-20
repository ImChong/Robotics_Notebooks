---
type: entity
tags: [quadruped, legged, hardware, platform, locomotion]
status: complete
updated: 2026-07-20
related:
  - ./humanoid-robot.md
  - ./anymal.md
  - ./paper-discrete-terrain-minimal-proximity-sensing.md
  - ./boston-dynamics.md
  - ./paper-autonomous-spot-nebula-exploration.md
  - ./paper-spot-rl-distributional-sim2real.md
  - ./patent-boston-dynamics-legged-control-stack.md
  - ./unitree.md
  - ./paper-barkour-quadruped-agility-benchmark.md
  - ./paper-apt-rl-agile-perceptive-quadruped-locomotion.md
  - ./legged-gym.md
  - ../tasks/locomotion.md
  - ../tasks/hybrid-locomotion.md
  - ../methods/hipan.md
  - ../concepts/sim2real.md
  - ../overview/notable-commercial-robot-platforms.md
sources:
  - ../../sources/repos/notable-commercial-robot-platforms.md
  - ../../sources/papers/locomotion_rl.md
summary: "四足机器人是以四条腿与环境间歇接触的腿足平台，侧重崎岖地形移动与户外部署；常与 RL locomotion、Sim2Real、分层导航结合。"
---

# 四足机器人（Quadruped Robot）

## 一句话定义

四足机器人是以四条腿与环境形成间歇接触的腿足平台，侧重崎岖地形移动与户外部署，常与强化学习 locomotion、Sim2Real 及分层导航结合。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Quadruped | Quadruped Robot | 四足平台，静态稳定裕度通常大于双足 |
| RL | Reinforcement Learning | 四足 loco 并行仿真训练成熟 |
| PD | Proportional–Derivative | 策略输出经 PD 转关节力矩 |
| Sim2Real | Simulation to Real | 四足零样本迁移案例丰富 |
| WBC | Whole-Body Control | 四足亦可用全身/质心级协调 |

## 核心特征

1. **支撑域更大**：四足站立时通常可将投影支撑多变更为三角形或四边形支撑，同等尺度下较双足 **更易满足静态或拟静态稳定**，对底层平衡控制的容错相对友好（仍需处理打滑、柔性地形与动态步态）。
2. **任务侧重 locomotion**：主流产品 narrative 以移动、巡检、测绘为主；末端操作能力通常弱于强调双臂的人形平台（例外：加装机械臂的「移动操作」配置）。
3. **高频控制与感知**：优秀产品往往在 **500Hz–1000Hz** 级关节闭环与 **LiDAR / 深度 / IMU** 融合上下功夫，工业场景还对防护等级与可靠性（MTBF）敏感。
4. **仿真训练范式成熟**：大量工作沿用 **并行仿真 + 域随机化 + 特权信息蒸馏** 的路线，把神经网络策略迁移到真机；这与本库中 [Sim2Real](../concepts/sim2real.md)、[Legged Gym](./legged-gym.md) 一脉相承。敏捷评测侧可对照 [Barkour 基准与开源机体](./paper-barkour-quadruped-agility-benchmark.md)（障碍课 + Menagerie MJCF + 开源硬件栈）。

## 主流平台速览

| 平台 | 组织 | 典型定位 | 本库延伸阅读 |
|------|------|-----------|----------------|
| **Spot** | Boston Dynamics | 工业巡检与商业化四足标杆 | [Boston Dynamics](./boston-dynamics.md)；研究侧见 [NeBula 探索](./paper-autonomous-spot-nebula-exploration.md)、[RL Sim2Real](./paper-spot-rl-distributional-sim2real.md)、[控制专利栈](./patent-boston-dynamics-legged-control-stack.md) |
| **ANYmal** | ANYbotics / ETH | 高端工业与顶尖学术 RL 载体 | [ANYmal](./anymal.md) |
| **Go2 / B2** | Unitree | 科研与量产带宽大、生态活跃 | [Unitree](./unitree.md) |

更多品牌索引见 [市面知名机器人平台纵览](../overview/notable-commercial-robot-platforms.md)。

## 与人形机器人的区别

两者同属腿足机器人，但默认约束不同：

- **稳定性**：四足在多数站姿下更易利用多腿支撑；人形双足支撑面窄、重心高，对动态平衡与上身协调要求更苛刻。人形侧的系统梳理见 [人形机器人](./humanoid-robot.md)。
- **任务边界**：四足以移动与场景穿越为主流卖点；人形更常承接 **操作 + 全身协调（loco-manipulation）** 的产品叙事。
- **硬件与成本曲线**：四足在相近动态性能下往往可比人形更早做到 **户外批量部署与巡检闭环**，但加装双臂操作臂后系统复杂度会显著上升。

## 核心挑战

1. **地形与接触不确定性**：碎石、湿滑、台阶边缘会导致接触力学突变，需要稳健步态与估计。
2. **Sim2Real**：执行器模型、关节间隙与地面摩擦的仿真误差仍会磨损策略迁移效果。
3. **续航与热管理**：长时间巡检需权衡功率、电池与算力占用。
4. **结构化环境中的机动**：窄通道、限高与三维障碍需要分层策略或全身姿态协调；参见 [HiPAN](../methods/hipan.md)。

## 相关工具链与生态

- **系统学习路线**：[四足控制学习策展](./quadruped-control-curriculum.md)（URDF → SysID → PPO/DR → Sim2Real → 导航闭环；基于 [MATRiX](./matrix-simulation-platform.md)）
- **仿真与 RL 范式**：[Legged Gym](./legged-gym.md)、[MuJoCo](./mujoco.md)、[Isaac Gym / Isaac Lab](./isaac-gym-isaac-lab.md)、[MATRiX](./matrix-simulation-platform.md)
- **多步态 bio-inspired RL**：[Learning to Adapt（Nature MI 2025）](./paper-learning-to-adapt-bio-inspired-quadruped-gait.md)（RaiSim + [ihcr/learning_to_adapt](https://github.com/ihcr/learning_to_adapt)）
- **离散地形最小感知**：[足底 ToF + RL（ETH RSL, arXiv:2606.31912）](./paper-discrete-terrain-minimal-proximity-sensing.md)（踏石/沟/平衡木，无相机 LiDAR）
- **Stanford 开源四足**：[Stanford Doggo / Pupper](./stanford-doggo-and-pupper.md)（含 **Pupper v3** 文档站、RL/VLM 与 CS 123）；**早期模型控制**见 [easy_quadruped](./easy-quadruped.md)（MuJoCo Trot 闭环，与 v3 monorepo 不同栈）
- **任务扩展**：[Hybrid Locomotion](../tasks/hybrid-locomotion.md)（轮腿、步态切换等多模式）
- **总任务入口**：[Locomotion](../tasks/locomotion.md)

## 关联页面

- [人形机器人](./humanoid-robot.md)
- [Boston Dynamics](./boston-dynamics.md)
- [Autonomous Spot / NeBula](./paper-autonomous-spot-nebula-exploration.md)
- [Spot RL Sim2Real](./paper-spot-rl-distributional-sim2real.md)
- [BD 足式控制专利栈](./patent-boston-dynamics-legged-control-stack.md)
- [ANYmal](./anymal.md)
- [Unitree](./unitree.md)
- [Legged Gym](./legged-gym.md)
- [Locomotion 任务](../tasks/locomotion.md)
- [HiPAN（四足分层导航）](../methods/hipan.md)
- [Barkour（敏捷评测与开源四足）](./paper-barkour-quadruped-agility-benchmark.md)
- [离散地形最小感知（足底 ToF）](./paper-discrete-terrain-minimal-proximity-sensing.md)
- [市面知名机器人平台纵览](../overview/notable-commercial-robot-platforms.md)
- [Sim2Real](../concepts/sim2real.md)
- [四足控制学习策展](./quadruped-control-curriculum.md)
- [MATRiX 仿真平台](./matrix-simulation-platform.md)
- [RoamerX 导航栈](./roamerx-navigation.md)

## 参考来源

- [notable-commercial-robot-platforms](../../sources/repos/notable-commercial-robot-platforms.md)
- [locomotion_rl](../../sources/papers/locomotion_rl.md)
- [Autonomous Spot 论文摘录（arXiv:2010.09259）](../../sources/papers/autonomous_spot_arxiv_2010_09259.md)
- [Spot RL 论文摘录（arXiv:2504.17857）](../../sources/papers/spot_rl_distributional_sim2real_arxiv_2504_17857.md)
- [Boston Dynamics 足式专利摘录](../../sources/patents/boston_dynamics_legged_robot_patents.md)
