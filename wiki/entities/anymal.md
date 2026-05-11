---
type: entity
tags: [robot, quadruped, hardware, platform, eth]
status: complete
updated: 2026-04-21
related:
  - ./humanoid-robot.md
  - ./quadruped-robot.md
  - ../tasks/locomotion.md
  - ../entities/legged-gym.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/papers/locomotion_rl.md
summary: "ANYmal 是由 ETH Zurich 和 ANYbotics 开发的高性能四足机器人平台。它以 SEA 力控执行器和极高的防护等级著称，是机器人强化学习与 Sim2Real 技术研究的标志性载体。"
---

# ANYmal 四足机器人

**ANYmal** 是由苏黎世联邦理工学院（ETH Zurich）的机器人系统实验室（Robotic Systems Lab, RSL）研发，并随后由衍生公司 ANYbotics 成功商业化的高性能四足机器人。它不仅在学术界（特别是足式机器人强化学习、Sim2Real 领域）具有极高的统治力，同时在工业巡检（如海上钻井平台、化工矿区）领域也是首屈一指的标杆产品。

## 核心硬件特征

1. **极其强悍的防护与鲁棒性**：
   ANYmal 从设计之初就瞄准了最恶劣的工业环境。其机身具有极高的防尘防水等级（IP67），能够在大雨、泥泞甚至浅水中涉水行走，同时也防爆、抗摔。这使其在商业化巡检中能够做到真正的 24/7 全天候部署。
2. **串联弹性执行器 (SEA, Series Elastic Actuator)**：
   有别于目前市面上普及的准直接驱动（QDD）电机，ANYmal 采用了专门开发的 SEA。在电机和输出轴之间加入了一层物理弹簧。
   - **优点**：SEA 提供了天然的机械碰撞缓冲，极大地保护了减速器免受冲击力破坏；同时，通过测量弹簧的形变，可以实现极其精确的关节力矩反馈（Torque Feedback）。这使得 ANYmal 在执行阻抗控制（Impedance Control）和接触力控制时具备无可比拟的优势。
   - **挑战**：SEA 引入了额外的非线性动力学和共振问题，对底层控制算法的要求远高于 QDD 电机。
3. **顶尖的传感器与算力配置**：
   ANYmal C/D 系列集成了全景激光雷达（LiDAR）、多个深度相机、热成像仪以及高精度的 RTK/IMU，内置了强大的工控机和 GPU，支撑起其完全自主的 SLAM 导航与避障能力。

## 强化学习的“黄埔军校”

ANYmal 被誉为机器人强化学习的“黄埔军校”，它是无数里程碑式 RL 论文的御用实验平台：
- **Learning Agile Locomotion**：ETH 团队在 2020 年展示了 ANYmal 如何通过深度强化学习，在自然界复杂地形中学会类似动物的高动态恢复和翻越动作，彻底确立了 RL 在四足 locomotion 中的主流地位。
- **RMA (Rapid Motor Adaptation)**：通过在仿真中对地形和动态参数进行特权训练（Privileged Training），再使用适配模块（Adaptation Module）在线估计隐式环境特征，使 ANYmal 展现出了极强的越野与地形自适应能力。
- **盲走能力**：ANYmal 证明了仅依靠本体感受（Proprioception，即关节位置、速度和 IMU 数据），机器人就能在深雪、高草、碎石等复杂地形下稳健行走。

## 与 Unitree 产品的区别

| 维度 | ANYmal (ANYbotics) | Unitree (Go2 / B2) |
|------|--------------------|--------------------|
| **市场定位** | 高端重型工业巡检 / 顶尖学术科研 | 极高性价比普及型科研 / 消费级与行业级混合 |
| **驱动器架构** | 串联弹性执行器 (SEA) | 准直接驱动电机 (QDD) |
| **物理特性** | 较重（约 50kg），负载能力强，防护极高 | 较轻（Go2 约 15kg），运动速度极快，爆发力强 |
| **软件生态** | 基于 Raisim / OCS2 的开源强化学习框架 | 开源 Unitree SDK / Isaac Gym (legged_gym) |

## 对机器人技术生态的深远贡献

ANYmal 系列及其背后的 ETH RSL 团队是“**数据驱动足式控制（Data-driven Legged Locomotion）**”路线的绝对先驱。他们向全球研究者证明了：在仿真环境（如 Raisim 或 Isaac Gym）中进行大规模并行训练的神经网络策略，只要通过合理的域随机化（Domain Randomization）和精确的执行器建模（Actuator Network），就可以无缝、稳定地迁移（Sim2Real）到极端复杂的真实户外地形中。这一范式目前已被包括人形机器人在内的整个具身智能行业广泛采纳。

## 关联页面
- [四足机器人](./quadruped-robot.md)
- [人形机器人 (Humanoid Robot)](./humanoid-robot.md)
- [Locomotion 任务](../tasks/locomotion.md)
- [Legged Gym](../entities/legged-gym.md)
- [Sim2Real 概念](../concepts/sim2real.md)

## 参考来源
- Hutter, M., et al. (2016). *ANYmal - a highly mobile and dynamic quadrupedal robot*.
- Lee, J., et al. (2020). *Learning quadrupedal locomotion over challenging terrain*.
- [sources/papers/locomotion_rl.md](../../sources/papers/locomotion_rl.md)
