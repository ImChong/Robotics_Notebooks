---
type: entity
tags: [humanoid, hardware, open-source, robotics, research]
status: complete
updated: 2026-05-23
related:
  - ./humanoid-robot.md
  - ./roboto-origin.md
  - ./asimov-v1.md
  - ../overview/robot-open-source-wechat-issue01-curator.md
  - ../overview/robot-open-source-wechat-issue02-curator.md
  - ./berkeley-humanoid-lite.md
  - ./odri-solo-and-bolt.md
  - ./fourier-grx-n1.md
  - ./tienkung-humanoid-open-source.md
  - ./agibot-lingxi-x1.md
  - ./openloong.md
  - ./open-duck-mini.md
  - ../queries/humanoid-hardware-selection.md
  - ../roadmaps/humanoid-control-roadmap.md
sources:
  - ../../sources/papers/humanoid_hardware.md
  - ../../sources/repos/roboto_origin.md
  - ../../sources/repos/asimov-v1.md
  - ../../sources/blogs/wechat_jixie_robot_open_source_treasury_issue01_10_robots.md
  - ../../sources/blogs/wechat_jixie_robot_open_source_treasury_issue02_10_robots.md
  - ../../sources/repos/openloong.md
summary: "主流开源人形机器人硬件方案对比：梳理 Berkeley Humanoid、Roboto Origin、Asimov v1、ODRI 与商业平台的机械结构、执行器选型及开源生态，为研究者提供低成本入门指南。"
---

# 开源人形机器人硬件方案对比

随着具身智能的爆发，人形机器人的硬件门槛正在迅速降低。对于预算有限的实验室或个人研究者，**开源硬件方案 (Open-source Humanoid Hardware)** 是验证算法的首选。

## 微信策展索引（更多开源整机入口）

第三方清单文将 **傅利叶 N1、智元灵犀 X1、天工、ODRI、Berkeley Humanoid Lite** 等与 **ROBOTIS 教育/人形线** 并列收录，已拆成独立实体页，入口见：[机器人开源宝库（微信策展第01期）](../overview/robot-open-source-wechat-issue01-curator.md)。本页下文的 **Berkeley / ODRI** 小节可与 [Berkeley Humanoid Lite](./berkeley-humanoid-lite.md)、[ODRI Solo / Bolt](./odri-solo-and-bolt.md) 交叉阅读。

**第02期**补充 **Reachy2、Poppy、InMoov、Stanford Doggo/Pupper、myCobot 320、myAGV、TidyBot2、Kinova Gen3、Franka Research 3、PAROL6** 等实体索引，见 [机器人开源宝库（微信策展第02期）](../overview/robot-open-source-wechat-issue02-curator.md)。

## 核心方案对比

| 方案名称 | 主导机构 | 自由度 (DOF) | 核心执行器 | 成本估算 | 软件生态 |
|------|------|-----|-----------|---------|---------|
| **Berkeley Humanoid** | UC Berkeley | 12-14 | 准直接驱动 (QDD) | < 5,000 USD | 基于 Python/C++ 的简易控制框架 |
| **Roboto Origin** | Roboparty | 20+ | QDD + 舵机混合 | < 3,000 USD | ROS2 支持，兼容简单平衡算法 |
| **ODRI (Bolt/Solo)** | Max Planck | 6-12 (双腿/四足) | 基于 T-Motor 改造 | 中等 | OCS2 / Pinocchio 深度支持 |
| **Unitree H1 (SDK版)** | Unitree | 19 | 商业级 QDD | > 50,000 USD | Isaac Gym (legged_gym) 生态极强 |
| **Asimov v1** | Asimov Inc. | 25 主动 + 2 被动（公开 README） | 铝结构 + MJF 尼龙；关节驱动以官方设计为准 | DIY Kit 目标价量级约 **1.5 万 USD**（以官网为准） | 单仓含 CAD/电气/**MuJoCo**；运控 API/策略仍在路线图 |
| **OpenLoong 青龙** | 人形机器人（上海）/ 开放原子 | **43**（公开硬件 README） | 全尺寸公版；**EtherCAT** 关节总线；五指灵巧手 | 全栈开源图纸（制造门槛高，非 DIY 低价档） | **Framework**（ROS-free C++）+ **Dyn-Control**（MPC/WBC+MuJoCo）+ Isaac Gym/ROS 并行栈；详见 [OpenLoong](./openloong.md) |

## 1. Berkeley Humanoid (准直接驱动派)
- **特点**：极其强调低成本和维修便捷性。它证明了使用廉价的无刷电机和 3D 打印结构，也能完成稳定的动态行走。
- **优点**：动力学透明度高，非常适合做强化学习的 Sim2Real 验证。

## 2. Roboto Origin (科普与原型派)
- **特点**：由国内开源社区驱动，旨在打造人人都能拥有的“第一台人形机器人”。
- **优点**：文档详尽，组装门槛低。

## 2b. Asimov v1（全栈单仓 + 手册/BOM 外链）

- **特点**：机械子装配、电气线束与 PCB、**MuJoCo** 模型与板载软件集中在同一 GitHub 仓库；公开 **双板计算**（Raspberry Pi 5 + Radxa CM5）与多段 **CAN** 带宽规格，便于做通信与实时性规划。
- **优点**：「制造—仿真—机载软件」对齐成本低；提供 **DIY Kit** 与 **自采 BOM** 两条路径。
- **局限**：全身 **locomotion 策略**、统一 **API** 等在官方路线图中仍属推进项，算法侧需自建或与社区贡献结合。
- **详情**：[Asimov v1](./asimov-v1.md)

## 2c. OpenLoong 青龙（全尺寸公版 + 国家级全栈开源）

- **特点**：业内首个强调 **全尺寸公版机** 开源叙事之一；硬件 PDF 分模块（腰/胸/头/腿足），软件 **ROS-free C++ Framework** 与 **MPC+WBC MuJoCo 栈** 并行；开放原子基金会项目。
- **优点**：EtherCAT 工业总线、43 DOF、训推/白虎数据集与 [OpenLoong-Brain](./openloong.md) 大模型技能线，适合整机系统工程与 WBC/MPC 研究。
- **局限**：制造与 EtherCAT 集成门槛高；多软件栈（Framework / Dyn-Control / ROS / Isaac Gym）环境版本需自行对齐。
- **详情**：[OpenLoong（青龙·公版机）](./openloong.md)

## 3. ODRI 架构 (学术严谨派)
- **特点**：Open Dynamic Robot Initiative。虽然其人形版本较少，但其开源的执行器模块（Actuator Modules）被广泛借鉴。
- **优点**：力控精度极高，代码质量达工业级。

## 选型建议

- **如果你想验证 RL 算法**：首选 **Berkeley Humanoid** 类方案，因为其 QDD 电机的动力学建模最为简单。
- **如果你想研究全身协调 (WBC)**：建议寻找支持更高自由度的平台，或者在仿真中使用 **ODRI** 模型进行先行验证。
- **如果你想做低成本娱乐双足 / BDX 复刻**：见 [Open Duck Mini](./open-duck-mini.md)（~42 cm、BOM &lt;$400、MuJoCo Playground + Pi Zero 2W 部署；非全尺寸人形，但 sim2real 管线完整）。

## 2d. Open Duck Mini（迷你娱乐双足 / DIY）

- **特点：** Disney BDX 角色的开源迷你版；Feetech 舵机 + Onshape CAD + 四仓分工（Hub / Playground / 参考运动 / Runtime）。
- **优点：** 社区活跃、文档与预训练 ONNX 公开；适合学习 **BAM 执行器辨识 + 模仿奖励** 在廉价硬件上的 sim2real。
- **局限：** 舵机扭矩与背隙限制动态性能；与 Berkeley / 青龙等全尺寸研究平台不可直接类比。
- **详情：** [Open Duck Mini](./open-duck-mini.md)

## 关联页面
- [人形机器人 (Humanoid Robot)](./humanoid-robot.md)
- [Roboto Origin（开源人形机器人基线）](./roboto-origin.md)
- [Asimov v1](./asimov-v1.md)
- [OpenLoong（青龙·公版机）](./openloong.md)
- [Open Duck Mini](./open-duck-mini.md)
- [机器人开源宝库（微信策展第01期）](../overview/robot-open-source-wechat-issue01-curator.md)
- [机器人开源宝库（微信策展第02期）](../overview/robot-open-source-wechat-issue02-curator.md)
- [人形机器人硬件怎么选](../queries/humanoid-hardware-selection.md)
- [Humanoid Control Roadmap](../roadmaps/humanoid-control-roadmap.md)

## 参考来源
- [humanoid_hardware.md](../../sources/papers/humanoid_hardware.md)
- [roboto_origin.md](../../sources/repos/roboto_origin.md)
- [asimov-v1.md](../../sources/repos/asimov-v1.md)
- [openloong.md](../../sources/repos/openloong.md)
- [wechat_jixie_robot_open_source_treasury_issue01_10_robots.md](../../sources/blogs/wechat_jixie_robot_open_source_treasury_issue01_10_robots.md)
- [wechat_jixie_robot_open_source_treasury_issue02_10_robots.md](../../sources/blogs/wechat_jixie_robot_open_source_treasury_issue02_10_robots.md)
- 各开源项目 GitHub Readme 与 Wiki。
