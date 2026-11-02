---
type: overview
tags: [overview, quadruped, open-source, locomotion, technology-map, wechat]
status: complete
updated: 2026-11-01
related:
  - ../entities/spotmicro-ros-mike4192.md
  - ../entities/mangdang-quadruped-robot.md
  - ../entities/legged-control-ocs2.md
  - ../entities/quadruped-robot.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/blogs/wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md
summary: "古月居转载的四足机器人开源项目综述的阅读坐标：36 个 GitHub 项目按极简运动学、ROS、优化控制、强化学习、轮足、研究级硬件六条流派排列，新建 24 / 复用 12 个独立节点，并列出文章与 README 不一致之处。"
---

# 四足机器人开源项目技术地图：从极简运动学到研究级硬件（36 个 GitHub 项目）

> **本页定位**：为 [四足机器人开源项目深度解析](https://mp.weixin.qq.com/s/pRGoPm41qYCNyHPWR6Dl6A)（2026-11-01，古月居转载敢敢AUTOHUB）提供按流派组织的阅读坐标；文中 36 个项目均有唯一详情节点。

## 一句话观点

**文章的主线是「从极简运动学到研究级硬件」：越往右越贵、越依赖算力与硬件，开源项目的价值在于看清不同流派的工程权衡。**（文章结论的概括；其具体项目描述有若干与 README 不符，见下表）

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MPC | Model Predictive Control | 模型预测控制 |
| WBC | Whole-Body Control | 全身控制 |
| RL | Reinforcement Learning | 强化学习 |
| IK | Inverse Kinematics | 逆运动学 |
| QDD | Quasi-Direct Drive | 准直驱 |

## 为什么单独做这张地图

- 综述一次点名 36 个仓库，库内已有 12 个节点散落在各处，需要一个统一入口；
- 文章对若干项目的描述与仓库 README 不一致（如 CHAMP 并非凸 MPC、legged_control 已停止维护），需要显式标注，避免把转述当事实。

## 分组索引

### 极简运动学与状态机

| 项目 | 节点 | 星标（文章） |
|------|------|-------------|
| Stanford Pupper | [stanford-doggo-and-pupper](../entities/stanford-doggo-and-pupper.md)（复用） | 1676 |
| Stanford Doggo | [stanford-doggo-and-pupper](../entities/stanford-doggo-and-pupper.md)（复用） | 2508 |
| Petoi OpenCat | [petoi-bittle-opencat](../entities/petoi-bittle-opencat.md)（复用） | 4713 |

### ROS 生态

| 项目 | 节点 | 星标（文章） |
|------|------|-------------|
| Mini Pupper | [mini-pupper](../entities/mini-pupper.md)（复用） | 36 |
| SpotMicro（mike4192） | [spotmicro-ros-mike4192](../entities/spotmicro-ros-mike4192.md) | 2043 |
| MangDang QuadrupedRobot | [mangdang-quadruped-robot](../entities/mangdang-quadruped-robot.md) | 1088 |

### 优化控制（MPC / WBC）

| 项目 | 节点 | 星标（文章） |
|------|------|-------------|
| legged_control | [legged-control-ocs2](../entities/legged-control-ocs2.md) | （文中未给） |
| CHAMP | [champ-quadruped-controller](../entities/champ-quadruped-controller.md) | 2196 |
| Quadruped-PyMPC | [quadruped-pympc](../entities/quadruped-pympc.md) | 446 |
| TOWR | [towr-trajectory-optimization](../entities/towr-trajectory-optimization.md) | 1049 |
| DIAL-MPC | [dial-mpc](../entities/dial-mpc.md) | 952 |

### 强化学习

| 项目 | 节点 | 星标（文章） |
|------|------|-------------|
| Isaac Lab | [isaac-lab](../entities/isaac-lab.md)（复用） | （文中未给） |
| Legged Gym | [legged-gym](../entities/legged-gym.md)（复用） | 1800+ |
| Spot Mini Mini | [spot-mini-mini](../entities/spot-mini-mini.md) | 919 |
| Rex-Gym | [rex-gym](../entities/rex-gym.md) | 1089 |
| rl-mpc-locomotion | [rl-mpc-locomotion](../entities/rl-mpc-locomotion.md) | 946 |
| ABS（Agile But Safe） | [abs-agile-but-safe](../entities/abs-agile-but-safe.md) | （文中未给） |
| rl_sar | [rl-sar](../entities/rl-sar.md) | 1230 |

### 轮足机器人

| 项目 | 节点 | 星标（文章） |
|------|------|-------------|
| Wheel Legged Genesis | [wheel-legged-genesis](../entities/wheel-legged-genesis.md)（复用） | 340 |
| Isaac-RL-Two-wheel-Legged-Bot | [isaac-rl-two-wheel-legged-bot](../entities/isaac-rl-two-wheel-legged-bot.md)（复用） | 287 |
| Micro-Wheeled_leg-Robot | [paper-notebook-micro-wheeled-leg-robot](../entities/paper-notebook-micro-wheeled-leg-robot.md)（复用） | （文中未给） |
| Wheel-Legged-Gym | [wheel-legged-gym](../entities/wheel-legged-gym.md) | （文中未给） |
| legged-robots-manipulation | [legged-robots-manipulation](../entities/legged-robots-manipulation.md) | （文中未给） |
| Ascend Quadruped | [ascend-quadruped](../entities/ascend-quadruped.md) | （文中未给） |
| Wheeled Biped | [wheeled-bipedal-robot](../entities/wheeled-bipedal-robot.md) | （文中未给） |
| FLORES | [flores-wheel-legged-robot](../entities/flores-wheel-legged-robot.md) | 25 |
| FOC 双轮腿机器人 | [foc-wheel-legged-robot](../entities/foc-wheel-legged-robot.md) | 1557 |
| LocoLeggedWheel | [locoleggedwheel](../entities/locoleggedwheel.md) | 115 |
| WheelLeggedRobotSimulation | [wheel-legged-robot-simulation](../entities/wheel-legged-robot-simulation.md) | 73 |
| 启明-1 Venus-1 | [venus-1-four-link-wheel-legged-robot](../entities/venus-1-four-link-wheel-legged-robot.md) | 60 |
| awesome-wheeled-legged | [awesome-wheeled-legged](../entities/awesome-wheeled-legged.md) | 29 |

### 研究级硬件与驱动

| 项目 | 节点 | 星标（文章） |
|------|------|-------------|
| Solo（Open Robot Actuator） | [odri-solo-and-bolt](../entities/odri-solo-and-bolt.md)（复用） | 1371 |
| OpenQuadruped | [open-quadruped](../entities/open-quadruped.md)（复用） | 462 |
| moteus | [moteus](../entities/moteus.md)（复用） | 1137 |

### 其他

| 项目 | 节点 | 星标（文章） |
|------|------|-------------|
| Barkour Robot | [paper-barkour-quadruped-agility-benchmark](../entities/paper-barkour-quadruped-agility-benchmark.md)（复用） | 336 |
| Sesame Robot | [sesame-robot](../entities/sesame-robot.md) | 1309 |
| 菠萝狗 Py-Apple Dog | [py-apple-quadruped](../entities/py-apple-quadruped.md) | 1224 |

## 文章说法与 README 核对

| 项目 | 文章说法 | README / 核查 | 处理 |
|------|----------|---------------|------|
| CHAMP | 文章称含 `convex_mpc` 与摩擦锥 QP | README 称控制框架基于 MIT 的模式调制 + 阻抗控制分层控制器，未提 MPC | 以 README 为准 |
| legged_control | 文章称为成熟 ROS1 项目 | README 声明软件已不再维护，作者转向新框架 | 文章未提，已补 |
| ABS | 文章称 CMU+ETH 联合、MPPI 规划 | README 为 RSS 2024，署名含 Tairan He、Chong Zhang、Guanya Shi 等（未标机构）；未见 MPPI / ETH 联合字样；CC BY-NC 4.0 禁商用 | 文章细节未核实 |
| Barkour Robot | 文章称完整代码尚未开源 | README 称仓库含设计资产（CAD / PCBA / 装配）与固件、底层核心软件 | RL 训练代码是否开放未核实 |
| WheelLeggedRobotSimulation | 文章称支持多种仿真器 | README 只提 MATLAB（PID / LQR / MPC） | 以 README 为准 |
| rl_sar | 文章归入「轮足」 | README 称面向四足、轮式、人形的通用 RL 部署框架 | 文章归类偏窄 |
| Ascend Quadruped / Wheeled Biped | 文章称完整开源 | 入库日 git ls-remote 失败，无法核对 | 标未核实 |

> 星标数与「已开源」类表述为入库日（2026-11-01）读取的 README 与 `git ls-remote` 结果；星标数为文章表述。

## 关联页面

- [四足机器人总览](../entities/quadruped-robot.md)
- [运动控制任务页](../tasks/locomotion.md)
- [开源 QDD 执行器项目对比](../comparisons/open-source-qdd-actuator-projects.md)

## 参考来源

- [wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md](../../sources/blogs/wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md)
