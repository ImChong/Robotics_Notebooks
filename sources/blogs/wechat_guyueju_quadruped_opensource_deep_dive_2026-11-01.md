# 四足机器人开源项目深度解析：从入门到研究级的完整技术图谱

> 来源归档（blog / 微信公众号）

- **标题：** 四足机器人开源项目深度解析：从入门到研究级的完整技术图谱
- **类型：** blog
- **来源账号：** 古月居（转载），文内标注 The following article is from 敢敢AUTOHUB，Author 敢敢不敢写bug
- **原始链接：** https://mp.weixin.qq.com/s/pRGoPm41qYCNyHPWR6Dl6A
- **发表日期：** 2026-11-01
- **入库日期：** 2026-11-01
- **抓取方式：** `scripts/fetch_wechat_article.py`（Chromium + 微信内置 UA），正文 18453 字符
- **一句话说明：** 按「极简运动学 / ROS / 优化控制 / 强化学习 / 轮足 / 研究级硬件」六条流派梳理 GitHub 上 20 多个四足开源项目，附资源汇总与趋势判断。
- **说明：** 文中点名 **36 个 GitHub 仓库**（Stanford Pupper 与 Stanford Doggo 在库内共用一个节点）：**新建 24、复用 12 个节点**，每个项目唯一节点。星标数为文章表述，入库日未复核；README 核查见各节点与下表。

## 项目 → 节点

| # | 流派 | 项目 | 仓库 | 星标（文章） | 节点 | 状态 | 备注 |
|---|------|------|------|-------------|------|------|------|
| 01 | 极简运动学与状态机 | Stanford Pupper | [stanfordroboticsclub/StanfordQuadruped](https://github.com/stanfordroboticsclub/StanfordQuadruped) | 1676 | [stanford-doggo-and-pupper](../../wiki/entities/stanford-doggo-and-pupper.md) | 复用 | Pupper 与 Doggo 在库内共用此节点（该节点已同时覆盖两个仓库） |
| 02 | 极简运动学与状态机 | Stanford Doggo | [Nate711/StanfordDoggoProject](https://github.com/Nate711/StanfordDoggoProject) | 2508 | [stanford-doggo-and-pupper](../../wiki/entities/stanford-doggo-and-pupper.md) | 复用 | 同上 |
| 03 | 极简运动学与状态机 | Petoi OpenCat | [PetoiCamp/OpenCat-Quadruped-Robot](https://github.com/PetoiCamp/OpenCat-Quadruped-Robot) | 4713 | [petoi-bittle-opencat](../../wiki/entities/petoi-bittle-opencat.md) | 复用 |  |
| 04 | ROS 生态 | Mini Pupper | [mangdangroboticsclub/mini_pupper](https://github.com/mangdangroboticsclub/mini_pupper) | 36 | [mini-pupper](../../wiki/entities/mini-pupper.md) | 复用 |  |
| 05 | ROS 生态 | SpotMicro（mike4192） | [mike4192/spotMicro](https://github.com/mike4192/spotMicro) | 2043 | [spotmicro-ros-mike4192](../../wiki/entities/spotmicro-ros-mike4192.md) | 新建 |  |
| 06 | ROS 生态 | MangDang QuadrupedRobot | [mangdangroboticsclub/QuadrupedRobot](https://github.com/mangdangroboticsclub/QuadrupedRobot) | 1088 | [mangdang-quadruped-robot](../../wiki/entities/mangdang-quadruped-robot.md) | 新建 |  |
| 07 | 优化控制（MPC / WBC） | legged_control | [qiayuanl/legged_control](https://github.com/qiayuanl/legged_control) | （文中未给） | [legged-control-ocs2](../../wiki/entities/legged-control-ocs2.md) | 新建 |  |
| 08 | 优化控制（MPC / WBC） | CHAMP | [chvmp/champ](https://github.com/chvmp/champ) | 2196 | [champ-quadruped-controller](../../wiki/entities/champ-quadruped-controller.md) | 新建 |  |
| 09 | 优化控制（MPC / WBC） | Quadruped-PyMPC | [iit-DLSLab/Quadruped-PyMPC](https://github.com/iit-DLSLab/Quadruped-PyMPC) | 446 | [quadruped-pympc](../../wiki/entities/quadruped-pympc.md) | 新建 |  |
| 10 | 优化控制（MPC / WBC） | TOWR | [ethz-adrl/towr](https://github.com/ethz-adrl/towr) | 1049 | [towr-trajectory-optimization](../../wiki/entities/towr-trajectory-optimization.md) | 新建 |  |
| 11 | 优化控制（MPC / WBC） | DIAL-MPC | [LeCAR-Lab/dial-mpc](https://github.com/LeCAR-Lab/dial-mpc) | 952 | [dial-mpc](../../wiki/entities/dial-mpc.md) | 新建 |  |
| 12 | 强化学习 | Isaac Lab | [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) | （文中未给） | [isaac-lab](../../wiki/entities/isaac-lab.md) | 复用 |  |
| 13 | 强化学习 | Legged Gym | [leggedrobotics/legged_gym](https://github.com/leggedrobotics/legged_gym) | 1800+ | [legged-gym](../../wiki/entities/legged-gym.md) | 复用 |  |
| 14 | 强化学习 | Spot Mini Mini | [OpenQuadruped/spot_mini_mini](https://github.com/OpenQuadruped/spot_mini_mini) | 919 | [spot-mini-mini](../../wiki/entities/spot-mini-mini.md) | 新建 |  |
| 15 | 强化学习 | Rex-Gym | [nicrusso7/rex-gym](https://github.com/nicrusso7/rex-gym) | 1089 | [rex-gym](../../wiki/entities/rex-gym.md) | 新建 |  |
| 16 | 强化学习 | rl-mpc-locomotion | [silvery107/rl-mpc-locomotion](https://github.com/silvery107/rl-mpc-locomotion) | 946 | [rl-mpc-locomotion](../../wiki/entities/rl-mpc-locomotion.md) | 新建 |  |
| 17 | 强化学习 | ABS（Agile But Safe） | [LeCAR-Lab/ABS](https://github.com/LeCAR-Lab/ABS) | （文中未给） | [abs-agile-but-safe](../../wiki/entities/abs-agile-but-safe.md) | 新建 |  |
| 18 | 强化学习 | rl_sar | [fan-ziqi/rl_sar](https://github.com/fan-ziqi/rl_sar) | 1230 | [rl-sar](../../wiki/entities/rl-sar.md) | 新建 |  |
| 19 | 轮足机器人 | Wheel Legged Genesis | [Albusgive/wheel_legged_genesis](https://github.com/Albusgive/wheel_legged_genesis) | 340 | [wheel-legged-genesis](../../wiki/entities/wheel-legged-genesis.md) | 复用 |  |
| 20 | 轮足机器人 | Isaac-RL-Two-wheel-Legged-Bot | [jaykorea/Isaac-RL-Two-wheel-Legged-Bot](https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot) | 287 | [isaac-rl-two-wheel-legged-bot](../../wiki/entities/isaac-rl-two-wheel-legged-bot.md) | 复用 |  |
| 21 | 轮足机器人 | Micro-Wheeled_leg-Robot | [MuShibo/Micro-Wheeled_leg-Robot](https://github.com/MuShibo/Micro-Wheeled_leg-Robot) | （文中未给） | [paper-notebook-micro-wheeled-leg-robot](../../wiki/entities/paper-notebook-micro-wheeled-leg-robot.md) | 复用 | 按项目名匹配到库内待深读占位节点（该节点未记 GitHub 链接） |
| 22 | 轮足机器人 | Wheel-Legged-Gym | [clearlab-sustech/Wheel-Legged-Gym](https://github.com/clearlab-sustech/Wheel-Legged-Gym) | （文中未给） | [wheel-legged-gym](../../wiki/entities/wheel-legged-gym.md) | 新建 |  |
| 23 | 轮足机器人 | legged-robots-manipulation | [aCodeDog/legged-robots-manipulation](https://github.com/aCodeDog/legged-robots-manipulation) | （文中未给） | [legged-robots-manipulation](../../wiki/entities/legged-robots-manipulation.md) | 新建 |  |
| 24 | 轮足机器人 | Ascend Quadruped | [DRCL-USC/Ascend_Quadruped](https://github.com/DRCL-USC/Ascend_Quadruped) | （文中未给） | [ascend-quadruped](../../wiki/entities/ascend-quadruped.md) | 新建 |  |
| 25 | 轮足机器人 | Wheeled Biped | [Derek-TH-Wang/wheeled_bipedal_robot](https://github.com/Derek-TH-Wang/wheeled_bipedal_robot) | （文中未给） | [wheeled-bipedal-robot](../../wiki/entities/wheeled-bipedal-robot.md) | 新建 |  |
| 26 | 轮足机器人 | FLORES | [ZhichengSong6/FLORES](https://github.com/ZhichengSong6/FLORES) | 25 | [flores-wheel-legged-robot](../../wiki/entities/flores-wheel-legged-robot.md) | 新建 |  |
| 27 | 轮足机器人 | FOC 双轮腿机器人 | [Skythinker616/foc-wheel-legged-robot](https://github.com/Skythinker616/foc-wheel-legged-robot) | 1557 | [foc-wheel-legged-robot](../../wiki/entities/foc-wheel-legged-robot.md) | 新建 |  |
| 28 | 轮足机器人 | LocoLeggedWheel | [zhaozijie2022/LocoLeggedWheel](https://github.com/zhaozijie2022/LocoLeggedWheel) | 115 | [locoleggedwheel](../../wiki/entities/locoleggedwheel.md) | 新建 |  |
| 29 | 轮足机器人 | WheelLeggedRobotSimulation | [Rengar-Yang/WheelLeggedRobotSimulation](https://github.com/Rengar-Yang/WheelLeggedRobotSimulation) | 73 | [wheel-legged-robot-simulation](../../wiki/entities/wheel-legged-robot-simulation.md) | 新建 |  |
| 30 | 轮足机器人 | 启明-1 Venus-1 | [track-tech/Venus-1__Four-link-wheel-legged-robot](https://github.com/track-tech/Venus-1__Four-link-wheel-legged-robot) | 60 | [venus-1-four-link-wheel-legged-robot](../../wiki/entities/venus-1-four-link-wheel-legged-robot.md) | 新建 |  |
| 31 | 轮足机器人 | awesome-wheeled-legged | [XinLang2019/awesome-wheeled-legged](https://github.com/XinLang2019/awesome-wheeled-legged) | 29 | [awesome-wheeled-legged](../../wiki/entities/awesome-wheeled-legged.md) | 新建 |  |
| 32 | 研究级硬件与驱动 | Solo（Open Robot Actuator） | [open-dynamic-robot-initiative/open_robot_actuator_hardware](https://github.com/open-dynamic-robot-initiative/open_robot_actuator_hardware) | 1371 | [odri-solo-and-bolt](../../wiki/entities/odri-solo-and-bolt.md) | 复用 |  |
| 33 | 研究级硬件与驱动 | OpenQuadruped | [adham-elarabawy/open-quadruped](https://github.com/adham-elarabawy/open-quadruped) | 462 | [open-quadruped](../../wiki/entities/open-quadruped.md) | 复用 |  |
| 34 | 研究级硬件与驱动 | moteus | [mjbots/moteus](https://github.com/mjbots/moteus) | 1137 | [moteus](../../wiki/entities/moteus.md) | 复用 | 仅出现在附录清单 |
| 35 | 其他 | Barkour Robot | [google-deepmind/barkour_robot](https://github.com/google-deepmind/barkour_robot) | 336 | [paper-barkour-quadruped-agility-benchmark](../../wiki/entities/paper-barkour-quadruped-agility-benchmark.md) | 复用 | 文中称「完整代码尚未开源」；README 实为设计资产（CAD / PCBA / 装配）与固件、底层代码，见地图页核对 |
| 36 | 其他 | Sesame Robot | [dorianborian/sesame-robot](https://github.com/dorianborian/sesame-robot) | 1309 | [sesame-robot](../../wiki/entities/sesame-robot.md) | 新建 |  |
| 37 | 其他 | 菠萝狗 Py-Apple Dog | [ToanTech/py-apple-quadruped-robot](https://github.com/ToanTech/py-apple-quadruped-robot) | 1224 | [py-apple-quadruped](../../wiki/entities/py-apple-quadruped.md) | 新建 |  |

## 文章说法与 README 核对（有差异的项）

| 项目 | 文章说法 | README / 核查 | 处理 |
|------|----------|---------------|------|
| CHAMP | 文章称含 `convex_mpc` 与摩擦锥 QP | README 称控制框架基于 MIT 的模式调制 + 阻抗控制分层控制器，未提 MPC | 以 README 为准 |
| legged_control | 文章称为成熟 ROS1 项目 | README 声明软件已不再维护，作者转向新框架 | 文章未提，已补 |
| ABS | 文章称 CMU+ETH 联合、MPPI 规划 | README 为 RSS 2024，署名含 Tairan He、Chong Zhang、Guanya Shi 等（未标机构）；未见 MPPI / ETH 联合字样；CC BY-NC 4.0 禁商用 | 文章细节未核实 |
| Barkour Robot | 文章称完整代码尚未开源 | README 称仓库含设计资产（CAD / PCBA / 装配）与固件、底层核心软件 | RL 训练代码是否开放未核实 |
| WheelLeggedRobotSimulation | 文章称支持多种仿真器 | README 只提 MATLAB（PID / LQR / MPC） | 以 README 为准 |
| rl_sar | 文章归入「轮足」 | README 称面向四足、轮式、人形的通用 RL 部署框架 | 文章归类偏窄 |
| Ascend Quadruped / Wheeled Biped | 文章称完整开源 | 入库日 git ls-remote 失败，无法核对 | 标未核实 |

## 对 wiki 的映射

- 阅读坐标：[四足机器人开源项目技术地图：从极简运动学到研究级硬件（36 个 GitHub 项目）](../../wiki/overview/quadruped-open-source-projects-technology-map.md)
- 原文趋势判断（文档质量参差、硬件依赖强、仿真与现实差距）仅作文章观点收录，未独立验证。
