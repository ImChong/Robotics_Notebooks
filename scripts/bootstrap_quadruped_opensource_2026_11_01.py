#!/usr/bin/env python3
"""Bootstrap ingest: 古月居转载《四足机器人开源项目深度解析：从入门到研究级的完整技术图谱》（2026-11-01）。

- 原作者：敢敢AUTOHUB / 敢敢不敢写bug；抓取：`scripts/fetch_wechat_article.py`。
- 文中点名 36 个 GitHub 项目（Stanford Pupper 与 Stanford Doggo 在库内共用一个节点）：
  新建 24 个独立 `wiki/entities/` 节点，复用 12 个已有节点；每个项目一个唯一节点。
- 核查（入库日 2026-11-01）：`git ls-remote` + 浅克隆 README（仅读，不运行）；
  文章说法与 README 不一致处在节点「文章说法核对」一节标出。
- 用法：python3 scripts/bootstrap_quadruped_opensource_2026_11_01.py <抓取的 txt>
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TODAY = date.today().isoformat()
URL = "https://mp.weixin.qq.com/s/pRGoPm41qYCNyHPWR6Dl6A"
BLOG = "wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md"
MAP = "quadruped-open-source-projects-technology-map.md"
MAP_TITLE = "四足机器人开源项目技术地图：从极简运动学到研究级硬件（36 个 GitHub 项目）"

G_MIN = "极简运动学与状态机"
G_ROS = "ROS 生态"
G_OPT = "优化控制（MPC / WBC）"
G_RL = "强化学习"
G_WL = "轮足机器人"
G_HW = "研究级硬件与驱动"
G_OTHER = "其他"
GROUP_ORDER = [G_MIN, G_ROS, G_OPT, G_RL, G_WL, G_HW, G_OTHER]

LOC = "../tasks/locomotion.md"
QR = "../entities/quadruped-robot.md"
MPC = "../methods/model-predictive-control.md"
NMPC = "../methods/nonlinear-model-predictive-control.md"
RL = "../methods/reinforcement-learning.md"
WBC = "../concepts/whole-body-control.md"
S2R = "../concepts/sim2real.md"
DR = "../concepts/domain-randomization.md"
LGYM = "../entities/legged-gym.md"
RSL = "../entities/rsl-rl.md"
ILAB = "../entities/isaac-lab.md"
IGYM = "../entities/isaac-gym.md"
OCS2 = "../entities/ocs2.md"
ACADOS = "../entities/acados.md"
MPPI = "../methods/mppi.md"
LOCOMANIP = "../tasks/loco-manipulation.md"
QDD = "../comparisons/open-source-qdd-actuator-projects.md"
SFOC = "../entities/simplefoc.md"
DIALP = "../entities/paper-notebook-full-order-sampling-based-mpc-for-torque-level-l.md"
ROBOTLAB = "../entities/robot-lab.md"

# 复用节点（库内已有）：repo -> (wiki 节点, 文中项目名)
REUSE: list[dict[str, Any]] = [
    dict(
        repo="stanfordroboticsclub/StanfordQuadruped",
        name="Stanford Pupper",
        node="stanford-doggo-and-pupper",
        stars="1676",
        group=G_MIN,
        note="Pupper 与 Doggo 在库内共用此节点（该节点已同时覆盖两个仓库）",
    ),
    dict(
        repo="Nate711/StanfordDoggoProject",
        name="Stanford Doggo",
        node="stanford-doggo-and-pupper",
        stars="2508",
        group=G_MIN,
        note="同上",
    ),
    dict(
        repo="PetoiCamp/OpenCat-Quadruped-Robot",
        name="Petoi OpenCat",
        node="petoi-bittle-opencat",
        stars="4713",
        group=G_MIN,
        note="",
    ),
    dict(
        repo="mangdangroboticsclub/mini_pupper",
        name="Mini Pupper",
        node="mini-pupper",
        stars="36",
        group=G_ROS,
        note="",
    ),
    dict(
        repo="isaac-sim/IsaacLab",
        name="Isaac Lab",
        node="isaac-lab",
        stars="（文中未给）",
        group=G_RL,
        note="",
    ),
    dict(
        repo="leggedrobotics/legged_gym",
        name="Legged Gym",
        node="legged-gym",
        stars="1800+",
        group=G_RL,
        note="",
    ),
    dict(
        repo="open-dynamic-robot-initiative/open_robot_actuator_hardware",
        name="Solo（Open Robot Actuator）",
        node="odri-solo-and-bolt",
        stars="1371",
        group=G_HW,
        note="",
    ),
    dict(
        repo="adham-elarabawy/open-quadruped",
        name="OpenQuadruped",
        node="open-quadruped",
        stars="462",
        group=G_HW,
        note="",
    ),
    dict(
        repo="mjbots/moteus",
        name="moteus",
        node="moteus",
        stars="1137",
        group=G_HW,
        note="仅出现在附录清单",
    ),
    dict(
        repo="google-deepmind/barkour_robot",
        name="Barkour Robot",
        node="paper-barkour-quadruped-agility-benchmark",
        stars="336",
        group=G_OTHER,
        note="文中称「完整代码尚未开源」；README 实为设计资产（CAD / PCBA / 装配）与固件、底层代码，见地图页核对",
    ),
    dict(
        repo="Albusgive/wheel_legged_genesis",
        name="Wheel Legged Genesis",
        node="wheel-legged-genesis",
        stars="340",
        group=G_WL,
        note="",
    ),
    dict(
        repo="jaykorea/Isaac-RL-Two-wheel-Legged-Bot",
        name="Isaac-RL-Two-wheel-Legged-Bot",
        node="isaac-rl-two-wheel-legged-bot",
        stars="287",
        group=G_WL,
        note="",
    ),
    dict(
        repo="MuShibo/Micro-Wheeled_leg-Robot",
        name="Micro-Wheeled_leg-Robot",
        node="paper-notebook-micro-wheeled-leg-robot",
        stars="（文中未给）",
        group=G_WL,
        note="按项目名匹配到库内待深读占位节点（该节点未记 GitHub 链接）",
    ),
]

# 新建节点
NEW: list[dict[str, Any]] = [
    dict(
        slug="spotmicro-ros-mike4192",
        inst="independent",
        name="SpotMicro（mike4192）",
        repo="mike4192/spotMicro",
        stars="2043",
        forks="484",
        last="2021-03-20",
        group=G_ROS,
        tags=["quadruped", "ros", "spot-micro", "diy"],
        one="mike4192/spotMicro 是基于 ROS 的 Spot Micro 四足机器人运动控制源码：实现坐/站/姿态角/行走控制，并带 URDF、TF2 与里程计、SLAM 等配套。",
        article="文章把它称作「开源界的波士顿动力」，强调完整 ROS 集成：`spot_micro_motion_cmd` 节点订阅 `/cmd_vel`，转成步态参数与足端轨迹，再经逆运动学发布关节角；并有完整 TF 树。",
        readme="README 目录列有 ROS 节点说明、URDF、TF2 发布与里程计、SLAM 等章节，概述称实现了 3D 打印 Spot Micro 的坐、站、姿态角与行走控制。",
        caveat="库内 `spotmicro` 节点对应的是 SpotMicroAI（GitLab 主仓库），与本仓库不同，故单独建节点。节点名 `spot_micro_motion_cmd` 与 `/cmd_vel` 流程来自文章，未在 README 前 700 字内逐项核对。",
        related=[QR, LOC, "../entities/spotmicro.md"],
        abbr=[
            ("ROS", "Robot Operating System", "机器人中间件"),
            ("TF", "Transform", "ROS 坐标变换库"),
            ("IK", "Inverse Kinematics", "逆运动学"),
        ],
    ),
    dict(
        slug="mangdang-quadruped-robot",
        inst="independent",
        name="MangDang QuadrupedRobot",
        repo="mangdangroboticsclub/QuadrupedRobot",
        stars="1088",
        forks="",
        last="2026-10-29",
        group=G_ROS,
        tags=["quadruped", "ros2", "mini-pupper", "sim2real"],
        one="mangdangroboticsclub/QuadrupedRobot 是 MangDang 的 Mini Pupper 系列软件仓库，README 当前主打 Mini Pupper 4：3-DoF coreless 智能舵机、ROS 2 Jazzy + micro-ROS、SLAM 与 Nav2。",
        article="文章只在附录清单列出（1088 星），未展开描述。",
        readme="README 称支持在仿真中训练神经网络策略并部署到实机（Sim2Real）、与 BAM（Better Actuator Models）兼容以刻画关节摩擦，并提供 ROS 2 Jazzy 与 micro-ROS 模板。",
        caveat="与库内 `mini-pupper`（mini_pupper 仓库）是同一厂商的不同仓库，故分开建节点。",
        related=[QR, "../entities/mini-pupper.md", S2R],
        abbr=[
            ("BAM", "Better Actuator Models", "更好的执行器模型"),
            ("ROS 2", "Robot Operating System 2", "第二代机器人中间件"),
        ],
    ),
    dict(
        slug="legged-control-ocs2",
        inst="independent",
        name="legged_control",
        repo="qiayuanl/legged_control",
        stars="（文中未给）",
        forks="",
        last="2025-02-13",
        group=G_OPT,
        tags=["quadruped", "mpc", "wbc", "ocs2", "ros"],
        one="legged_control 是基于 OCS2 与 ros-control 的足式机器人 NMPC + WBC + 状态估计 + sim2real 框架（作者 Qiayuan Liao 等）。",
        article="文章称其在 Gazebo 中运行，包含 EKF/互补滤波状态估计、基于 SRBD 的 MPC、WBC 执行器与步态调度器；要求 ROS1 Noetic / Ubuntu 20.04。",
        readme="README 标题引用为「NMPC, WBC, state estimation, and sim2real framework for legged robots based on OCS2 and ros-controls」；顶部声明**软件已不再维护**，作者转向全新框架，并推荐带感知的 legged_perceptive。",
        caveat="文章未提「不再维护」；SRBD / SQP 等细节来自文章，未在 README 中逐项核对。",
        related=[OCS2, MPC, WBC, LOC],
        abbr=[
            ("OCS2", "Optimal Control for Switched Systems", "ETH 切换系统最优控制库"),
            ("NMPC", "Nonlinear Model Predictive Control", "非线性模型预测控制"),
            ("WBC", "Whole-Body Control", "全身控制"),
        ],
    ),
    dict(
        slug="champ-quadruped-controller",
        inst="independent",
        name="CHAMP",
        repo="chvmp/champ",
        stars="2196",
        forks="428",
        last="2024-07-04",
        group=G_OPT,
        tags=["quadruped", "ros", "controller", "gait"],
        one="CHAMP 是用于搭建新四足机器人与开发控制算法的 ROS 开源框架，控制框架基于 MIT Cheetah 的「模式调制 + 阻抗控制」分层控制器论文。",
        article="文章把它描述为实现了完整 MPC：SRBD 模型、`convex_mpc` 模块与摩擦锥线性化 QP，另提及 champ_setup_assistant 与多个预配置机器人模型。",
        readme="README 核心特性：基于 ROS 导航栈的自主能力、setup-assistant 配置新机器人、预配置 URDF（Anymal、MIT Mini Cheetah、Spot、LittleDog）、Gazebo 仿真，并兼容 SpotMicroAI 与 OpenQuadruped。README 称控制框架依据的是 MIT「Hierarchical controller for highly dynamic locomotion utilizing pattern modulation and impedance control」论文。",
        caveat="**文章的 MPC / `convex_mpc` / 摩擦锥 QP 描述与 README 不符**：README 指向的是模式调制 + 阻抗控制的分层控制器，未提 MPC。本节点以 README 为准；文章说法未核实，勿据此判断 CHAMP 含凸 MPC。",
        related=[QR, LOC, "../entities/open-quadruped.md", "../entities/spotmicro.md"],
        abbr=[
            ("CHAMP", "Quadruped Controller（CHAMP 框架）", "ROS 四足控制框架"),
            ("URDF", "Unified Robot Description Format", "机器人描述格式"),
        ],
    ),
    dict(
        slug="quadruped-pympc",
        inst="iit",
        name="Quadruped-PyMPC",
        repo="iit-DLSLab/Quadruped-PyMPC",
        stars="446",
        forks="",
        last="2026-10-15",
        group=G_OPT,
        tags=["quadruped", "mpc", "acados", "jax", "mujoco"],
        one="Quadruped-PyMPC（IIT DLS Lab）是基于单刚体模型、用 Python 写的 MPC：梯度式（acados）与采样式（JAX）两种实现，已在真机测试并兼容 MuJoCo。",
        article="文章称它提供基于 Python 的 MPC 与 WBC 实现，用 acados 求解器，适合学习。",
        readme="README：梯度式 MPC 在 i7-13700H 上计算不到 5 ms，可选积分器、足端力平滑、落脚点优化、实时迭代等；采样式 MPC 在 NVIDIA 4050 mobile GPU 上 10000 并行 rollout 不到 2 ms。",
        caveat="文章说 WBC 实现；README 前段只见 MPC 两种实现，WBC 未在已读段落出现，未核实。",
        related=[MPC, NMPC, ACADOS, MPPI, LOC],
        abbr=[
            ("MPC", "Model Predictive Control", "模型预测控制"),
            ("SRBD", "Single Rigid Body Dynamics", "单刚体动力学"),
            ("RTI", "Real-Time Iteration", "实时迭代"),
        ],
    ),
    dict(
        slug="towr-trajectory-optimization",
        inst="eth",
        name="TOWR",
        repo="ethz-adrl/towr",
        stars="1049",
        forks="",
        last="2021-03-29",
        group=G_OPT,
        tags=["legged", "trajectory-optimization", "cpp", "ros", "eth"],
        one="TOWR（ETH ADRL）是轻量、可扩展的足式机器人轨迹优化 C++ 库：提供一套变量、代价与约束基础集合，可组合出单足跳、双足行走与完整四足小跑循环，并同时优化步态与步时长（论文称 100 ms 内）。",
        article="文章仅在附录清单列出（1049 星）。",
        readme="README 称借助 ifopt 可用 Ipopt、Snopt 高性能求解器，已用于生成单足跳、双足行走和四足小跑；仓库带 BSD-3 许可徽章，最近提交 2021-03-29。",
        caveat="仓库近年无更新（最近提交 2021-03-29）。",
        related=[MPC, LOC, QR],
        abbr=[
            ("TOWR", "Trajectory Optimization for Walking Robots", "步行机器人轨迹优化"),
            ("ifopt", "Interface to Optimization Solvers", "优化求解器接口"),
        ],
    ),
    dict(
        slug="dial-mpc",
        inst="cmu",
        name="DIAL-MPC",
        repo="LeCAR-Lab/dial-mpc",
        stars="952",
        forks="",
        last="2025-05-28",
        group=G_OPT,
        tags=["legged", "mpc", "sampling-based", "jax", "diffusion"],
        one="DIAL-MPC（Diffusion-Inspired Annealing for Legged MPC）是 training-free 的全阶、力矩级采样式 MPC，直接在 Brax 物理仿真中采样 rollout，不需要降阶建模、线性化或预设接触序列（ICRA 2025 Best Paper Finalist）。",
        article="文章仅在附录清单列出（952 星）。",
        readme="README 标注 ICRA 2025 Best Paper Finalist，对应论文「Full-Order Sampling-Based MPC for Torque-Level Locomotion Control via Diffusion-Style Annealing」（arXiv:2409.15610），Apache-2.0 徽章，基于 JAX。",
        caveat="",
        related=[DIALP, MPC, MPPI, LOC],
        abbr=[
            ("DIAL", "Diffusion-Inspired Annealing for Legged", "扩散式退火（足式 MPC）"),
            ("MPC", "Model Predictive Control", "模型预测控制"),
        ],
    ),
    dict(
        slug="spot-mini-mini",
        inst="independent",
        name="Spot Mini Mini",
        repo="OpenQuadruped/spot_mini_mini",
        stars="919",
        forks="191",
        last="2022-10-05",
        group=G_RL,
        tags=["quadruped", "rl", "pybullet", "gym", "spot-micro"],
        one="Spot Mini Mini 是为 Spot Micro 社区做的 PyBullet 仿真与 OpenAI Gym 环境，用于在不损坏机器人的前提下做强化学习与控制实验。",
        article="文章称其核心贡献是展示域随机化：随机化质量、质心、连杆长度、摩擦、电机延迟、传感器噪声与地形等，以提高策略鲁棒性。",
        readme="README 动机段说明作者为 Spot Micro 社区开发 PyBullet 仿真器并可作 Gym 环境；顶部声明开发因雇主 NDA 于 2020 年 11 月暂停。",
        caveat="文章列举的具体随机化项（电机延迟、齿轮间隙等）未在已读 README 段落中逐项核对。",
        related=[RL, DR, QR, "../entities/spotmicro.md"],
        abbr=[
            ("DR", "Domain Randomization", "域随机化"),
            ("PyBullet", "Python Bullet Physics", "物理仿真库"),
        ],
    ),
    dict(
        slug="rex-gym",
        inst="independent",
        name="Rex-Gym",
        repo="nicrusso7/rex-gym",
        stars="1089",
        forks="",
        last="2021-05-15",
        group=G_RL,
        tags=["quadruped", "rl", "openai-gym", "ppo", "3d-printed"],
        one="Rex-Gym 是 3D 打印开源四足机器人 Rex 的 OpenAI Gym 训练环境集合：含 Rex URDF、PPO 学习智能体与训练 / 回放脚本，目标是仿真学到的策略无需再手调即可迁移到真机。",
        article="文章称其提供多个训练任务与精心设计的奖励函数（前进速度、能量、冲击力、姿态偏差、关节速度等惩罚），并使用课程学习。",
        readme="README 称仓库包含用于训练 Rex 的 Gym 环境、Rex URDF 模型、PPO 学习智能体与批量训练 / 策略回放 CLI；相关仓库 rexctl（控制 CLI）、rex-models（URDF 可视化）。",
        caveat="奖励函数项与课程学习在 README 前段未见，来自文章，未核实。",
        related=[RL, QR, LOC, "../concepts/reward-design.md"],
        abbr=[
            ("PPO", "Proximal Policy Optimization", "近端策略优化"),
            ("URDF", "Unified Robot Description Format", "机器人描述格式"),
        ],
    ),
    dict(
        slug="rl-mpc-locomotion",
        inst="independent",
        name="rl-mpc-locomotion",
        repo="silvery107/rl-mpc-locomotion",
        stars="946",
        forks="",
        last="2026-04-26",
        group=G_RL,
        tags=["quadruped", "rl", "mpc", "isaac-gym", "aliengo"],
        one="rl-mpc-locomotion 提供快速仿真与 RL 训练框架：高层策略网络动态预测 MPC 控制器的权重参数，低层是参考 MIT Cheetah Software、用 Python 重写的 MPC。",
        article="文章称它是完整教学框架（配套 UG-Thesis-EN 教学 PPT），实现凸 MPC、WBC，并用 PPO 学习 MPC 权重，基于 Isaac Gym 并行训练。",
        readme="README 称 RL 训练使用 NVIDIA Isaac Gym 并行、使用 Unitree Aliengo 模型，并在真实 Aliengo 上实现 sim2real（**sim2real 不在本代码库内**）；依赖 Python 3.8、PyTorch 1.10、Isaac Gym Preview 4。",
        caveat="文章提到的 WBC 与教学 PPT 细节未在 README 前段核对。",
        related=[RL, MPC, IGYM, LOC],
        abbr=[
            ("RL", "Reinforcement Learning", "强化学习"),
            ("MPC", "Model Predictive Control", "模型预测控制"),
        ],
    ),
    dict(
        slug="abs-agile-but-safe",
        inst="cmu",
        name="ABS（Agile But Safe）",
        repo="LeCAR-Lab/ABS",
        stars="（文中未给）",
        forks="",
        last="2024-07-11",
        group=G_RL,
        tags=["quadruped", "rl", "safety", "navigation", "legged-gym"],
        one="ABS 是「Agile But Safe: Learning Collision-Free High-Speed Legged Locomotion」（RSS 2024）的官方实现：在 legged_gym / rsl_rl 之上训练高速且无碰撞的四足运动。",
        article="文章称其由 CMU 与 ETH 联合开发，3–4 m/s 奔跑，感知层用 RealSense D435i 与轻量 CNN 输出局部占据栅格，规划层用 MPPI，控制层为 RL 低层控制器。",
        readme="README 标注 RSS 2024，作者含 Tairan He、Chong Zhang、Guanya Shi 等（README 未标机构）；代码按 CC BY-NC 4.0 许可（禁止商用），继承 Legged Gym 与 RSL RL（ETH Zurich / NVIDIA）的许可；建议先跑通 RSL 版 Legged Gym。",
        caveat="文章的「CMU+ETH 联合」「MPPI 规划」「D435i + CNN 占据栅格」等在已读 README 段落中未见，来自文章，未核实；README 另限制商业用途。",
        related=[LGYM, RSL, RL, LOC, S2R],
        abbr=[
            ("ABS", "Agile But Safe", "敏捷但安全"),
            ("RSS", "Robotics: Science and Systems", "机器人顶会"),
        ],
    ),
    dict(
        slug="rl-sar",
        inst="independent",
        name="rl_sar",
        repo="fan-ziqi/rl_sar",
        stars="1230",
        forks="",
        last="2026-08-16",
        group=G_RL,
        tags=["rl", "sim2real", "deployment", "ros", "mujoco", "isaac"],
        one="rl_sar（sar = simulation and real）是机器人强化学习算法的仿真验证与实物部署框架，适用于四足、轮式与人形机器人，同时支持 IsaacGym / IsaacSim、ROS1 / ROS2、libtorch / onnxruntime、Gazebo / MuJoCo。",
        article="文章把它作为「国产轮足机器人强化学习实践」，称支持宇树 Go2、B2、Aliengo 等轮足平台，高层 MLP + 低层 PD，轮腿协同奖励，零样本迁移。",
        readme="README 称框架面向四足、轮式、人形机器人，覆盖 locomotion 与 dance，并列出支持机器人、预训练策略与 Gazebo / MuJoCo / 实机状态表（Apache-2.0 徽章）。",
        caveat="README 把它定位为通用 RL 仿真到实机部署框架；文章把它归入「轮足」并称支持 Go2/B2/Aliengo 轮足平台，该归类与具体机型列表未逐项核对（以 README 支持列表为准）。库内 `robot-lab` 是另一个项目。",
        related=[S2R, RL, LOC, ROBOTLAB],
        abbr=[
            ("SAR", "Simulation And Real", "仿真与实机"),
            ("ONNX", "Open Neural Network Exchange", "神经网络交换格式"),
        ],
    ),
    dict(
        slug="wheel-legged-gym",
        inst="sustech",
        name="Wheel-Legged-Gym",
        repo="clearlab-sustech/Wheel-Legged-Gym",
        stars="（文中未给）",
        forks="",
        last="2024-06-19",
        group=G_WL,
        tags=["wheel-legged", "rl", "legged-gym", "isaac-gym"],
        one="Wheel-Legged-Gym（南方科技大学 CLEAR Lab）是基于 legged_gym 与 rsl_rl 的轮足机器人强化学习环境。",
        article="文章称其简单易上手，有清晰代码结构与文档，含完整训练流程并提供预训练模型。",
        readme="README 致谢 legged_gym 与 rsl_rl，列出相关论文（Learning to Walk in Minutes…、Concurrent Training of a Control Policy and a State Estimator…），安装依赖 Python 3.8 与 Isaac Gym Preview 4。",
        caveat="「预训练模型」「文档详细」来自文章，未核实。",
        related=[LGYM, RSL, RL, LOC],
        abbr=[("RL", "Reinforcement Learning", "强化学习")],
    ),
    dict(
        slug="legged-robots-manipulation",
        inst="independent",
        name="legged-robots-manipulation",
        repo="aCodeDog/legged-robots-manipulation",
        stars="（文中未给）",
        forks="",
        last="2025-07-10",
        group=G_WL,
        tags=["wheel-legged", "loco-manipulation", "rl", "unitree-b2w"],
        one="legged-robots-manipulation 是 (轮)足机器人移动操作仓库（IROS 2024 Oral 项目的部分实现），基于 legged_gym，含 airbot、go2_arx、b2w_z1、aliengo_z1 与 b2w。",
        article="文章把它描述为基于宇树 B2W 的轮足移动操作项目，在统一 RL 框架中整合腿部与机械臂控制。",
        readme="README 称当前仓库为论文的部分实现、仍在建设，Isaac Lab 版本与端到端视觉版本为待办；项目页为 wheel-legged-loco-manipulation。",
        caveat="仓库未声明 LICENSE 文件；内容是论文部分实现，不是完整复现。",
        related=[LOCOMANIP, LGYM, RL, LOC],
        abbr=[
            ("B2W", "Unitree B2-W", "宇树轮足平台（文章称）"),
            (
                "IROS",
                "IEEE/RSJ International Conference on Intelligent Robots and Systems",
                "机器人会议",
            ),
        ],
    ),
    dict(
        slug="ascend-quadruped",
        inst="usc",
        name="Ascend Quadruped",
        repo="DRCL-USC/Ascend_Quadruped",
        stars="（文中未给）",
        forks="",
        last="",
        group=G_WL,
        tags=["wheel-legged", "hardware", "usc"],
        one="Ascend Quadruped 在文中被描述为南加州大学开发的低成本轮足平台，完整开源机械与电路设计（文章说法）。",
        article="文章仅一句话：南加州大学低成本轮足平台，完整开源机械和电路设计。",
        readme="入库日 `git ls-remote` 访问 github.com/DRCL-USC/Ascend_Quadruped 失败（仓库可能已更名、迁移或私有），**未能核对 README**。",
        caveat="本节点只有文章一句话；所有细节均未核实，勿作为开源状态依据。",
        related=[QR, LOC],
        abbr=[("USC", "University of Southern California", "南加州大学")],
        reachable=False,
    ),
    dict(
        slug="wheeled-bipedal-robot",
        inst="independent",
        name="Wheeled Biped",
        repo="Derek-TH-Wang/wheeled_bipedal_robot",
        stars="（文中未给）",
        forks="",
        last="",
        group=G_WL,
        tags=["wheel-legged", "biped", "ros"],
        one="Wheeled Biped（Derek-TH-Wang/wheeled_bipedal_robot）在文中被描述为双足轮式机器人，展示轮腿结合的另一种形态（文章说法）。",
        article="文章仅一句话：双足轮式机器人，展示了轮腿结合的另一种形态。",
        readme="入库日 `git ls-remote` 访问该仓库失败（仓库可能已更名、迁移或私有），**未能核对 README**。",
        caveat="本节点只有文章一句话；所有细节均未核实，勿作为开源状态依据。",
        related=[QR, LOC],
        abbr=[],
        reachable=False,
    ),
    dict(
        slug="flores-wheel-legged-robot",
        inst="independent",
        name="FLORES",
        repo="ZhichengSong6/FLORES",
        stars="25",
        forks="",
        last="2026-05-22",
        group=G_WL,
        tags=["wheel-legged", "reconfigurable", "rl", "ral"],
        one="FLORES 对应 RA-L 2026 论文「A Reconfigured Wheel-Legged Robot for Enhanced Steering and Adaptability」：通过关节 DoF 与可重构腿设计增强转向与适应性。",
        article="文章称其采用模块化腿部设计、可动态调整构型，主动转向关节与智能轮向控制实现类似汽车的阿克曼转向。",
        readme="README 标注发表于 IEEE RA-L 2026（请引用同行评审版本），并给出机械设计与训练流程图：HIM 策略先在 Isaac Gym 带奖励塑形训练，再迁移到 MuJoCo 做 sim-to-sim，第二阶段为实机部署。",
        caveat="「阿克曼转向」等机理细节来自文章，未在 README 已读段落核对；仓库无 LICENSE 文件。",
        related=[LOC, S2R, RL],
        abbr=[
            ("RA-L", "IEEE Robotics and Automation Letters", "期刊"),
            (
                "HIM",
                "Hybrid Internal Model",
                "Hybrid Internal Model 策略（全称为补充，非 README 原文）",
            ),
        ],
    ),
    dict(
        slug="foc-wheel-legged-robot",
        inst="independent",
        name="FOC 双轮腿机器人",
        repo="Skythinker616/foc-wheel-legged-robot",
        stars="1557",
        forks="",
        last="2023-11-23",
        group=G_WL,
        tags=["wheel-legged", "foc", "stm32", "esp32", "hardware"],
        one="foc-wheel-legged-robot 是完整的双轮腿机器人开源项目：SolidWorks 机械、MATLAB/Simulink/Simscape 算法仿真、STM32 CAN 无刷电机驱动板（FOC）、ESP32 + MPU6050 主控、Linux 图传与 Android 遥控 APP。",
        article="文章称其采用五连杆结构，FOC 驱动实现力控，完整开源 STM32 驱动板与控制算法。",
        readme="README 列出上述六部分与对应目录（solidworks / matlab / stm32-foc / esp32-controller / linux-fpv / android）；含 LICENSE 文件。",
        caveat="「五连杆」来自文章，已读 README 前段未见，未核实。",
        related=[SFOC, QDD, LOC],
        abbr=[
            ("FOC", "Field-Oriented Control", "磁场定向控制"),
            ("CAN", "Controller Area Network", "控制器局域网总线"),
        ],
    ),
    dict(
        slug="locoleggedwheel",
        inst="independent",
        name="LocoLeggedWheel",
        repo="zhaozijie2022/LocoLeggedWheel",
        stars="115",
        forks="",
        last="2026-07-09",
        group=G_WL,
        tags=["wheel-legged", "rl", "isaac-lab", "sim2real", "go2w"],
        one="LocoLeggedWheel（LocoWheeledLegged）是基于 NVIDIA Isaac Lab 的轮足机器人 RL 运动控制项目，重点优化奖励函数与训练稳定性，并提供宇树 Go2W 的 sim2real 部署接口与配置。",
        article="文章称基于 Isaac Lab，强调 Sim-to-Real 迁移。",
        readme="README 指出 Isaac Gym 版本见 DreamWaQ_Go2W；安装需 Isaac Sim 5.x（README 同时出现 v5.5.1 与 5.1.0 两种写法）；无 LICENSE 文件。",
        caveat="README 中 Isaac Sim 版本写法不一致，部署前以官方文档为准。",
        related=[ILAB, RL, S2R, LOC],
        abbr=[("Go2W", "Unitree Go2-W", "宇树轮足版 Go2（README 所称）")],
    ),
    dict(
        slug="wheel-legged-robot-simulation",
        inst="independent",
        name="WheelLeggedRobotSimulation",
        repo="Rengar-Yang/WheelLeggedRobotSimulation",
        stars="73",
        forks="",
        last="2026-01-13",
        group=G_WL,
        tags=["wheel-legged", "matlab", "simulation", "lqr", "mpc"],
        one="WheelLeggedRobotSimulation 是面向新手的五连杆轮腿机器人 MATLAB 仿真平台：含 PID、LQR、MPC 三种控制方式，运动学与动力学两种控制模型，LQR 结合 MRAC 做简单自适应控制。",
        article="文章称是轮腿机器人仿真文件集合，支持多种仿真器。",
        readme="README 称整体结构简单、适合入门，手动运行代码在 Init.m；作者自述代码可能有问题并欢迎指正。",
        caveat="**文章称「支持多种仿真器」，README 只提 MATLAB**；本节点以 README 为准。",
        related=[LOC, "../methods/lqr-ilqr.md", MPC],
        abbr=[
            ("LQR", "Linear Quadratic Regulator", "线性二次调节器"),
            ("MRAC", "Model Reference Adaptive Control", "模型参考自适应控制"),
        ],
    ),
    dict(
        slug="venus-1-four-link-wheel-legged-robot",
        inst="independent",
        name="启明-1 Venus-1",
        repo="track-tech/Venus-1__Four-link-wheel-legged-robot",
        stars="60",
        forks="",
        last="2025-05-09",
        group=G_WL,
        tags=["wheel-legged", "four-link", "stm32", "hardware", "pid"],
        one="启明-1（Venus-1）是四连杆轮腿机器人的个人本科毕设开源项目：SolidWorks 机械、STM32H723 + FreeRTOS、小米 CyberGear 关节电机、DJI C610 + M2006 轮电机、全 PID 控制、SBUS 遥控。",
        article="文章称是启明-1 四连杆轮腿机器人，完整机械和控制开源。",
        readme="README 声明这是个人本科毕业设计，**存在设计缺陷，请谨慎复刻**；遵循 GPL 3.0，量产与商用需授权；结构为 3 mm 3240 环氧板，测试含重心自适应、跌落缓冲、主动悬挂、跳跃。",
        caveat="GPL 3.0 且商用需授权；作者自述有设计缺陷。",
        related=[QDD, LOC],
        abbr=[
            ("PID", "Proportional–Integral–Derivative", "比例积分微分控制"),
            ("SBUS", "Serial Bus", "遥控接收机串行总线"),
        ],
    ),
    dict(
        slug="awesome-wheeled-legged",
        inst="independent",
        name="awesome-wheeled-legged",
        repo="XinLang2019/awesome-wheeled-legged",
        stars="29",
        forks="",
        last="2025-07-31",
        group=G_WL,
        tags=["wheel-legged", "awesome-list", "resources"],
        one="awesome-wheeled-legged 是轮足机器人资源清单：开源项目、论文、概览 PPT、硬件平台。",
        article="文章称其汇总了开源代码、学术论文与商业产品。",
        readme="README 目录为：Open Source Projects、Research Papers、OverView PPT、Hardware Platforms、License。",
        caveat="这是资源索引而非可运行项目。",
        related=[LOC, "../entities/open-source-humanoid-hardware.md"],
        abbr=[],
    ),
    dict(
        slug="sesame-robot",
        inst="independent",
        name="Sesame Robot",
        repo="dorianborian/sesame-robot",
        stars="1309",
        forks="",
        last="2026-10-15",
        group=G_OTHER,
        tags=["quadruped", "esp32", "maker", "3d-printed"],
        one="Sesame Robot 是基于 ESP32 的开源创客四足机器人：强调表情与动作，8 个舵机（每腿 2 个），含 CAD / STL、制作与接线指南、基础与扩展固件及调试固件。",
        article="文章仅在附录清单列出（1309 星）。",
        readme="README 称制作需基础焊接、约 50–60 美元硬件与 3D 打印机，使用 Arduino IDE；含 LICENSE 文件。",
        caveat="",
        related=[QR, "../entities/open-quadruped.md"],
        abbr=[("ESP32", "ESP32 Microcontroller", "乐鑫 Wi-Fi/蓝牙 MCU")],
    ),
    dict(
        slug="py-apple-quadruped",
        inst="independent",
        name="菠萝狗 Py-Apple Dog",
        repo="ToanTech/py-apple-quadruped-robot",
        stars="1224",
        forks="",
        last="2022-04-11",
        group=G_OTHER,
        tags=["quadruped", "low-cost", "diy", "open-source"],
        one="菠萝狗（Py-Apple Dog）是低成本、可 DIY 的大型全套四足机器人软硬件开源项目，主站聚合四个子项目：Dynamics 控制软件、Controller 万能控制器、Structure 机械结构、BLDC 无刷四足。",
        article="文章仅在附录清单列出（1224 星）。",
        readme="README 列出四个子项目（py-apple-dynamics、py-apple-controller、py-apple-structure、py-apple-bldc-quadruped-robot）及性能 / 遥控 / 平衡 / 二次开发 / AI 功能表。",
        caveat="本节点只对应「主站」仓库；四个子项目未单独建节点（文章未点名）。最近提交 2022-04-11。",
        related=[QR, LOC],
        abbr=[("BLDC", "Brushless DC Motor", "无刷直流电机")],
    ),
]

# 与 README 不一致等需在地图页突出的核对结论
CHECKS = [
    (
        "CHAMP",
        "文章称含 `convex_mpc` 与摩擦锥 QP",
        "README 称控制框架基于 MIT 的模式调制 + 阻抗控制分层控制器，未提 MPC",
        "以 README 为准",
    ),
    (
        "legged_control",
        "文章称为成熟 ROS1 项目",
        "README 声明软件已不再维护，作者转向新框架",
        "文章未提，已补",
    ),
    (
        "ABS",
        "文章称 CMU+ETH 联合、MPPI 规划",
        "README 为 RSS 2024，署名含 Tairan He、Chong Zhang、Guanya Shi 等（未标机构）；未见 MPPI / ETH 联合字样；CC BY-NC 4.0 禁商用",
        "文章细节未核实",
    ),
    (
        "Barkour Robot",
        "文章称完整代码尚未开源",
        "README 称仓库含设计资产（CAD / PCBA / 装配）与固件、底层核心软件",
        "RL 训练代码是否开放未核实",
    ),
    (
        "WheelLeggedRobotSimulation",
        "文章称支持多种仿真器",
        "README 只提 MATLAB（PID / LQR / MPC）",
        "以 README 为准",
    ),
    (
        "rl_sar",
        "文章归入「轮足」",
        "README 称面向四足、轮式、人形的通用 RL 部署框架",
        "文章归类偏窄",
    ),
    (
        "Ascend Quadruped / Wheeled Biped",
        "文章称完整开源",
        "入库日 git ls-remote 失败，无法核对",
        "标未核实",
    ),
]


def _yaml_list(items: list[str]) -> str:
    return "\n".join(f"  - {x}" for x in items)


def _repo_src(p: dict[str, Any]) -> str:
    reach = (
        "仓库可达（`git ls-remote`，浅克隆读 README，不运行代码）"
        if p.get("reachable", True)
        else "入库日 `git ls-remote` 失败，未能核对"
    )
    last = f"- **最近提交：** {p['last']}\n" if p.get("last") else ""
    return f"""# {p["name"]} 仓库

> 来源归档（repo）

- **标题：** {p["name"]}
- **类型：** repo
- **链接：** <https://github.com/{p["repo"]}>
- **入库日期：** {TODAY}
- **线索来源：** [{BLOG}](../blogs/{BLOG})（古月居转载，原作者敢敢AUTOHUB）
- **一句话说明：** {p["one"]}
{last}- **核查：** {reach}
- **文章星标（文章表述）：** {p["stars"]}{("；分支 " + p["forks"]) if p.get("forks") else ""}
- **沉淀到 wiki：** [`wiki/entities/{p["slug"]}.md`](../../wiki/entities/{p["slug"]}.md)

## README 摘录（入库日人工阅读）

{p["readme"]}
"""


def _entity(p: dict[str, Any]) -> str:
    rel = p["related"] + [f"../overview/{MAP}"]
    abbr = (
        "\n".join(f"| {a} | {b} | {c} |" for a, b, c in p["abbr"])
        or "| — | — | 本项目无需额外缩写 |"
    )
    cav = f"\n## 文章说法核对\n\n- **文章：** {p['article']}\n- **README / 核查：** {p['readme']}\n- **差异与保留：** {p['caveat'] or '未发现与已读 README 的冲突。'}\n"
    rel_lines = "\n".join(f"- [{r.split('/')[-1][:-3]}]({r})" for r in rel)
    last = f"| **最近提交** | {p['last']} |\n" if p.get("last") else ""
    summary = p["one"].replace('"', "'")
    return f"""---
type: entity
project_id: {p["slug"]}
tags:
{_yaml_list(["repo", p["inst"], *p["tags"]])}
status: complete
updated: {TODAY}
code: https://github.com/{p["repo"]}
related:
{_yaml_list(rel)}
sources:
  - ../../sources/repos/{p["slug"].replace("-", "_")}.md
  - ../../sources/blogs/{BLOG}
summary: "{summary}"
---

# {p["name"]}

**{p["name"]}**（[GitHub](https://github.com/{p["repo"]})）来自 [古月居转载的四足机器人开源项目综述]({URL})，属「{p["group"]}」组。

## 一句话定义

**{p["one"]}**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
{abbr}

## 为什么重要

- 综述把它放在「{p["group"]}」这条技术路线下，作为可读源码的参照项目；横向对比见 [{MAP_TITLE}](../overview/{MAP})。
- 文中星标：{p["stars"]}{("；分支 " + p["forks"]) if p.get("forks") else ""}（文章表述，入库日未复核）。

## 核心信息

| 项 | 内容 |
|----|------|
| **仓库** | [{p["repo"]}](https://github.com/{p["repo"]}) |
{last}| **可达性** | {"`git ls-remote` 可达" if p.get("reachable", True) else "`git ls-remote` 失败，未核对"}（{TODAY}） |
| **文章归类** | {p["group"]} |
{cav}
## 关联页面

{rel_lines}

## 参考来源

- [{p["slug"].replace("-", "_")}.md](../../sources/repos/{p["slug"].replace("-", "_")}.md)
- [{BLOG}](../../sources/blogs/{BLOG})
- [原文（微信公众号）]({URL})
"""


def _blog() -> str:
    rows = []
    for g in GROUP_ORDER:
        for r in REUSE:
            if r["group"] == g:
                rows.append(
                    (
                        g,
                        r["name"],
                        r["repo"],
                        r["stars"],
                        f"[{r['node']}](../../wiki/entities/{r['node']}.md)",
                        "复用",
                        r["note"],
                    )
                )
        for p in NEW:
            if p["group"] == g:
                rows.append(
                    (
                        g,
                        p["name"],
                        p["repo"],
                        p["stars"],
                        f"[{p['slug']}](../../wiki/entities/{p['slug']}.md)",
                        "新建",
                        "",
                    )
                )
    body = "\n".join(
        f"| {i:02d} | {a} | {b} | [{c}](https://github.com/{c}) | {d} | {e} | {f} | {n} |"
        for i, (a, b, c, d, e, f, n) in enumerate(rows, 1)
    )
    checks = "\n".join(f"| {a} | {b} | {c} | {d} |" for a, b, c, d in CHECKS)
    return f"""# 四足机器人开源项目深度解析：从入门到研究级的完整技术图谱

> 来源归档（blog / 微信公众号）

- **标题：** 四足机器人开源项目深度解析：从入门到研究级的完整技术图谱
- **类型：** blog
- **来源账号：** 古月居（转载），文内标注 The following article is from 敢敢AUTOHUB，Author 敢敢不敢写bug
- **原始链接：** {URL}
- **发表日期：** 2026-11-01
- **入库日期：** {TODAY}
- **抓取方式：** `scripts/fetch_wechat_article.py`（Chromium + 微信内置 UA），正文 18453 字符
- **一句话说明：** 按「极简运动学 / ROS / 优化控制 / 强化学习 / 轮足 / 研究级硬件」六条流派梳理 GitHub 上 20 多个四足开源项目，附资源汇总与趋势判断。
- **说明：** 文中点名 **36 个 GitHub 仓库**（Stanford Pupper 与 Stanford Doggo 在库内共用一个节点）：**新建 {len(NEW)}、复用 {len({r["node"] for r in REUSE})} 个节点**，每个项目唯一节点。星标数为文章表述，入库日未复核；README 核查见各节点与下表。

## 项目 → 节点

| # | 流派 | 项目 | 仓库 | 星标（文章） | 节点 | 状态 | 备注 |
|---|------|------|------|-------------|------|------|------|
{body}

## 文章说法与 README 核对（有差异的项）

| 项目 | 文章说法 | README / 核查 | 处理 |
|------|----------|---------------|------|
{checks}

## 对 wiki 的映射

- 阅读坐标：[{MAP_TITLE}](../../wiki/overview/{MAP})
- 原文趋势判断（文档质量参差、硬件依赖强、仿真与现实差距）仅作文章观点收录，未独立验证。
"""


def _map() -> str:
    sections = []
    for g in GROUP_ORDER:
        lines = []
        for r in REUSE:
            if r["group"] == g:
                lines.append(
                    f"| {r['name']} | [{r['node']}](../entities/{r['node']}.md)（复用） | {r['stars']} |"
                )
        for p in NEW:
            if p["group"] == g:
                lines.append(
                    f"| {p['name']} | [{p['slug']}](../entities/{p['slug']}.md) | {p['stars']} |"
                )
        sections.append(
            f"### {g}\n\n| 项目 | 节点 | 星标（文章） |\n|------|------|-------------|\n"
            + "\n".join(lines)
            + "\n"
        )
    checks = "\n".join(f"| {a} | {b} | {c} | {d} |" for a, b, c, d in CHECKS)
    first = [f"../entities/{p['slug']}.md" for p in NEW[:3]]
    return f"""---
type: overview
tags: [overview, quadruped, open-source, locomotion, technology-map, wechat]
status: complete
updated: {TODAY}
related:
{_yaml_list(first + ["../entities/quadruped-robot.md", "../tasks/locomotion.md"])}
sources:
  - ../../sources/blogs/{BLOG}
summary: "古月居转载的四足机器人开源项目综述的阅读坐标：36 个 GitHub 项目按极简运动学、ROS、优化控制、强化学习、轮足、研究级硬件六条流派排列，新建 {len(NEW)} / 复用 {len({r["node"] for r in REUSE})} 个独立节点，并列出文章与 README 不一致之处。"
---

# {MAP_TITLE}

> **本页定位**：为 [四足机器人开源项目深度解析]({URL})（2026-11-01，古月居转载敢敢AUTOHUB）提供按流派组织的阅读坐标；文中 36 个项目均有唯一详情节点。

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

{chr(10).join(sections)}
## 文章说法与 README 核对

| 项目 | 文章说法 | README / 核查 | 处理 |
|------|----------|---------------|------|
{checks}

> 星标数与「已开源」类表述为入库日（{TODAY}）读取的 README 与 `git ls-remote` 结果；星标数为文章表述。

## 关联页面

- [四足机器人总览](../entities/quadruped-robot.md)
- [运动控制任务页](../tasks/locomotion.md)
- [开源 QDD 执行器项目对比](../comparisons/open-source-qdd-actuator-projects.md)

## 参考来源

- [{BLOG}](../../sources/blogs/{BLOG})
"""


def _patch_reuse() -> None:
    blog = f"  - ../../sources/blogs/{BLOG}\n"
    mapl = f"  - ../overview/{MAP}\n"
    for node in sorted({r["node"] for r in REUSE}):
        path = ROOT / "wiki/entities" / f"{node}.md"
        text = path.read_text(encoding="utf-8")
        head, sep, body = text.partition("\n---\n")
        if not sep:
            raise SystemExit(f"frontmatter 缺失: {path}")
        lines = head.split("\n")

        def _append(key: str, item: str) -> None:
            if item.strip() in (x.strip() for x in lines):
                return
            idx = next((i for i, x in enumerate(lines) if x.startswith(f"{key}:")), None)
            if idx is None:
                lines.extend([f"{key}:", item.rstrip("\n")])
                return
            if lines[idx].strip() != f"{key}:":
                raise SystemExit(f"{path}: {key} 为行内写法，需手工补")
            j = idx + 1
            indent = "  "
            if j < len(lines) and lines[j].lstrip().startswith("- "):
                indent = lines[j][: len(lines[j]) - len(lines[j].lstrip())]
            while j < len(lines) and lines[j].lstrip().startswith("- "):
                j += 1
            lines.insert(j, indent + item.strip())

        _append("sources", blog)
        _append("related", mapl)
        lines = [f"updated: {TODAY}" if x.startswith("updated:") else x for x in lines]
        path.write_text("\n".join(lines) + sep + body, encoding="utf-8")


def main() -> None:
    txt = Path(sys.argv[1]).read_text(encoding="utf-8")
    slugs: set[str] = set()
    for p in NEW:
        if p["slug"] in slugs:
            raise SystemExit(f"slug 重复: {p['slug']}")
        slugs.add(p["slug"])
        if (ROOT / "wiki/entities" / f"{p['slug']}.md").exists():
            raise SystemExit(f"节点已存在: {p['slug']}")
        for r in p["related"]:
            if not (ROOT / "wiki/entities" / r).resolve().exists():
                raise SystemExit(f"关联页不存在: {r} ({p['slug']})")
    for r in REUSE:
        if not (ROOT / "wiki/entities" / f"{r['node']}.md").exists():
            raise SystemExit(f"复用节点不存在: {r['node']}")
    raw = ROOT / "sources/raw" / BLOG
    raw.write_text(
        f"# 四足机器人开源项目深度解析\n\n> 原始抓取（{TODAY}，`scripts/fetch_wechat_article.py`），仅供溯源；来源：{URL}\n\n{txt}",
        encoding="utf-8",
    )
    (ROOT / "sources/blogs" / BLOG).write_text(_blog(), encoding="utf-8")
    (ROOT / "wiki/overview" / MAP).write_text(_map(), encoding="utf-8")
    for p in NEW:
        (ROOT / "sources/repos" / f"{p['slug'].replace('-', '_')}.md").write_text(
            _repo_src(p), encoding="utf-8"
        )
        (ROOT / "wiki/entities" / f"{p['slug']}.md").write_text(_entity(p), encoding="utf-8")
    _patch_reuse()
    print(
        f"Wrote 1 blog, 1 map, {len(NEW)} new entities, patched {len({r['node'] for r in REUSE})} reuse nodes"
    )


if __name__ == "__main__":
    main()
