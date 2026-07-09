# HighTorque Robotics（GitHub 组织）

> 来源归档

- **标题：** HighTorque Robotics
- **类型：** repo（组织入口）
- **机构：** 广州高擎机电科技有限公司（High Torque Technology Co., Ltd.）
- **链接：** <https://github.com/HighTorque-Robotics>
- **官网：** <https://hightorque.cn/>
- **入库日期：** 2026-07-09
- **一句话说明：** 国产桌面级人形与开源机械臂平台商；GitHub 组织提供 Mini Pi Plus 全身跟踪/RL/sim2real 管线、Panthera-HT 六轴臂全栈开源，以及 RoboCup 人形小尺寸赛队工作区。

## 组织概况（GitHub API，2026-07-09）

| 字段 | 值 |
|------|-----|
| 创建时间 | 2024-04-24 |
| 公开仓库 | 36 |
| Followers | 216+ |

公司背景（官网 / RoboCup 介绍 PDF）：2020 年成立于广州；全链路自研关节模组、减速器、电机、FOC 与运控算法；定位「具身智能时代的 PC」——把高性能人形从实验室搬到开发者桌面。

## 产品线与代表仓库

### 1. Mini Pi / Mini Pi Plus（小型人形）

| 仓库 | Stars | 说明 |
|------|-------|------|
| [Mini-Pi-Plus_BeyondMimic](https://github.com/HighTorque-Robotics/Mini-Pi-Plus_BeyondMimic) | 170 | 基于 [BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking) 改造；**Isaac Sim 5.0 + Isaac Lab 2.2 + rsl_rl**；内置 Pi Plus 资产与 retarget 数据，**零调参 sim-to-real 运动跟踪** |
| [livelybot_pi_rl_baseline](https://github.com/HighTorque-Robotics/livelybot_pi_rl_baseline) | 82 | **Isaac Gym Preview 4 + legged_gym 风格 PPO**；含 Isaac Gym → MuJoCo **sim2sim** |
| [Pi_Isaaclab](https://github.com/HighTorque-Robotics/Pi_Isaaclab) | 14 | **Teacher–Student** 框架，基于 Isaac Lab（LeggedLab + rsl_rl） |
| [Mini-Pi-Plus_PBHC](https://github.com/HighTorque-Robotics/Mini-Pi-Plus_PBHC) | 34 | PBHC 相关训练栈（README 待补） |
| [sim2real](https://github.com/HighTorque-Robotics/sim2real) | 10 | ROS Noetic **真机 sim2real 部署**（配合 gitee `hightorque_rl` master） |
| [sim2real-inference_code](https://github.com/HighTorque-Robotics/sim2real-inference_code) | 6 | 推理侧部署代码 |
| [robot_urdf](https://github.com/HighTorque-Robotics/robot_urdf) | 12 | Mini Pi **最新 URDF** 持续更新 |
| [hi_dynamic_control](https://github.com/HighTorque-Robotics/hi_dynamic_control) | 18 | **Hi 人形**（串并联踝关节）**非线性 MPC + WBC**；依赖 OCS2 + ROS1；Gazebo / 真机 launch |
| [RoboCup_Workspace](https://github.com/HighTorque-Robotics/RoboCup_Workspace) | 17 | **RoboCup 人形小尺寸** Pi+ 赛队：行为、网络、规划、IO、视觉模块与 launch |
| [HoST](https://github.com/HighTorque-Robotics/HoST) | 1 | RSS 2025 起身控制官方实现的 **Mini Pi 扩展 fork**（主仓 InternRobotics/HoST） |
| [Policy-Format-Convert](https://github.com/HighTorque-Robotics/Policy-Format-Convert) | 3 | 人形策略落地：**ONNX → RKNN / TensorRT** |

**硬件规格（Mini Pi Plus，官方商店 / RoboCup 资料）：** 高约 **65 cm**，重约 **7–11 kg**，**26–27 DoF**，自研集成伺服关节；面向科研、教学与 **RoboCup Humanoid Small Size**；ICRA 2026 全球发布 Mini Pi Plus。

### 2. Panthera-HT（开源六轴机械臂）

| 仓库 | Stars | 说明 |
|------|-------|------|
| [Panthera-HT_Main](https://github.com/HighTorque-Robotics/Panthera-HT_Main) | 176 | 项目主仓：教学/创客六轴臂介绍、能力矩阵、仓库索引 |
| [Panthera-HT_SDK](https://github.com/HighTorque-Robotics/Panthera-HT_SDK) | 63 | **C++ / Python SDK**：位姿/速度/力矩/阻抗、动力学、轨迹、主从遥操 |
| [Panthera-HT_ROS2](https://github.com/HighTorque-Robotics/Panthera-HT_ROS2) | 53 | **MoveIt 驱动** + 笛卡尔空间阻抗控制 |
| [Panthera-HT_Model](https://github.com/HighTorque-Robotics/Panthera-HT_Model) | 15 | SolidWorks / 钣金 / 3D 打印 STL + **BOM** |
| [Panthera-HT_lerobot](https://github.com/HighTorque-Robotics/Panthera-HT_lerobot) | 8 | **LeRobot** 数据采集与模仿学习推理 |
| [Panthera_HT_SDK_Extensions](https://github.com/HighTorque-Robotics/Panthera_HT_SDK_Extensions) | 5 | 手眼标定、视觉伺服等扩展 |
| [Panthera_digital_twin](https://github.com/HighTorque-Robotics/Panthera_digital_twin) | 2 | 数字孪生可视化环境 |

源自 [Ragtime_Panthera](https://github.com/Ragtime-LAB/Ragtime_Panthera) 社区项目，与高擎联合完善；**ICRA 2026 WBCD Challenge** 官方标准平台之一。Hub：<https://hightorque.cn/Panthera-HT_Hub/>

### 3. 基础设施与其它

| 仓库 | 说明 |
|------|------|
| [livelybot_hardware_sdk](https://github.com/HighTorque-Robotics/livelybot_hardware_sdk) | 硬件通信 SDK |
| [HT_GVHMR-and-GMR](https://github.com/HighTorque-Robotics/HT_GVHMR-and-GMR) | GVHMR + GMR 重定向使用说明 |
| [footstep_rl](https://github.com/HighTorque-Robotics/footstep_rl) | Mini Hi **步态 RL** |
| [LiteArm-A1](https://github.com/HighTorque-Robotics/LiteArm-A1) | 轻型臂产品线 |
| [hightorque_ros2](https://github.com/HighTorque-Robotics/hightorque_ros2) | ROS2 项目早期骨架 |

## 技术栈摘要

| 方向 | 典型栈 |
|------|--------|
| 全身运动跟踪 | Isaac Sim/Lab + BeyondMimic + GMR retarget + rsl_rl → ONNX → MuJoCo sim2sim → 真机 |
| 经典 RL 行走 | Isaac Gym + PPO（legged_gym 风格）→ JIT/ONNX → MuJoCo sim2sim → ROS sim2real |
| 模型基控制 | OCS2 NMPC + WBC（hi_dynamic_control）；ROS1 Noetic + Gazebo |
| 机械臂 | 自研 SDK（C++/Python）/ ROS2 MoveIt / LeRobot 模仿学习 |
| 策略部署 | ONNX → RKNN（边缘）/ TensorRT |

## 对 wiki 的映射

- 沉淀实体页：[wiki/entities/hightorque-robotics.md](../../wiki/entities/hightorque-robotics.md)
- 交叉更新：[wiki/entities/paper-host-humanoid-standingup.md](../../wiki/entities/paper-host-humanoid-standingup.md)（HoST Mini Pi 扩展）
- 交叉更新：[wiki/entities/open-source-humanoid-hardware.md](../../wiki/entities/open-source-humanoid-hardware.md)（小型开源人形选型）
- 交叉更新：[wiki/entities/humanoid-robot.md](../../wiki/entities/humanoid-robot.md)
