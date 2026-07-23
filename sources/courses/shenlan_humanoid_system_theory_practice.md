# 人形机器人系统 — 理论与实践（深蓝学院）课程大纲

- **类型：** course
- **来源：** 具身智能研究室（微信公众号）课程大纲整理；深蓝学院「人形机器人系统 - 理论与实践」
- **课程链接：** <https://www.shenlanxueyuan.com/course/802/task/33927/show>
- **收录日期：** 2026-07-23
- **一句话说明：** 以 Unitree G1 为主平台，串起 **RL 双足行走 → LiDAR 建图定位 → A\*/DWA 导航 → TARE/FAR 自主探索 → RealSense+YOLO 感知 → RoboCup 足球 → 大模型/VLN/NaVid** 的人形系统工程闭环。

## 为什么值得保留

- 与本库 [四足控制策展](../../wiki/entities/quadruped-control-curriculum.md) 形成 **足式 → 人形** 对照：本课更强调 **G1 软硬件、导航规划与足球感知**，而非四足动力学/SysID。
- 八章 + 实践作业可直接映射到 wiki 的 **概念 / 方法 / 实体 / 任务** 节点；公众号大纲图即本归档的章节源。
- 第 5 章 **TARE / FAR**、第 8 章 **NaVid** 等开源栈可交叉到 CMU Exploration 与 VLN 复现策展。

## 章节大纲（8 章 + 实践作业）

### 第 1 章 人形机器人技术发展现状与课程介绍

| 节 | 主题 |
|----|------|
| 1.1 | 人形机器人发展历史 |
| 1.2 | 人形机器人算法研究现状 |
| 1.3 | G1 人形机器人硬件组成 |
| 1.4 | G1 人形机器人软件服务实现 |
| 1.5 | 课程内容与项目进度安排 |
| **实践** | G1 仿真环境搭建与运动控制 |

### 第 2 章 基于强化学习的人形机器人行走控制

| 节 | 主题 |
|----|------|
| 2.1 | 人形机器人双足行走理论基础 |
| 2.2 | 强化学习原理与 PPO 算法 |
| 2.3 | 人形机器人双足行走强化学习训练 |
| 2.4 | 人形机器人双足行走 Sim2Real 演示 |
| **实践** | 基于强化学习的 G1 行走控制模型训练 |

### 第 3 章 基于 Lidar 的人形机器人建图与定位

| 节 | 主题 |
|----|------|
| 3.1 | 定位建图方案与硬件介绍 |
| 3.2 | 基于激光雷达的建图方案 |
| 3.3 | 基于激光雷达的定位方案 |
| 3.4 | 基于里程计与激光雷达的融合定位 |
| **实践** | G1 激光雷达建图与定位 |

### 第 4 章 人形机器人的全局路径规划与局部避障

| 节 | 主题 |
|----|------|
| 4.1 | 动态障碍物剔除与二维导航地图制作 |
| 4.2 | 基于 A\* 算法的全局路径规划 |
| 4.3 | 基于 DWA 算法的局部路径规划 |
| 4.4 | 仿真环境下的路径规划实践 |
| **实践** | 基于 A\* 与 DWA 的路径规划与避障 |

### 第 5 章 基于 TarePlanner 与 FarPlanner 的机器人自主探索

| 节 | 主题 |
|----|------|
| 5.1 | 机器人自主探索任务 |
| 5.2 | TarePlanner |
| 5.3 | FarPlanner |
| 5.4 | 仿真环境下的机器人自主探索实践 |
| **实践** | 仿真环境下的机器人自主探索实践 |

### 第 6 章 基于 RealSense 的人形机器人感知系统

| 节 | 主题 |
|----|------|
| 6.1 | RealSense 深度相机介绍 |
| 6.2 | YOLO 系列算法原理 |
| 6.3 | 足球场仿真环境介绍 |
| 6.4 | 球门与足球场线交点检测识别 |
| **实践** | 基于 YOLO11 的足球、球门与场地线检测识别 |

### 第 7 章 人形机器人 RoboCup 仿真足球赛

| 节 | 主题 |
|----|------|
| 7.1 | 感知后处理与坐标变换 |
| 7.2 | 空间视觉定位 — 线匹配 |
| 7.3 | 空间视觉定位 — EKF 融合 |
| 7.4 | 人形机器人足球实践 |
| **实践** | 人形机器人 RoboCup 仿真足球赛 |

### 第 8 章 大模型赋能人形机器人

| 节 | 主题 |
|----|------|
| 8.1 | 大模型赋能人形机器人的实现方法 |
| 8.2 | 搭建人形机器人智能语音交互系统 |
| 8.3 | VLN 核心概念与开发流程 |
| 8.4 | NaVid 算法核心框架与实机部署演示 |
| **实践** | 人形机器人语音交互导航系统 |

## 对 wiki 的映射

| 课程主题 | wiki 页面 |
|---------|-----------|
| 课程总览 | [humanoid-system-curriculum](../../wiki/entities/humanoid-system-curriculum.md) |
| 1.1 发展历史 | [humanoid-robot-history](../../wiki/overview/humanoid-robot-history.md) |
| 1.2 算法研究现状 | [humanoid-algorithm-research-status](../../wiki/overview/humanoid-algorithm-research-status.md) |
| 1.3 G1 硬件 | [unitree-g1](../../wiki/entities/unitree-g1.md) |
| 1.4 G1 软件服务 | [unitree-g1-software-stack](../../wiki/entities/unitree-g1-software-stack.md) |
| 2.1 双足理论基础 | [lip-zmp](../../wiki/concepts/lip-zmp.md) |
| 2.2 PPO / RL | [ppo](../../wiki/methods/ppo.md)、[reinforcement-learning](../../wiki/methods/reinforcement-learning.md) |
| 2.3 双足 RL 训练 | [humanoid-rl-cookbook](../../wiki/queries/humanoid-rl-cookbook.md)、[humanoid-locomotion](../../wiki/tasks/humanoid-locomotion.md) |
| 2.4 Sim2Real | [sim2real](../../wiki/concepts/sim2real.md) |
| 3.1–3.3 建图定位栈 | [navigation-slam-autonomy-stack](../../wiki/overview/navigation-slam-autonomy-stack.md)、[slam-toolbox](../../wiki/entities/slam-toolbox.md)、[fast-lio](../../wiki/entities/fast-lio.md) |
| 3.4 里程计–激光融合 | [lidar-odometry-fusion](../../wiki/methods/lidar-odometry-fusion.md) |
| 4.1 动态障碍剔除 | [dynamic-obstacle-filtering](../../wiki/concepts/dynamic-obstacle-filtering.md) |
| 4.2 A\* | [a-star](../../wiki/methods/a-star.md) |
| 4.3 DWA | [dwa](../../wiki/methods/dwa.md) |
| 4.4 规划实践 | [python-robotics](../../wiki/entities/python-robotics.md)、[navigation2](../../wiki/entities/navigation2.md) |
| 5.1 自主探索任务 | [autonomous-exploration](../../wiki/tasks/autonomous-exploration.md) |
| 5.2 TARE | [tare-planner](../../wiki/entities/tare-planner.md) |
| 5.3 FAR | [far-planner](../../wiki/entities/far-planner.md) |
| 6.1 RealSense | [intel-realsense](../../wiki/entities/intel-realsense.md) |
| 6.2 YOLO | [object-detection](../../wiki/methods/object-detection.md)、[paper-yolo](../../wiki/entities/paper-yolo-unified-realtime-detection.md) |
| 6.3 足球场仿真 | [soccer-field-simulation](../../wiki/concepts/soccer-field-simulation.md) |
| 6.4 球门/场地线检测 | [soccer-field-line-detection](../../wiki/methods/soccer-field-line-detection.md) |
| 7.1 感知后处理与坐标 | [perception-coordinate-postprocessing](../../wiki/concepts/perception-coordinate-postprocessing.md) |
| 7.2 线匹配定位 | [visual-line-matching-localization](../../wiki/methods/visual-line-matching-localization.md) |
| 7.3 EKF 融合定位 | [visual-line-ekf-fusion](../../wiki/methods/visual-line-ekf-fusion.md)、[ekf](../../wiki/formalizations/ekf.md) |
| 7.4 足球实践 | [humanoid-soccer](../../wiki/tasks/humanoid-soccer.md) |
| 8.1 大模型赋能 | [large-model-empowered-humanoids](../../wiki/overview/large-model-empowered-humanoids.md) |
| 8.2 语音交互 | [humanoid-voice-interaction](../../wiki/methods/humanoid-voice-interaction.md) |
| 8.3 VLN | [vision-language-navigation](../../wiki/tasks/vision-language-navigation.md) |
| 8.4 NaVid | [paper-vln-10-navid](../../wiki/entities/paper-vln-10-navid.md) |

## 相关外部入口

- 深蓝学院课程页：<https://www.shenlanxueyuan.com/course/802/task/33927/show>
- CMU Exploration（TARE/FAR 开发环境）：<https://www.cmu-exploration.com/>
- 旧笔记索引：[know-how.md](../notes/know-how.md)（深蓝学院章节列表）
