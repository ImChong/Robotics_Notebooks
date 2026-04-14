---
type: roadmap
tags: [fullstack, humanoid, perception, planning, deployment, integration]
status: stable
---

# 路线B：机器人全栈工程师扩展路线

> **目标**：在路线A（运动控制成长路线）基础上，从"控制切入口"扩展到感知、规划、软件系统、部署与整机集成，成为能独立交付"会走 + 会看 + 会规划 + 能上真机"的全栈机器人工程师。
>
> **前置条件**：完成路线A L0-L3（基础运动控制、WBC/RL 基础，能在仿真中训练出可用策略）。

---

## 路线B 全局地图

```
L0  运动控制基础（路线A前置）
    ↓
L1  感知层：状态估计 + 视觉感知
    ↓
L2  规划层：运动规划 + 任务规划
    ↓
L3  软件系统：ROS2 + 中间件 + 实时通信
    ↓
L4  仿真与数字孪生
    ↓
L5  硬件认知：执行器 + 传感器 + 驱动
    ↓
L6  部署、调试、测试、安全
    ↓
L7  整机集成：感知→规划→控制全链路闭环
```

---

## L1 · 感知层（状态估计 + 视觉）

### L1.1 机器人状态估计

**目标**：理解并能实现基于 IMU + 运动学的 EKF 基座姿态估计。

核心知识：
- EKF 基础：预测步（IMU 积分）+ 更新步（接触运动学约束）
- 浮动基状态：$[p_b, q_b, v_b, q_j, \dot{q}_j]$
- 接触检测：力矩估计 / 接触力阈值 / 概率模型
- InEKF（不变 EKF）：李群 $SE_2(3)$ 上的状态估计，比标准 EKF 更鲁棒

工具：
```bash
# 典型状态估计实现路径
1. 了解 bloesch-rsll-state-est（ETH 经典EKF）
2. 了解 contact_estimation（接触概率模型）
3. Pinocchio + IMU 数据融合
```

关键论文：Bloesch 2013（RSL EKF）、Hartley InEKF 2020

**阶段验收**：在 Isaac Sim 中给仿真机器人写一个简单 EKF，base 速度估计误差 < 0.1 m/s。

### L1.2 视觉感知基础

**目标**：理解深度图、点云处理、高度图生成；能从 RGB-D 数据生成地形高度图。

核心知识：
- 深度传感器：结构光 / ToF / 立体视觉 / LiDAR
- 点云处理：下采样 / 地面分割 / 高度图生成
- 视觉里程计：ORB-SLAM3 / DROID-SLAM 原理
- 地形感知在 RL 中的应用：高度图 → privileged observation

工具：
```python
# 高度图生成示例
import open3d as o3d
import numpy as np
pcd = o3d.io.read_point_cloud("scan.pcd")
# 体素降采样 → 投影到网格 → 生成高度图
```

**阶段验收**：从 Isaac Sim 的深度图数据生成 11×11 高度图，可视化验证与真实地形一致。

---

## L2 · 规划层

### L2.1 运动规划基础

**目标**：理解路径规划→轨迹规划→步位规划的层次结构；能调用现有库生成无碰撞轨迹。

核心知识：
- **导航层**：A\* / Dijkstra / RRT（全局路径规划）
- **局部规划**：DWA（动态窗口法）/ TEB / MPPI
- **步位规划**：基于 Capture Point 的反应式 + MPC 多步预测
- **接触序列规划**：离散接触点 → 连续轨迹优化

工具栈：
- Nav2（ROS2 导航框架）
- OMPL（采样规划库）
- Drake DirectCollocation（接触轨迹优化）

**阶段验收**：用 Nav2 让机器人在带障碍物的地图中从 A 走到 B。

### L2.2 任务规划基础

**目标**：理解行为树（BT）和有限状态机（FSM），能用 BT 编排多步任务。

核心知识：
- FSM：状态定义 + 转移条件 + 典型步态切换实现
- 行为树：Sequence / Selector / Condition / Action 节点
- 任务分解：高层任务 → 子任务 → 基本动作
- LLM + Robotics：用语言模型生成任务规划

工具：
- BehaviorTree.CPP（C++/Python BT 框架）
- py_trees（Python BT）

**阶段验收**：用行为树实现"走到目标点 → 检测障碍 → 绕行 → 到达"的完整任务序列。

---

## L3 · 软件系统

### L3.1 ROS2 核心概念

**目标**：理解 ROS2 通信模型，能用 ROS2 完成机器人感知→控制的完整数据流。

核心知识：
- **话题（Topic）**：异步发布/订阅，适合传感器数据
- **服务（Service）**：同步请求/响应，适合一次性任务
- **动作（Action）**：带反馈的异步任务，适合长时运动
- **参数服务器**：运行时配置管理
- **生命周期节点**：受控节点启停，用于安全关键系统

实践：
```bash
# 标准开发流程
ros2 pkg create --build-type ament_cmake my_robot_pkg
colcon build
source install/setup.bash
ros2 run my_robot_pkg my_node
```

**阶段验收**：用 ROS2 实现 IMU 数据订阅 → 状态估计节点 → 关节命令发布的完整链路。

### L3.2 实时通信与中间件

**目标**：了解机器人低延迟通信的核心要求，理解 EtherCAT / LCM / CycloneDDS。

核心知识：
- **实时性要求**：控制循环通常 1kHz，通信延迟 < 1ms
- **EtherCAT**：工业标准现场总线，用于伺服驱动器通信
- **LCM（Lightweight Communications）**：MIT Cheetah 使用，低延迟 UDP 多播
- **DDS（Data Distribution Service）**：ROS2 底层，支持 QoS 配置
- **共享内存**：同机器高速数据共享，延迟 < 10μs

**阶段验收**：了解 Unitree SDK 的通信协议（LCM 或 CycloneDDS），能读取关节状态并发送命令。

---

## L4 · 仿真与数字孪生

### L4.1 仿真平台选型与使用

**目标**：能在 MuJoCo / Isaac Lab 中建立自定义机器人场景，实现感知→控制闭环。

| 平台 | 适合 | 注意 |
|------|------|------|
| MuJoCo | 精度高，接触物理稳定 | 单线程（MJX 除外） |
| Isaac Lab | GPU 并行，大规模 RL | 依赖 NVIDIA GPU |
| Genesis | 速度极快，可微分 | 新兴，API 还在变化 |
| Drake | 精确物理+控制设计 | 学习曲线陡 |

**阶段验收**：在 Isaac Lab 中从 URDF 导入自定义场景，训练一个能走 5m 的简单策略。

### L4.2 数字孪生概念

**目标**：了解什么是数字孪生及其在机器人测试中的价值。

核心概念：
- **数字孪生**：与真实机器人同步更新的虚拟模型
- **软在环（SIL）**：策略在仿真中运行，真实传感器输入
- **硬在环（HIL）**：真实控制器运行，仿真环境反馈
- **param sweeping**：在孪生中批量测试参数，无需真机风险

**阶段验收**：能描述 SIL / HIL 的差异，并用 Isaac Sim 实现一个简单的 SIL 测试。

---

## L5 · 硬件认知

### L5.1 执行器与驱动

**目标**：理解关节执行器的控制接口，能调整 PD 增益实现稳定关节控制。

核心知识：
- **电机类型**：直流无刷（BLDC）/ 行星减速器 / 谐波减速器
- **控制模式**：位置控制 / 速度控制 / 力矩控制（优先）
- **PD 增益调参**：$\tau = k_p(q_{des} - q) + k_d(\dot{q}_{des} - \dot{q})$
- **Unitree SDK 接口**：`MotorState.q`, `MotorState.dq`, `MotorCmd.tau`
- **执行器网络（ActuatorNet）**：用神经网络建模非线性执行器动力学（Hwangbo 2019）

**阶段验收**：在真实 Unitree Go1/G1 上手动调 kp/kd，实现稳定站立。

### L5.2 传感器认知

**目标**：了解机器人主要传感器的特性和局限。

| 传感器 | 用途 | 局限 |
|--------|------|------|
| IMU（6轴） | 姿态估计、加速度 | 积分漂移、震动噪声 |
| 关节编码器 | 关节位置/速度 | 分辨率、延迟 |
| 力/力矩传感器 | 接触力直接测量 | 昂贵、噪声 |
| RGB-D（RealSense/ZED） | 深度感知、地形 | 计算量大、光照敏感 |
| LiDAR | 精确 3D 环境扫描 | 重量、功耗 |

---

## L6 · 部署与安全

### L6.1 真机部署流程

**目标**：掌握从仿真策略到真机部署的完整流程，具备安全操作意识。

```
仿真策略 → 导出 ONNX/TorchScript
         → 真机推理测试（吊绳）
         → 低速平地验证
         → 逐步放开速度/地形
         → 全参数测试
```

部署 Checklist（参考 [humanoid-rl-cookbook](../wiki/queries/humanoid-rl-cookbook.md)）：
- [ ] 策略推理频率 ≥ 控制频率
- [ ] Observation 归一化参数一致
- [ ] 动作裁剪（关节限位）
- [ ] 紧急停止按钮可达
- [ ] 首次运行全程有人监控

### L6.2 调试技能

**目标**：能系统性诊断"策略在仿真里好，真机上差"的问题。

常见 sim2real 失败原因：
1. **执行器动力学差异**：仿真 PD 增益与真机不符 → 检查 kp/kd/力矩限制
2. **传感器噪声不匹配**：仿真无 IMU 噪声 → 加入噪声层
3. **延迟差异**：仿真无通信延迟 → 加 1-2 步 action latency
4. **接触建模差异**：地面硬度/摩擦系数 → 域随机化扩大范围
5. **执行器饱和**：真机力矩饱和但仿真无限制 → 加力矩裁剪

---

## L7 · 整机集成

### L7.1 感知→规划→控制全链路

**目标**：能独立完成"看到目标 → 规划路径 → 走过去"的完整整机演示。

全链路架构：
```
传感器层       → RGB-D + IMU + 关节编码器
      ↓
感知层         → 高度图 + 目标检测 + 状态估计
      ↓
规划层         → 全局路径 + 步位规划 + 步态选择
      ↓
控制层         → MPC/RL 策略 + WBC 接触力分配
      ↓
执行层         → 关节力矩命令 → 执行器驱动
```

**阶段验收**：在 Isaac Sim 中完成端到端演示：机器人从 RGB-D 视频流中识别目标，自动规划路径并行走到达。

### L7.2 系统集成工程技能

- **配置管理**：YAML/TOML 集中管理所有模块参数
- **日志系统**：时序数据记录（关节状态/控制命令/估计值），便于复现问题
- **可视化**：RViz2 / Foxglove Studio 实时监控
- **单元测试**：每个模块有独立测试，集成测试验证接口兼容性
- **CI**：GitHub Actions 自动运行仿真冒烟测试

---

## 推荐学习顺序

| 阶段 | 时间（估计） | 核心产出 |
|------|-------------|---------|
| L1 状态估计 | 2-3 周 | EKF 在 Isaac Sim 中验证 |
| L1 视觉基础 | 2-3 周 | 高度图生成 pipeline |
| L2 运动规划 | 2-3 周 | Nav2 避障导航 |
| L3 ROS2 | 2-4 周 | 完整传感→控制 ROS2 节点 |
| L4 仿真 | 1-2 周 | 自定义场景 Isaac Lab |
| L5 硬件 | 持续 | Unitree SDK 实操 |
| L6 部署 | 2-3 周 | 真机策略上线 |
| L7 集成 | 4-6 周 | 端到端整机演示 |

---

## 关联页面

- [路线A：运动控制成长路线](./route-a-motion-control.md) — 本路线的前置
- [State Estimation](../wiki/concepts/state-estimation.md) — L1 状态估计深度参考
- [Footstep Planning](../wiki/concepts/footstep-planning.md) — L2 步位规划
- [Gait Generation](../wiki/concepts/gait-generation.md) — L2 步态生成
- [Sim2Real](../wiki/concepts/sim2real.md) — L6 真机部署的核心挑战
- [Humanoid RL Cookbook](../wiki/queries/humanoid-rl-cookbook.md) — L6 部署 Checklist

---

## 一句话记忆

> 路线B 在运动控制基础上加了"眼睛"（感知）、"大脑"（规划）和"神经系统"（ROS2/中间件），最终目标是端到端整机——从传感器信号到关节力矩，全链路自己能跑通。
