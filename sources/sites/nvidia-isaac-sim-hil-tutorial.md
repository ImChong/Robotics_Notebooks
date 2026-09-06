# Getting Started With Isaac Sim — Hardware-in-the-Loop (HIL)

> 来源归档

- **标题：** Hardware-in-the-Loop (HIL) Fundamentals — Leveraging ROS 2 and Hardware-in-the-Loop in Isaac Sim
- **类型：** course / 教程模块
- **URL：** https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-sim/latest/leveraging-ros-2-and-hil-in-isaac-sim/01-hardware-in-the-loop-hil-fundamentals.html
- **课程：** Getting Started With Isaac Sim（NVIDIA Physical AI Learning）
- **模块目录：** `leveraging-ros-2-and-hil-in-isaac-sim/`（含 Jetson 环境、Isaac ROS 部署等后续章节）
- **门户：** https://docs.nvidia.com/learning/physical-ai/
- **入库日期：** 2026-09-06
- **一句话说明：** 官方 HIL 入门：在 **仿真环境** 中连接 **真实硬件**（传感器、控制器、Jetson 等），验证软硬件集成；SIL 之后、全尺寸部署之前的工程验证环。

## 模块要点（2026-09-06 抓取）

### HIL 定义

- 将软件接到 **真实硬件组件**，同时用仿真复现其将面对的环境（工厂地面、户外地形等）。
- 安全、可重复、低成本地测试控制算法，无需完整物理系统。

### HIL vs SIL

| 缩写 | 焦点 |
|------|------|
| **SIL** | 算法与逻辑在 **纯仿真** 环境验证 |
| **HIL** | 在仿真环境中纳入 **真实硬件**，测软硬件集成 |

- 管线顺序：**SIL → HIL → 部署**；HIL 可暴露 SIL 遗漏的硬件约束（如 **内存不足**、算力不够）。

### 收益与应用

- **成本与时间：** 减少物理样机迭代；早期发现问题。
- **安全：** 可测危险或难复现场景（极端天气、故障模式）。
- **行业：** 工业自动化、汽车 ECU、航空航天等。

### 有效 HIL 配置要点

- **先虚拟后实物：** 无完整硬件即可用虚拟模型启动测试，加速迭代。
- **组件集成：** 验证目标硬件是否具备运行最终软件所需的 **算力与内存**。
- **迭代测试：** 基于结果持续 refine，再进入物理测试。

### 课程后续（同模块族）

- NVIDIA Jetson 平台概览与环境搭建
- 在 Jetson 上部署 Isaac ROS
- 与 ROS 2 联用的 HIL 工作流

## 命名消歧

本模块 **HIL = Hardware-in-the-Loop**，与以下 **不同含义** 的 HIL 无关：

- [Hybrid Imitation Learning](../../wiki/methods/hil-hybrid-imitation-learning.md)（TOG 2026 跑酷模仿）
- [HIL-HARC](../../wiki/entities/paper-hil-harc.md)（真机在线 RL 论文缩写碰撞）

## 对 wiki 的映射

- 概念页：**`wiki/concepts/hardware-in-the-loop.md`**
- [Software-in-the-Loop](../../wiki/concepts/software-in-the-loop.md)
- [Isaac Sim](../../wiki/entities/isaac-sim.md)
- [NVIDIA Physical AI Learning](../../wiki/entities/nvidia-physical-ai-learning.md)
