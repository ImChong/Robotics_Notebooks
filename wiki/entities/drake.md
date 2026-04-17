---
title: Drake（机器人仿真与控制工具链）
type: entity
status: complete
created: 2026-04-14
updated: 2026-04-14
summary: MIT 开发的机器人动力学仿真、优化与控制框架，以严格的数学基础和符号计算著称，常用于轨迹优化、MPC 和接触力学研究。
sources:
  - ../../sources/papers/optimal_control.md
  - ../../sources/papers/robot_kinematics_tools.md
---

# Drake

## 是什么

Drake 是由 MIT Robotics (Russ Tedrake 团队) 开发的开源机器人工程平台，提供：
- **多体动力学仿真**（刚体 + 接触）
- **轨迹优化工具**（SNOPT, IPOPT, OSQP 后端）
- **系统级控制框架**（diagram/system 层级）
- **符号自动微分**（与 MathematicalProgram 集成）

GitHub: [RobotLocomotion/drake](https://github.com/RobotLocomotion/drake)

---

## 核心模块

### 1. MultibodyPlant
Drake 的动力学核心：

```python
from pydrake.multibody.plant import MultibodyPlant
plant = MultibodyPlant(time_step=0.001)
parser = Parser(plant)
parser.AddModelFromFile("robot.urdf")
plant.Finalize()
```

- 支持 URDF / SDF / SDFormat
- 刚体接触：HydroElastic Contact（连续接触模型），更物理真实
- 可输出质量矩阵 M(q)、科氏矩阵 C(q,v)、重力 g(q)

### 2. MathematicalProgram（优化）
统一的数学规划接口，支持：
- LP / QP / SOCP / SDP / NLP
- 后端：OSQP, SNOPT, IPOPT, Gurobi, MOSEK

```python
prog = MathematicalProgram()
x = prog.NewContinuousVariables(4, "x")
prog.AddQuadraticCost(x.dot(Q).dot(x))
prog.AddLinearConstraint(A @ x <= b)
result = Solve(prog)
```

### 3. 轨迹优化（DirectCollocation / DIRCOL）
全身轨迹优化的主要工具：

```python
dirtran = DirectTranscription(plant, context, num_time_samples=21)
dirtran.AddRunningCost(u.dot(u))
dirtran.AddFinalCost(...)
result = Solve(dirtran.prog())
```

### 4. LCM 通信
Drake 使用 LCM（Lightweight Communications and Marshalling）作为进程间通信协议，配合 `drake-ros` 可接入 ROS2。

---

## Drake vs 其他仿真器

| 特性 | Drake | MuJoCo | Isaac Sim | Gazebo |
|------|-------|--------|-----------|--------|
| **接触模型** | HydroElastic（更精确） | 柔性接触 | GPU 刚性接触 | ODE/Bullet |
| **优化集成** | 内置，一流 | 需外部 | 无 | 无 |
| **并行仿真** | 不支持 GPU 并行 | 支持（有限） | 支持，万级 | 不支持 |
| **数学严格性** | 极高（符号微分） | 中 | 低 | 低 |
| **RL 训练速度** | 慢（非 GPU） | 中 | 极快 | 慢 |
| **轨迹优化** | 极强 | 弱 | 无 | 无 |
| **适用场景** | 轨迹优化 / MPC / 研究 | RL 训练 / 学术 | 大规模并行 RL | ROS 集成 |

---

## 典型使用场景

### 1. 轨迹优化 + WBC
- 用 DirectCollocation 优化 humanoid 的跳跃/上楼轨迹
- 生成参考轨迹给 WBC 执行

### 2. MPC 控制器设计
- MultibodyPlant 提供精确动力学，作为 MPC 的预测模型
- 配合 OSQP 实时求解

### 3. 系统辨识（SysID）
- Drake 的符号动力学可用于参数辨识（关节摩擦、质量）
- 生成辨识实验的最优激励轨迹

### 4. 教学 / 研究
- MIT 6.832（Underactuated Robotics）课程的官方工具
- Russ Tedrake 的《Underactuated Robotics》教材配套

---

## 安装与快速开始

```bash
pip install drake
```

或使用 Docker：
```bash
docker pull robotlocomotion/drake:latest
```

与 Python 接口（pydrake）：
```python
import pydrake
from pydrake.all import *
```

---

## 参考来源

- Tedrake et al., *Drake: Model-Based Design and Verification for Robotics* (2019) — Drake 系统论文
- Tedrake, *Underactuated Robotics* (MIT 6.832 课程教材) — 最佳学习资源
- Mastalli et al., *Crocoddyl* (2020) — 类似工具，对比参考

---

## 关联页面

- [Trajectory Optimization](../methods/trajectory-optimization.md) — Drake 的核心应用场景
- [Model Predictive Control (MPC)](../methods/model-predictive-control.md) — Drake 常用于 MPC 的预测模型和求解器
- [Optimal Control (OCP)](../concepts/optimal-control.md) — Drake 实现的核心理论基础
- [Whole-Body Control](../concepts/whole-body-control.md) — Drake 生成的轨迹供 WBC 追踪
- [Crocoddyl](./crocoddyl.md) — 功能类似的另一个轨迹优化框架（更偏 DDP/iLQR）
- [MuJoCo](./mujoco.md) — 互补工具，Drake 用于优化，MuJoCo 用于 RL 训练
