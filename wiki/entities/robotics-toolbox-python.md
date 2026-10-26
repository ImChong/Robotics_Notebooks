---
type: entity
project_id: robotics-toolbox-python
project: https://petercorke.github.io/robotics-toolbox-python/
code: https://github.com/petercorke/robotics-toolbox-python
tags: [software, python, robotics, kinematics, inverse-kinematics, dynamics, trajectory-generation, manipulation, mobile-robot, qut]
status: complete
updated: 2026-10-26
related:
  - ../formalizations/forward-kinematics.md
  - ../formalizations/inverse-kinematics.md
  - ../formalizations/robot-jacobian.md
  - ./modern-robotics-book.md
  - ./pinocchio.md
sources:
  - ../../sources/repos/robotics-toolbox-python.md
  - ../../sources/sites/robotics-toolbox-python-docs.md
summary: "Peter Corke 的 Python 机器人学工具箱，覆盖串联机械臂建模、运动学、动力学和轨迹生成，也提供移动机器人规划、定位、建图与 SLAM 算法。"
---

# Robotics Toolbox for Python

**一句话：** Robotics Toolbox for Python（RTB-P）是面向研究与教育的 Python 工具箱，用同一套可运行 API 表达机械臂与移动机器人的常见建模、运动学、动力学和规划算法。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RTB-P | Robotics Toolbox for Python | 本页介绍的 Python 机器人学工具箱 |
| DH | Denavit–Hartenberg | 用参数化连杆坐标变换描述串联机械臂 |
| ETS | Elementary Transform Sequence | 工具箱采用的基本变换序列表示 |
| URDF | Unified Robot Description Format | 可导入的机器人连杆与关节描述格式 |
| FK | Forward Kinematics | 从关节配置计算末端位姿 |
| IK | Inverse Kinematics | 根据目标末端位姿求关节配置 |

## 项目与维护者

Peter Corke 的官方 QUT 学术主页将他列为 QUT 荣休杰出教授，并说明他创建了用于机器人学教学与研究的开源 Python/MATLAB 软件。该主页确认的是作者的 QUT 关联；代码开放与 MIT 许可则以项目仓库为准。

## 为什么重要

- **把理论变成可检查的程序：** 可直接构造机器人模型并运行 FK、IK、雅可比、动力学与轨迹示例，适合教学、原型验证和算法比较。
- **模型入口灵活：** 官方 README 支持 DH 建模、URDF 导入和内置模型；示例包括 Franka Panda。
- **同一工具箱兼顾移动机器人：** 除机械臂外，还提供运动模型、路径规划、定位、建图和 SLAM 的实现，适合课堂实验和小规模原型。

## 核心结构与用法

| 模块 | 官方材料列出的能力 | 适合的使用阶段 |
|------|------------------|----------------|
| 机器人模型 | 串联机械臂、DH/ETS 模型、URDF 导入与内置模型 | 模型搭建与算法验证 |
| 运动学 | 正向运动学、雅可比与数值/解析逆运动学 | 末端位姿计算、IK 原型 |
| 动力学与轨迹 | 刚体动力学算法、关节轨迹生成 | 运动分析与轨迹实验 |
| 移动机器人 | 运动模型、路径规划、定位、建图与 SLAM | 教学示例与算法原型 |
| 可视化 | Matplotlib、Swift 等后端展示机器人运动 | 调试和直观检查 |

最小示例沿用官方 README 的 Panda 用法：

```python
import roboticstoolbox as rtb

robot = rtb.models.Panda()
T = robot.fkine(robot.qr)   # ready pose 的末端位姿
sol = robot.ik_LM(T)        # Levenberg–Marquardt 数值 IK
```

安装核心包可用 `pip install roboticstoolbox-python`。Swift 可视化、QP IK、碰撞检查、命令行工具和解析 IK 推导依赖以官方安装文档中的可选 extras 为准。

## 工程实践

- 用同一个 URDF 或 DH 模型计算 FK、雅可比和 IK，可先验证坐标系、关节顺序和末端定义，再迁入 ROS 2 或控制器工程。
- 官方工具箱提供的是机器人算法与建模 API；部署到实机时仍需由应用补齐驱动、状态估计、实时调度、安全监控和硬件通信。
- 可将 FK 结果与独立实现（如 [Pinocchio](./pinocchio.md)）交叉核验；IK 则应检查求解状态、残差、关节限位和碰撞约束，不能只看是否返回关节值。
- 官方 README 要求 Python 3.10 或更新版本。可选依赖按需安装；例如 Windows 上碰撞检查依赖 Coal，而官方说明该依赖没有 PyPI Windows wheel。

## 局限与风险

1. **示例库不等同于完整机器人系统：** 工具箱不替代 ROS 2 驱动、控制器、安全层或实时部署方案。
2. **算法结果受模型质量约束：** URDF/DH 的轴向、零位、单位或末端定义错误会直接污染 FK、雅可比与 IK。
3. **安装能力取决于 extras 与平台：** 按需安装依赖，并在目标操作系统上核对官方说明；不要将可选可视化或碰撞依赖当作核心包保证。
4. **性能数字需按目标机器复测：** 官方 README 的速度描述不代表任意平台上的端到端控制周期保证。

## 关联页面

- [正向运动学](../formalizations/forward-kinematics.md) — 机器人模型到末端位姿的可执行计算
- [逆运动学](../formalizations/inverse-kinematics.md) — 数值 IK 目标函数、初值与约束检查
- [机器人雅可比矩阵](../formalizations/robot-jacobian.md) — 速度/力映射及操纵度分析
- [Modern Robotics 教材](./modern-robotics-book.md) — 配套理论语言与螺旋理论背景
- [Pinocchio](./pinocchio.md) — C++ 刚体动力学库，可作模型与动力学计算对照

## 参考来源

- [Robotics Toolbox for Python 仓库归档](../../sources/repos/robotics-toolbox-python.md)
- [Robotics Toolbox for Python 官方文档归档](../../sources/sites/robotics-toolbox-python-docs.md)
- [官方 GitHub 仓库与 README](https://github.com/petercorke/robotics-toolbox-python)
- [QUT 官方学术主页：Peter Corke](https://www.qut.edu.au/about/our-people/academic-profiles/peter.corke)
- [官方文档](https://petercorke.github.io/robotics-toolbox-python/)
- [Corke & Haviland, “Not your grandmother’s toolbox – the Robotics Toolbox reinvented for Python,” ICRA 2021](https://ieeexplore.ieee.org/document/9561366)

## 推荐继续阅读

- [官方 Wiki：Models](https://github.com/petercorke/robotics-toolbox-python/wiki/Models)
- [官方 Wiki：Kinematics](https://github.com/petercorke/robotics-toolbox-python/wiki/Kinematics)
- [官方示例目录](https://github.com/petercorke/robotics-toolbox-python/tree/main/examples)
