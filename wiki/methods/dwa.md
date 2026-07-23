---
type: method
tags: [path-planning, local-planning, obstacle-avoidance, navigation, mobile-robot]
status: complete
updated: 2026-07-23
related:
  - ./a-star.md
  - ../entities/python-robotics.md
  - ../entities/navigation2.md
  - ../concepts/dynamic-obstacle-filtering.md
  - ../entities/humanoid-system-curriculum.md
sources:
  - ../../sources/courses/shenlan_humanoid_system_theory_practice.md
  - ../../sources/repos/python_robotics.md
  - ../../sources/repos/navigation2.md
summary: "DWA 局部路径规划：在速度空间动态窗口内采样可达 (v,ω)，用朝向/速度/间隙代价选最优控制，实现实时避障；Nav2 DWB 为其工程继承。"
---

# DWA（Dynamic Window Approach）局部路径规划

## 一句话定义

**动态窗口法（DWA）** 在机器人加速度与障碍约束形成的 **速度窗口** 内采样轨迹，按目标朝向、速度与空隙评价选最优 \((v,\omega)\)，实现局部避障——课程第 4.3 节。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DWA | Dynamic Window Approach | 速度空间采样局部规划 |
| DWB | Dynamically Windowed Blind/Base | Nav2 中 DWA 系局部控制器 |
| cmd_vel | Command Velocity | ROS 速度指令话题 |
| Footprint | Robot Footprint | 碰撞检测用机器人外形 |
| TEB | Timed Elastic Band | 对照局部优化规划器 |

## 为什么重要

- 全局 [A\*](./a-star.md) 不懂差速/人形速度限幅；DWA 把 **动力学可达** 与 **即时障碍** 放进同一评分。
- 课程实践「A\* + DWA 避障」是经典两层导航作业。
- 运行时动态体进 obstacle costmap；静态层仍靠 [动态障碍物滤波](../concepts/dynamic-obstacle-filtering.md) 保持干净。

## 主要技术路线

| 路线 | 说明 | 工程入口 |
|------|------|----------|
| 经典 DWA | 速度窗口采样 + 多目标打分 | PythonRobotics / 教材实现 |
| DWB（Nav2） | 插件化轨迹生成与批评器 | [Navigation2](../entities/navigation2.md) |
| 与 TEB/MPC 对照 | 优化式局部轨迹 vs 采样式 | 狭窄通道、高动态场景选型 |
| 人形速度桥接 | DWA `cmd_vel` → 行走策略接口 | [G1 软件栈](../entities/unitree-g1-software-stack.md) |

## 核心原理

1. 由当前速度与加加速度限制截出动态窗口。
2. 向前仿真多条圆弧/线段轨迹，做 footprint 碰撞检测。
3. 代价常含：到全局路径的偏差、到目标朝向、速度奖励、障碍间隙。
4. 输出瞬时速度指令，下一周期重规划。

## 工程实践

- 算法直觉：[PythonRobotics](../entities/python-robotics.md) DWA 示例。
- 工程：[Navigation2](../entities/navigation2.md) `dwb_core` / 其它局部控制器插件。
- 人形若底层是行走策略而非差速底盘，需把 DWA 输出映射为 **速度命令接口**（见 [G1 软件栈](../entities/unitree-g1-software-stack.md)）。

## 局限与风险

- 局部最优与狭窄通道振荡；参数（评分权重、仿真时域）敏感。
- 高速或非完整约束强时，可考虑 TEB/MPC 局部器。

## 关联页面

- [A\*](./a-star.md)
- [人形系统课程策展](../entities/humanoid-system-curriculum.md)

## 参考来源

- [深蓝学院人形系统课程大纲](../../sources/courses/shenlan_humanoid_system_theory_practice.md)
- [PythonRobotics 归档](../../sources/repos/python_robotics.md)
- [Navigation2 归档](../../sources/repos/navigation2.md)

## 推荐继续阅读

- Fox, Burgard, Thrun, *The Dynamic Window Approach to Collision Avoidance*, IEEE RAM 1997
