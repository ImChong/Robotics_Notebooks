---
type: method
tags: [robotics, kinematics, retargeting, humanoid]
status: complete
updated: 2026-04-27
related:
  - ../concepts/motion-retargeting.md
  - ./beyondmimic.md
sources:
  - ../../sources/papers/motion_control_projects.md
summary: "GMR (General Motion Retargeting) 是一种高效的通用动作重定向方法，主要解决从人类动捕数据到异构机器人骨架的几何映射问题。"
---

# GMR: 通用动作重定向

**GMR (General Motion Retargeting)** 是运动控制流程中的“前端”模块，负责将人类或其他来源的动作序列转换为机器人可理解的关节角度序列。

## 核心原理

GMR 主要基于**运动学 (Kinematics)** 优化。它不考虑力学项，而是通过最小化几何误差来实现姿态复现。

### 优化目标
1. **关键点位置匹配**：让机器人的手掌、脚掌、肘部等关键点位置尽可能贴近参考轨迹。
2. **关节限位约束**：确保生成的角度不超出机器人的物理极限。
3. **平滑性约束**：减少相邻帧之间的角度突变。

## 主要技术路线

| 模块 | 核心方法 | 关键约束 |
|------|---------|---------|
| **骨架映射** | 关节树匹配 / 重排 | 处理人机自由度不一致 |
| **几何对齐** | 关键点 IK (Inverse Kinematics) | 最小化手/足位置与参考轨迹误差 |
| **数值求解** | 基于 QP 的优化器 | 满足关节限位与角速度连续性 |
| **后处理** | 时间平滑 + 静态稳定性筛选 | 减少高频噪声，剔除极度失稳片段 |

## 关键局限与避坑指南

根据《开源运动控制项目》文档的点评，GMR 的使用必须注意其“非物理性”：

### 1. 缺乏动力学一致性
GMR 只管姿态“像不像”，不管“能不能站稳”。
- **表现**：重定向后的轨迹可能出现脚悬空、质心超出支撑多边形的情况。
- **后果**：直接把 GMR 输出给底层 PD 控制器，机器人极大概率摔倒。

### 2. 接触不连续性
由于没有建模接触力，GMR 输出的轨迹在脚触地瞬间可能存在穿透或虚位。

### 3. 速度与加速度跳变
几何最优不代表导数最优。

## 工业界最佳实践

**GMR 只是起点，不是终点。** 

一个完整的重定向流水线应为：
$$
\text{Raw MoCap} \xrightarrow{GMR} \text{Kinematic Trajectory} \xrightarrow{\text{Dynamic Filter}} \text{Feasible Trajectory}
$$

- **动力学过滤层**：通过 QP 优化（如 HALO 方式）或全动力学优化，补上质量、惯性和接触力约束。
- **RL 细化**：将 GMR 轨迹作为参考，通过 [BeyondMimic](./beyondmimic.md) 等框架训练具有鲁棒性的 RL 策略。

## 参考来源

- [sources/papers/motion_control_projects.md](../../sources/papers/motion_control_projects.md) — 飞书公开文档《开源运动控制项目》总结。
- [GMR 源码仓库](https://github.com/YanjieZe/GMR)

## 关联页面

- [Motion Retargeting (动作重定向)](../concepts/motion-retargeting.md) — 任务概览。
- [BeyondMimic](./beyondmimic.md) — 动作模仿学习通常以重定向后的轨迹作为输入。
