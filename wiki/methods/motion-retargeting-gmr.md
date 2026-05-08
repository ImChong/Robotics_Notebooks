---
type: method
tags: [robotics, kinematics, retargeting, humanoid]
status: complete
updated: 2026-05-08
related:
  - ../concepts/motion-retargeting.md
  - ./neural-motion-retargeting-nmr.md
  - ./beyondmimic.md
sources:
  - ../../sources/papers/motion_control_projects.md
  - ../../sources/papers/exoactor.md
  - ../../sources/papers/neural_motion_retargeting_nmr.md
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

## 反例：什么时候不该用 GMR

并非所有流水线都从重定向获益。[ExoActor (BAAI, 2026)](./exoactor.md) 在视频生成 → 动作估计 → 动作跟踪的流水线上做了消融：

- **现象**：在估计出来的 SMPLX 轨迹上叠加 GMR / OmniRetarget 后，全身运动确实更平滑、抖动更少，但同时引入了明显的全局空间偏差。
- **原因**：上游动作估计本身有全局位置漂移和脚滑，重定向尝试"修正"这些伪影时反而破坏了原本的轨迹；同时人机肢长比例差异会放大步长与位置积累误差。
- **结论**：在该流水线下，作者选择**直接把人体动作喂给 SONIC**，跳过中间重定向阶段。

这说明 GMR 在 MoCap → 机器人这种"源动作干净"的链路上是收益项，但在"源动作本身就来自上游估计/生成模型"的链路上，需要额外评估它是否会放大上游噪声。

## 互补视角：NMR 把 GMR 放进数据管线

[NMR（神经运动重定向与人形全身控制）](./neural-motion-retargeting-nmr.md) 仍用 GMR 生成**运动学初轨迹**，再通过 **CEPR**（聚类、并行 RL 跟踪专家、仿真 rollout）得到物理更一致的 **人机配对** 监督，最后训练 CNN–Transformer 做整段推断。可将这条路线理解为：**GMR 负责覆盖与几何对齐，仿真 RL 负责把轨迹拉回可行流形，神经网络负责快速、时序一致的前向重定向**。

## 参考来源

- [sources/papers/motion_control_projects.md](../../sources/papers/motion_control_projects.md) — 飞书公开文档《开源运动控制项目》总结。
- [sources/papers/exoactor.md](../../sources/papers/exoactor.md) — ExoActor 的重定向消融提供"什么时候不该用 GMR"的反例。
- [sources/papers/neural_motion_retargeting_nmr.md](../../sources/papers/neural_motion_retargeting_nmr.md) — NMR 以 GMR 为 CEPR 前端的神经重定向工作。
- [GMR 源码仓库](https://github.com/YanjieZe/GMR)

## 关联页面

- [Motion Retargeting (动作重定向)](../concepts/motion-retargeting.md) — 任务概览。
- [BeyondMimic](./beyondmimic.md) — 动作模仿学习通常以重定向后的轨迹作为输入。
- [ExoActor](./exoactor.md) — 视频生成驱动的人形控制流水线，给出"何时跳过 GMR"的反例。
- [NMR（神经运动重定向与人形全身控制）](./neural-motion-retargeting-nmr.md) — 用 GMR + 仿真 RL 构造监督的学习式重定向。
