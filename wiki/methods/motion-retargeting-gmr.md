---
type: method
tags: [robotics, kinematics, retargeting, humanoid]
status: complete
updated: 2026-05-12
related:
  - ../concepts/motion-retargeting.md
  - ./neural-motion-retargeting-nmr.md
  - ./beyondmimic.md
  - ./sonic-motion-tracking.md
sources:
  - ../../sources/papers/motion_control_projects.md
  - ../../sources/papers/exoactor.md
  - ../../sources/papers/neural_motion_retargeting_nmr.md
summary: "GMR (General Motion Retargeting) 是一种高效的通用动作重定向方法，主要解决从人类动捕数据到异构机器人骨架的几何映射问题。"
---

# GMR: 通用动作重定向

**GMR (General Motion Retargeting)** 是运动控制流程中的“前端”模块，负责将人类或其他来源的动作序列转换为机器人可理解的关节角度序列。

> **命名说明**：本知识库中的 **GMR** 均指 *General Motion Retargeting*（通用动作重定向），与统计学里的 **Gaussian Mixture Regression（高斯混合回归）** 缩写相同但无关。

## 核心原理

GMR 主要基于**运动学 (Kinematics)** 优化。它不考虑力学项，而是通过最小化几何误差来实现姿态复现。

### 优化目标
1. **关键点位置匹配**：让机器人的手掌、脚掌、肘部等关键点位置尽可能贴近参考轨迹。
2. **关节限位约束**：确保生成的角度不超出机器人的物理极限。
3. **平滑性约束**：减少相邻帧之间的角度突变。

## 单条流水线的结构（Mermaid）

下图概括「人体参考 → 机器人关节序列」在**几何层**上的典型数据流；与具体实现中的 IK 配置、关键点权重、是否滑窗等细节可能略有出入，但便于在工程里对齐模块边界。

```mermaid
flowchart TD
  subgraph inputs["人体侧参考（多源）"]
    M1[动捕 BVH / FBX 等]
    M2[SMPL / SMPL-X 序列]
    M3[单目视频 → 姿态估计<br/>如 GVHMR 等]
  end

  M1 --> N[格式归一与坐标对齐]
  M2 --> N
  M3 --> N

  N --> T[骨架子树 / DoF 对应<br/>人机拓扑对齐]
  T --> IK[任务空间误差 + IK / QP<br/>手 足 肘等关键点]
  IK --> C[关节限位 + 角速度上限等<br/>硬约束 / 罚项]
  C --> S[时间维平滑 / 异常帧剔除]
  S --> Q[机器人关节轨迹 q_t]

  Q --> OUT1[离线：导出 pickle / CSV<br/>供模仿学习或后处理]
  Q --> OUT2[在线：低延迟流<br/>对接遥操作或 RL tracking]
```

## 在整条控制栈中的位置（Mermaid）

GMR 解决的是「**长得像**」；是否「**站得稳、力矩可行**」要在下游处理。

```mermaid
flowchart LR
  subgraph upstream["上游"]
    U[干净 MoCap / 估计人体序列]
  end

  subgraph kinematic["几何层（本页）"]
    G[GMR 运动学重定向]
  end

  subgraph downstream["下游（常需另建模块）"]
    D1[动力学滤波 / QP 接触修补]
    D2[RL tracking 或 WBC 参考跟踪]
    D3[直接大规模 tracking<br/>部分流水线可跳过重定向]
  end

  U --> G
  G --> D1
  G --> D2
  D1 --> D2
  U -.->|估计噪声大时评估消融| D3
```

## 主要技术路线

| 模块 | 核心方法 | 关键约束 |
|------|---------|---------|
| **骨架映射** | 关节树匹配 / 重排 | 处理人机自由度不一致 |
| **几何对齐** | 关键点 IK (Inverse Kinematics) | 最小化手/足位置与参考轨迹误差 |
| **数值求解** | 基于 QP 的优化器 | 满足关节限位与角速度连续性 |
| **后处理** | 时间平滑 + 静态稳定性筛选 | 减少高频噪声，剔除极度失稳片段 |

## 开源实现侧的工程要点

官方仓库自述中强调的能力（便于与论文/代码对齐，**不等价**于「已解决动力学可行性」）：

- **CPU 上实时重定向**，面向全身遥操作与在线闭环（例如与 [TWIST](https://github.com/YanjieZe/TWIST) 等管线配套）。
- **多机型、多人体数据格式**：同一套接口切换目标人形机器人；支持常见动捕导出与视频估计链路（仓库持续扩展具体格式与 MJCF/URDF）。
- **与 RL tracking 协同**：README 写明针对 RL 跟踪策略做过工程调参；并提供与 [BeyondMimic](./beyondmimic.md) 等工具链对接的 **pickle → CSV** 等转换脚本思路。
- **默认关节角速度上限等保护**：减少「几何可行但电机跟不上」的极端指令（仍以机器人标定与控制器实测为准）。

论文与报告可优先查阅：[arXiv:2505.02833](https://arxiv.org/abs/2505.02833)、技术报告 [arXiv:2510.02252](https://arxiv.org/abs/2510.02252)；录用信息以官方仓库徽章为准（如 ICRA 2026 标注）。

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

### 4. 何时怀疑「重定向在帮倒忙」
若上游是人体 **估计/生成** 轨迹（全局漂移、脚滑、时序不连贯），几何重定向可能在「对齐人机比例」的同时 **放大** 空间误差；此时应做消融对比「重定向 → 跟踪」与「跳过中间层 → 跟踪」，参见下文 ExoActor 反例与 [SONIC](./sonic-motion-tracking.md) 讨论。

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
- Ze Y., et al. *GMR: General Motion Retargeting* — [arXiv:2505.02833](https://arxiv.org/abs/2505.02833)；技术报告 [arXiv:2510.02252](https://arxiv.org/abs/2510.02252)。
- [GMR 源码仓库](https://github.com/YanjieZe/GMR) — 功能列表、支持的机器人与数据格式、与 TWIST / MimicKit 等生态链接。

## 关联页面

- [Motion Retargeting (动作重定向)](../concepts/motion-retargeting.md) — 任务概览。
- [BeyondMimic](./beyondmimic.md) — 动作模仿学习通常以重定向后的轨迹作为输入。
- [ExoActor](./exoactor.md) — 视频生成驱动的人形控制流水线，给出"何时跳过 GMR"的反例。
- [NMR（神经运动重定向与人形全身控制）](./neural-motion-retargeting-nmr.md) — 用 GMR + 仿真 RL 构造监督的学习式重定向。
- [SONIC（规模化运动跟踪）](./sonic-motion-tracking.md) — 与「跳过重定向、直接 tracking」路线对照阅读。
