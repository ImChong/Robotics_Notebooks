---
type: concept
tags: [optimization, collision-avoidance, motion-planning, signed-distance, safety]
status: complete
updated: 2026-06-23
related:
  - ../entities/curobo.md
  - ../concepts/safety-filter.md
  - ../methods/smooth-navigation-path-generation.md
  - ../concepts/constrained-optimization.md
sources:
  - ../../sources/courses/numerical_optimization_foundations_robotics.md
summary: "碰撞距离计算：将几何碰撞检测表述为优化或距离场查询，是 TrajOpt、NMPC 与 cuRobo 类 GPU 规划的核心子问题。"
---

# Collision Distance Optimization（碰撞距离优化）

**碰撞距离优化**：求机器人构型 $q$ 与障碍物之间 **最小距离** 或 **有符号距离（SDF）**，并在 TrajOpt / NMPC 中作为不等式 $d(q) \ge d_{\min}$ 或软惩罚项；高效实现是 GPU 运动规划（如 [cuRobo](../entities/curobo.md)）的关键。

## 一句话定义

> 规划不仅要「路径短」，还要「离障碍够远」——碰撞距离把几何安全变成可微/可优化的约束或代价。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDF | Signed Distance Field | 有符号距离场，内外侧符号相反 |
| NLP | Nonlinear Programming | 轨迹级碰撞约束形成 NLP |
| NMPC | Nonlinear Model Predictive Control | 在线避障约束 |
| CBF | Control Barrier Function | 用距离构造安全不变集 |
| GPU | Graphics Processing Unit | 批量 SDF / 碰撞并行查询 |

## 核心结构

### 距离查询

- **Primitive 距离**：球、胶囊、OBB 间闭式距离
- **Mesh / SDF 场**：预计算体素或 GPU 查表
- **连续碰撞检测（CCD）**：轨迹段上最小距离

### 进入优化

| 形式 | 用法 |
|------|------|
| 硬约束 $d(q_k) \ge d_{\min}$ | NMPC / 约束 TrajOpt |
| 软惩罚 $\sum \max(0, d_{\min}-d)^2$ | 罚函数 / GNC |
| 线性化约束 | 凸 MPC 子问题 |

## 机器人中的用法

- **cuRobo**：GPU 批量 SDF + L-BFGS TrajOpt
- **Safety Filter / CBF**：距离或其 Lie 导数构造 barrier（见 [Safety Filter](./safety-filter.md)）
- **课程 5.5 实践**：复杂障碍环境安全导航 = 距离约束 + 路径优化

## 常见误区

- **距离可微 everywhere**：非光滑处需 smoothing 或 subgradient。
- **只查离散点**：稀疏采样可能漏检，需 CCD 或稠密约束。
- **SDF 分辨率**：过粗导致「假安全」。

## 与其他页面的关系

- [cuRobo](../entities/curobo.md)
- [Smooth Navigation Path Generation](../methods/smooth-navigation-path-generation.md)
- [Convex Relaxation in Robotics](../methods/convex-relaxation-robotics.md)
- [Numerical Optimization Curriculum](../entities/numerical-optimization-curriculum.md)

## 推荐继续阅读

- [Safety Filter](./safety-filter.md)
- [CLF-CBF in WBC](../queries/clf-cbf-in-wbc.md)

## 参考来源

- [sources/courses/numerical_optimization_foundations_robotics.md](../../sources/courses/numerical_optimization_foundations_robotics.md) — 第 3 章 3.7、第 5 章 5.5
