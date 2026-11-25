---
type: tech_map_node
status: complete
topic: [model-based-control]
summary: "机器人运动学技术栈模块：覆盖正运动学（FK）、逆运动学（IK）、雅可比矩阵（Jacobian）与奇异位形/可操作度分析，是控制、规划与动作重定向的几何语言。"
sources:
  - ../../../sources/papers/modern_robotics_textbook.md
  - ../../../sources/blogs/wechat_goodman_forward_kinematics_poe.md
  - ../../../sources/repos/robotics-toolbox-python.md
updated: 2026-11-24
---

# 运动学（技术栈模块）

运动学只回答「几何上怎么动」，不管力与质量：**正运动学** 由关节角算末端位姿，**逆运动学** 反过来由目标位姿求关节角，**雅可比** 把关节速度与末端速度联系起来，也是奇异位形、可操作度与力映射的分析工具。它是控制与规划共同的语言底座。

## 在全栈中的位置

```mermaid
flowchart LR
    SE3["刚体运动 SE(3) & 线性代数"] --> FK["正运动学 FK：q → T_ee ∈ SE(3)"]
    FK --> JAC["雅可比矩阵 J(q)：qd ↔ V_ee & τ = J^T F"]
    JAC --> IK["逆运动学 IK：目标位姿 → 关节构型 q"]
    JAC --> DYN["进入动力学 M(q) 与 WBC 任务空间投影"]
    IK --> RET["服务 Motion Retargeting 与末端规划"]
```

## 核心内容与站内入口

- **形式化核心**：[正运动学](../../../wiki/formalizations/forward-kinematics.md) · [逆运动学](../../../wiki/formalizations/inverse-kinematics.md) · [机器人雅可比](../../../wiki/formalizations/robot-jacobian.md)
- **工程可行性对比**：[运动学可行 vs 动力学可行](../../../wiki/concepts/kinematic-vs-dynamic-feasibility.md)

## 上下游模块

- **上游依赖**：[线性代数](../math/linear-algebra.md) · [刚体运动](rigid-body-motion.md) · [全栈技术域总览](../../overview.md)
- **下游应用**：[动力学](dynamics.md) · [WBC 模块](../control/whole-body-control.md) · [Motion Retargeting](../il/motion-retarget.md)

## 参考来源与收录边界

- [Modern Robotics 教材摘录](../../../sources/papers/modern_robotics_textbook.md)
- [PoE 正运动学讲义归档](../../../sources/blogs/wechat_goodman_forward_kinematics_poe.md)
- [Robotics Toolbox for Python 归档](../../../sources/repos/robotics-toolbox-python.md)
- 指数积公式与数值 IK 算法细节见对应 `wiki/formalizations/` 页面。
