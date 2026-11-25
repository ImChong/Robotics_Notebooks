---
type: tech_map_node
status: complete
topic: [model-based-control, locomotion]
summary: "MPC（模型预测控制）技术栈模块：使用简化或非线性动力学模型滚动求解带约束最优控制问题，生成腿足落脚点、质心轨迹与接触力剖面。"
sources:
  - ../../../sources/papers/mpc.md
  - ../../../sources/papers/wbic_mpc_mini_cheetah_arxiv_1909_06586.md
updated: 2026-11-24
---

# MPC（模型预测控制）

**MPC** 在每个控制周期里用模型向前滚动预测一段时间，在满足约束的前提下解一个优化问题，只执行第一步再重新求解。它是「有模型、要满足硬约束、还要实时」这类运控问题的默认答案——腿足机器人的落脚点规划、质心轨迹与接触力分配大多走这条路。

## 在全栈中的位置

```mermaid
flowchart LR
    MOD["动力学简化模型（LIP / SRBD / Centroidal）"] --> OCP["有限时域 OCP（QP / NMPC / 采样）"]
    STATE["状态估计（位姿 / 速度 / 接触状态）"] --> OCP
    OCP -->|"仅执行首步参考（轨迹与接触力）"| WBC["低层 WBC / TSID 1kHz 跟踪"]
    WBC -->|"下一拍重新观测滚动求解"| STATE
```

## 核心内容与站内入口

- **方法原理与公式**：[模型预测控制](../../../wiki/methods/model-predictive-control.md) · [非线性 MPC](../../../wiki/methods/nonlinear-model-predictive-control.md)
- **工程调参与求解器选型**：[MPC 调参指南](../../../wiki/queries/mpc-tuning-guide.md) · [求解器选型](../../../wiki/queries/mpc-solver-selection.md)
- **路线对照与论文导航**：[MPC vs RL](../../../wiki/comparisons/mpc-vs-rl.md) · [MPC 论文导航](../../../references/papers/mpc.md)

## 上下游模块

- **上游依赖**：[动力学](../robotics/dynamics.md) · [线性代数](../math/linear-algebra.md) · [全栈技术域总览](../../overview.md)
- **下游对接**：[WBC 模块](whole-body-control.md) · [人形 Locomotion](humanoid-locomotion.md)

## 参考来源与收录边界

- [sources/papers/mpc.md](../../../sources/papers/mpc.md) — MPC 经典文献与足式应用归档
- [sources/papers/wbic_mpc_mini_cheetah_arxiv_1909_06586.md](../../../sources/papers/wbic_mpc_mini_cheetah_arxiv_1909_06586.md) — Mini Cheetah MPC+WBIC 原文归档
- 算法细节与对比实验以对应 `wiki/` 页面为准。
