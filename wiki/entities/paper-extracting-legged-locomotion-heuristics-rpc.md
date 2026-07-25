---
type: entity
tags: [paper, mpc, locomotion, quadruped, mit, heuristics, control]
status: complete
updated: 2026-07-25
related:
  - ./paper-bledt-rpc-thesis.md
  - ./paper-wbic-mpc-mini-cheetah.md
  - ./mit-mini-cheetah.md
  - ../methods/model-predictive-control.md
  - ../concepts/gait-generation.md
sources:
  - ../../sources/papers/extracting_legged_locomotion_heuristics_rpc_icra_2020.md
  - ../../sources/papers/bledt_rpc_thesis_mit_2020.md
summary: "Bledt & Kim ICRA 2020：离线探索代价空间提取 RPC 正则启发式，在线适应；Mini Cheetah 上不改控制结构即可增强能力。"
---

# Extracting Legged Locomotion Heuristics with RPC

## 一句话定义

**Bledt & Kim（MIT，ICRA 2020，[DOI:10.1109/ICRA40945.2020.9197488](https://doi.org/10.1109/ICRA40945.2020.9197488)）** 给出从仿真中**提取腿足 locomotion 正则启发式**的框架：离线充分探索代价空间 → 拟合命令/最优动作/状态的简单模型 → 在线参数适应；在 **Mini Cheetah** 上**不改控制器结构与增益**即可增强能力。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RPC | Regularized Predictive Control | 正则化预测控制框架 |
| MPC | Model Predictive Control | 优化控制母体 |
| ICRA | International Conference on Robotics and Automation | 发表会议 |
| CoM | Center of Mass | 常出现在启发式状态特征中 |
| GRF | Ground Reaction Force | 优化动作常见输出 |

## 为什么重要

- 把「调 MPC 代价」从玄学变成可重复的数据驱动提取。
- 强调**保留物理直觉**的简单模型，而不是黑箱替代控制器。
- 是 [Bledt 博士论文](./paper-bledt-rpc-thesis.md) 的会议浓缩版。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 麻省理工（MIT） |
| **平台** | Mini Cheetah |
| **开源** | **方法工具链未单独开源** |

## 核心原理

```mermaid
flowchart LR
  sim["离线仿真探索代价空间"] --> fit["拟合简单启发式模型"]
  fit --> online["在线参数适应"]
  online --> rpc["RPC / 预测控制"]
  rpc --> bot["Mini Cheetah"]
```

1. 约束/隔离特定状态与动作，暴露有意义的启发式候选。
2. 用简单关系连接期望命令、最优控制动作与机器人状态。
3. 在线适应吸收模型简化与参数不确定。

## 源码运行时序图

**不适用**（无官方独立可运行提取管线仓库）。

## 工程实践

| 项 | 建议 |
|----|------|
| 流程 | 先固定控制器结构，只替换/增加正则项 |
| 数据 | 离线覆盖速度、扰动、地形代理任务 |
| 验收 | 真机对比：相同增益下通过性/速度/扰动恢复 |

## 评测

| 维度 | 要点 |
|------|------|
| 硬件 | Mini Cheetah 验证能力提升 |
| 约束 | 不修改控制器结构或增益 |
| 目标 | 近似复杂动力学并容忍模型误差 |

## 结论

**总判：** 这篇短文是 RPC 落地说明书——教你如何**挖启发式**而不是只报一个调好的代价。

- 真影响：可迁移的启发式提取流程。
- 次要代价：离线计算；简单模型表达力上限。
- 部署：与现有 MPC/WBIC 叠用，见导航系统文。

## 局限与风险

- 启发式可能过拟合离线任务分布。
- IEEE 全文访问受限时以摘要+博士论文互补。

## 关联页面

- [Bledt RPC 论文](./paper-bledt-rpc-thesis.md)
- [WBIC+MPC](./paper-wbic-mpc-mini-cheetah.md)
- [MIT Mini Cheetah](./mit-mini-cheetah.md)
- [Gait generation](../concepts/gait-generation.md)

## 参考来源

- [论文归档](../../sources/papers/extracting_legged_locomotion_heuristics_rpc_icra_2020.md)
- [Bledt 论文归档](../../sources/papers/bledt_rpc_thesis_mit_2020.md)

## 推荐继续阅读

- IEEE：<https://ieeexplore.ieee.org/document/9197488>
