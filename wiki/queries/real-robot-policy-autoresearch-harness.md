---
type: query
tags: [autoresearch, real-world-rl, coding-agents, policy-improvement, auto-reset, manipulation, data-flywheel]
status: complete
summary: 把真机策略开发改造成 coding agent 可编排的自改进闭环时，环境侧（自动 reset/verify）、策略改进范式选型（启发式/BC/RL）、rollout 预算与机队 scaling 各该怎么落地——以 ENPIRE 的 EN–PI–R–E 框架为骨架的实操选型指南。
sources:
  - ../../sources/papers/enpire_nvidia_gear_2026.md
  - ../../sources/sites/nvidia-research-enpire.md
related:
  - ../methods/enpire.md
  - ../methods/behavior-cloning.md
  - ../methods/reinforcement-learning.md
  - ../concepts/simulation-evaluation-infrastructure.md
  - ../concepts/data-flywheel.md
  - ../concepts/embodied-scaling-laws.md
  - ../tasks/manipulation.md
---

# 真机策略 autoresearch 闭环搭建指南

> **Query 产物**：本页由以下问题触发：「如何把真机策略开发改造成 coding agent 可编排的自改进（autoresearch）闭环？需要哪些前提、怎么选策略改进范式、怎么读 scaling？」
> 综合来源：[ENPIRE](../methods/enpire.md)、[Behavior Cloning](../methods/behavior-cloning.md)、[Reinforcement Learning](../methods/reinforcement-learning.md)、[仿真评测基础设施](../concepts/simulation-evaluation-infrastructure.md)、[Data Flywheel](../concepts/data-flywheel.md)

## TL;DR

数字世界里 agent 已能自动搜算法；真机侧的瓶颈不是某个 SOTA 网络结构，而是缺**可重复、可验证、可复位**的物理闭环。要让 coding agent 接手策略迭代，第一刀砍在**环境工程**而不是模型：没有自动 reset + 自动 verification，agent 就只能低频人工试验。[ENPIRE](../methods/enpire.md) 的 **EN–PI–R–E** 四模块给出一个可照搬的骨架。

```
任务能否交给 agent 自改进？
│
├─ 能自动复位到随机初始态并确认就绪吗？
│   └─ 否 → 先做 EN（Environment）：reset 行为 + 复位成功判据
│
├─ 能不靠人自动判成功/失败吗？
│   └─ 否 → 先做自动 verification：检测/分割/几何 → 二值 reward
│
└─ 两者都有 → 进入 PI/R/E：选范式 → 并行 rollout → 跨假设演化
```

## 第一步：环境（EN）是一等公民

agent 能跑多密的 trial，上限由环境而非模型决定。两件事必须先就位：

| 前提 | 要求 | 落地判据 |
|------|------|---------|
| **自动 reset** | 随机初始态采样 + 复位行为 + 复位成功确认 | 无人值守能连续跑 N 次 trial 不需人介入 |
| **自动 verification** | 多视角检测/分割/几何融合成**二值 reward** | reward 判定与人工标注一致率足够高 |
| **`env.py` 接口** | `reset / get_observation / get_reward / step` 工具化 | agent 可纯代码驱动整条交互 |

verification 是最容易被低估的部分：换传感器布局或安全约束，判分器往往需要**重新标定**——把它当成与策略同等重要的工程资产维护。详见[仿真评测基础设施](../concepts/simulation-evaluation-infrastructure.md)。

## 第二步：策略改进范式（PI）怎么选

同一 harness 下可比较并组合多条路线，选型由**真机成功率**驱动，而非纯仿真曲线：

| 范式 | 适用场景 | 起步成本 | 参考 |
|------|---------|---------|------|
| **启发式 / code-as-policy** | 任务有清晰几何/接触结构（如 Push-T） | 最低，无需数据 | 先写无神经网络 baseline 验证环境 |
| **Behavior Cloning** | 有演示数据，想要稳定起点 | 中，依赖演示质量 | [Behavior Cloning](../methods/behavior-cloning.md) |
| **Offline / Online RL** | 接触丰富、需在线纠偏（插针、扎带） | 高，需 reward + 安全约束 | [Reinforcement Learning](../methods/reinforcement-learning.md) |

实践顺序通常是：**先用启发式跑通环境闭环 → BC 拿到可用起点 → RL/online mix 抬高成功率**。ENPIRE 在 Push-T、插针、GPU 插拔、扎/剪扎带等任务上报告**约 99% pass@8**，但这些数字绑定在**特定判分器、复位策略与 trial 预算**上，不能脱离 harness 泛化。

## 第三步：rollout（R）与机队 scaling（E）

- **R — rollout**：在预算内并行多机 trial，**强制留存 state/action/video/trace**；日志是 E 阶段演化的唯一依据。
- **E — evolution**：读日志做失败分析，跨 agent 分支比较，**采纳提升成功率的 recipe、剪枝失败假设**。

机队变大不是免费的，要把**物理 scaling** 与 **token scaling** 分开优化：

| 指标 | 含义 | 读法 |
|------|------|------|
| **MRU**（Mean Robot Utilization）| 机器人时间利用率 | agent 读日志/写代码/等 LLM 时机器人空转 → MRU 下降 |
| **MTU**（Mean Token Utilization）| LLM token 吞吐利用率 | 机队变大 → 协调开销与 token 成本上升 |

ENPIRE 用 **AutoEnvBench** 比较不同 coding agent（Codex / Claude Code / Kimi Code 等）在同任务上的**墙钟研究进展曲线**——评估的是「agent 推进研究的速度」，而非单次策略分数。更大并行能更快抬高成功率，但 **token-to-success** 与协调开销同步上升，需在两条 scaling 轴上权衡。这与[具身规模法则](../concepts/embodied-scaling-laws.md)的数据/模型缩放律互补。

## 常见误区

- **「有 coding agent 就能跳过环境工程」**：恰恰相反，reset/verify 接口才是核心贡献；没有自动判分与复位，agent 只能低频人工试验。
- **「成功率数字可脱离 harness 泛化」**：高成功率建立在特定 verification、复位策略与预算上；换传感器布局需重新标定。
- **「更多机器人/agent 一定更快」**：并行能加速，但 MRU/MTU 揭示 GPU 利用率与 token 消耗的同步上升，物理与 token scaling 要分开读。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| EN–PI–R–E | Environment–Policy Improvement–Rollout–Evolution | ENPIRE 四模块真机 autoresearch 束具 |
| BC | Behavior Cloning | 从演示或日志监督学习策略 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| MRU | Mean Robot Utilization | 机器人时间利用率（物理 scaling 维度）|
| MTU | Mean Token Utilization | 协作时 LLM token 吞吐利用率（token scaling 维度）|
| pass@8 | pass at 8 | 8 次独立 rollout 中至少一次成功的通过率 |

## 参考来源

- [sources/papers/enpire_nvidia_gear_2026.md](../../sources/papers/enpire_nvidia_gear_2026.md)
- [sources/sites/nvidia-research-enpire.md](../../sources/sites/nvidia-research-enpire.md)
- NVIDIA GEAR, *ENPIRE: Agentic Robot Policy Self-Improvement in the Real World*, 项目页, 2026. <https://research.nvidia.com/labs/gear/enpire/>

## 关联页面

- [ENPIRE](../methods/enpire.md) — 本页骨架来源：EN–PI–R–E 真机自改进束具
- [Behavior Cloning](../methods/behavior-cloning.md) — PI 范式之一，常作 RL 之前的稳定起点
- [Reinforcement Learning](../methods/reinforcement-learning.md) — online/offline RL 在 harness 内与启发式、BC 公平对比
- [仿真评测基础设施](../concepts/simulation-evaluation-infrastructure.md) — 自动 verification 与受控 reset 的工程背景
- [Data Flywheel](../concepts/data-flywheel.md) — rollout 日志回流驱动策略迭代的飞轮视角
- [具身规模法则](../concepts/embodied-scaling-laws.md) — MRU/MTU 机队 scaling 与数据/模型缩放律互补
- [Manipulation](../tasks/manipulation.md) — Push-T / 插针 / 扎带等灵巧操作任务语境
