---
type: entity
tags: [paper, quadruped, spot, curriculum]
status: complete
updated: 2026-09-14
arxiv: "2609.09492"
related:
  - ../tasks/locomotion.md
  - ./paper-ebert-nonlinear-normal-modes.md
  - ../queries/sim2real-closed-loop-engineering.md
sources:
  - ../../sources/papers/actuator_dynamics_curricula_arxiv_2609_09492.md
summary: "Actuator Dynamics Curricula（arXiv:2609.09492）：high joint stiffness early then anneal to identified stiffness; Spot stand-to-handstand sim2real；截至入库日未见官方代码。"
---

# Actuator Dynamics Curricula（arXiv:2609.09492）

**Actuator Dynamics Curricula**（*Actuator Dynamics Curricula for Narrow-Viability Tasks in Legged Robot Learning*，[arXiv:2609.09492](https://arxiv.org/abs/2609.09492)）由 **萨克森工业大学联盟；格罗宁根大学（University of Groningen）；特文特大学（University of Twente）** 提出（公众号周更 ingest 见 [策展索引](../../sources/blogs/wechat_shenlan_weekly_humanoid_quadruped_2026-09-14.md)）。

## 一句话定义

面向窄可行域腿式任务的执行器动力学课程学习 — high joint stiffness early then anneal to identified stiffness。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| PD | Proportional-Derivative | 比例微分关节控制 |
| sim2real | Simulation-to-Real | 仿真到真机 |

## 为什么重要

窄可行域任务对执行器带宽与刚度极敏感；直接真机刚度训练易崩溃。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 萨克森工业大学联盟；格罗宁根大学（University of Groningen）；特文特大学（University of Twente） |
| **开源** | **未见/待发布**（步骤 2.5 核查：截至 2026-09-14 无可运行官方仓库） |

## 核心原理

课程从高关节刚度开始扩大可探索集，再 anneal 到系统辨识刚度；Spot 站立→手倒立迁移。

### 流程总览

```mermaid
flowchart LR
  highK[高刚度仿真] --> policy[RL 策略]
  policy --> anneal[刚度退火]
  anneal --> idK[辨识刚度]
  idK --> spot[Spot 真机]
```

## 源码运行时序图

**不适用** — 截至 **2026-09-14** arXiv 与常见项目页 **未见** 官方可运行代码仓库。

## 工程实践

| 项 | 说明 |
|----|------|
| 开源状态 | 未见官方仓库；以 arXiv 为准 |
| 复现入口 | 论文方法与超参；代码发布后再补 `sources/repos/` |
| 部署注意 | 刚度 schedule 与 reward 塑形需同步；手倒立安全停机制必备。 |

## 实验与评测

Spot stand-to-handstand 成功率；刚度 ablation。

## 结论

执行器动力学课程让 Spot 在窄可行域特技上实现 sim2real。

1. 高刚度早期扩大可行域。
2. 退火到辨识参数匹配真机。
3. 手倒立是窄可行域试金石。
4. 执行器模型误差是主要风险。
5. 课程优于一次性域随机。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| 固定刚度训练 | 窄域易失败 |
| 纯域随机 | 样本效率低 |

## 局限与风险

仅 Spot 平台；其他执行器曲线需重调课程。

## 关联页面

- [locomotion](../tasks/locomotion.md)
- [./paper-ebert-nonlinear-normal-modes.md](./paper-ebert-nonlinear-normal-modes.md)
- [sim2real-closed-loop-engineering](../queries/sim2real-closed-loop-engineering.md)

## 参考来源

- [actuator_dynamics_curricula_arxiv_2609_09492.md](../../sources/papers/actuator_dynamics_curricula_arxiv_2609_09492.md)
- [公众号周更策展](../../sources/blogs/wechat_shenlan_weekly_humanoid_quadruped_2026-09-14.md)

## 推荐继续阅读

- [https://arxiv.org/abs/2609.09492](https://arxiv.org/abs/2609.09492)
