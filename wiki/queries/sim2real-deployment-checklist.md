---
type: query
tags: [sim2real, deployment, checklist, redirect]
status: complete
summary: "【已合并】真机部署检查清单已并入 Sim2Real 工程 Checklist 的「快速部署检查」节；本页保留检索与旧链接兼容。"
updated: 2026-06-30
sources:
  - ../../sources/papers/sim2real.md
related:
  - ./sim2real-checklist.md
  - ../concepts/sim2real.md
---

# Sim2Real 真机部署清单（已合并）

> **Query 产物**：原问题「真机部署 RL 策略前后要检查什么？」已合并至 [Sim2Real 工程 Checklist](./sim2real-checklist.md#快速部署检查)；本页保留旧链接与检索兼容。
>
> **本页内容已合并至** [Sim2Real 工程 Checklist](./sim2real-checklist.md#快速部署检查)。
>
> 原触发问题：「真机部署 RL 策略前后要检查什么？」——训练端 / 部署端 / 调试端三阶段检查项现位于母页 **「快速部署检查」** 节；完整五阶段流水线（建模 → DR → 验证 → SysID → 上机 → Gap 闭环）见同一母页后续章节。

## 一句话定义

真机部署 RL 策略的检查项已并入 [Sim2Real 工程 Checklist](./sim2real-checklist.md#快速部署检查)，避免与全流程清单重复维护。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| CAN | Controller Area Network | 电机/关节常用的现场总线通信协议 |

## 参考来源

- [Sim2Real 工程 Checklist](./sim2real-checklist.md) — 合并后的 canonical 页
- [Sim2Real 概念页](../concepts/sim2real.md)

## 关联页面

- [Sim2Real 工程 Checklist](./sim2real-checklist.md#快速部署检查) — **请从此进入**
- [Sim2Real Gap 缩减指南](./sim2real-gap-reduction.md)
- [RL 策略真机调试 Playbook](./robot-policy-debug-playbook.md)
- [Sim2Real](../concepts/sim2real.md)
