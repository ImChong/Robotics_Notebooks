---
type: entity
project_id: leash
arxiv: "2611.14115"
code: https://github.com/zhuqinfeng1999/LEASH
project: https://zhuqinfeng1999.github.io/LEASH/
tags: [paper, leash, robotics]
status: draft
updated: 2026-11-14
summary: "LEASH 以任务门、动作门、提交门检查机器人计划和不可逆动作，并用 RoboRisk 测量闭环伤害风险。"
related:
  - ../concepts/safety-filter.md
  - ../concepts/robot-safety-state-machine.md
original:
  - ../../sources/papers/leash_arxiv_2611_14115.md
sources:
  - ../../sources/papers/leash_arxiv_2611_14115.md
  - ../../sources/repos/leash.md
  - ../../sources/sites/leash.md
---

# LEASH: Stay Safe in Agentic Robotics｜分层安全门保障机器人智能体

## 一句话定义

LEASH 以任务门、动作门、提交门检查机器人计划和不可逆动作，并用 RoboRisk 测量闭环伤害风险。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| LEASH | Lease-Enforced Agentic Safety Harness | 本研究涉及的LEASH术语 |
| HR | Harm Rate | 本研究涉及的HR术语 |
| SSR | Safe Success Rate | 本研究涉及的SSR术语 |

## 为什么重要

RoboRisk 覆盖任务拒绝、计划修复、执行危险、注入、顺序记忆五类风险，92 项仿真与 13 项真机任务。

## 核心原理与方法

1. RoboRisk 覆盖任务拒绝、计划修复、执行危险、注入、顺序记忆五类风险，92 项仿真与 13 项真机任务。
2. 规划器和三个安全门共享统一安全规则，以实时摄像头上下文判断执行风险。
3. 同时看 Harm Rate 和 Safe Success Rate，安全拒绝危险任务也可视为成功。

## 流程总览

```mermaid
flowchart LR
  A0["任务请求"]
  A1["Task gate"]
  A2["Agent 规划"]
  A3["Action/Commit gate"]
  A4["机器人执行反馈"]
  A0 --> A1
  A1 --> A2
  A2 --> A3
  A3 --> A4
```

> 方法流程根据摘要/官网整理，尚待论文原文逐图核实。

## 实验与评测

文章报道 GPT-6 Astra 仿真中伤害率 11.23%→0%、安全成功率 78.99%→80.80%（每配置 276 次）；真机每配置 13 次，未观察到伤害不能证明普遍安全。

上述结果均为作者报告或用户附件转载，**不是本次复测**。

## 结论

**LEASH 以任务门、动作门、提交门检查机器人计划和不可逆动作，并用 RoboRisk 测量闭环伤害风险。**

1. RoboRisk 覆盖任务拒绝、计划修复、执行危险、注入、顺序记忆五类风险，92 项仿真与 13 项真机任务。
2. 规划器和三个安全门共享统一安全规则，以实时摄像头上下文判断执行风险。
3. 依赖审查模型与视觉感知；未知危害可能漏检，未独立复现实验。

## 与其他工作对比

比较本文与相关基线时应在相同任务、数据、样本量、传感器和部署条件下评估；不能把论文中的单个数字解释为真实部署的通用进步。待全文表格核对后补足逐项量化对照。

## 源码运行时序图

```mermaid
sequenceDiagram
  participant A0 as benchmark.native
  participant A1 as Planner
  participant A2 as Safety gates
  participant A3 as RoboRisk
  A0->>A1: 启动评测/训练
  A1->>A2: 处理环境与模型输入
  A2->>A3: 执行模型或输出指标
  A3-->>A0: 返回状态
````

**README 入口：** `python -m benchmark.native P3-01 --seed 0`。图仅表示已核对到的入口与概念性组件顺序，不代表已运行或逐行审阅。

## 工程实践

- 对齐版本、传感器、动作/观测空间和评测预算后再尝试复现。
- 正式应用应增加跨分布扰动与任务安全边界检查；原文完整复现步骤待核对。
- 本次通读状态：**未完成**；页面仍为 draft。

## 局限与风险

依赖审查模型与视觉感知；未知危害可能漏检，未独立复现实验。

未对全部论文正文/附录及运行细节完成审阅，任何进一步量化外推均不成立。

## 关联页面

- [safety-filter](../concepts/safety-filter.md)
- [robot-safety-state-machine](../concepts/robot-safety-state-machine.md)

## 参考来源

- [原论文及阅读进度](../../sources/papers/leash_arxiv_2611_14115.md)
- [官方源码入口](../../sources/repos/leash.md)
- [官方项目页](../../sources/sites/leash.md)
- [用户附件 13 篇原始资讯整理](../../sources/blogs/embodied-research-digest-2026-11-12.md)

## 推荐继续阅读

- [arXiv 原文](https://arxiv.org/abs/2611.14115)
- [官方源码](https://github.com/zhuqinfeng1999/LEASH)
