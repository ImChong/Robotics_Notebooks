---
type: entity
project_id: claude-fable-5-2
project: "https://www.anthropic.com/claude-fable-and-mythos-5-2"
tags: [embodied-ai, robotics, agents, foundation-model, benchmark]
status: draft
updated: 2026-11-11
summary: "Anthropic Claude Fable 5.2 官方展示 RoboDojo Sim 最高努力档 64.9% 和多项真机操作演示；不等于公开机器人控制策略或独立复现。"
original:
  - ../../sources/blogs/anthropic-claude-fable-5-2-2026-11-09.md
sources:
  - ../../sources/blogs/anthropic-claude-fable-5-2-2026-11-09.md
related:
  - ./embodied-first-exam.md
  - ./paper-world-in-context-learning-roboscript.md
---

# Claude Fable 5.2（克劳德 Fable 5.2）

## 一句话定义

Claude Fable 5.2 是 Anthropic 发布的模型，官方以机器人仿真测评和真机操作视频展示其具身任务能力；现有公开信息不能将其等同于已开放权重的机器人动作策略。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| LLM | Large Language Model | 机器人任务决策中使用的大模型类别 |
| Sim | Simulation | RoboDojo 仿真而非真实机器人试验 |
| SR | Success Rate | 对特定评测协议统计的任务成功率 |

## 为什么重要

- 通用模型与机器人环境连接的能力正从静态推理扩展到交互任务；需要区分 benchmark 数值与演示视频。
- 可与[具身代码代理评测](./embodied-first-exam.md)比较：能力表述、观察权限、任务池和努力档不同。

## 流程总览（按官方评测叙述抽象）

```mermaid
flowchart LR
  T["RoboDojo Sim 机器人任务"] --> M["Claude Fable 5.2（Max effort）"]
  M --> A["模型驱动的任务尝试"]
  A --> S["仿真执行及反馈"]
  S --> M
  S --> R["按 RoboDojo 协议统计结果"]
```

这是**官方介绍层级**的任务闭环图，不代表已公开内部 agent 架构、tool schema 或动作接口。

## 核心原理与实验

官方在 RoboDojo Sim **最高努力档**报告 **64.9%**。博客展示真机魔方、旋瓶盖、绘画等任务，但未将这些演示等价定义为 RoboDojo 的同一任务分母或可靠性统计。

## 结论

**可以把它作为通用模型具身能力的公开演示和 benchmark 线索，但不宜直接当成可部署的机器人控制器。**

1. 对外引用 64.9% 时必须标注 RoboDojo Sim 和 Max effort。
2. 真机演示说明有成功案例，不自动给出重复试验成功率。
3. 要开展选型，应另核对机器人 API、观测、延迟、动作安全与复现许可。

## 工程实践

- 从官方博客追踪 benchmark 协议；未见可据此直接安装的官方机器人控制部署包。
- 若比较不同 agent，必须控制 effort、工具权限、成功判据与试验次数。

## 局限与风险

公开介绍没有构成可独立复现的完整具身评估研究；完整指标表、附录、控制接口和运行入口待逐项查证。本节点暂为草稿。

## 源码运行时序图

**不适用：** 官方发布博客未提供可核验的 Fable 5.2 机器人任务运行代码入口，不能推测其真实调用栈。

## 关联页面

- [Embodied First Exam](./embodied-first-exam.md)：标准化 coding-agent 具身测评
- [WICL / RoboScript](./paper-world-in-context-learning-roboscript.md)：明确的仿真反馈与 agent 交互机制

## 参考来源

- [Anthropic 官方博客归档](../../sources/blogs/anthropic-claude-fable-5-2-2026-11-09.md)

## 推荐继续阅读

- [Claude Fable and Mythos 5.2 — Anthropic](https://www.anthropic.com/claude-fable-and-mythos-5-2)
