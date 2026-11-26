---
type: method
tags: [robotics, evaluation, benchmark]
status: complete
updated: 2026-11-26
summary: "固定任务与预算做可复现回归，或用配对盲测比较开放式真机表现；两类评测分别报告成功、进度与失效。"
sources:
  - ../../sources/papers/roboarena_arxiv_2506_18123.md
original:
  - ../../sources/papers/roboarena_arxiv_2506_18123.md
related:
  - ../concepts/ai-agent-evaluation.md
  - ../overview/hub-embodied-eval-benchmark.md
  - ../queries/embodied-eval-benchmark-selection-loop.md
  - ./roboarena.md
---

# Embodied Agent Benchmarking · 具身智能体评测方法

## 一句话定义

将任务、环境、交互接口和预算写成可复核的协议，再用轨迹证据衡量智能体是否完成物理目标，而不只看它是否给出了正确的计划。

## 英文缩写速查

| 缩写 | 英文全称 | 说明 |
|---|---|---|
| API | Application Programming Interface | 评测环境暴露的观测、动作及复位接口 |
| BT | Bradley–Terry | 用配对偏好推断相对能力的统计模型，本页不指行为树 |
| CI | Confidence Interval | 根据重复样本估计指标的不确定性 |

## 为什么重要

同一模型换了摄像头、动作分频、超时限制或人工接管条件，即可能得到不同结果。把这些差异与模型一起报告，才能区分模型能力、环境变化和评测流程变化。固定集合适合稳定回归；只用一个固定集合又容易遗漏真实环境长尾。

## 核心机制

| 设计 | 适用场景 | 必须保留的证据 |
|---|---|---|
| 固定任务与复位 | 版本回归、单因素消融 | 任务与场景版本、同一初始条件/种子、共同的终止判据 |
| 同场景匿名 A/B | 比较可在相同本体运行的通用策略 | 成对复位、超时、匿名分派、每路进度、偏好与轨迹 |
| 仿真或日志回放 | 快速筛选和定位接口故障 | 环境版本、扰动范围或回放限制；不能据此宣称真机闭环可靠 |

```mermaid
flowchart LR
  goal["定义目标、成功条件与风险边界"] --> freeze["固定接口、预算和复位规则"]
  freeze --> mode{"回答哪类问题"}
  mode -->|"版本回归"| fixed["固定任务集合与种子"]
  mode -->|"开放场景比较"| paired["匿名 A/B、对子内复位"]
  fixed --> traces["记录轨迹、超时与接管"]
  paired --> traces
  traces --> metrics["分层报告、比较与复核失败"]
```

[RoboArena](./roboarena.md) 的实例是 DROID 真机操作：中央服务器分配匿名策略对，同一评测员让两路策略顺序运行并尽量匹配初始条件，随后提交进度、主观偏好、解释及轨迹。论文的全局排序依赖跨任务聚合；配对偏好和绝对任务成功率是不同输出。

## 主要技术路线

- **固定任务复测：** 用同一组复位和种子检验版本回归。
- **开放场景配对：** 用匿名策略对检验同场景相对能力，保留偏好及独立成功证据。
- **分阶段筛选：** 从日志或仿真定位接口问题，再对关键结果做真机复核。

## 工程实践

1. 在运行前定好任务版本、初始摆位、公平的观测条件、动作表示与控制频率；同步记录模型、环境和策略端点版本。
2. 定义完成、部分完成、超时、危险终止、传感故障与人为接管的归类，并明确分母。接管后的成功不等同于自主成功。
3. 先按任务、本体及场景报告样本量与成功率，再给聚合数；同一场景的多次运行有关联时，保留成对结果，用场景或试验对为单位估计不确定性。
4. 保存首帧、指令、时间戳、动作、终止原因和复核用记录；明确哪个环节有权停机。将开放盲测与固定回归并行使用，不让单一排名替代诊断。

## 局限与风险

RoboArena 的实证覆盖单一 DROID 类配置及其策略池；不能直接推广为异构本体、动态任务或导航的同尺度排行。对子内复位也不是完全相同的物理状态；开放评测还要检查评测员、任务分布和偏好标注变化。此页为方法导航，不是独立评测实现，未重新运行原论文真机实验。

## 关联页面

- [具身智能评测与基准总览](../overview/hub-embodied-eval-benchmark.md)
- [具身评测选择闭环](../queries/embodied-eval-benchmark-selection-loop.md)
- [AI Agent 评测](../concepts/ai-agent-evaluation.md)
- [RoboArena：真机配对评测实例](./roboarena.md)

## 参考来源

- [RoboArena 论文全文核对归档](../../sources/papers/roboarena_arxiv_2506_18123.md)

## 推荐继续阅读

- [RoboArena v2 正文、实验与附录](https://arxiv.org/html/2506.18123v2)
