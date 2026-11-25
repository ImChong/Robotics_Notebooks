---
type: entity
project_id: robo-swe-2611-16116
tags: [paper, robot-software, benchmark, coding-agent]
status: complete
updated: 2026-11-21
arxiv: "2611.16116"
project: https://evalium.ai/robo-swe
original:
  - ../../sources/papers/robo-swe-2611-16116.md
sources:
  - ../../sources/papers/robo-swe-2611-16116.md
  - ../../sources/sites/robo-swe-2611-16116.md
related:
  - ../overview/embodied-frontier-29-papers-2026-11-16.md
  - ./paper-maniladder-manipulation-benchmark.md
summary: "从 ROS 2 和仿真等真实机器人仓库的 Issue、修复前代码及测试构造可执行的代码智能体修复任务。"
---

# Robo-SWE: An Executable Issue-Resolution Dataset Mined from Robot Software Repositories｜Robo-SWE：可执行机器人软件问题修复数据集

## 一句话定义

**Robo-SWE：可执行机器人软件问题修复数据集**：从 ROS 2 和仿真等真实机器人仓库的 Issue、修复前代码及测试构造可执行的代码智能体修复任务。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| SWE | Software Engineering | 软件工程 |
| F2P | Fail-to-Pass | 修复前失败的测试 |
| P2P | Pass-to-Pass | 已有测试不回归 |

## 为什么重要

从 ROS 2 和仿真等真实机器人仓库的 Issue、修复前代码及测试构造可执行的代码智能体修复任务。 比起单一 benchmark 排名，研究与工程落地需要分清楚是否具有动作控制闭环、是否能复现以及硬件/数据前提。

## 核心信息

| 条目 | 原始证据 |
|---|---|
| 论文 | [arXiv:2611.16116v1](https://arxiv.org/abs/2611.16116) |
| 官方项目 | [主页](https://evalium.ai/robo-swe) |
| 开源入口 | 未核实该研究完整可运行官方代码 |

## 核心原理

在合并 PR 中筛出可验证 issue 与新增测试，重建修复前容器；用 base→新增测试应失败→加入开发者修复应通过的 F2P/P2P 测试准入门槛筛选，最终向 Agent 隐藏真实修复 patch。

### 流程总览

```mermaid
flowchart LR
  A0["机器人仓库 merged PR"]
  A1["Issue 与开发者测试提取"]
  A2["构造修复前容器"]
  A3["F2P/P2P 三状态检验"]
  A4["Agent 修复与执行评分"]
  A0 --> A1
  A1 --> A2
  A2 --> A3
  A3 --> A4
```

流程图来自一手论文方法链；与实际源码包之间的映射如下。

## 源码运行时序图

**不适用：** 当前官方材料未提供经核实的可运行源码入口，暂不编造训练或推理代码调用关系。

## 工程实践

1. 依据原文方法核对输入传感器、控制/评估接口、样本分布及实验硬件。
2. 没有核实的官方实现时不把其他仓库替代成作者实际源码。
3. 将论文成绩、真机误差、系统延迟与安全要求分开验收，复现前明确失败标准。

## 实验与评测

原文称从 61,655 个 merged PR 筛候选、最终 262 项通过执行准入；项目当前展示 236 个任务及 28 个仓库，需区分论文静态构造与公开榜单口径。

## 结论

**从 ROS 2 和仿真等真实机器人仓库的 Issue、修复前代码及测试构造可执行的代码智能体修复任务。**

1. 保留相同模型与数据预算下的可比口径，不跨基准简单求平均。
2. 优先复核最可能出现数据泄漏、接触失败或部署延迟的环节。
3. 对论文未检验的部署条件单列验证计划。

## 与其他工作对比

- **对比通用软件工程与机器人编码基准（§VI Related Work）：** 不同于面向通用 Python/多语言软件的 `SWE-bench` / `Multi-SWE-bench`，也不同于仅汇集静态 ROS 缺陷的 `ROBUST`（221 bugs）、只测单一 ROS 2 导航栈仿真行为预言的 `RoboBugBench`（33 repairs），以及侧重从零写控制脚本的 `Code as Policies` / `EmbodiedSWE` / `ROSDevEval`，Robo-SWE 覆盖 28 个真实机器人仓库（ROS 2、仿真、算法、学习四域）的 **236 个容器化可执行 F2P/P2P 真实 Issue 修复任务**。
- **对比不同模型–Agent 组合在 Robo-SWE 上的表现（Table II，Nov 2026）：** 7 组主流模型–harness 配置的总解决率分布在 **43.9%–56.5%**（`Opus 5.5` 56.5%、`Sol max` 54.1%、`Sol medium` 52.5%、`Astra` 51.4%、`GLM-5.3` 46.2%、`Sonnet 5.5` 45.8%、`Kimi K3.1 Pro` 43.9%；单次 API 等价成本均值 \$0.10–\$1.24）。论文明确指出各主模型 95% 自举区间高度重叠，且评测对象是「模型 + harness」联合体，不能抽离 harness 作纯模型排名。

## 局限与风险

容器测试通过只说明软件 issue 修复不一定保证真机行为正确；不同模型 harness 与软件栈和任务数量不可直接横比。

## 关联页面

- [具身评测基准选型闭环](../overview/hub-embodied-eval-benchmark.md) — 将可执行软件修复评测与机器人策略成功率、sim↔real 校准分层；F2P/P2P 通过不能替代真机行为验证。
- [本期 29 篇具身论文选型地图](../overview/embodied-frontier-29-papers-2026-11-16.md)
- [ManiLadder：任务难度分层评测](./paper-maniladder-manipulation-benchmark.md)

## 参考来源

- [原论文核对归档](../../sources/papers/robo-swe-2611-16116.md)
- [项目页开放状态](../../sources/sites/robo-swe-2611-16116.md)

## 推荐继续阅读

- [arXiv HTML](https://arxiv.org/html/2611.16116v1)
- [论文 PDF](https://arxiv.org/pdf/2611.16116)
