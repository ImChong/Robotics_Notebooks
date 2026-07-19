---
type: overview
tags: [topic, embodied-eval-benchmark, benchmark, evaluation, mllm, world-model, sim2real]
status: complete
updated: 2026-07-18
related:
  - ../queries/embodied-eval-benchmark-selection-loop.md
  - ../concepts/sim-vs-real-eval-gap.md
  - ../entities/robo-bench.md
  - ../entities/esi-bench.md
  - ../entities/ewmbench.md
  - ../entities/paper-gigaworld-1-policy-evaluation.md
  - ../concepts/simulation-evaluation-infrastructure.md
sources:
  - ../../sources/papers/robo_bench_arxiv_2510_17801.md
  - ../../sources/papers/ewmbench.md
  - ../../sources/papers/esi_bench_arxiv_2605_18746.md
summary: "具身评测基准选型闭环专题枢纽：把具身大脑/MLLM 认知评测 → 世界模型预测保真度评测 → 策略任务成功率评测 → sim↔real 评测 gap 校准 四层评测，从分散的评测基准实体页收拢为一条可导航的选型链，统一各层测什么、用什么代表性基准、指标的可复现性/真实代表性/过程 vs 结果/成本取舍入口。"
---

# 具身评测基准选型闭环（专题汇总）

> **专题定位**：本页是「MLLM 认知评测 → 世界模型预测保真度评测 → 策略任务成功率评测 → sim↔real 评测 gap 校准」四层具身评测基准的统一入口，把近周密集 ingest 的 RoboBench / ESI-Bench / EWMBench / GigaWorld-1 等评测基准从分散的实体页收拢为一条可导航的选型链。它是「[具身大模型分类学选型闭环](./topic-embodied-foundation-model.md)」的评测姊妹篇——前者回答「选哪一类具身大模型」，本专题回答「怎么评测/证明它」。

## 一句话定义

**具身评测基准选型闭环** 指按 **具身大脑/MLLM 认知评测 → 世界模型预测保真度评测 → 策略任务成功率评测 → sim↔real 评测 gap 校准** 逐层分工的评测谱系，各层共享「测什么 / 用什么代表性基准 / 指标怎么读」的方法学底座，但在可复现性、真实代表性、过程 vs 结果指标、成本上各有取舍，需按评测目的组合选型。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MLLM | Multimodal Large Language Model | 多模态大模型，作为具身大脑的认知评测对象 |
| EWM | Embodied World Model | 具身世界模型，视频生成保真度评测对象 |
| WM | World Model | 世界模型，时序预演与预测保真度 |
| VLA | Vision-Language-Action | 视觉-语言-动作策略，成功率评测对象 |
| WMES | World Model Evaluation Score | GigaWorld-1 世界模型评估综合分 |
| gap | Sim-to-Real Evaluation Gap | 仿真评测结论外推真机的偏差 |

## 为什么重要

- **补一条贯通的评测选型视角**：仓库已有各评测基准的实体页，但缺「从认知到真机逐层测什么、各基准边界与取舍」的统一决策入口。
- **暴露评测层间取舍矛盾**：仿真基准易复现 vs 真机代表性、任务成功率 vs 过程/中间指标、世界模型视频质量 ≠ 下游策略收益、MLLM 认知评分 ≠ 可执行动作能力——这些矛盾只有并置在一条链上才看得清（详见事实库对应矛盾检测规则）。
- **与选型闭环同向**：选出一类具身大模型后，唯有可信评测才能证明其收益，评测选型是模型选型的验收环节。

## 四层评测选型闭环

| 层次 | 测什么 | 代表基准 | 站内入口 |
|------|--------|----------|----------|
| ① 认知评测 | MLLM 作为 embodied brain 的感知/规划/推理能力 | RoboBench、ESI-Bench | [RoboBench](../entities/robo-bench.md)、[ESI-Bench](../entities/esi-bench.md) |
| ② 预测保真度评测 | 世界模型视频生成的时序/轨迹/语义保真度 | EWMBench、GigaWorld-1 WMBench | [EWMBench](../entities/ewmbench.md)、[GigaWorld-1 策略评估](../entities/paper-gigaworld-1-policy-evaluation.md) |
| ③ 策略成功率评测 | 下游 VLA/策略的任务成功率与泛化 | GigaWorld-1 评估器、仿真闭环 | [GigaWorld-1 策略评估](../entities/paper-gigaworld-1-policy-evaluation.md)、[仿真评测基建](../concepts/simulation-evaluation-infrastructure.md) |
| ④ sim↔real gap 校准 | 评测结论能否外推到真机 | real-to-sim 相关性、代表性代价 | [仿真 vs 真机评测 gap](../concepts/sim-vs-real-eval-gap.md) |
| 端到端 | 四层如何逐层选型取舍 | 选型决策树 | [评测基准选型闭环 Query](../queries/embodied-eval-benchmark-selection-loop.md) |

## 评测选型的关键取舍

- **可复现性 vs 真实代表性**：仿真基准在吞吐/可控/可复现上占优，代价是牺牲真实接触、感知噪声与长尾分布的代表性；评测结论能否外推真机取决于 real-to-sim 相关性。
- **过程指标 vs 结果指标**：任务成功率（结果）直观但掩盖长尾失败模式；过程/中间指标可归因但可能与真实收益脱钩。
- **代理指标 ≠ 下游收益**：世界模型视频质量高 ≠ 下游策略收益高、MLLM 认知评分高 ≠ 可执行动作能力强，跨层用代理指标要警惕。
- **单任务过拟合 vs 跨任务泛化**：基准饱和 ≠ 真实场景就绪，评测集泄漏与分布漂移会致虚高。

## 与其他专题的关系

- **[具身大模型分类学选型闭环](./topic-embodied-foundation-model.md)**：模型选型的家族谱系，本专题为其验收环节。
- **[仿真到现实（Sim2Real）](./topic-sim2real.md)**：④ 层 sim↔real gap 校准与 sim2real 迁移共享同一物理根因。

## 关联页面

- [具身大模型评测基准选型闭环 Query](../queries/embodied-eval-benchmark-selection-loop.md)
- [仿真 vs 真机评测 gap](../concepts/sim-vs-real-eval-gap.md)
- [RoboBench](../entities/robo-bench.md)
- [ESI-Bench](../entities/esi-bench.md)
- [EWMBench](../entities/ewmbench.md)
- [GigaWorld-1 策略评估](../entities/paper-gigaworld-1-policy-evaluation.md)
- [仿真评测基础设施](../concepts/simulation-evaluation-infrastructure.md)

## 参考来源

- [RoboBench 论文](../../sources/papers/robo_bench_arxiv_2510_17801.md) — MLLM 具身大脑五维评测
- [EWMBench 论文](../../sources/papers/ewmbench.md) — 具身世界模型视频生成评测
- [ESI-Bench 论文](../../sources/papers/esi_bench_arxiv_2605_18746.md) — 具身空间智能评测
- 本页归纳自 [评测基准选型闭环 Query](../queries/embodied-eval-benchmark-selection-loop.md) 及各评测基准实体/概念页
