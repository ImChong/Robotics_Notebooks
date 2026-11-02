---
type: entity
project_id: interventional-memory-evaluation-vla
tags:
  - paper
  - benchmark
  - evaluation
status: complete
updated: 2026-11-01
arxiv: "2610.23372"
project: https://anonymous.4open.science/r/IMEP-ChainBench-DC83/
related:
  - ../concepts/simulation-evaluation-infrastructure.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../overview/embodied-station-2026-11-01-56-papers-technology-map.md
  - ../queries/embodied-eval-benchmark-selection-loop.md
sources:
  - ../../sources/papers/interventional_memory_evaluation_vla_arxiv_2610_23372.md
  - ../../sources/sites/interventional-memory-evaluation-vla.md
  - ../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md
summary: "Interventional Memory Evaluation VLA（arXiv:2610.23372）：IMEP在检索接口替换事实、保留事实或施加微小数值扰动，分开判断决策与执行；ChainBench补充历史依赖任务，强调用事实特异性证据验证记忆作用。"
---

# Interventional Memory Evaluation VLA（arXiv:2610.23372）

**Interventional Memory Evaluation VLA**（*Beyond Success Rate: Interventional Evaluation of Memory in Vision–Language–Action Systems*，[arXiv:2610.23372](https://arxiv.org/abs/2610.23372)，[项目页](https://anonymous.4open.science/r/IMEP-ChainBench-DC83/)）来自 [具身智能小站 2026-11-01 盘点](../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md)，属「评测与基准」组。

## 一句话定义

**IMEP在检索接口替换事实、保留事实或施加微小数值扰动，分开判断决策与执行；ChainBench补充历史依赖任务，强调用事实特异性证据验证记忆作用。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Interventional Memory Evaluation VLA | Beyond Success Rate: Interventional Evaluation of Memory in Vision–Language–Action Systems | 本文方法 / 基准的简称 |

## 为什么重要

- 成功率变高，不能证明记忆真的被使用。
- 开源结论：**项目页未列 GitHub**（步骤 2.5，2026-11-01）。项目页为 anonymous.4open.science 匿名链接，curl 返回 HTTP 401，无法核查代码。
- 同期同组工作见 [动态评测、技能编译与环境潜变量适配：具身智能小站 11-01 期 56 篇阅读坐标](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)。

## 流程总览

```mermaid
flowchart TD
  A["配对单元：相同 任务-种子 或多回合链，跨条件复用"] --> B["记忆流程：先前回合写入记忆 → 查询时 VLA 读取 → 决策 → 执行"]
  B --> C["在检索接口施加条件"]
  C --> C1["访问/oracle：A 参考记忆，N 阻断读取或清空，O 给出正确事实/决策"]
  C --> C2["控制：C 精确重复，U 一个 ULP 数值扰动"]
  C --> C3["事实替换：S 同事实记录，T 不同事实记录"]
  C1 --> D["分别读出：决策正确率（可观测时）与任务成功率，不合并"]
  C2 --> D
  C3 --> D
  D --> E["Q1 净收益 D_N；Q2 条件；Q3 结果敏感度 ρ 对比重复下限；Q4 事实特异性 θ = ρ_T − ρ_S"]
  F["解释检查：信息（历史是否必需）、投递（读取/替换覆盖率）、有效性（记录是否仍有效）"] --> E
  G["ChainBench（LIBERO）：写入回合揭示物体所在抽屉 → 干扰回合写入其他记录 → 查询时抽屉关闭、观测相同"] --> H["先决策选定一个抽屉，再由冻结的 π0.5 执行"]
  H --> D
```

图依据：论文 §3.1–3.5 / Fig.2。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.23372](https://arxiv.org/abs/2610.23372) |
| **开源** | **项目页未列 GitHub** |
| **要点** | IMEP在检索接口替换事实、保留事实或施加微小数值扰动，分开判断决策与执行；ChainBench补充历史依赖任务，强调用事实特异性证据验证记忆作用。 |
| **文内指标** | 公众号未给具体数字，本页不引用。 |

## 源码运行时序图

**不适用（项目页未列 GitHub）** — 项目页为 anonymous.4open.science 匿名链接，curl 返回 HTTP 401，无法核查代码。 不编造运行脚本；代码公开后再补。

## 实验与评测

- 公众号未给具体数字，本页不引用。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [动态评测、技能编译与环境潜变量适配：具身智能小站 11-01 期 56 篇阅读坐标](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**成功率变高，不能证明记忆真的被使用：IMEP在检索接口替换事实、保留事实或施加微小数值扰动，分开判断决策与执行；ChainBench补充历史依赖任务，强调用事实特异性证据验证记忆作用。**

1. 开源边界：**项目页未列 GitHub** — 项目页为 anonymous.4open.science 匿名链接，curl 返回 HTTP 401，无法核查代码。
2. 核心机制：IMEP在检索接口替换事实、保留事实或施加微小数值扰动，分开判断决策与执行；ChainBench补充历史依赖任务，强调用事实特异性证据验证记忆作用。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [simulation-evaluation-infrastructure](../concepts/simulation-evaluation-infrastructure.md)
- [vla](../methods/vla.md)
- [manipulation](../tasks/manipulation.md)
- [embodied-station-2026-11-01-56-papers-technology-map](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)
- [具身大模型评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md) — 评测基准在选型闭环中的定位与取舍

## 参考来源

- [interventional_memory_evaluation_vla_arxiv_2610_23372.md](../../sources/papers/interventional_memory_evaluation_vla_arxiv_2610_23372.md)
- [wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md](../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md)
- [arXiv:2610.23372](https://arxiv.org/abs/2610.23372)
- [arXiv PDF](https://arxiv.org/pdf/2610.23372)：已通读全文主文：IMEP 协议、干预与读出、ChainBench（§3、§4；附录仅查阅相关设定）

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.23372)
- [项目页](https://anonymous.4open.science/r/IMEP-ChainBench-DC83/)
