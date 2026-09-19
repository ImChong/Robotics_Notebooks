---
type: entity
tags:
  - paper
  - vla
  - aerial
  - benchmark
  - agent
status: complete
updated: 2026-09-19
arxiv: "2607.10180"
related:
  - ../methods/vla.md
  - ../overview/vla-weekly-trends-2026-08-10-part1-technology-map.md
  - ../queries/embodied-eval-benchmark-selection-loop.md
sources:
  - ../../sources/papers/activefly_bench_arxiv_2607_10180.md
  - ../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md
summary: "空中具身问答与 VLA 对齐基准：拆分问答、观察行为规划与语言引导控制三层，真实+仿真户外数据与实机验证。"
---

# ActiveFly-Bench（arXiv:2607.10180）

**ActiveFly-Bench**（[arXiv:2607.10180](https://arxiv.org/abs/2607.10180)）收录于 [多模空间 · 一周 VLA 研究趋势简析（2026.08.10–08.16）第一篇](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md) **类 Agent、数采评估** 段。

## 一句话定义

**空中具身问答与 VLA 对齐基准：拆分问答、观察行为规划与语言引导控制三层，真实+仿真户外数据与实机验证。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言多模态模型 |
| VLN | Vision-Language Navigation | 视觉–语言导航 |
| WAM | World Action Model | 联合未来与动作生成的具身策略 |
| TTT | Test-Time Training | 部署阶段无标注数据的在线适配 |

## 为什么重要

- 空中具身问答与 VLA 对齐基准：拆分问答、观察行为规划与语言引导控制三层，真实+仿真户外数据与实机验证。
- 开源状态：**待核实**（步骤 2.5，入库日 2026-09-19）。
- 与 [vla weekly trends 2026 08 10 part1 technology map](../overview/vla-weekly-trends-2026-08-10-part1-technology-map.md) 同批工作可横向对照。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2607.10180](https://arxiv.org/abs/2607.10180) |
| **项目页** | https://lvmolvmo.github.io/ActiveFly |
| **开源** | **待核实** |
| **文内评测** | ActiveFly-Bench（自建） |

## 源码运行时序图

**不适用**（截至入库日未提供可运行官方代码入口，或仓库尚未公开）。


## 实验与评测

- **文内口径：** ActiveFly-Bench（自建）
- **读法：** 本页为 [公众号策展](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md) 摘要；逐项指标以 **原文 PDF** 为准。

## 与其他工作对比

> 下表只做**定位对照**：本页与下列同批各页均为 [公众号策展](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md) 的索引级摘要，未逐条核对原文实验表，彼此**不共享同一评测协议**，不可据此横比数字。

| 对照 | 差异读法 |
|------|----------|
| [CMU-Drive / V2V-VLA](./paper-cmu-drive-v2v-vla.md) | 同批「评测 · 多智能体」段的另一条自建闭环基准，**域与协作轴相反**：ActiveFly 测单机空中 agent 的**纵向分层**（问答 → 观察行为规划 → 语言引导控制），CMU-Drive 测多车之间的**横向协同**与通信策略。两者都是自建协议，也都不能与 LIBERO 系桌面操作榜混读 |
| [RecoverFly](./paper-recoverfly-aerial-vln.md) | 同批同为空中具身，一测一训：本页出的是**基准**（怎么测），RecoverFly 出的是**训练范式**（失败感知 RL 后训练）。两页之间缺的正是「用哪套口径判定纠错成功」——RecoverFly 报 TravelUAV，与本基准不同协议 |
| [VLN 任务页](../tasks/vision-language-navigation.md) | 该页给「语言指令 → 三维空间动作」的任务定义；ActiveFly 把它从地面推到**空中 + 主动问答**，多出一层「该飞到哪去看」的观察行为规划，这层在常规 VLN 指标里不单列 |
| [具身评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md) | 选基准先定层：ActiveFly 横跨「认知问答」与「策略成功率」两层，且真实 + 仿真户外数据 + 实机验证三档混编——按该页的分层读法确认自己要测的是哪一档，再决定是否引入 |

## 结论

**ActiveFly-Bench 适合作为本期「类 Agent、数采评估」路线的快速索引页。**

1. 核心贡献：空中具身问答与 VLA 对齐基准：拆分问答、观察行为规划与语言引导控制三层，真实+仿真户外数据与实机验证。
2. 开源结论：**待核实** — 以项目页/仓库实际链接为准。
3. 横向对照见 [技术地图](../overview/vla-weekly-trends-2026-08-10-part1-technology-map.md)，避免与同 arXiv 重复造页。

## 关联页面

- [VLA（Vision-Language-Action）](../methods/vla.md)
- [一周 VLA 趋势技术地图](../overview/vla-weekly-trends-2026-08-10-part1-technology-map.md)
- [Generative World Models](../methods/generative-world-models.md)

## 参考来源

- [wechat_duomo_vla_weekly_trends_2026-08-10_part1.md](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md)
- [arXiv:2607.10180](https://arxiv.org/abs/2607.10180)

## 推荐继续阅读

- [VLA 方法页](../methods/vla.md)
- [arXiv PDF](https://arxiv.org/pdf/2607.10180)
