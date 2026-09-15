---
type: entity
tags: [curated-list, recursive-self-improvement, llm-agents, agent-harness, self-evolution, benchmarks]
status: complete
updated: 2026-09-15
related:
  - ../concepts/recursive-self-improvement.md
  - ../concepts/ai-auto-research.md
  - ./rsi-harness.md
  - ./paper-metarsi-v1.md
  - ./paper-harnessbank.md
  - ./karpathy-autoresearch.md
  - ./sol-pi.md
  - ./deepseek-harness.md
sources:
  - ../../sources/repos/awesome-rsi.md
  - ../../sources/sites/awesome-rsi-github-io.md
summary: "Prism-Shadow 维护的 Awesome RSI：按 RSI artifact（改什么）与 RSI mode（在线/离线评测）策展 50+ 方法、29 基准与 2 系统；双语静态站支持多维筛选与引用图谱，是 agent 层自进化文献的主索引入口。"
---

# Awesome RSI（Prism-Shadow 精选集）

**Awesome RSI**（GitHub：[Prism-Shadow/awesome-rsi](https://github.com/Prism-Shadow/awesome-rsi)，站点：[prism-shadow.github.io/awesome-rsi](https://prism-shadow.github.io/awesome-rsi/)）是一份 **Recursive Self-Improvement（RSI）** 文献与基准的 curated 列表：把「agent 如何改自己」拆成可检索的 **artifact × mode × feedback** 维度，并配套交互式引用图与中英导读。

## 一句话定义

面向 **LLM agent 自进化** 的 **方法与基准索引入口** — 先回答「改的是权重、harness、上下文、记忆还是技能」，再回答「评测是 online 还是 offline」。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RSI | Recursive Self-Improvement | 执行任务→学习→更新自身状态→用于后续任务的闭环 |
| DGM | Darwin Gödel Machine | 开放-ended harness 自改代表工作（清单收录） |
| GEPA | Genetic-Pareto / Reflective Prompt Evolution | 反思式 prompt 进化；清单 Context artifact 代表 |
| HGB | Harness Gene Bank | HarnessBank 的语义基因库（清单 Harness code 条目） |
| Online / Offline RSI mode | — | 基准协议：任务流中演化 vs 演化与评测分离 |

## 为什么重要

- **消歧「RSI」一词**：本库 [递归自改进](../concepts/recursive-self-improvement.md) 概念页侧重 **宏观 RSI**（AI 设计后继模型）；Awesome RSI 聚焦 **已发表 agent 机制** — prompt 进化、harness 自改、记忆蒸馏、技能库共演化等 **可对照实现**。
- **artifact 先行选型**：同一篇工作常跨多 artifact（如 Gödel Agent 同时标 Harness code 与 Context）；站点多 tag 筛选比单维 README 更适合 **「我想改 harness 不改权重」** 的部署约束。
- **基准与方法对齐**：29 条 benchmark 标注被测 workflow 中演化的 artifact 与 Online/Offline 协议 — 读 [HarnessBank](./paper-harnessbank.md) / [MetaRSI-v1](./paper-metarsi-v1.md) 时可回查同类评测语境。
- **与 Auto-Research 分工**：[AI Auto-Research](../concepts/ai-auto-research.md) 覆盖 **学术全生命周期**；Awesome RSI 深耕 **agent 状态自更新** 子空间，二者在 S3 实验自动化与 harness 进化处交叉。

## 核心结构（怎么读）

### RSI 闭环（清单共识）

```mermaid
flowchart LR
  T[执行任务]
  L[轨迹与反馈]
  U[更新状态]
  N[后继任务使用新状态]
  T --> L --> U --> N
  N --> T
```

**状态** 五类 + 其它：Model parameters · Harness code · Context · Memory · Skill（及 benchmark 侧的 Other artifacts）。

### Methods 页（`#methods`）怎么用

| 步骤 | 做法 |
|------|------|
| 定 artifact | 权重封闭 → 优先 Harness / Context / Memory / Skill 筛选 |
| 叠 protocol | 需要任务流内学习 → 对照 Online 基准；可离线演化再评测 → Offline |
| 读代表行 | Harness：`Gödel Agent`、`Darwin Gödel Machine`、`Proteus`；Context：`GEPA`、`ReasoningBank`；Memory：`A-MEM`、`Dynamic Cheatsheet` |
| 追邻域 | 打开 Citation graph，从种子论文扩邻域 |

### 与本库实体对照（非 exhaustive）

| Awesome RSI 线索 | 站内已有沉淀 |
|------------------|--------------|
| Harness code 自改 | [RSI-Harness](./rsi-harness.md)、[HarnessBank](./paper-harnessbank.md)、[DeepSeek Harness](./deepseek-harness.md) |
| 三算子 RSI 框架 | [MetaRSI-v1](./paper-metarsi-v1.md) |
| 最小实验环 / 训练脚本自改 | [karpathy/autoresearch](./karpathy-autoresearch.md) |
| Harness 效率再 scale | [SoL-Pi](./sol-pi.md) |
| 宏观 RSI 与具身跟随 | [递归自改进](../concepts/recursive-self-improvement.md) |

## 局限与使用注意

- **清单滞后**：awesome 依赖社区 PR；以 arXiv / 官方仓为准。
- **非可运行栈**：无统一训练代码；各条目开源状态需 **逐条** 核项目页（见 [sources/repos/awesome-rsi.md](../../sources/repos/awesome-rsi.md) 步骤 2.5）。
- **与「完全 RSI」保持距离**：多数条目是 **部分 artifact 自更新** 或 **长程经验改进**，不等同于 Anthropic 文内的「自主设计后继模型」。
- **机器人读者**：清单主体为 **软件 agent**；具身相关条目稀疏 — 真机闭环仍看 [真机 autoresearch harness](../queries/real-robot-policy-autoresearch-harness.md)。

## 关联页面

- [递归自改进（RSI）](../concepts/recursive-self-improvement.md) — 宏观 RSI 与具身跟随假设
- [AI Auto-Research](../concepts/ai-auto-research.md) — 学术研究自动化全谱
- [RSI-Harness](./rsi-harness.md) · [MetaRSI-v1](./paper-metarsi-v1.md) · [HarnessBank](./paper-harnessbank.md)
- [karpathy/autoresearch](./karpathy-autoresearch.md) · [SoL-Pi](./sol-pi.md)

## 参考来源

- [sources/repos/awesome-rsi.md](../../sources/repos/awesome-rsi.md)
- [sources/sites/awesome-rsi-github-io.md](../../sources/sites/awesome-rsi-github-io.md)

## 推荐继续阅读

- [Awesome RSI Methods 页](https://prism-shadow.github.io/awesome-rsi/#methods)
- [Understanding RSI 导读（站点 blog）](https://prism-shadow.github.io/awesome-rsi/#blog/understanding-rsi)
- [GitHub 仓库 README](https://github.com/Prism-Shadow/awesome-rsi)
