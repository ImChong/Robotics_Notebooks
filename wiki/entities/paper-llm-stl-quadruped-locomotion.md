---
type: entity
tags: [paper, quadruped, llm, stl, ppo, usc, uf]
status: complete
updated: 2026-09-15
arxiv: "2609.07111"
related:
  - ../tasks/locomotion.md
  - ../methods/ppo.md
  - ./paper-adapt-text-driven-humanoid.md
sources:
  - ../../sources/papers/llm_stl_quadruped_locomotion_arxiv_2609_07111.md
summary: "LLM-STL Quadruped（arXiv:2609.07111）：LLM generates parametric STL specs; expert trajectories set params; STL robustness as smooth reward for PPO；截至入库日未见官方代码。"
---

# LLM-STL Quadruped（arXiv:2609.07111）

**LLM-STL Quadruped**（*From LLM-Generated Specifications to Learned Quadruped Locomotion*，[arXiv:2609.07111](https://arxiv.org/abs/2609.07111)）由 **南加州大学（USC）；佛罗里达大学（University of Florida）** 提出（公众号周更 ingest 见 [策展索引](../../sources/blogs/wechat_shenlan_weekly_humanoid_quadruped_2026-09-14.md)）。

## 一句话定义

从大模型生成的形式化规约到四足运动策略学习 — LLM generates parametric STL specs。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| STL | Signal Temporal Logic | 信号时序逻辑 |
| LLM | Large Language Model | 大语言模型 |
| PPO | Proximal Policy Optimization | 近端策略优化 |

## 为什么重要

腿式任务规约难手写；LLM 可生成 STL 模板，鲁棒度量化奖励便于 RL。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 南加州大学（USC）；佛罗里达大学（University of Florida） |
| **开源** | **未见/待发布**（步骤 2.5 核查：截至 2026-09-14 无可运行官方仓库） |

## 核心原理

LLM 输出参数化 STL；专家轨迹拟合参数；PPO 优化策略使 STL robustness 最大化作为平滑奖励。

### 流程总览

```mermaid
flowchart LR
  lang[自然语言任务] --> llm[LLM]
  llm --> stl[参数化 STL]
  expert[专家轨迹] --> params[参数初始化]
  params --> stl
  stl --> reward[鲁棒度奖励]
  reward --> ppo[PPO 四足策略]
```

## 源码运行时序图

**不适用** — 截至 **2026-09-14** arXiv 与常见项目页 **未见** 官方可运行代码仓库。

## 工程实践

| 项 | 说明 |
|----|------|
| 开源状态 | 未见官方仓库；以 arXiv 为准 |
| 复现入口 | 论文方法与超参；代码发布后再补 `sources/repos/` |
| 部署注意 | STL 参数边界需验证；LLM 幻觉会导致不可满足规约。 |

## 实验与评测

多四足 locomotion 任务；与手工 reward 对比。

## 结论

LLM→STL→鲁棒度奖励链路把语言任务描述接到四足 PPO 学习。

1. STL 提供可验证任务语义。
2. 专家轨迹锚定参数。
3. 鲁棒度奖励比稀疏事件更平滑。
4. LLM 需人工/仿真过滤坏规约。
5. 适用中等复杂度步态任务。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| 手工 reward 工程 | 难扩展语言任务 |
| 纯 LLM 高层规划 | 无低层可执行保证 |

## 局限与风险

复杂接触 STL 表达力有限；LLM 错误规约需检测。

## 关联页面

- [locomotion](../tasks/locomotion.md)
- [ppo](../methods/ppo.md)
- [./paper-adapt-text-driven-humanoid.md](./paper-adapt-text-driven-humanoid.md)

## 参考来源

- [llm_stl_quadruped_locomotion_arxiv_2609_07111.md](../../sources/papers/llm_stl_quadruped_locomotion_arxiv_2609_07111.md)
- [公众号周更策展](../../sources/blogs/wechat_shenlan_weekly_humanoid_quadruped_2026-09-14.md)

## 推荐继续阅读

- [https://arxiv.org/abs/2609.07111](https://arxiv.org/abs/2609.07111)
