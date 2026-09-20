---
type: entity
tags: [paper, llm, planning, manipulation]
status: complete
updated: 2026-09-20
arxiv: "2304.11477"
related:
  - ./paper-autotamp.md
  - ./paper-saycan.md
  - ../concepts/llm-robotics-control-interfaces.md
sources:
  - ../../sources/blogs/wechat_lumina_embodied_practice_part1_llm_planner_2026-09-20.md
summary: "LLM+P（arXiv:2304.11477）：LLM 生成 PDDL 问题描述，交给经典符号规划器求可执行计划。"
---

# LLM+P

**LLM+P**（[arXiv:2304.11477](https://arxiv.org/abs/2304.11477)）收录于 Lumina [Embodied-AI-Guide 微信专辑](../../wiki/overview/embodied-ai-guide-wechat-album-curator.md)。本页为独立详情节点；实验数字以原文为准。

## 一句话定义

**让 LLM 写规划问题，让 PDDL 求解器保证语法与可行性。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| LLM | Large Language Model | 大语言模型 |
| IL | Imitation Learning | 模仿学习 |
| BC | Behavior Cloning | 行为克隆 |

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | — |
| **arXiv** | [2304.11477](https://arxiv.org/abs/2304.11477) |
| **开源** | 未列官方 GitHub（截至 2026-09-20） |

## 结论

LLM+P 适合需要可验证规划的长程任务；代价是 PDDL 建模与状态同步成本。

- 软 grounding 无保证 vs 符号规划有保证
- 与 MetaCtrl 等后续工作共享 LLM+符号谱系
- 无官方仓时以论文 PDDL 为准

## 源码运行时序图

**不适用（无官方可运行仓库链接）**

## 关联页面

- [paper-autotamp](./paper-autotamp.md)
- [paper-saycan](./paper-saycan.md)
- [llm-robotics-control-interfaces](../concepts/llm-robotics-control-interfaces.md)

## 参考来源

- [wechat_lumina_embodied_practice_part1_llm_planner_2026-09-20.md](../../sources/blogs/wechat_lumina_embodied_practice_part1_llm_planner_2026-09-20.md)

## 推荐继续阅读

- [arXiv:2304.11477](https://arxiv.org/abs/2304.11477)
