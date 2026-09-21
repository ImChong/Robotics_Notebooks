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

## 实验与评测

- **本页为索引级节点**（Lumina Embodied-AI-Guide 微信专辑）：正文固化定位与开源边界，**未转存原文实验表**。
- **回原文须核对的证据**：本页结论已点明卖点是「可验证规划」——回原文须核对 (a) PDDL 转写的正确率与 (b) 符号规划器的求解成功率：二者**串联**，总成功率不能只看后者。截至入库未列官方 GitHub，复现以论文给出的 PDDL 为准。
- **读法：** 先对齐本体、任务集与成功判定，再读任何数字；勿把专辑摘要当实验结论。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **对照工作** | 与纯 LLM 软 grounding 路线对照：后者对可行性无保证，本文用符号规划器换取保证，代价是 PDDL 建模与状态同步成本；与 [AutoTAMP](./paper-autotamp.md) 共享「LLM + 符号」谱系 |
| **横比口径** | 「可验证」是二值性质而非成功率数字，与端到端 VLA 的成功率不是同一维度，不可直接横比。 |
| **开源状态** | **未列官方 GitHub（截至 2026-09-20）** — 复现前以项目页 / 官方仓实际链接为准 |

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
