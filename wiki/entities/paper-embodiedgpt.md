---
type: entity
tags: [paper, llm, embodied, planning]
status: complete
updated: 2026-09-20
arxiv: "2305.15021"
related:
  - ./paper-palm-e-embodied-language-model.md
  - ./paper-saycan.md
sources:
  - ../../sources/blogs/wechat_lumina_embodied_practice_part1_llm_planner_2026-09-20.md
summary: "EmbodiedGPT（arXiv:2305.15021）：面向具身场景的 GPT 式规划模型。"
---

# EmbodiedGPT

**EmbodiedGPT**（[arXiv:2305.15021](https://arxiv.org/abs/2305.15021)）收录于 Lumina [Embodied-AI-Guide 微信专辑](../../wiki/overview/embodied-ai-guide-wechat-album-curator.md)。本页为独立详情节点；实验数字以原文为准。

## 一句话定义

**在具身数据上微调 LLM，输出可交给下游控制器的计划。**

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
| **arXiv** | [2305.15021](https://arxiv.org/abs/2305.15021) |
| **开源** | 待核实 |

## 实验与评测

- **本页为索引级节点**（Lumina Embodied-AI-Guide 微信专辑）：正文固化定位与开源边界，**未转存原文实验表**。
- **回原文须核对的证据**：本页结论已点明「输出仍是语言/计划」——因此被评测的是**规划质量**而非动作执行精度；回原文须核对其规划评测基准，以及是否包含真机闭环而非纯离线打分。
- **读法：** 先对齐本体、任务集与成功判定，再读任何数字；勿把专辑摘要当实验结论。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **对照工作** | 与 RT-2 的「统一高低层」路线对照：本文停在计划层、需外接技能执行器，RT-2 直接输出动作 token |
| **横比口径** | 规划层基准与动作层基准分属两类，不可混比；规划分数高不代表本体可执行。 |
| **开源状态** | **待核实** — 复现前以项目页 / 官方仓实际链接为准 |

## 结论

EmbodiedGPT 代表具身专用 LM 规划器方向。

- 输出仍是语言/计划
- 与 RT-2 统一高低层对照

## 源码运行时序图

**不适用**

## 关联页面

- [paper-palm-e-embodied-language-model](./paper-palm-e-embodied-language-model.md)
- [paper-saycan](./paper-saycan.md)

## 参考来源

- [wechat_lumina_embodied_practice_part1_llm_planner_2026-09-20.md](../../sources/blogs/wechat_lumina_embodied_practice_part1_llm_planner_2026-09-20.md)

## 推荐继续阅读

- [arXiv:2305.15021](https://arxiv.org/abs/2305.15021)
