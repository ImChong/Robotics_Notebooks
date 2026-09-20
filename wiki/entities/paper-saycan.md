---
type: entity
tags: [paper, llm, planning, manipulation, google]
status: complete
updated: 2026-09-20
arxiv: "2204.01691"
code: https://github.com/google-research/google-research/tree/master/saycan
related:
  - ./paper-palm-e-embodied-language-model.md
  - ./paper-rt-2.md
  - ../methods/saycan.md
  - ../methods/vla.md
  - ../concepts/llm-robotics-control-interfaces.md
sources:
  - ../../sources/blogs/wechat_lumina_embodied_practice_part1_llm_planner_2026-09-20.md
summary: "SayCan（arXiv:2204.01691，Google）：LLM 生成子任务候选，价值/成功率估计作 affordance 过滤器，分层规划真机厨房任务。"
---

# SayCan（Do As I Can）

**SayCan（Do As I Can）**（[arXiv:2204.01691](https://arxiv.org/abs/2204.01691)，[代码](https://github.com/google-research/google-research/tree/master/saycan)）收录于 Lumina [Embodied-AI-Guide 微信专辑](../../wiki/overview/embodied-ai-guide-wechat-album-curator.md)。本页为独立详情节点；实验数字以原文为准。

## 一句话定义

**用语言模型的常识做计划，用学得的价值函数做「能不能做」的物理过滤器。**

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
| **机构** | Google Research |
| **arXiv** | [2204.01691](https://arxiv.org/abs/2204.01691) |
| **开源** | 部分开源（google-research 子目录；非完整内部栈） |

## 实验与评测

- **本页为索引级节点**（Lumina Embodied-AI-Guide 微信专辑）：正文固化定位与开源边界，**未转存原文实验表**。
- **回原文须核对的证据**：本页结论已点明「价值函数/成功率估计是系统关键，不是 LLM 参数规模」——其直接证据是**去掉 affordance 过滤器的消融**，回原文须核对该消融；真机厨房任务成功率则要连同技能库覆盖面一起读。
- **读法：** 先对齐本体、任务集与成功判定，再读任何数字；勿把专辑摘要当实验结论。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **对照工作** | 与 RT-2 / OpenVLA 的端到端路线对照：本文输出离散子技能、需预置技能库，端到端 VLA 直接产动作；取舍是可解释性与分层复用 ↔ 技能覆盖上限 |
| **横比口径** | 成功率受预置技能库覆盖面支配，技能库不同的系统不可横比；且本页开源一栏为部分开源（研究片段，完整栈依赖内部技能库），可复现范围有限。 |
| **开源状态** | **部分开源（google-research 子目录；非完整内部栈）** — 复现前以项目页 / 官方仓实际链接为准 |

## 结论

SayCan 确立了 LLM+机器人最可落地的分工：规划在上、affordance 裁剪在中、技能执行在下。

- 不要把 SayCan 当成端到端 VLA 的前身——输出仍是离散子技能
- 价值函数/成功率估计是系统关键，不是 LLM 参数规模
- 与 OpenVLA/RT-2 对照：何时需要统一高低层
- 复现优先读 google-research 子目录与原文消融

## 源码运行时序图

**不适用（无独立可运行官方仓库；SayCan 代码为研究片段，完整栈依赖内部技能库）**

## 关联页面

- [paper-palm-e-embodied-language-model](./paper-palm-e-embodied-language-model.md)
- [paper-rt-2](./paper-rt-2.md)
- [saycan](../methods/saycan.md)
- [vla](../methods/vla.md)
- [llm-robotics-control-interfaces](../concepts/llm-robotics-control-interfaces.md)

## 参考来源

- [wechat_lumina_embodied_practice_part1_llm_planner_2026-09-20.md](../../sources/blogs/wechat_lumina_embodied_practice_part1_llm_planner_2026-09-20.md)

## 推荐继续阅读

- [arXiv:2204.01691](https://arxiv.org/abs/2204.01691)
