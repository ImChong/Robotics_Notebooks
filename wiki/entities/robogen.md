---

type: entity
tags: [entity, simulator, data-generation, llm, embodied-ai, procedural, stanford]
status: complete
updated: 2026-06-22
related:
  - ./sapien.md
  - ./genesis-sim.md
  - ../methods/imitation-learning.md
  - ../overview/sim-platforms-decade-technology-map.md
sources:
  - ../../sources/blogs/wechat_shenlan_sim_platforms_top8_decade.md
summary: "LLM 驱动的机器人任务与场景自动生成框架：程序化扩展仿真任务与训练数据，面向 VLA 等数据饥渴范式，减轻人工环境搭建成本。"
---

# RoboGen

**RoboGen** 是面向 **机器人学习数据扩展** 的 **自动生成框架**，利用大语言模型与仿真器联动 **程序化生成任务、场景与演示轨迹**。

## 一句话定义

> RoboGen 把「写环境、写任务」从纯人工劳动变成 **可规模化的生成流水线**：用 LLM 提出任务与物体配置，在仿真中自动实例化并采集数据，缓解 VLA/IL 对多样交互数据的饥渴。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| LLM | Large Language Model | 大语言模型，用于任务/场景提案 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |
| IL | Imitation Learning | 模仿学习 |
| Sim | Simulation | 仿真环境 |

## 为什么重要

具身数据标注成本极高。RoboGen 代表 **生成式仿真数据** 支路：

1. **任务自动提案**：LLM 生成可操作任务描述与成功条件。
2. **场景实例化**：对接物理仿真后端布置物体与初始状态。
3. **与 VLA 数据飞轮对齐**：在 [ManiSkill2](./maniskill2.md)、[SAPIEN](./sapien.md) 等栈之上扩展 **组合多样性**，而非单一手工环境。

[十年仿真盘点](../../sources/blogs/wechat_shenlan_sim_platforms_top8_decade.md) 将其与 CARLA、Genesis 等并列为 **近年各方向重要贡献**。

## 核心结构/机制

- **LLM 规划层**：任务语言 → 结构化场景/技能规格。
- **仿真执行层**：调用仿真器运行并记录轨迹。
- **质量过滤**：失败重试与可行性检查（以论文实现为准）。

## 常见误区或局限

- **误区：RoboGen = 新物理引擎** — 它是 **数据生成框架**，依赖既有仿真后端。
- **局限：物理可行性** — 生成任务须经仿真验证；高保真日常活动评测见 [BEHAVIOR-1K](./behavior-1k.md)。

## 关联页面

- [SAPIEN](./sapien.md)
- [Genesis](./genesis-sim.md)
- [模仿学习](../methods/imitation-learning.md)
- [十年仿真平台技术地图](../overview/sim-platforms-decade-technology-map.md)

## 参考来源

- [sources/blogs/wechat_shenlan_sim_platforms_top8_decade.md](../../sources/blogs/wechat_shenlan_sim_platforms_top8_decade.md)
- Wang et al., *RoboGen: Towards Unleashing Infinite Data for Automated Robot Learning via Generative Simulation* — [arXiv](https://arxiv.org/abs/2311.01445)

## 推荐继续阅读

- [RoboGen 项目页](https://robogen-ai.github.io/)
- [机器人训练栈分层地图](../overview/robot-training-stack-layers-technology-map.md)
