---
type: entity
tags: [company, embodied-foundation-model, dataset, scaling, generalist-ai, foundation-policy, cross-embodiment]
title: Generalist AI（机器人）
status: complete
summary: "Generalist AI 是宣称以超大规模真实交互数据预训练具身基础模型的商业团队；公开材料侧重 GEN 系列迭代、物理常识与多末端「千手」泛化，引用应以官网博客为准且注意闭源边界。"
updated: 2026-07-24
related:
  - ./generalist-gen1-thousand-hands.md
  - ../concepts/embodied-scaling-laws.md
  - ../concepts/foundation-policy.md
  - ../methods/octo-model.md
  - ../overview/topic-cross-embodiment.md
sources:
  - ../../sources/blogs/generalist_thousand_hands.md
  - ../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md
---

# Generalist AI（机器人方向）

## 一句话定义

**Generalist AI**：聚焦具身智能与通用机器人策略的商业实体；对外叙事强调 **海量人类 / 机器人交互数据** 上的预训练、规模定律验证，以及 GEN 系列（GEN-0 → GEN-1）向 **多末端 / 多工具接口** 的扩展。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GEN-0 / GEN-1 | Generalist Embodied Model 0/1 | 公司公开的具身基础模型代际 |
| EFM | Embodied Foundation Model | 具身基础模型；其产品叙事核心 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略方向（对照开源路线） |
| EE | End Effector | 末端执行器；「千手」博文的扩展轴 |

## 与其他「Generalist」用语区分

知识库中另有「通用策略（generalist policy）」泛指任意跨任务模型；本页仅指 **Generalist AI 公司**，避免与 [Octo](../methods/octo-model.md)、[VLA](../methods/vla.md)、[Open X-Embodiment](../concepts/open-x-embodiment.md) 等开源或开放数据路线混淆。

## 为什么重要

- **商业侧 embodied scaling 样本：** 与开源 OXE / Octo 对照，提供「超大规模 in-house 数据 + 闭源模型」的产业叙事锚点。
- **多末端轴补全跨具身图景：** 2026-07「千手」博文把跨具身细化为 **工具/末端接口多样性**，见 [GEN-1 千手](./generalist-gen1-thousand-hands.md)。
- **引用纪律：** 成功率、小时数、变体数为官方自报；**确认未开源** 代码与数据集，不可当作可复现方法论文。

## 公开产品线脉络（博客）

| 节点 | 要点 | 入口 |
|------|------|------|
| **GEN-0**（约 2025-11） | 主张机器人侧 scaling laws / 预训练时代叙事 | [GEN-0 博文](https://generalistai.com/blog/gen-0) |
| **GEN-1**（约 2026-04） | 「mastery」阈值叙事；半百万小时级交互数据；后训练约 1h 机器人数据等自报 | [GEN-1 博文](https://generalistai.com/blog/gen-1) |
| **千手**（2026-07） | ~9k 末端变体；task-vector 诊断；任务中途换手 | [本库实体页](./generalist-gen1-thousand-hands.md) |
| **Physical Commonsense** | 「物理常识 / 暗物质」姊妹叙事 | [博文](https://generalistai.com/blog/physical-commonsense) |

## 数据与就绪度

- **数据 / 重定向就绪度：** 对外强调海量人类可穿戴交互预训练 + 少量机器人后训练；具体数据形态与跨本体适配 **未公开**，不可直接用于重定向或复现实验。
- **开源：** 截至 2026-07-24，公司站与相关博文 **未见** GitHub / Hugging Face 训练推理入口。

## 核心原理（对外可核对部分）

公司不公开架构配方；外部读者可核对的主张主要是：

1. **规模化真实交互预训练** 驱动通才物理策略（对照 [Embodied Scaling Laws](../concepts/embodied-scaling-laws.md)）。
2. **末端多样性** 作为接触物理数据轴（对照 [跨具身专题](../overview/topic-cross-embodiment.md)）。
3. **与开源 foundation policy** 的选型分工：要复现选 OXE/Octo/π 系开源代码；本页仅作产业对照。

## 工程实践

| 场景 | 建议 |
|------|------|
| 写综述 / 时代叙事 | 可引 GEN 系列作为商业 scaling 样本，并链到 [三个时代 Query](../queries/robot-learning-three-eras-narrative.md) |
| 做跨末端研究 | 借鉴「千手」的 **task-vector 诊断** 与 **mid-episode tool swap** 评测设计，用自有开源栈实现 |
| 产线选型 | **不要**假设可下载 GEN-1；评估闭源 API/集成需直接对接厂商 |

## 局限与风险

- 营销与技术边界模糊；定量结果缺第三方复现。
- 「物理 AGI / Cambrian explosion」为愿景修辞，工程上仍受数据、对齐与安全约束。
- 勿把公司名与开源 generalist policy 文献混为一谈。

## 关联页面

- [GEN-1 千手：跨末端执行器泛化](./generalist-gen1-thousand-hands.md)
- [Embodied Scaling Laws](../concepts/embodied-scaling-laws.md)
- [Foundation Policy](../concepts/foundation-policy.md)
- [跨具身迁移（专题）](../overview/topic-cross-embodiment.md)
- [Octo](../methods/octo-model.md)

## 参考来源

- [Towards Machines with a Thousand Hands（来源归档）](../../sources/blogs/generalist_thousand_hands.md)
- [ted_xiao_embodied_three_eras_primary_refs.md](../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md)
- GEN-0：<https://generalistai.com/blog/gen-0>
- GEN-1：<https://generalistai.com/blog/gen-1>

## 推荐继续阅读

- [Towards Machines with a Thousand Hands](https://generalistai.com/blog/towards-machines-with-a-thousand-hands) — 多末端扩展主文
- [GEN-1 官方博文](https://generalistai.com/blog/gen-1) — mastery 与数据引擎叙事
