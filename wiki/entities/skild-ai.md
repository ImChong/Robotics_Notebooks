---
type: entity
tags: [company, embodied-foundation-model, in-context-learning, foundation-policy, manipulation, skild-ai, cmu]
title: Skild AI
status: complete
updated: 2026-08-26
related:
  - ./skild-s1.md
  - ../concepts/robot-in-context-learning.md
  - ../concepts/foundation-policy.md
  - ./generalist-ai-robotics.md
  - ../overview/overseas-embodied-ai-labs-landscape-2026.md
  - ../overview/notable-commercial-robot-platforms.md
  - ./paper-notebook-locoformer-generalist-locomotion-via-long-contex.md
sources:
  - ../../sources/sites/skild-ai.md
  - ../../sources/blogs/skild_s1_in_context_learning.md
summary: "Skild AI 是 CMU 体系衍生的具身基础模型公司，主张 omni-bodied 单一脑控制任意机器人；2026-08 公开旗舰操作模型 S1（视频上下文 ICL），代码与数据确认未开源。"
---

# Skild AI

| 字段 | 内容 |
|------|------|
| **机构** | 斯齐尔德（Skild AI） |
| **类型** | 商业具身基础模型公司 |
| **公开锚点** | [skild.ai](https://www.skild.ai/)；旗舰博文 [S1](https://www.skild.ai/blogs/s1) |
| **学术前序** | LocoFormer（arXiv:2509.23745，Liu / Pathak / Agarwal） |
| **开源** | **确认未开源**（[github.com/skild-ai](https://github.com/skild-ai) 0 公开仓；截至 2026-08-26） |

## 一句话定义

**Skild AI**：主张 **omni-bodied** 物理智能的商业团队——同一套策略脑不绑定单一机型或任务；对外技术锚点从 2025 运动域 **LocoFormer**（上下文里累积在线经验）推进到 2026-08 操作域 **S1**（一条视频示范、权重不变）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| S1 | Skild S1 | 2026-08 旗舰操作基础模型，视频 in-context 指定任务 |
| ICL | In-Context Learning | 不更新权重、从上下文示范归纳映射 |
| VLA | Vision-Language-Action | S1 对照基线是语言条件 VLA |
| CMU | Carnegie Mellon University | 公司衍生叙事中的研究所母体 |

## 为什么重要

- **产业 ICL 样本：** 与 [Generalist AI](./generalist-ai-robotics.md) 的 **涌现 physical prompting** 对照，Skild 把 ICL 写成 **预训练目标本身**（任务只通过示范指定）。
- **跨域迁移叙事：** 先在 locomotion 验证长上下文适应，再搬到 manipulation 长程未见任务，见 [S1](./skild-s1.md)。
- **引用纪律：** 成功率、数据小时数为官方自报；**确认未开源**，不可替代 Octo / π / OpenVLA 做实验。

## 公开产品线脉络

| 节点 | 要点 | 入口 |
|------|------|------|
| 公司 thesis | omni-bodied brain；安防巡检 / 移动操作 API / 精细装箱 | [站点归档](../../sources/sites/skild-ai.md) |
| **LocoFormer**（2025-09） | 运动通才 + 上下文累积经验；未见本体/环境在线适应 | [论文笔记占位](./paper-notebook-locoformer-generalist-locomotion-via-long-contex.md) |
| **S1**（2026-08） | 视频 prompt；未见任务最长约 10 分钟；相对语言 VLA 未见档约 7× | [S1 实体](./skild-s1.md) |

## 工程实践

| 场景 | 建议 |
|------|------|
| 写综述 / 选型 | 把 Skild 与 Generalist、π 并列为 **闭源 ICL / 通才策略** 对照，不要假设可下载权重 |
| 复现 ICL | 用开源 one-shot IL / [RoboTTT](./paper-robottt-test-time-training-vla-context.md) / SynthICL；S1 只提供评测轴（已见 vs 未见、短程 vs 10 min） |
| LocoFormer 代码 | 官方未开源；[lucidrains/locoformer](https://github.com/lucidrains/locoformer) 是社区 WIP，非官方配方 |

## 局限与风险

- 站点以品牌与应用为主，技术细节几乎只在 S1 博客；训练配方声明「后续再写」。
- 「omni-bodied」是愿景修辞；公开演示未给出跨本体定量表。
- GitHub org 空仓容易被误读成「即将开源」——入库日应按 **确认未开源** 处理。

## 关联页面

- [S1：机器人 In-Context Learning](./skild-s1.md)
- [机器人 In-Context Learning](../concepts/robot-in-context-learning.md)
- [Foundation Policy](../concepts/foundation-policy.md)
- [Generalist AI](./generalist-ai-robotics.md) — 另一条闭源通才 / ICL 产业线
- [海外具身智能实验室地图（2026）](../overview/overseas-embodied-ai-labs-landscape-2026.md)
- [LocoFormer（论文笔记占位）](./paper-notebook-locoformer-generalist-locomotion-via-long-contex.md)
- [HOST](./paper-host-one-shot-human-video.md) — 开源单视频 one-shot 对照，不是本公司产品

## 参考来源

- [Skild AI 公司站点归档](../../sources/sites/skild-ai.md)
- [S1 博客归档](../../sources/blogs/skild_s1_in_context_learning.md)

## 推荐继续阅读

- 公司首页：<https://www.skild.ai/>
- S1 原文：<https://www.skild.ai/blogs/s1>
- Liu, Pathak, Agarwal, *LocoFormer*（[arXiv:2509.23745](https://arxiv.org/abs/2509.23745)）
