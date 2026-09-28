---
type: entity
tags: [company, vla, manipulation, evaluation, physical-intelligence]
title: PI Robot Olympics 微调演示
status: complete
updated: 2026-09-28
related:
  - ./paper-pistar06-recap.md
  - ./paper-pi-human-to-robot.md
  - ../methods/π0-policy.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/blogs/pi_robot_olympics_2025-12-22.md
  - ../../sources/sites/pi-website-technical-articles.md
summary: "2025-12-22 博客：微调 π₀.₆ 尝试 Holson 的 Robot Olympics。五项中三项金、两项银；自报平均成功率 52%、进度 72%。无预训练 VLM 对照进度 9%。确认未开源。"
---

# Robot Olympics：用 π₀.₆ 微调硬操作

Physical Intelligence 在 2025-12-22 的博客 [Moravec's Paradox and the Robot Olympics](https://www.pi.website/blog/olympics) 里，用当时的 **π₀.₆** 微调去尝试 Benjie Holson 提出的一组日常操作。任务不是团队自己挑选的，用来看通才预训练之后，少量新数据能覆盖哪些精细技能。

## 一句话定义

> **一组外部规定的家务挑战，用来检验 π₀.₆ 微调能不能在数小时数据内做出此前没有演示过的操作。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 被微调的 π₀.₆ |
| VLM | Vision-Language Model | 对照：没有机器人预训练的标准 VLM |
| SFT | Supervised Fine-Tuning | 本文实际使用的适配方式 |

## 为什么重要

Moravec 悖论在这里被说成数据问题：认知任务能从网上的解释里学，拿刀、擦锅学不到，因为人不会把这些动作写成文本。通才模型的用处是提供一块物理先验，使新任务不必从零开始堆数据。这篇不是新算法，它给出这条先验在外部任务单上的一次微调结果，并明确写出硬件做不到的项目。

## 核心原理

五项挑战各有铜、银、金。作者尽量按原始设定布置，但部分任务用了固定底座，原文面向移动机器人。策略来自微调 π₀.₆；作者写明没有为了刷成功率去跑他们另一篇关于 RL 可靠性与速度的工作。无 π₀.₆、只微调标准 VLM 的对照用来隔离机器人预训练。

| 项 | 博客中的处理 |
|----|----------------|
| 全身 / 门 | 金：拉开并穿过自闭门 |
| 洗衣 | 金级翻衬衫袖口受夹爪宽度限制；银为翻袜子（约 8 小时数据），另做铜级翻面 T 恤折叠 |
| 工具 | 金：从桌面拿起钥匙并插入；银：花生酱三明治；铜：喷壶加纸巾擦窗 |
| 指尖 | 银：套拾便袋；金级剥橙用了工具，作者不计成功 |
| 湿滑 | 金：水与海绵洗油锅 |

## 工程实践

多数任务数据少于 9 小时。成功并不稳定：作者自报平均成功率 **52%**、任务进度 **72%**。VLM 对照没有完成任何任务，平均进度 **9%**。两项金级因夹爪几何做不到，其中剥橙依赖额外工具，被明确排除。

## 局限与风险

- 确认未开源，也不是独立论文实验。52% / 72% 是作者平均，不含他们故意没做的 RL 优化。
- 固定底座与「用工具剥橙不计分」说明硬件边界会改写奖牌，不能读成模型已通过全部金级。
- 对照 VLM 的具体检查点未在博客展开，只能支持「没有机器人预训练则这组任务失败」，不能支持任意 VLM 都是 9%。

## 关联页面

- [π\*₀.₆ / RECAP](./paper-pistar06-recap.md)
- [人视频迁移](./paper-pi-human-to-robot.md)
- [π₀](../methods/π0-policy.md)
- [操作任务](../tasks/manipulation.md)

## 参考来源

- [pi_robot_olympics_2025-12-22](../../sources/blogs/pi_robot_olympics_2025-12-22.md)
- [PI 官网技术文章索引](../../sources/sites/pi-website-technical-articles.md)

## 推荐继续阅读

- [博客原文](https://www.pi.website/blog/olympics)
