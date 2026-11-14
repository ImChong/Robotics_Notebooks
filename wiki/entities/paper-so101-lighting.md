---
type: entity
project_id: so101-lighting
arxiv: "2611.13810"
tags: [paper, so101-lighting, robotics]
status: draft
updated: 2026-11-14
summary: "SO-101 光照对照实验发现环境照度下降 11.5 倍时，自动曝光使模型输入图像变化远小于照度变化。"
related:
  - ../methods/vla.md
  - ../methods/action-chunking.md
original:
  - ../../sources/papers/so101_lighting_arxiv_2611_13810.md
sources:
  - ../../sources/papers/so101_lighting_arxiv_2611_13810.md
---

# Camera equalisation of an 11.5x drop in environmental illuminance: four imitation-learning policies on an SO-101 arm｜自动曝光如何影响 SO-101 光照鲁棒性评测

## 一句话定义

SO-101 光照对照实验发现环境照度下降 11.5 倍时，自动曝光使模型输入图像变化远小于照度变化。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| SO-101 | SO-101 Robot Arm | 本研究涉及的SO-101术语 |
| ACT | Action Chunking Transformer | 本研究涉及的ACT术语 |
| CI | Confidence Interval | 本研究涉及的CI术语 |

## 为什么重要

四种策略 ACT、SmolVLA、Diffusion Policy 和 π0.5 均用同一抓放任务 50 条演示训练。

## 核心原理与方法

1. 四种策略 ACT、SmolVLA、Diffusion Policy 和 π0.5 均用同一抓放任务 50 条演示训练。
2. 在约 520 与 45 lux 各运行每策略 10 次。
3. 同步记录环境 lux、曝光/白平衡与输入图像统计，避免误用照度当图像域偏移量。

## 流程总览

```mermaid
flowchart LR
  A0["环境照度变化"]
  A1["相机自动曝光"]
  A2["统计实际输入亮度"]
  A3["四策略 rollout"]
  A4["成败与置信区间"]
  A0 --> A1
  A1 --> A2
  A2 --> A3
  A3 --> A4
```

> 方法流程根据摘要/官网整理，尚待论文原文逐图核实。

## 实验与评测

常亮 20/40 成功，低照度 19/40（-2.5 个百分点，95% CI -23 至 +19）；前相机图像均值亮度仅下降约 3%，腕相机约 19%，相机开启自动曝光。

上述结果均为作者报告或用户附件转载，**不是本次复测**。

## 结论

**SO-101 光照对照实验发现环境照度下降 11.5 倍时，自动曝光使模型输入图像变化远小于照度变化。**

1. 四种策略 ACT、SmolVLA、Diffusion Policy 和 π0.5 均用同一抓放任务 50 条演示训练。
2. 在约 520 与 45 lux 各运行每策略 10 次。
3. 样本量小且曝光未锁定，不能证明策略对真实低照图像干扰鲁棒。

## 与其他工作对比

比较本文与相关基线时应在相同任务、数据、样本量、传感器和部署条件下评估；不能把论文中的单个数字解释为真实部署的通用进步。待全文表格核对后补足逐项量化对照。

## 源码运行时序图

**不适用：** 未核对可运行源码。[公开数据入口](https://huggingface.co/datasets/fysical/so101-lighting-autoexposure)，不能凭摘要编造执行调用栈。

## 工程实践

- 对齐版本、传感器、动作/观测空间和评测预算后再尝试复现。
- 正式应用应增加跨分布扰动与任务安全边界检查；原文完整复现步骤待核对。
- 本次通读状态：**未完成**；页面仍为 draft。

## 局限与风险

样本量小且曝光未锁定，不能证明策略对真实低照图像干扰鲁棒。

未对全部论文正文/附录及运行细节完成审阅，任何进一步量化外推均不成立。

## 关联页面

- [vla](../methods/vla.md)
- [action-chunking](../methods/action-chunking.md)

## 参考来源

- [原论文及阅读进度](../../sources/papers/so101_lighting_arxiv_2611_13810.md)
- [用户附件 13 篇原始资讯整理](../../sources/blogs/embodied-research-digest-2026-11-12.md)

## 推荐继续阅读

- [arXiv 原文](https://arxiv.org/abs/2611.13810)
- [官方数据](https://huggingface.co/datasets/fysical/so101-lighting-autoexposure)
