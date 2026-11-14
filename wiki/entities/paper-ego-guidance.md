---
type: entity
project_id: ego-guidance
arxiv: "2611.13685"
code: https://github.com/llong-cs/EgoGuidance
project: https://llong-cs.github.io/EgoGuidance/
tags: [paper, ego-guidance, robotics]
status: draft
updated: 2026-11-14
summary: "Ego Guidance 用 SPAR 共享动作预测和未来视觉特征预测，允许无动作标签第一视角视频改进机器人策略表征。"
related:
  - ../methods/vla.md
  - ../concepts/flow-matching-embodied-policy.md
original:
  - ../../sources/papers/ego_guidance_arxiv_2611_13685.md
sources:
  - ../../sources/papers/ego_guidance_arxiv_2611_13685.md
  - ../../sources/repos/ego-guidance.md
  - ../../sources/sites/ego-guidance.md
---

# Ego Guidance: Enhancing Robot Policy Learning with Action-Free Human Egocentric Videos｜无动作标签 Ego 视频辅助机器人策略

## 一句话定义

Ego Guidance 用 SPAR 共享动作预测和未来视觉特征预测，允许无动作标签第一视角视频改进机器人策略表征。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| SPAR | Shared Prediction on Action and Representation | 本研究涉及的SPAR术语 |
| VLM | Vision-Language Model | 本研究涉及的VLM术语 |
| Ego | Egocentric Video | 本研究涉及的Ego术语 |

## 为什么重要

预训练视觉语言模型上共享机器人动作监督和人类第一视角视觉未来预测。

## 核心原理与方法

1. 预训练视觉语言模型上共享机器人动作监督和人类第一视角视觉未来预测。
2. 训练期使用双分支表征学习并关注视频筛选和分布差异。
3. 推理时不必运行辅助未来视觉预测分支，仅部署动作策略。

## 流程总览

```mermaid
flowchart LR
  A0["机器人演示与 Ego 视频"]
  A1["共享视觉语言编码"]
  A2["未来视觉特征监督"]
  A3["动作预测训练"]
  A4["部署动作策略"]
  A0 --> A1
  A1 --> A2
  A2 --> A3
  A3 --> A4
```

> 方法流程根据摘要/官网整理，尚待论文原文逐图核实。

## 实验与评测

摘要报告分布外任务与视频扩充的收益；新闻附件所列五项真机任务共 50 次测试，模型成功 20 次、基线 22 次，不支持所有真机任务均获提升的说法。

上述结果均为作者报告或用户附件转载，**不是本次复测**。

## 结论

**Ego Guidance 用 SPAR 共享动作预测和未来视觉特征预测，允许无动作标签第一视角视频改进机器人策略表征。**

1. 预训练视觉语言模型上共享机器人动作监督和人类第一视角视觉未来预测。
2. 训练期使用双分支表征学习并关注视频筛选和分布差异。
3. 无动作标签视频与机器人执行域存在差距，少量真机结果难支持广泛泛化。

## 与其他工作对比

比较本文与相关基线时应在相同任务、数据、样本量、传感器和部署条件下评估；不能把论文中的单个数字解释为真实部署的通用进步。待全文表格核对后补足逐项量化对照。

## 源码运行时序图

```mermaid
sequenceDiagram
  participant A0 as train.sh
  participant A1 as Pretrain config
  participant A2 as SPAR backbone
  participant A3 as Action/visual objectives
  A0->>A1: 启动评测/训练
  A1->>A2: 处理环境与模型输入
  A2->>A3: 执行模型或输出指标
  A3-->>A0: 返回状态
````

**README 入口：** `bash scripts/train.sh configs/pretrain/cotrain_openaoe.yaml`。图仅表示已核对到的入口与概念性组件顺序，不代表已运行或逐行审阅。

## 工程实践

- 对齐版本、传感器、动作/观测空间和评测预算后再尝试复现。
- 正式应用应增加跨分布扰动与任务安全边界检查；原文完整复现步骤待核对。
- 本次通读状态：**未完成**；页面仍为 draft。

## 局限与风险

无动作标签视频与机器人执行域存在差距，少量真机结果难支持广泛泛化。

未对全部论文正文/附录及运行细节完成审阅，任何进一步量化外推均不成立。

## 关联页面

- [vla](../methods/vla.md)
- [flow-matching-embodied-policy](../concepts/flow-matching-embodied-policy.md)

## 参考来源

- [原论文及阅读进度](../../sources/papers/ego_guidance_arxiv_2611_13685.md)
- [官方源码入口](../../sources/repos/ego-guidance.md)
- [官方项目页](../../sources/sites/ego-guidance.md)
- [用户附件 13 篇原始资讯整理](../../sources/blogs/embodied-research-digest-2026-11-12.md)

## 推荐继续阅读

- [arXiv 原文](https://arxiv.org/abs/2611.13685)
- [官方源码](https://github.com/llong-cs/EgoGuidance)
