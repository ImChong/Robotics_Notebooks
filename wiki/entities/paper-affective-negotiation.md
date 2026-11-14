---
type: entity
project_id: affective-negotiation
arxiv: "2611.14247"
code: https://github.com/monurkeskin/Personalized-Affective-Perception-via-Continual-Learning-TAFFC-2026
tags: [paper, affective-negotiation, robotics]
status: draft
updated: 2026-11-14
summary: "通过 CLIFER 持续适应谈判者表情的情绪效价-唤醒表征，并用差分反馈调整机器人的谈判效用目标。"
related:
  - ../concepts/agent-memory-and-dreaming.md
  - ../methods/vla.md
original:
  - ../../sources/papers/affective_negotiation_arxiv_2611_14247.md
sources:
  - ../../sources/papers/affective_negotiation_arxiv_2611_14247.md
  - ../../sources/repos/affective-negotiation.md
---

# Personalized Affective Perception via Continual Learning in Human-Agent Negotiation｜持续学习个性化情感感知的谈判机器人

## 一句话定义

通过 CLIFER 持续适应谈判者表情的情绪效价-唤醒表征，并用差分反馈调整机器人的谈判效用目标。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| CLIFER | Continual Learning with Imagination for Facial Expression Recognition | 本研究涉及的CLIFER术语 |
| VA | Valence-Arousal | 本研究涉及的VA术语 |
| NAOqi | NAO robot software runtime | 本研究涉及的NAOqi术语 |

## 为什么重要

从表情预测个性化 valence-arousal 状态。

## 核心原理与方法

1. 从表情预测个性化 valence-arousal 状态。
2. 异步连续学习和生成式回放维护 CLIFER 情感表征。
3. 把情绪方向和强度变化转成目标效用反馈并调节谈判策略。

## 流程总览

```mermaid
flowchart LR
  A0["人脸图像"]
  A1["CLIFER 连续学习"]
  A2["VA 差分反馈"]
  A3["调节目标效用"]
  A4["代理谈判行为"]
  A0 --> A1
  A1 --> A2
  A2 --> A3
  A3 --> A4
```

> 方法流程根据摘要/官网整理，尚待论文原文逐图核实。

## 实验与评测

66 人配对研究报告代理收益和参与者让步率上升，参与者效用与 Nash 距离无显著改善；无独立表情真值证明情绪识别精度提高。

上述结果均为作者报告或用户附件转载，**不是本次复测**。

## 结论

**通过 CLIFER 持续适应谈判者表情的情绪效价-唤醒表征，并用差分反馈调整机器人的谈判效用目标。**

1. 从表情预测个性化 valence-arousal 状态。
2. 异步连续学习和生成式回放维护 CLIFER 情感表征。
3. 行为收益不等于准确读懂情绪；部署依赖旧式机器人软件/感知资源且存在伦理边界。

## 与其他工作对比

比较本文与相关基线时应在相同任务、数据、样本量、传感器和部署条件下评估；不能把论文中的单个数字解释为真实部署的通用进步。待全文表格核对后补足逐项量化对照。

## 源码运行时序图

```mermaid
sequenceDiagram
  participant A0 as verify_all_results.py
  participant A1 as Public data
  participant A2 as CLIFER assets
  participant A3 as Outcome checks
  A0->>A1: 启动评测/训练
  A1->>A2: 处理环境与模型输入
  A2->>A3: 执行模型或输出指标
  A3-->>A0: 返回状态
````

**README 入口：** `python verify_all_results.py（仅公开结果校验）`。图仅表示已核对到的入口与概念性组件顺序，不代表已运行或逐行审阅。

## 工程实践

- 对齐版本、传感器、动作/观测空间和评测预算后再尝试复现。
- 正式应用应增加跨分布扰动与任务安全边界检查；原文完整复现步骤待核对。
- 本次通读状态：**未完成**；页面仍为 draft。

## 局限与风险

行为收益不等于准确读懂情绪；部署依赖旧式机器人软件/感知资源且存在伦理边界。

未对全部论文正文/附录及运行细节完成审阅，任何进一步量化外推均不成立。

## 关联页面

- [agent-memory-and-dreaming](../concepts/agent-memory-and-dreaming.md)
- [vla](../methods/vla.md)

## 参考来源

- [原论文及阅读进度](../../sources/papers/affective_negotiation_arxiv_2611_14247.md)
- [官方源码入口](../../sources/repos/affective-negotiation.md)
- [用户附件 13 篇原始资讯整理](../../sources/blogs/embodied-research-digest-2026-11-12.md)

## 推荐继续阅读

- [arXiv 原文](https://arxiv.org/abs/2611.14247)
- [官方源码](https://github.com/monurkeskin/Personalized-Affective-Perception-via-Continual-Learning-TAFFC-2026)
