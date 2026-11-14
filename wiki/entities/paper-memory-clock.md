---
type: entity
project_id: memory-clock
arxiv: "2611.14488"
code: https://github.com/bennidict23/memory-clock
tags: [paper, memory-clock, robotics]
status: draft
updated: 2026-11-14
summary: "按时间采样的 VLA 记忆可能把暂停当作任务进度；用零事件干预与事件索引暴露和缓解 temporal shortcuts。"
related:
  - ../methods/vla.md
  - ../concepts/agent-memory-and-dreaming.md
original:
  - ../../sources/papers/memory_clock_arxiv_2611_14488.md
sources:
  - ../../sources/papers/memory_clock_arxiv_2611_14488.md
  - ../../sources/repos/memory-clock.md
---

# Is Your Robot's Memory Just a Clock? Temporal Shortcuts in Memory-Augmented VLA Policies｜机器人的记忆可能只是时钟

## 一句话定义

按时间采样的 VLA 记忆可能把暂停当作任务进度；用零事件干预与事件索引暴露和缓解 temporal shortcuts。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| VLA | Vision-Language-Action | 本研究涉及的VLA术语 |
| RoboMME | RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies | 本研究涉及的RoboMME术语 |
| Null event | Task-neutral event | 本研究涉及的Null event术语 |

## 为什么重要

在 RoboMME 计数任务中加入不改变物体或任务状态的暂停和空抓动作。

## 核心原理与方法

1. 在 RoboMME 计数任务中加入不改变物体或任务状态的暂停和空抓动作。
2. 对照正常运行、暂停帧进入/不进入记忆、时间戳单独缩放，定位虚假进度来源。
3. 在测试时按关键事件而非时钟均匀采样记忆；仅删除时间编码不能消除全部捷径。

## 流程总览

```mermaid
flowchart LR
  A0["观察与时间戳"]
  A1["插入暂停/空抓"]
  A2["时序记忆采样"]
  A3["策略判断进度"]
  A4["事件索引对照"]
  A0 --> A1
  A1 --> A2
  A2 --> A3
  A3 --> A4
```

> 方法流程根据摘要/官网整理，尚待论文原文逐图核实。

## 实验与评测

PickXtimes 两次 5 秒暂停后 FrameSamp-Modul 从 47/50 降至 16/50；不把暂停帧写入记忆且停止时间推进时恢复为 45/50。论文摘要称另一任务 44/50→10/50。

上述结果均为作者报告或用户附件转载，**不是本次复测**。

## 结论

**按时间采样的 VLA 记忆可能把暂停当作任务进度；用零事件干预与事件索引暴露和缓解 temporal shortcuts。**

1. 在 RoboMME 计数任务中加入不改变物体或任务状态的暂停和空抓动作。
2. 对照正常运行、暂停帧进入/不进入记忆、时间戳单独缩放，定位虚假进度来源。
3. 事件检测器依赖环境任务信号；RoboMME 仿真结果不能直接代表开放世界真机。

## 与其他工作对比

比较本文与相关基线时应在相同任务、数据、样本量、传感器和部署条件下评估；不能把论文中的单个数字解释为真实部署的通用进步。待全文表格核对后补足逐项量化对照。

## 源码运行时序图

```mermaid
sequenceDiagram
  participant A0 as run_eval.sh
  participant A1 as RoboMME
  participant A2 as Memory sampler
  participant A3 as VLA policy
  A0->>A1: 启动评测/训练
  A1->>A2: 处理环境与模型输入
  A2->>A3: 执行模型或输出指标
  A3-->>A0: 返回状态
````

**README 入口：** `bash run_eval.sh --task PickXtimes --event pause --len 100 --memory visible`。图仅表示已核对到的入口与概念性组件顺序，不代表已运行或逐行审阅。

## 工程实践

- 对齐版本、传感器、动作/观测空间和评测预算后再尝试复现。
- 正式应用应增加跨分布扰动与任务安全边界检查；原文完整复现步骤待核对。
- 本次通读状态：**未完成**；页面仍为 draft。

## 局限与风险

事件检测器依赖环境任务信号；RoboMME 仿真结果不能直接代表开放世界真机。

未对全部论文正文/附录及运行细节完成审阅，任何进一步量化外推均不成立。

## 关联页面

- [vla](../methods/vla.md)
- [agent-memory-and-dreaming](../concepts/agent-memory-and-dreaming.md)

## 参考来源

- [原论文及阅读进度](../../sources/papers/memory_clock_arxiv_2611_14488.md)
- [官方源码入口](../../sources/repos/memory-clock.md)
- [用户附件 13 篇原始资讯整理](../../sources/blogs/embodied-research-digest-2026-11-12.md)

## 推荐继续阅读

- [arXiv 原文](https://arxiv.org/abs/2611.14488)
- [官方源码](https://github.com/bennidict23/memory-clock)
