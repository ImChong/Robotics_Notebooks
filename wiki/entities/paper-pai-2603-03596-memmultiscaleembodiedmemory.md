---
type: entity
tags:
- paper
- pi
- vla
- memory
status: complete
updated: '2026-10-05'
arxiv: '2603.03596'
summary: MEM 用短期视频记忆保留局部动态，用长期语言记忆保留任务事件，在 VLA 延迟预算内支持分钟级长程操作。
related:
- ../entities/awesome-physical-ai-natnew.md
- ../overview/awesome-physical-ai-technology-map.md
- ../methods/vla.md
- ../tasks/manipulation.md
- ./paper-rcl-2511-14759-0-6-a-vla-that-learns-from-experience.md
- ./paper-hi-robot.md
sources:
- ../../sources/papers/pai_awesome_2603_03596_mem-multi-scale-embodied-memory.md
- ../../sources/repos/awesome-physical-ai-union-catalog.md
- ../../sources/repos/awesome-physical-ai-natnew.md
- ../../sources/repos/awesome-physical-ai-aichr.md
- ../../sources/sites/pi-memory-rlt-fast.md
---

# MEM：多尺度具身记忆

## 一句话定义

MEM 用短期视频记忆保留局部动态，用长期语言记忆保留任务事件，在 VLA 延迟预算内支持分钟级长程操作。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
| --- | --- | --- |
| VLA | Vision-Language-Action | 视觉和语言条件下生成动作 |
| MEM | Multi-Scale Embodied Memory | 秒级视觉与分钟级语言记忆 |
| VLM | Vision-Language Model | 高层语义理解与记忆更新模型 |

## 为什么重要

- 当前观测无法回答“此前做过什么”和“物体刚才怎么动”；两类记忆需要不同分辨率。
- 把所有历史帧塞进模型会增加算力与时延，MEM 用不同尺度的表示控制成本。

## 核心原理

| 记忆 | 机制 | 主要信息 |
| --- | --- | --- |
| 短期 | 高效视频编码器保留数秒视觉历史 | 动态、遮挡、抓取滑动与现场适应 |
| 长期 | 高层策略更新自然语言记忆，条件化后续策略 | 已完成步骤、失败事件、跨分钟任务状态 |

论文将系统实例化为 **π₀.₆-MEM**。高层更新语义记忆，低层同时消费近期观测与记忆条件；语言压缩降低历史成本，但不保留精确接触几何，因此不能替代短期视频。

## 源码运行时序图

**不适用**：本次可读原始论文，但官网项目页访问受限，未确认官方可运行训练/推理实现；这里不以 openpi 的存在推定 MEM 已开放。

## 工程实践

1. 用需要历史的任务评估：隐藏信息、重复子任务、已完成步骤确认。
2. 比较无记忆、仅视频、仅语言与双尺度四种设置；保持任务/数据量一致。
3. 同时记录任务完成率、记忆更新错误、推理时延和历史长度。
4. 原始论文可读；本次官网返回 403，完整官方训练/部署代码 **未确认**。

## 评测与指标

论文报告厨房整理、烹饪等最长约 **15 分钟**任务。读实验时区分短期记忆改善局部操作、长期记忆改善子任务状态，以及模型/数据变化的收益；每项对比须绑定论文相应设置。

## 结论

**MEM 的关键是用两种表示分别保存局部动态和长期语义。**

1. 用确实需要历史信息的任务测记忆价值。
2. 逐尺度消融，避免把额外数据收益归到记忆结构。
3. 部署时检查记忆更新可靠性与延迟预算。

## 与其他工作对比

[Hi Robot](paper-hi-robot.md) 重点是语言指令分层和现场纠正；MEM 重点是历史状态的保留。仅靠高层拆解任务不足以证明记住已执行步骤，比较时要加入需历史信息的任务与逐尺度记忆消融。

## 局限与风险

- 长期语言记忆可能遗漏或误记事件；失误会持续影响后续动作。
- “15 分钟任务”是论文中的评测范围，不代表任何任务均可无限记忆。
- 当前未确认可运行官方实现，不把论文结构图当作源码运行时序图。

## 关联页面

- [π₀.₆](./paper-rcl-2511-14759-0-6-a-vla-that-learns-from-experience.md)
- [Hi Robot](./paper-hi-robot.md)
- [VLA](../methods/vla.md)

## 参考来源

- [PI 一手资料补核](../../sources/sites/pi-memory-rlt-fast.md)

## 推荐继续阅读

- [MEM 论文](https://arxiv.org/abs/2603.03596)
