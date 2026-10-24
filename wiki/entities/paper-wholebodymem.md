---
type: entity
tags: [paper, humanoid, VLA, object-memory, loco-manipulation, keyframe-memory]
status: draft
updated: 2026-10-24
project_id: wholebodymem-prehistory-memory-humanoid-loco-manipulation
arxiv: "2610.31166"
project: https://memory-before-goal.github.io/
related:
  - ../methods/vla.md
  - ../tasks/loco-manipulation.md
sources:
  - ../../sources/papers/wholebodymem_arxiv_2610_31166.md
  - ../../sources/sites/wholebodymem.md
summary: "WholeBodyMem equips humanoid VLA policies with object-guided keyframe memory of pre-goal events for dynamic loco-manipulation."
---

# WholeBodyMem：人形移动操作的目标前历史记忆

**一句话定义：** WholeBodyMem 让人形 VLA 保留执行目标前观察到的物体事件，减少遮挡、物体变化和视角转动造成的历史线索丢失。

## 核心方法

SAM3 跟踪任务物体，并由物体事件从历史观察中自适应选择关键帧。动作专家通过交叉注意力读取关键帧和物体 token；token 编码目标物体的外观、几何与时间信息。论文称其目标是记住与当前任务有关的关键历史，而不是简单堆叠更多帧。

## HuMeGo 基准与结果

HuMeGo 包含六项仿真和六项真实 Unitree G1 任务。论文摘要报告，WholeBodyMem 将基础策略成功率从仿真 29.1% 提升至 72.6%，真实机器人从 36.7% 提升至 88.3%；SIMPLE 标准任务从 70.2% 提升至 86.9%。比较结果以论文所定义的任务和评测口径为准。

## 方法栈

`相机历史 → SAM3 物体跟踪与事件筛选 → 关键帧 + 物体 token → 动作专家交叉注意力 → 全身动作`

## 工程启示

- 记忆检索可由目标物体事件触发，保留的帧数不必与历史长度成正比。
- 物体 token 同时带入外观、几何和时间信息，帮助策略区分“看到过什么”与“何时发生”。
- 应分别评估物体跟踪、关键帧选择、记忆读取与最终任务成功率，定位记忆模块的失败来源。

## 参考来源

- [WholeBodyMem 论文归档](../../sources/papers/wholebodymem_arxiv_2610_31166.md)
- [WholeBodyMem 项目页归档](../../sources/sites/wholebodymem.md)
- [arXiv:2610.31166](https://arxiv.org/abs/2610.31166)
- [官方项目页](https://memory-before-goal.github.io/)
