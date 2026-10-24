---
type: entity
tags: [paper, humanoid, motion-retargeting, loco-manipulation, contact-aware-control, reinforcement-learning]
status: draft
updated: 2026-10-24
project_id: str-spatiotemporal-retargeting-humanoid-loco-manipulation
arxiv: "2610.30540"
related:
  - ../concepts/motion-retargeting.md
  - ../tasks/loco-manipulation.md
  - ../methods/reinforcement-learning.md
sources:
  - ../../sources/papers/str_arxiv_2610_30540.md
summary: "STR allocates motion-retargeting effort to task-critical keyframes, body parts, and object contacts, then trains contact-aware humanoid control."
---

# STR：面向人形移动操作的时空偏好动作重定向

**一句话定义：** STR 把重定向优化重点放在任务关键帧、关键身体部位与物体接触上，再将重定向结果用于人形控制。

## 核心方法

1. **Spatiotemporal Retargeting：** VLM 选择与任务相关的动作信号和身体部位；信号变化用于定位关键帧，关键帧获得更多求解迭代，相关部位获得更高交互权重。
2. **Mechanics-Aware Contact Anchoring（MCA）：** 用近似物体 wrench 模型调整接触位置，保留指定的物体表面方向，提升几何接触一致性。
3. **Spatiotemporal Reinforcement Learning（STRL）：** 以自适应关键阶段采样和接触奖励训练下游控制器。

## 实验

论文摘要报告 Unitree G1 仿真与 Unitree R1 真机实验。STR 改善接触覆盖与连续性并降低重定向计算成本；生成的参考动作提升了多种物体交互任务的完成表现。具体数字和比较口径应以论文实验表为准。

## 工程启示

- 人体动作重定向可按任务重要性分配求解预算，避免每帧采用相同优化强度。
- 对物体交互任务，身体关键部位、接触点和接触方向都应作为显式约束核验。
- 重定向质量与下游控制要分开评估：参考轨迹接触更准确，并不自动保证真机任务成功。

## 局限与开放状态

本次核查的 arXiv 页面未列官方项目页或代码链接，因此不据此判断代码是否已开源。摘要未提供完整实验数据，量化结论需回到论文表格核对。

## 参考来源

- [STR 论文归档](../../sources/papers/str_arxiv_2610_30540.md)
- [arXiv:2610.30540](https://arxiv.org/abs/2610.30540)
