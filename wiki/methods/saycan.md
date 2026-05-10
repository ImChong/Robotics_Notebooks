---
type: method
tags: [llm, robotics, affordance, planning, language]
status: complete
updated: 2026-05-10
related:
  - ./vla.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md
summary: "SayCan 用语言模型生成高层子任务候选，用语义价值函数估计在环境中的可行性，将常识推理与机器人局部可达性结合。"
---

# SayCan（Do As I Can）

## 一句话定义

**SayCan**：把大型语言模型当作高层任务分解器，把学得的价值函数或成功率估计当作**物理 affordance 过滤器**，两者结合输出当前环境下可执行的指令序列。

## 主要技术路线

- **分层接口**：LLM 产出候选子技能序列；学得的价值或成功率估计裁剪「物理上可行」的一步（affordance grounding）。
- **与端到端 VLA 对照**：SayCan 仍是「规划器 + 底层策略」叙事；端到端策略见 [VLA](./vla.md)、[Robotics Transformer](./robotics-transformer-rt-series.md)；抽象层见 [Foundation Policy](../concepts/foundation-policy.md)。

## 关联页面

- [VLA](./vla.md)
- [Foundation Policy](../concepts/foundation-policy.md)

## 参考来源

- Ahn et al., *Do As I Can, Not As I Say: Grounding Language in Robotic Affordances*, https://arxiv.org/abs/2204.01691
- [ted_xiao_embodied_three_eras_primary_refs.md](../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md)
