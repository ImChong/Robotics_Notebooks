---
type: entity
tags: [hardware, dexterity, manipulation, robot-hand, industry]
status: complete
updated: 2026-04-21
related:
  - ./allegro-hand.md
  - ../tasks/manipulation.md
  - ../methods/in-hand-reorientation.md
sources:
  - ../../sources/papers/humanoid_hardware.md
summary: "Shadow Hand 是一款仿生程度极高的五指灵巧手，拥有 20 个主动驱动自由度，是研究拟人化灵巧操作与复杂手内重定向的标杆性硬件。"
---

# Shadow Hand (灵巧手)

**Shadow Hand** 由英国 Shadow Robot Company 开发，是目前世界上最接近人类手部功能的灵巧手平台之一。它拥有 5 根手指和 20 个主动驱动关节（总计 24 个自由度），常被用于具身智能、远程手术操作和极限灵巧性的研究。

## 硬件特性

1. **拟人化设计**：完全遵循人类手的比例与关节分布，甚至包含指间关节的耦合关系。
2. **高自由度**：20 个主动关节，使得它能完成扣纽扣、玩魔方、打字等人类级别的复杂动作。
3. **驱动方式**：
   - **气动版**：利用气动人工肌肉，具有天然的柔韧性。
   - **电动版**：采用电机加绳索传动（Tendons），响应更精确。
4. **全集成传感器**：每个关节都有高精度编码器，指尖可集成高度敏感的力/触觉传感器。

## 为什么在科研中重要

- **极致测试**：如果一个算法能在 Shadow Hand 上跑通，通常能证明该算法在高维动作空间中的可扩展性。
- **强化学习标杆**：OpenAI 著名的“单手解魔方”项目即是基于 Shadow Hand 完成，证明了 [域随机化](../concepts/sim2real.md) 对极高维度硬件的有效性。

## 关联页面
- [Allegro Hand 实体](./allegro-hand.md) — 更轻量化的 4 指备选方案
- [Manipulation 任务](../tasks/manipulation.md)
- [手内重定向 (In-hand Reorientation)](../methods/in-hand-reorientation.md)

## 参考来源
- Shadow Robot Company Official Website.
- OpenAI, et al. (2019). *Solving Rubik’s Cube with a Robot Hand*.
