---
type: overview
tags: [humanoid, actuator, reflected-inertia, qdd, backdrivability, category-hub]
status: complete
updated: 2026-06-02
summary: "Actuator 102 · 03 — N² 反射惯量：100:1 减速使输出端惯量放大 10000 倍；透明性光谱从谐波到 QDD，决定绊障退让与 RL 力感知。"
related:
  - ./humanoid-actuator-102-technology-map.md
  - ./humanoid-actuator-102-split-architecture.md
  - ./humanoid-hardware-101-actuation-sensing-chain.md
sources:
  - ../../sources/blogs/wechat_human_five_humanoid_actuator_102.md
  - ../../sources/raw/wechat_humanoid_actuator_102_2026-06-02.md
---

# Actuator 102 · 03：减速与反射惯量

> **图谱分类节点**：**IV 齿轮减速权衡**。

## N² 陷阱

减速比 **N** 时：力矩 ∝ **N**，反射到输出端的电机惯量 ∝ **N²**。

- **100:1** → 输出端「感觉」惯量放大 **10000×**。
- 绊到障碍时腿需 **快速退让**；惯量过大 → 像钢杆 → 冲击摧毁齿面。

## 透明性光谱（文意）

从高减速 **谐波/RV**（高力矩密度、低扭矩透明）到 **QDD / 低减速行星**（高透明、利接触感知与 RL）。

与 [Hardware 101 · 传动链](./humanoid-hardware-101-actuation-sensing-chain.md) 中 QDD vs 谐波表一致，本文从 **行走退让** 角度展开。

## 设计含义

- 减速比常取 **6:1–50:1**（按关节与架构）。
- 输出端反射惯量目标：**<0.1 kg·m²**（文内 70 kg 人形语境）。
- 低惯量转子 + 中等减速，常优于高惯量转子 + 极低减速。

## 关联页面

- [柔顺与感知](./humanoid-actuator-102-compliance-sensing.md)
- [工业陷阱](./humanoid-actuator-102-industrial-actuator-trap.md)

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| QDD | Quasi-Direct Drive | 准直驱，低减速比、高背驱动性的作动方案 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |

## 参考来源

- [wechat_human_five_humanoid_actuator_102.md](../../sources/blogs/wechat_human_five_humanoid_actuator_102.md)
- [wechat_humanoid_actuator_102_2026-06-02.md](../../sources/raw/wechat_humanoid_actuator_102_2026-06-02.md)
