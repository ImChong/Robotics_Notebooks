---
type: entity
tags: [humanoid, teleoperation, sim2real, motion-tracking, egocentric-vision, cmu, nvidia, gear]
status: complete
updated: 2026-05-14
related:
  - ./tairan-he.md
  - ../methods/sonic-motion-tracking.md
  - ../tasks/teleoperation.md
  - ../concepts/motion-retargeting.md
  - ../concepts/sim2real.md
  - ./mimickit.md
  - ./humanoid-robot.md
sources:
  - ../../sources/sites/zhengyiluo.md
summary: "罗正宜（Zhengyi Luo）为 NVIDIA GEAR Lab 高级研究科学家、CMU RI 博士（Kris Kitani）；工作横跨人形通用低层控制、人–人形遥操作、视觉 Sim2Real 与交互感知，是 HOVER、ASAP、OmniH2O、PDC 与 SONIC 等社区关键论文的核心作者之一。"
---

# Zhengyi Luo（罗正宜）

## 一句话定义

**Zhengyi Luo** 的研究把 **人形机器人的通用低层控制** 与 **视觉–语言–动作、Sim2Real 与遥操作数据闭环** 串在同一职业轨迹上：博士阶段提出并开源 **PHC / PULSE** 等「可扩展人形控制与表示」路线，随后在 NVIDIA GEAR 与 LECAR 网络合作者一道推进 **HOVER、ASAP、VIRAL、Doorman、SONIC** 等面向真机与大规模仿真的系统论文。

## 为什么重要

- **人形学习管线枢纽作者**：与 [Tairan He](./tairan-he.md) 等在 **OmniH2O、HOVER、ASAP、SONIC、VIRAL** 上高度重叠署名，是理解 **2023–2026 通用人形控制 + 视觉迁移** 论文簇的关键人物节点。
- **开源资产多**：PHC、PULSE、SimXR、EmbodiedPose、UniversalHumanoidControl 等仓库长期被社区用作 **仿真人形 baseline 与重定向参考**（以各仓库 README 为准）。

## 核心研究脉络（归纳）

1. **通用控制与表示**：论文与代码强调在单一策略或表示下覆盖多种步态 / 技能（博士论文题目 *Learning Universal Humanoid Control* 概括该主线）。
2. **人–机共享演示**：OmniH2O、实时全身遥操作等把 **接口设计、数据规模与策略学习** 绑在一起，对应 [Teleoperation](../tasks/teleoperation.md) 任务页讨论的问题域。
3. **视觉与 Sim2Real**：PDC、VIRAL、Doorman 等工作把 **像素观测、规模化仿真与真机迁移** 推到前台，与 [Sim2Real](../concepts/sim2real.md) 概念页互补。
4. **与合作者方法的交集**：例如 [CLoSD](https://guytevet.github.io/CLoSD-page/)、[MaskedManipulator](https://xbpeng.github.io/projects/MaskedManipulator/index.html) 体现与 **物理角色控制 / MimicKit 生态** 的交叉。

## 常见误区或局限

- **「Universal」有边界**：通用控制在分布外地形、外骨骼负载或新硬件上仍需 **再训练或对齐模块**（参见 ASAP 等论文主张，以原文为准）。
- **in submission / workshop 标注**：主页出版物区部分条目状态会更新，引用前请核对 **最新会议或 arXiv 版本**。

## 关联页面

- [Tairan He（何泰然）](./tairan-he.md)
- [SONIC（规模化运动跟踪）](../methods/sonic-motion-tracking.md)
- [Teleoperation](../tasks/teleoperation.md)
- [Motion Retargeting](../concepts/motion-retargeting.md)
- [Sim2Real](../concepts/sim2real.md)
- [MimicKit](./mimickit.md)
- [人形机器人](./humanoid-robot.md)

## 参考来源

- [Zhengyi Luo 个人主页原始资料](../../sources/sites/zhengyiluo.md)

## 推荐继续阅读

- [HOVER 项目页](https://hover-versatile-humanoid.github.io/)
- [ASAP 项目页](https://agile.human2humanoid.com/)
