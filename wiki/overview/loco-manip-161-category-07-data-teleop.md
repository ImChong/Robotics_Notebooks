---
type: overview
tags: [loco-manipulation, humanoid, category-hub, survey]
status: complete
updated: 2026-09-15
original:
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
summary: "人形 Loco-Manip 161 篇 · 07 数据采集与遥操作系统（7 篇）— 训练数据如何高效采集。"
related:
  - ./humanoid-loco-manip-161-papers-technology-map.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
  - ../../sources/papers/humanoid_loco_manip_161_catalog.md
---

# Loco-Manip 161 分类 07：数据采集与遥操作系统

> **分类页**：**07 数据采集与遥操作系统**；总地图见 [人形 Loco-Manip 161 篇技术地图](./humanoid-loco-manip-161-papers-technology-map.md)。

## 一句话定义

梳理人形全身移动操作所需的 **全身遥操作、外骨骼与无本体真机采集（UMI）** 数据闭环的 7 篇核心系统。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |
| WM | World Model | 学习环境动态以供想象/规划的世界模型 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |

## 为什么重要
训练数据如何高效采集

## 核心原理

```mermaid
flowchart LR
  SRC["人手/全身示范或感知目标"] --> IFACE["接口抽象: 上身指令 / 重定向 / 全身遥操"]
  IFACE --> POL["全身策略 / 控制执行"]
```

## 本组论文（7 篇）

| # | 工作 | Wiki 实体 |
|---|------|-----------|
| 128 | BifrostUMI | [paper-loco-manip-161-128-bifrostumi](../entities/paper-bifrost-umi.md) |
| 129 | HOMIE | [paper-loco-manip-161-040-homie](../entities/paper-loco-manip-161-040-homie.md) |
| 130 | HumanoidExo | [paper-loco-manip-161-130-humanoidexo](../entities/paper-loco-manip-161-067-humanoidexo.md) |
| 131 | Open-TeleVision | [paper-loco-manip-161-131-open-television](../entities/paper-loco-manip-161-131-open-television.md) |
| 132 | TWIST2 | [paper-loco-manip-161-132-twist2](../entities/paper-twist2.md) |
| 133 | TWIST | [paper-loco-manip-161-133-twist](../entities/paper-twist.md) |
| 134 | WT-UMI | [paper-loco-manip-161-134-wt-umi](../entities/paper-loco-manip-07-wt-umi.md) |

## 工程实践

- **工程落地要点：** 对比外骨骼/动作捕捉（TWIST/TWIST2）的延时与标定成本 vs 手持 UMI（WT-UMI/BifrostUMI）的跨形态便携性，优先围绕目标机器人手腕与视觉视野对齐采集界面。

## 常见误区或局限

- 综述目录页仅作主题归类索引；各项具体成功率、传感器型号与开源状态以各子实体页及原始论文 PDF 核验为准。

## 关联页面

- [人形 Loco-Manip 161 篇技术地图](./humanoid-loco-manip-161-papers-technology-map.md)
- [Loco-Manipulation 任务页](../tasks/loco-manipulation.md)

## 参考来源

- [wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- [humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md)

## 推荐继续阅读

- [运动小脑 64 篇技术地图](./humanoid-motion-cerebellum-technology-map.md)
