---
title: Motion Data Quality（动作数据质量维度）
type: concept
status: complete
created: 2026-06-19
updated: 2026-06-19
summary: 评估人形参考运动 / 演示数据可用性的四个轴——物理可行性、接触一致性、形态差距（morphology gap）、规模与多样性，及其与重定向必要性的因果关系。
tags: [dataset, motion-retargeting, data-quality, humanoid]
related:
  - ./motion-retargeting.md
  - ../comparisons/humanoid-reference-motion-datasets.md
  - ../queries/humanoid-training-data-pipeline.md
---

# Motion Data Quality（动作数据质量维度）

## 是什么

把一段参考运动 / 演示数据「能不能直接喂给人形策略训练」拆成**四个可独立评估的质量轴**。它回答的不是「数据多不多」，而是「数据**像不像**机器人能物理执行的东西」——这正是 [Motion Retargeting](./motion-retargeting.md) 是否可省略的判据。

> **一句话**：动作数据质量 = 物理可行性 × 接触一致性 ×（1 − 形态差距）× 规模多样性；任意一轴塌缩，下游 RL/IL 都会以「姿态像但执行不了」的方式失败。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MoCap | Motion Capture | 动作捕捉，参考运动主要来源 |
| Morphology Gap | Morphology Gap | 源（人/动物）与目标机器人在骨架/比例/动力学上的差距 |
| SMPL | Skinned Multi-Person Linear Model | 常见人体参数化模型与重定向源 |
| GRF | Ground Reaction Force | 地面反作用力，接触可行性核心量 |
| WBT | Whole-Body Tracking | 全身跟踪训练，重定向产物的主要消费侧 |
| IL | Imitation Learning | 模仿学习，演示数据的主要消费侧 |

---

## 四个评估轴

### 1. 物理可行性（Physical Feasibility）

源数据中的运动是否满足目标机体的动力学约束：质心加速度、力矩上限、摩擦锥、零力矩点（ZMP/DCM）是否在支撑域内。

- **典型病灶**：纯光学 MoCap（如 [AMASS](../entities/amass.md)）只记录**运动学**轨迹，**不含力 / 接触信息**；播放给机器人会出现穿地、滑步、需要不存在的力矩。
- **判据**：在物理仿真里用 PD/WBC 跟踪能否不摔倒、力矩是否饱和。
- **补救**：物理一致化层（QP / RL fine-tune），或直接选用**已过滤**的数据集（[PHUMA](../entities/dataset-bfm-phuma.md) 的 PhySINK 管线即专门做这一步）。

### 2. 接触一致性（Contact Consistency）

「哪只脚 / 哪只手何时接触」的相位标注是否与轨迹自洽。

- **典型病灶**：接触相位缺失或与足端高度矛盾 → 重定向后机器人飞起或穿地；人–物交互（[OMOMO](../entities/omomo-dataset.md)）中手与物体的接触错配会让 loco-manipulation 策略学到错误抓握时机。
- **判据**：足端 / 手端在标注接触段的法向距离与速度是否近零。
- **关系**：接触一致性是物理可行性的**前置条件**——接触错了，再多力矩修正也救不回来。

### 3. 形态差距（Morphology Gap）

源骨架与目标机器人在拓扑、肢体比例、关节限位、DoF 上的差距。

- **典型病灶**：人有 ~23 主关节、连续手指；G1 为 43 DoF 且无手指 → 直接关节映射会越限、末端轨迹偏差大。
- **判据**：源/目标关节对应关系是否唯一、ROM 重叠率、末端可达性。
- **关系**：形态差距越大，**重定向越不可省略**；差距小到可忽略时（如已是同型机器人执行数据 [Humanoid Everyday](../entities/humanoid-everyday-dataset.md)）才可跳过重定向直接做 IL。

### 4. 规模与多样性（Scale & Diversity）

数据量、动作种类、地形/物体/速度覆盖是否支撑泛化。

- **典型病灶**：小集（如 [LaFAN1](../entities/lafan1-dataset.md) ~4.6 h）做基准/recovery 原型可以，但单独支撑通用策略易过拟合；人体视频规模大、却常**标注/3D 信息弱**，需上游重建补足。
- **判据**：动作类目覆盖、长尾占比、与目标任务分布的重叠。
- **关系**：规模可放大其余三轴的收益，但**不能替代**它们——大而不可行的数据只会更系统地污染策略（参见 [Embodied Scaling Laws](./embodied-scaling-laws.md)）。

---

## 四轴与重定向必要性的因果链

```text
形态差距大 ──► 必须重定向（几何映射）
接触不一致 ──► 重定向后仍不可执行 ──► 需接触修复 + 物理一致化
物理不可行 ──► 物理过滤 / RL fine-tune（PhySINK、ReActor）
规模/多样性不足 ──► 上游补数据（视频重建、生成式增广）或降低任务野心
```

四轴并非独立打分相加，而是**串联的门**：前一轴不过，后一轴的投入大多浪费。因此选型时按 **形态差距 → 接触 → 物理 → 规模** 的顺序体检，比单看「数据有多大」更能预测训练成败。

---

## 与数据集选型的对照

| 质量轴 | AMASS | LaFAN1 | OMOMO | PHUMA | Humanoid Everyday |
|--------|-------|--------|-------|-------|-------------------|
| 物理可行性 | 弱（纯运动学） | 弱 | 弱 | **强（已过滤）** | 强（真机执行） |
| 接触一致性 | 中（需补相位） | 中 | **需对齐人–物** | 强 | 强 |
| 形态差距 | 大（人→机器人） | 大 | 大 | **已消除（G1/H1-2）** | 已消除 |
| 规模多样性 | **强** | 弱 | 中（HOI 专） | 强 | 中（操作专） |

> 详见 [人形参考运动与操作数据集选型](../comparisons/humanoid-reference-motion-datasets.md)。

---

## 常见误区

1. **「数据越多越好」**：规模不能补物理可行性，大而不可行的库会系统性污染策略。
2. **「重定向后就物理可行了」**：几何重定向只缩小形态差距，接触与力矩仍需动力学一致化层。
3. **「人体视频规模碾压 MoCap」**：视频量大但 3D / 接触信息弱，需上游重建，否则接触一致性轴塌缩。
4. **「真机数据无需体检」**：真机执行数据天然物理可行，但仍受任务分布偏窄（规模多样性轴）限制。

---

## 参考来源

- [AMASS 站点归档](../../sources/sites/amass-dataset.md)
- [PHUMA 仓库归档](../../sources/repos/phuma.md)
- [OMOMO 仓库归档](../../sources/repos/omomo_release.md)
- [Humanoid Everyday 项目页归档](../../sources/sites/humanoideveryday.md)

---

## 关联页面

- [Motion Retargeting](./motion-retargeting.md) — 形态差距驱动的几何映射层，本页四轴的主要「修复手段」入口
- [Motion Retargeting Pipeline](./motion-retargeting-pipeline.md) — 源归一 → 骨架对齐 → IK → 物理筛选的端到端工程链路
- [人形参考运动与操作数据集选型](../comparisons/humanoid-reference-motion-datasets.md) — 五套数据集在四质量轴上的对照
- [人形训练数据管线选型指南](../queries/humanoid-training-data-pipeline.md) — 从原始来源到训练输入的端到端决策树（本页是其质量评估子模块）
- [Whole-Body Tracking Pipeline](./whole-body-tracking-pipeline.md) — 重定向产物作训练数据的中段消费侧
- [Imitation Learning](../methods/imitation-learning.md) — 形态差距可忽略时直接消费演示数据的范式
- [Embodied Scaling Laws](./embodied-scaling-laws.md) — 规模多样性轴的放大效应与边界
- [Embodied Data Cleaning](./embodied-data-cleaning.md) — 失败/低质轨迹过滤的工程实践

## 一句话记忆

> **先看形态差距决定要不要重定向，再过接触与物理两道可行性门，最后才谈规模多样性——四轴串联，缺一轴则下游策略「姿态像而执行废」。**
