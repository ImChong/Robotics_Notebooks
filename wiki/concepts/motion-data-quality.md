---
title: Motion Data Quality（动作数据质量维度）
type: concept
status: complete
created: 2026-06-19
updated: 2026-11-07
summary: 评估人形参考运动 / 演示数据可用性的四个轴——物理可行性、接触一致性、形态差距（morphology gap）、规模与多样性，及其与重定向必要性的因果关系。
tags: [dataset, motion-retargeting, data-quality, humanoid]
sources:
  - ../../sources/papers/phuma_arxiv_2510_26236.md
  - ../../sources/papers/gmr_retargeting_matters_arxiv_2510_02252.md
  - ../../sources/papers/neural_motion_retargeting_nmr.md
  - ../../sources/sites/amass-dataset.md
  - ../../sources/repos/phuma.md
  - ../../sources/repos/omomo_release.md
  - ../../sources/sites/humanoideveryday.md
  - ../../sources/repos/exercises-dataset.md
original:
  - ../../sources/papers/phuma_arxiv_2510_26236.md
  - ../../sources/papers/gmr_retargeting_matters_arxiv_2510_02252.md
  - ../../sources/papers/neural_motion_retargeting_nmr.md
related:
  - ./motion-retargeting.md
  - ../comparisons/humanoid-reference-motion-datasets.md
  - ../queries/humanoid-training-data-pipeline.md
  - ../entities/exercises-dataset.md
---

# Motion Data Quality（动作数据质量维度）

## 一句话定义

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

## 为什么重要

- **数据质量决定策略上限**：PHUMA 的数据集对比实验（G1 与 H1-2）表明，规模与质量单独都不够——Humanoid-X 规模最大（231.4 h，其中可行 69.1 h）却因悬空 / 穿透 / 滑动等质量问题落后，LaFAN1 与 AMASS 干净但覆盖不足；两者兼得的 PHUMA（73.0 h）在所有运动类别上成功率最高，真机跟踪误差比 AMASS 训练的低 16.3%。
- **伪影会传到策略**：GMR 的受控实验显示，地面穿透、自穿透、关节值突变这三类重定向伪影会显著降低跟踪策略鲁棒性。
- **体检是便宜的、训练是昂贵的**：四轴体检几乎只需运动学计算，能在花 GPU 时间之前剔除不可用数据。

---

## 核心原理：四个评估轴

### 1. 物理可行性（Physical Feasibility）

源数据中的运动是否满足目标机体的动力学约束：质心加速度、力矩上限、摩擦锥、零力矩点（ZMP/DCM）是否在支撑域内。

- **典型病灶**：纯光学 MoCap（如 [AMASS](../entities/amass.md)）只记录**运动学**轨迹，**不含力 / 接触信息**；播放给机器人会出现穿地、滑步、需要不存在的力矩。
- **判据**：在物理仿真里用 PD/WBC 跟踪能否不摔倒、力矩是否饱和；或用 PHUMA 论文里的"满足条件帧百分比"指标——关节位置 / 速度在机械极限 98% 以内（Joint Feasibility）、接触时脚高于地面不超过 1 cm（Non-Floating）、不低于地面 1 cm（Non-Penetration）、接触时水平速度 <10 cm/s（Non-Skating）。
- **补救**：物理一致化层（QP / RL fine-tune），或直接选用**已过滤**的数据集（[PHUMA](../entities/dataset-bfm-phuma.md) 的筛选 + PhySINK 管线即专门做这一步）。注意 PHUMA 的"物理可靠"是**运动学层**的（关节极限、悬空、穿透、滑动），不是完整动力学验证，论文自述"没有完全消除"违规。

### 2. 接触一致性（Contact Consistency）

「哪只脚 / 哪只手何时接触」的相位标注是否与轨迹自洽。

- **典型病灶**：接触相位缺失或与足端高度矛盾 → 重定向后机器人飞起或穿地；人–物交互（[OMOMO](../entities/omomo-dataset.md)）中手与物体的接触错配会让 loco-manipulation 策略学到错误抓握时机；视频估计的动作常因全局平移估计错误出现悬空 / 穿地（PHUMA）。
- **判据**：足端 / 手端在标注接触段的法向距离与速度是否近零。PHUMA 的做法：对脚部网格顶点高度做多数投票估计全局地面平面，按脚跟 / 脚尖计算接触得分，丢弃接触不足（悬空）的 4 秒片段。
- **关系**：接触一致性是物理可行性的**前置条件**——接触错了，再多力矩修正也救不回来。

### 3. 形态差距（Morphology Gap）

源骨架与目标机器人在拓扑、肢体比例、关节限位、DoF 上的差距。

- **典型病灶**：人体（SMPL 一类骨架，约 24 个关节含根，推测未在本次来源逐项核对）有连续的手指与多自由度肩髋；Unitree G1 本体为 29 DoF（腿 2×6 + 腰 3 + 臂 2×7），带灵巧手版本为 43 DoF（GMR README 的支持机器人表）→ 直接关节映射会越限、末端轨迹偏差大。
- **判据**：源/目标关节对应关系是否唯一、ROM 重叠率、末端可达性。
- **关系**：形态差距越大，**重定向越不可省略**；差距小到可忽略时（如已是同型机器人执行数据 [Humanoid Everyday](../entities/humanoid-everyday-dataset.md)）才可跳过重定向直接做 IL。

### 4. 规模与多样性（Scale & Diversity）

数据量、动作种类、地形/物体/速度覆盖是否支撑泛化。

- **典型病灶**：小集（如 [LaFAN1](../entities/lafan1-dataset.md)，约 4.6 h 为库内归档数字，本次未在原文复核；PHUMA 称其"只有几小时"）做基准/recovery 原型可以，但单独支撑通用策略易过拟合；AMASS 虽统一了 15 个光学标记点 mocap 数据集（>40 h、>300 名受试者、>11,000 段动作，据 AMASS 摘要），PHUMA 仍评价其"以行走为主、多样性受限"；人体视频规模大、却常**标注/3D 信息弱**，需上游重建补足。
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

## 工程实践：四轴体检流程

1. **形态差距**：列出源骨架到目标机器人的关节对应与 ROM 重叠；差距大就先走重定向（GMR 等，见 [GMR vs NMR vs ReActor](../comparisons/gmr-vs-nmr-vs-reactor.md)）。
2. **接触**：估计地面平面并计算每帧接触得分（PHUMA 做法）；接触不足即丢弃该片段，而不是整段序列（4 秒切片，保留有效片段）。
3. **物理可行性**：先过运动学层指标（关节极限 98%、悬空 / 穿透 1 cm、滑动 10 cm/s——PHUMA；关节单步跳变、自碰帧占比、关节极限帧占比——NMR 的口径），再过 jerk 与 CoM–支撑基稳定性筛选；NMR 的 CEPR 把这些罚项硬阈值化（自相交帧占比 >0.05 或最低脚点平均离地 >0.10 m 的序列被丢弃）。
4. **规模多样性**：统计动作类别、时长与目标任务分布；PHUMA 报告"角运动 / 垂直 / 水平 / 静止"四类的成功率差异，提示数据缺口。
5. **验证**：用同一跟踪器分别训练，比较成功率与跟踪误差；注意起始帧会显著影响结果（GMR 论文），评测时固定起始位姿。

## 与数据集选型的对照

| 质量轴 | AMASS | LaFAN1 | OMOMO | PHUMA | Humanoid Everyday |
|--------|-------|--------|-------|-------|-------------------|
| 物理可行性 | 弱（纯运动学） | 弱 | 弱 | **较强（运动学层已过滤，非完整动力学验证）** | 强（真机执行） |
| 接触一致性 | 中（需补相位） | 中 | **需对齐人–物** | 强 | 强 |
| 形态差距 | 大（人→机器人） | 大 | 大 | **已消除（G1/H1-2）** | 已消除 |
| 规模多样性 | **强** | 弱 | 中（HOI 专） | 强 | 中（操作专） |

> 详见 [人形参考运动与操作数据集选型](../comparisons/humanoid-reference-motion-datasets.md)。

---

## 局限与常见误区

1. **「数据越多越好」**：规模不能补物理可行性，大而不可行的库会系统性污染策略。
2. **「重定向后就物理可行了」**：几何重定向只缩小形态差距，接触与力矩仍需动力学一致化层。
3. **「人体视频规模碾压 MoCap」**：视频量大但 3D / 接触信息弱，需上游重建，否则接触一致性轴塌缩。
4. **「真机数据无需体检」**：真机执行数据天然物理可行，但仍受任务分布偏窄（规模多样性轴）限制。
5. **「健身 exercise dataset ≈ 参考运动」**：[Exercises Dataset](../entities/exercises-dataset.md) 一类目录只有标签与 GIF，**没有**可执行状态轨迹——四轴中物理/接触/形态差距均不可评估为「通过」。

---

## 参考来源

- [PHUMA 论文通读归档](../../sources/papers/phuma_arxiv_2510_26236.md) — arXiv:2510.26236，物理感知筛选、PhySINK 与数据集对比
- [GMR（Retargeting Matters）通读归档](../../sources/papers/gmr_retargeting_matters_arxiv_2510_02252.md) — 重定向伪影对策略的影响
- [NMR 通读归档](../../sources/papers/neural_motion_retargeting_nmr.md) — CEPR 的硬阈值过滤
- 数据集条目以下列库内归档为准（二手归档，**未列入 original**，数字未逐一回到原文复核）：
- [AMASS 站点归档](../../sources/sites/amass-dataset.md)
- [PHUMA 仓库归档](../../sources/repos/phuma.md)
- [OMOMO 仓库归档](../../sources/repos/omomo_release.md)
- [Humanoid Everyday 项目页归档](../../sources/sites/humanoideveryday.md)
- [Exercises Dataset 仓库归档](../../sources/repos/exercises-dataset.md) — 非轨迹目录的反例

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
- [Exercises Dataset](../entities/exercises-dataset.md) — 健身目录/GIF；过不了四轴体检的对照
## 一句话记忆

> **先看形态差距决定要不要重定向，再过接触与物理两道可行性门，最后才谈规模多样性——四轴串联，缺一轴则下游策略「姿态像而执行废」。**

## 推荐继续阅读

- Lee et al., *PHUMA: Physically Reliable Humanoid Locomotion Dataset*：<https://arxiv.org/abs/2510.26236>
- Mahmood et al., *AMASS: Archive of Motion Capture as Surface Shapes*：<https://arxiv.org/abs/1904.03278>
- Araujo et al., *Retargeting Matters*：<https://arxiv.org/abs/2510.02252>
