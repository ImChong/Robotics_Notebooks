---
type: entity
tags: [paper, manipulation, tro-manip-survey, dexterous-manipulation, reinforcement-learning, zju, nus]
status: complete
updated: 2026-07-08
arxiv: "2602.21811"
summary: "DexRep 手物几何与空间交互表征 + DRL；40 物体训练在 5000+ 未见物体抓取达 87.9% 成功率。"
related:
  - ../overview/tro-manip-5-papers-technology-map.md
  - ../overview/tro-manip-category-02-representation.md
  - ../concepts/contact-rich-manipulation.md
  - ../queries/dexterous-manipulation-data-pipeline.md
sources:
  - ../../sources/papers/tro_manip_survey_03_dexrepnet_plus_plus.md
  - ../../sources/blogs/wechat_shenlan_tro_manip_5_papers_survey.md
  - ../../sources/papers/tro_manip_5_papers_catalog.md
---

# DexRepNet++

**DexRepNet++** 收录于 [深蓝具身智能 · T-RO 2026 操作学习精选](https://mp.weixin.qq.com/s/nswA-jCGC3kr9iQjhRRuXQ) **第 03/5** 篇，归类为 **02 三维与手物表征**。

## 一句话定义

提出 **DexRep** 手-物几何与空间交互表征，嵌入 DRL 框架学习灵巧操作策略，在极少训练物体上实现对数千未见物体的抓取泛化。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DRL | Deep Reinforcement Learning | 深度强化学习 |
| HOI | Hand-Object Interaction | 手-物交互 |
| Sim2Real | Simulation to Reality | 仿真到真机迁移 |

## 为什么重要

- 多指灵巧手 **高维动作空间 + 复杂接触** 使表征设计成为泛化瓶颈，而非仅 sample efficiency。
- DexRep 强调 **指节与物体表面的相对空间关系** 与表面几何，而非仅物体全局形状。
- 在抓取、手内重定向、双手交接三类任务上相对既有手物表征 **显著提升**。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 03/5 |
| 分组 | 02 三维与手物表征 |
| 机构 | 浙江大学（ZJU）；新加坡国立大学（NUS）等 |
| 出处 | IEEE T-RO 2026, vol. 42, pp. 799–818 · arXiv:2602.21811 |
| 论文/项目 | <https://arxiv.org/abs/2602.21811> · <https://lqts.github.io/DexRepNet2/> |

## 核心机制（归纳）

### 1）DexRep 表征

- 捕捉物体表面特征及手-物 **空间关系**。
- 分量包括 **Occupancy Feature**、**Surface Feature**（距离/法向等）、**Local-Geo Feature**（PointNet 等局部几何）。

### 2）三类灵巧任务

| 任务 | 策展亮点 |
|------|----------|
| 抓取 | 40 训练物体 → 5000+ 未见多类物体 **87.9%** 成功率 |
| 手内重定向 | 相对既有表征 **+20%–40%** |
| 双手交接 | 同上；真机多/单摄像头 **sim-to-real gap 较小** |

## 常见误区

1. DexRep 解决 **输入表征泛化**，不自动解决 **低层力控 / 触觉闭环**——须与接触控制页面对照。
2. 87.9% 等指标来自 **仿真 benchmark**；真机协议以原文为准。

## 实验与评测

- 仿真三类任务 + 真机部署（多/单摄像头）；完整指标见 PDF / 项目页。

## 与其他页面的关系

- 技术地图：[tro-manip-5-papers-technology-map.md](../overview/tro-manip-5-papers-technology-map.md)
- [Canonical Policy（同组姊妹篇）](./paper-tro-manip-02-canonical-policy.md)
- [Contact-Rich Manipulation](../concepts/contact-rich-manipulation.md)

## 参考来源

- [tro_manip_survey_03_dexrepnet_plus_plus.md](../../sources/papers/tro_manip_survey_03_dexrepnet_plus_plus.md)
- [wechat_shenlan_tro_manip_5_papers_survey.md](../../sources/blogs/wechat_shenlan_tro_manip_5_papers_survey.md)
- 论文：<https://arxiv.org/abs/2602.21811>

## 推荐继续阅读

- [灵巧操作数据管线与 RL 基建](../queries/dexterous-manipulation-data-pipeline.md)
- [T-Rex（触觉反应式灵巧 VLA）](./paper-trex-tactile-reactive-dexterous-manipulation.md)
