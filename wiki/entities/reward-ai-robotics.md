---
type: entity
tags: [company, embodied-foundation-model, manipulation, dexterous-manipulation, wearable-mocap, foundation-policy, cross-embodiment, reward-ai]
title: Reward AI（机器人）
status: complete
summary: "Reward AI 是宣称以 Omnibody 全栈（可穿戴 Hand、统一数据接口、OM-1 通才策略、跨本体高频控制）从人类自然操作学习机器人智能的商业团队；公开材料以 2026-09 OM-1 博客为主，前序学术关联 DexCap，确认未开源。"
updated: 2026-09-15
related:
  - ./reward-ai-om1.md
  - ./paper-notebook-dexcap-scalable-and-portable-mocap-data-collecti.md
  - ./generalist-ai-robotics.md
  - ./skild-ai.md
  - ../concepts/foundation-policy.md
  - ../overview/hub-cross-embodiment.md
sources:
  - ../../sources/blogs/rewardai_om1.md
  - ../../sources/sites/rewardai.md
---

# Reward AI（机器人方向）

## 一句话定义

**Reward AI**：聚焦 **人类同速灵巧操作** 的商业实体；对外叙事以 **Omnibody** 全栈为核心——可穿戴采集、统一多模态数据接口、**OM-1** 通才策略与跨工业臂/人形的控制层，强调 **仅人类示范、无遥操作与机上数据** 的学习路径。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| OM-1 | Omnibody Model 1 | 公司 2026-09 发布的通才操作策略 |
| EFM | Embodied Foundation Model | 具身基础模型；与 OM-1 产业定位同族 |
| VI | Visual-Inertial | 手部位姿跟踪常用基线 |
| EM | Electromagnetic Sensing | Omnibody 栈用于补高速跟踪的电磁传感 |
| IL | Imitation Learning | 人类示范驱动学习 |

## 为什么重要

- **DexCap 团队商业延续：** 公开材料将 OM-1 接在 [DexCap](./paper-notebook-dexcap-scalable-and-portable-mocap-data-collecti.md)（Stanford RSS 2024）之后，代表「学术可穿戴 mocap → 产业通才栈」的一条路径。
- **人类数据哲学：** 与遥操作/机上 RL 数据飞轮并列的第三种叙事——**自然人操作即训练集**，并对齐 **1x human speed** 产品标语。
- **跨本体产业样本：** 与 [Generalist AI](./generalist-ai-robotics.md)、[Skild AI](./skild-ai.md) 等同属闭源通才策略对照栏，机制各异（见 [OM-1](./reward-ai-om1.md) 对比表）。

## 公开产品线脉络（博客）

| 节点 | 要点 | 入口 |
|------|------|------|
| **DexCap**（2024，学术） | 可穿戴 mocap + DexIL；团队前序 | [DexCap 项目页](https://dex-cap.github.io/) |
| **OM-1 / Omnibody**（2026-09） | Hand + Data Interface + OM-1 + Control；<30 min 新任务数据 | [本库 OM-1 实体](./reward-ai-om1.md) |

## 数据与就绪度

- **采集：** Omnibody Hand（7-DoF）+ 触觉/接近/in-hand 视觉 + VI/EM 位姿与力。
- **训练：** 宣称单阶段、仅人类数据；具体架构与数据规模 **未公开**。
- **开源：** 截至 2026-09-15，公司站 **未见** 代码/权重入口；**勿与** [OpenMind OM1](https://github.com/OpenMind/OM1) **混淆**。

## 工程实践

| 场景 | 建议 |
|------|------|
| 写综述 / 数据采集轴 | 与 DexCap、UMI、AnyTeleop 等并列讨论 **可穿戴人类示范** |
| 跨本体研究 | 借鉴「统一策略接口 + 独立控制层」分层，用自有开源栈验证 |
| 产线选型 | **不要**假设可下载 OM-1；评估需直接对接厂商 |

## 局限与风险

- **闭源：** 无法复现 <30 分钟适应等主张。
- **证据形态：** 目前主要为单篇技术博客 + 视频，无 peer-reviewed 基准表。
- **命名冲突：** 检索「OM-1」须区分 Reward AI 策略 vs OpenMind runtime。

## 关联页面

- [OM-1 通才操作策略](./reward-ai-om1.md)
- [DexCap（前序）](./paper-notebook-dexcap-scalable-and-portable-mocap-data-collecti.md)
- [Generalist AI（公司）](./generalist-ai-robotics.md)
- [Skild AI（公司）](./skild-ai.md)
- [跨具身迁移](../overview/hub-cross-embodiment.md)
- [Foundation Policy](../concepts/foundation-policy.md)

## 参考来源

- [OM-1 博客来源归档](../../sources/blogs/rewardai_om1.md)
- [Reward AI 公司站归档](../../sources/sites/rewardai.md)

## 推荐继续阅读

- <https://www.rewardai.com/blog/OM-1/> — 官方技术叙事
- [DexCap: Scalable and Portable Mocap…](https://dex-cap.github.io/) — 团队前序开源学术线
