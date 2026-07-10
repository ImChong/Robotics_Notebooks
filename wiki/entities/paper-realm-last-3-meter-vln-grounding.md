---
type: entity
tags:
  - paper
  - vln
  - reverie
  - object-grounding
  - evaluation
  - plug-and-play
  - vision-language-action
  - adelaide
  - fudan
  - epfl
  - ruyi-dynamics
status: complete
updated: 2026-07-10
arxiv: "2607.03792"
summary: "REALM（arXiv:2607.03792）：揭示 REVERIE-CE 等连续 VLN 在 3 m 区域到达与实例可见接地之间的 Last-3-Meter Grounding Gap；提出可插拔末段精修模块 REALM、实例指标 ONS/GS/OracleGS 与 REVERIE-AIM 数据集，在四类 VLN 骨干上一致提升细粒度接地并验证真机 Stretch。"
related:
  - ../tasks/vision-language-navigation.md
  - ../entities/paper-vln-03-reverie.md
  - ../entities/paper-vln-09-etpnav.md
  - ../overview/vln-open-source-repro-paradigms.md
  - ../overview/vln-10-papers-technology-map.md
  - ../concepts/3d-spatial-vqa.md
sources:
  - ../../sources/papers/realm_last_3_meter_vln_arxiv_2607_03792.md
---

# REALM（Last-3-Meter VLN · 实例级接地）

**REALM**（*From Region Arrival to Instance-Level Grounding in Vision-and-Language Navigation*，arXiv:2607.03792）系统分析连续 **视觉–语言导航** 在 **REVERIE / REVERIE-CE** 设定下的评测盲区：智能体可在 **3 m 预标注终点** 附近停车却仍 **背对或看不见** 被指实例。作者将此称为 **Last-3-Meter Grounding Gap**，并给出 **可插拔末段精修模块 REALM**、**实例中心指标** 与 **REVERIE-AIM** 训练/评测集。

## 一句话定义

**把长视界 VLN 与末段「走近并对准可见实例」解耦**：上游任意导航策略负责区域到达，**REALM** 在 stop 后仅用 RGB 与原文指令做 **短视界精修 + 开放词汇接地**，弥合区域到达与实例级感知之间的最后数米鸿沟。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLN | Vision-and-Language Navigation | 依据自然语言指令在环境中导航的具身任务 |
| VLN-CE | VLN in Continuous Environments | 无离散导航图的连续动作空间 VLN |
| ONS | Object Navigation Success | 停车点到目标实例周围可导航点的最小测地距离成功率 |
| GS | Grounding Success | 最终视角检测框与 GT 的 IoU≥0.5 |
| VSP | Visibility-Aware Stop Penalty | 目标不可见时惩罚过早 stop 的训练项 |
| LoRA | Low-Rank Adaptation | 低秩适配，用于精修策略高效微调 |

## 为什么重要

- **评测与任务目标错位：** REVERIE 要求导航后 **定位并框出目标实例**，但主流 VLN-CE 成功判据仍主要看 **与区域 waypoint 的 3 m 距离**，忽略 **朝向与可见性**——高 SR 不等于可操作的感知接口。
- **量化鸿沟：** 在 REVERIE-AIM 上，ETPNav-FT SR=34.67% 时 ONS@0.1m 仅 6.32%，说明多数「成功」未真正贴近实例；Human 上界 ONS@0.5m 达 74.71%，改进空间巨大。
- **架构无关的末段模块：** REALM 不改上游 **ETPNav / UniNaVid / SmartWay** 等范式，适合快速叠加到现有 VLN 栈——与 [VLN 四范式复现路径](../overview/vln-open-source-repro-paradigms.md) 中 **UniNaVid** 路线直接相关（精修骨干即 UniNaVid + LoRA）。
- **真机信号：** Hello Robot Stretch 上 ONS@0.5m 8.33%→33.33%，表明末段 **可见性感知停车** 对物理噪声与分布偏移有实际价值。

## 核心结构

| 模块 | 作用 |
|------|------|
| **π_nav（上游 VLN）** | 长视界路径规划与场景推理；在 s_T_nav 发出 stop |
| **π_ref（REALM 精修）** | UniNaVid 视频–语言–动作骨干 + LoRA；短视界 `{forward, turn_left, turn_right, stop}` 重定位到 **近且可见** 的 s_T_ref |
| **VSP 损失** | 目标不可见时惩罚 stop margin，抑制过早终止 |
| **短语抽取** | BERT-Large 抽取式 QA 从完整指令剥离目标名词（如「去浴室清洁马桶圈」→「马桶圈」） |
| **OWLv2 接地** | 开放词汇检测输出最终边界框 |
| **REVERIE-AIM** | 连续轨迹重建 + ObjectNav 式 **实例中心终点** + 180K 末段短视界训练样本 |

### 流程总览

```mermaid
flowchart LR
  I[语言指令] --> N[π_nav 长视界 VLN]
  N --> S[stop @ s_T_nav]
  S --> R[π_ref 末段精修\nUniNaVid + LoRA + VSP]
  R --> V[s_T_ref 近且可见]
  V --> B[BERT 目标短语抽取]
  B --> O[OWLv2 边界框]
```

### 实例中心评测指标（与 SR 解耦）

| 指标 | 含义 | 诊断用途 |
|------|------|----------|
| **ONS@τ** | 停车点到实例周围可导航点最小测地距 ≤ τ | 区分「到区域 waypoint」与「到真实实例」 |
| **GS** | 检测 IoU≥0.5 | 接地模块质量（不依赖导航是否成功） |
| **OracleGS** | GT 框在最终帧可见且面积≥1% | 区分 **导航视角失败** vs **检测失败** |

## REVERIE-AIM 数据集要点

- **相对 REVERIE-CE：** 同一 Matterport3D 场景层次，将松散区域终点换为 **实例几何/可视区域内的采样点**；有效平均最小测地距 **1.32 m → 0.34 m**。
- **规模：** 训练 3,691 条长视界轨迹（53 场景、90 类物体）；验证 1,344 条（10 unseen 场景）；**180K** 末段 clip（路径倒数第二节点锚定 + 空间/朝向扰动）。
- **独特性（作者归纳）：** 同时满足 **连续环境 + 语言 + 实例级目标 + 物体邻近终点** 的 VLN 扩展集。

## 实验要点（索引级）

- **四类上游：** ETPNav-ZS/FT、UniNaVid-ZS、SmartWay——覆盖零样本/微调、图/视频、免训练范式。
- **一致增益：** REALM Full 在各骨干上全面提升 ONS/GS/OracleGS；ETPNav-ZS ONS@0.1m **相对翻倍**（7.07%→14.66%）；VSP 消融在各设置均带来稳定提升。
- **真机：** Stretch + RealSense D435if，12 episode；UniNaVid+REALM 相对 vanilla 在 ONS@0.5m、GS、OracleGS 均约 **3–4×** 量级提升（样本量小，作初步证据）。

## 常见误区或局限

- **误区：** REVERIE / REVERIE-CE 的 **SR** 可直接代表「找到并看清目标」；本文表明应并列报告 **ONS / OracleGS**。
- **误区：** 末段精修必须重训整条 VLN；REALM 设计为 **stop 后接管**，上游可冻结。
- **局限：** 绝对性能仍远低于 Human 上界；开放检测与短语抽取错误会传导到 GS；真机评测 episode 较少；未覆盖室外或 manipulator 级后续交互。

## 关联页面

- [视觉–语言导航（VLN）](../tasks/vision-language-navigation.md) — 任务总览；本页补足 **REVERIE 末段接地与评测协议**
- [REVERIE](../entities/paper-vln-03-reverie.md) — 远程指代表达导航任务起源
- [ETPNav](../entities/paper-vln-09-etpnav.md) — 本文主要上游骨干之一（拓扑规划 VLN-CE）
- [VLN 四范式开源复现](../overview/vln-open-source-repro-paradigms.md) — UniNaVid 精修骨干与导航 VLA 语境
- [3D 空间 VQA](../concepts/3d-spatial-vqa.md) — 同为「为看见而行动」的细粒度空间能力，侧重 QA 而非轨迹到达

## 参考来源

- [REALM 论文摘录（arXiv:2607.03792）](../../sources/papers/realm_last_3_meter_vln_arxiv_2607_03792.md)

## 推荐继续阅读

- Shi et al., *From Region Arrival to Instance-Level Grounding in Vision-and-Language Navigation* — [arXiv:2607.03792](https://arxiv.org/abs/2607.03792)
- Qi et al., *REVERIE: Remote Embodied Visual Referring Expression in Real Indoor Environments* — [arXiv:1904.10151](https://arxiv.org/abs/1904.10151)
- Liu et al., *GroundingMate: Aiding Object Grounding for Goal-Oriented Vision-and-Language Navigation* — WACV 2025（离散 REVERIE 接地辅助对照）
- Zhang et al., *Uni-NaVid: A Video-based Vision-Language-Action Model for Unifying Embodied Navigation Tasks* — [arXiv:2412.06224](https://arxiv.org/abs/2412.06224)（REALM 精修骨干）
