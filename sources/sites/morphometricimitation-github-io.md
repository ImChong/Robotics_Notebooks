# morphometricimitation.github.io（Morphometric Imitation 项目页）

- **标题：** Morphometric Imitation: From Morphology and Contact Aware Hand Retargeting to Sim-to-Real Visuomotor Policy
- **类型：** site / project-page
- **URL：** <https://morphometricimitation.github.io/>
- **配套论文：** [Morphometric Imitation（arXiv:2609.28660）](https://arxiv.org/abs/2609.28660) — 归档见 [`sources/papers/morphometric_imitation_arxiv_2609_28660.md`](../papers/morphometric_imitation_arxiv_2609_28660.md)
- **代码：** <https://github.com/tsadja/morphometric> — 归档见 [`sources/repos/morphometric.md`](../repos/morphometric.md)
- **入库日期：** 2026-09-26

## 一句话摘要

UC Berkeley 官方页：**One human demo · Any multi-fingered hand · Zero-shot sim-to-real visuomotor policy**；三阶段 MMO → 残差 RL → visuomotor IL；真机视频 **10 类物体 × 3 实例**、**1× 速度全自主**；交互 3D 展示 MANO morph → 物 grasp → 机器人手迁移。

## 公开信息要点（截至入库日）

- **机构：** UC Berkeley EECS（Tomlin / Malik * equal advising）。
- **TLDR：** 单次人类示范；支持 **三/四/五指** 灵巧手（页内 **Allegro、Sharpa、Dex3** 视频对照）。
- **阶段展示：** Morphometric Optimization 三步 UI（morph 人手 → retarget 物 grasp → 转到机器人手）；Residual RL 对比 MANO / kinematic / RL result（Hammer、Apple、Cube 等）。
- **定量（摘要/页）：** visuomotor **89.3%** on **300 real trials / 30 objects / 10 categories**；MMO contact F1 **≥+8** vs 最强五基线；下游动态重定向 **≤+35 pt** SR。
- **Code 按钮：** 指向 `github.com/tsadja/morphometric`（仓库 README：**Code will be released soon**）。

## 为何值得保留

- **非 PDF 证据：** 多手同一条人类 hammer/apple/cube 示范的 **kinematic vs RL** 并排视频，是理解 MMO+残差 RL 分工的直观材料。
- **sim-to-real 卖点：** 项目页强调 **fully autonomous、zero-shot sim-to-real**，与仅 sim 高 SR 的 dexterous 线形成对照。

## 关联资料

- 论文归档：[`sources/papers/morphometric_imitation_arxiv_2609_28660.md`](../papers/morphometric_imitation_arxiv_2609_28660.md)
- 代码占位：[`sources/repos/morphometric.md`](../repos/morphometric.md)
