# RAGA 项目页（miraymen.github.io/raga）

- **标题：** RAGA: Real Time Ray Traced Gaussian Shadow Casting for 3DGS Avatar–Scene Interaction
- **类型：** site / project-page
- **URL：** <https://miraymen.github.io/raga/>
- **arXiv：** <https://arxiv.org/abs/2606.29329>
- **会议：** ECCV 2026（页面标注）
- **入库日期：** 2026-09-10
- **配套论文：** [RAGA（arXiv:2606.29329）](../papers/raga_arxiv_2606_29329.md)

## 一句话摘要

Tübingen AI Center / MPI / Imperial / KAUST / Snap 合作的 **ECCV 2026** 项目页：在 **纯 3DGS** 中为动画 avatar 做 **实时光线追踪阴影**（~50 FPS），对比 3DGRT 与 RaySplat，展示单人、多人与 avatar–物体交互。

## 公开信息要点（截至 2026-09-10 核查）

- **机构：** Tübingen AI Center, University of Tübingen；MPI for Informatics；Imperial College London；KAUST；Snap Inc.（Mir, Guler, Wang, Wonka, Zhou, Pons-Moll）。
- **卖点：** No Mesh · Fully in Gaussian Space · Exact Ray–Gaussian Integrals · ~50 FPS。
- **方法：** shadow ray 从场景 Gaussian 指向光源，穿过 avatar Gaussians 累积 transmittance；**normalized line integral** vs icosahedron proxy（3DGRT）与浅层 hit（RaySplat）。
- **反例：** mesh 场景提取有损；SMPL 人体 proxy 丢衣发；mesh 无法支持插入的 3DGS 物体。
- **代码 / 数据（步骤 2.5）：** 页面 **无** GitHub / Hugging Face / Zenodo / Code 按钮或 Footer 链接（仅 Bulma CSS 依赖与作者主页）。**确认未开源**。

## 关联

- Wiki：[paper-raga-real-time-ray-traced-gaussian-shadow-casting](../../wiki/entities/paper-raga-real-time-ray-traced-gaussian-shadow-casting.md)
- 交叉：[Generative World Models](../../wiki/methods/generative-world-models.md)、[RL-GSBridge](../../wiki/entities/paper-sa-2409-20291-rl-gsbridge-3d-gaussian-splatting-based-real2sim.md)
