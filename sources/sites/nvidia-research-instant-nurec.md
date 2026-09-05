# NVIDIA Research — Instant NuRec（SIL）

> 来源归档（ingest）

- **标题：** Instant NuRec: Feed-Forward 3D Gaussian Reconstruction for Driving Scene Simulation
- **类型：** site（官方项目页）
- **发布方：** NVIDIA Spatial Intelligence Lab（SIL）；Research：Jiahui Huang、Jiawei Ren、Michal Tyszkiewicz 等；Engineering：Bjoern Haefner、Qi Wu、Janick Martinez Esturo 等
- **原始链接：** <https://research.nvidia.com/labs/sil/projects/instant-nurec/>
- **配套论文：** arXiv:2607.14203 — 归档见 [sources/papers/instant_nurec_arxiv_2607_14203.md](../papers/instant_nurec_arxiv_2607_14203.md)
- **代码：** <https://github.com/NVIDIA/instant-nurec> — 归档见 [sources/repos/nvidia-instant-nurec.md](../repos/nvidia-instant-nurec.md)
- **入库日期：** 2026-09-05
- **一句话说明：** 多相机驾驶序列 **单次前向** → 分层 3DGS（静/动/天空）；页面给重建画廊、赛道长片段、**AlpaSim 闭环** 与相对逐场景 NuRec 的 PSNR / 检测 / 时间条。

## 摘录要点（与论文分工）

- **Abstract 条：** 10–20 s 多相机约 **1.5 s**；Waymo PSNR 比最强对照高 **2.01 dB**；接入 NuRec 与 AlpaSim。
- **方法轮播：** 位姿多视图 → 交替注意力 ViT → 共享 latent → 深度 / 语义 / 运动 / 天空 / 3DGS → 可选逐场景精修 → 仿真。
- **量化条（页面对照 NuRec / Instant Dense / Instant Selective）：**
  - PSNR：**34.38 / 29.93 / 29.77**
  - Detection precision：**0.970 / 0.955 / 0.946**
  - Detection recall：**0.955 / 0.940 / 0.929**
  - 重建时间：~**75 min / 1.5 s / 1.5 s**
- **扩展卡片：** 单目前视仍可重姿态 / 俯视一致；LiDAR 前向扩展在记录与平移轨迹上保住主几何。
- **引用：** `@techreport{nvidia2026instantnurec, ... year = {2026}}`

## 论文 / 代码状态（步骤 2.5，2026-09-05）

- 论文：<https://arxiv.org/abs/2607.14203>
- **已开源（部分）：** 项目页与文档指向 [NVIDIA/instant-nurec](https://github.com/NVIDIA/instant-nurec)（Apache-2.0）+ HF `nvidia/instant-nurec`。仓内可跑 **静态推理导出**；动态层与训练不在独立 CLI。详见 [nvidia-instant-nurec.md](../repos/nvidia-instant-nurec.md)。
- 产品栈：[docs.nvidia.com/nurec](https://docs.nvidia.com/nurec/)（Omniverse NuRec 26.04，Instant 为推荐初始化）。

## 对 wiki 的映射

- [Instant NuRec 论文实体](../../wiki/entities/paper-instant-nurec.md)
- [NVIDIA Omniverse NuRec](../../wiki/entities/nvidia-nurec.md)
- [官方仓归档](../repos/nvidia-instant-nurec.md)
- [仿真评测基础设施](../../wiki/concepts/simulation-evaluation-infrastructure.md)
