# MAMMA 项目页（mamma.is.tue.mpg.de）

> 来源归档（ingest 配套站点）

- **URL：** <https://mamma.is.tue.mpg.de/>
- **对应论文：** [MAMMA: Markerless Accurate Multi-person Motion Acquisition](https://arxiv.org/abs/2506.13040)（CVPR 2026 Oral）
- **补充新闻稿：** <https://www.mpg.de/27039732/avatars-without-marker-suits-using-only-smartphone-cameras>（Max Planck Society，2026-10-15）
- **机构：** Max Planck Institute for Intelligent Systems, Tübingen / Carnegie Mellon University
- **入库日期：** 2026-06-09
- **一句话说明：** 官方落地页：MammaNet 架构、MAMMASyn 合成数据、跨视角匹配、接触优化、定量/定性对比、Vicon 对标与 iPhone 野外 demo。

## 页面要点（2026-06 快照）

- **核心叙述：** 多视角视频 → **MammaNet** 预测 512 稠密 landmark（μ, σ, 可见性 p，以及人–人/人–地 **接触概率**）→ **对称极线距离 + Hungarian** 跨视角匹配 → **多阶段 SMPL-X 优化**（含接触能量）；**不依赖 pose 回归初始化**。
- **MammaNet：** 每 landmark 独立 learnable query；极端 OOD 姿态（如瑜伽）与 **部分 mask** 下仍稳健。
- **MAMMASyn：** 2.5M crops；子集 S / I / H；基于 BEDLAM 扩展 32 相机数字孪生 + 自采交互 mocap。
- **Multiview matching：** Fundamental matrix 对称极线距离；Hungarian 一对一；环一致图链接跨视角身份。
- **Optimization 三阶段：** 重投影 → pose/shape + Geman-McClure → 接触优化（排斥+吸引）；多视角平均接触概率映射到体表。
- **定量结果（MPJPE/PVE mm）：** MAMMA 在 RICH、Harmony4D、CHI3D、MAMMAEval-S/D、MOYO 上优于 Look-Ma* / CameraHMR / SMPLify；**MAMMA-C** 为带接触项变体。
- **Vicon 对比：** 37 held-out marker 评测，与 MoSh++ 管线误差差 **< 1 mm** 量级（页面述「less than 1mm difference」；论文摘要说 1.6 mm）。
- **野外：** 4 台 iPhone 室内/室外恢复人体运动。
- **数据集入口：** 注册账号后下载；含 dance、multi-person、iPhone、eval、synthetic。
- **BibTeX：** `@inproceedings{cuevas2026mamma, ... CVPR 2026}`

## 官方新闻稿补充（2026-10-15）

- 项目页标注 MAMMA 为 **CVPR 2026 Oral** 和 **Award Candidate（Top ~1.75%）**。
- MPI 新闻稿称，MAMMA 与 marker-and-suit 系统的动作精度差异不到 1 mm，并且处理整套数据的速度接近传统系统所需**人工工作流的 3 倍**。这里的速度是端到端人工工作流对比，不是神经网络单帧推理速度。
- 该新闻稿说明，标准手机相机（包括一组 iPhone）可作为多视角采集设备；系统仍依赖多机位视频与相机标定，重点支持最多两人的近距离接触交互。
- **指标口径：** 新闻稿的“<1 mm”是媒体概括；论文对 37 个 held-out markers 与 MoSh++ 流程的误差差异报告约 **1.6 mm**。两者来自不同表述，知识页保留论文数值作为可复核评测口径。

## 对 wiki 的映射

- 与 [sources/papers/mamma_arxiv_2506_13040.md](../papers/mamma_arxiv_2506_13040.md) 配对
- 实体页：[wiki/entities/paper-mamma-markerless-motion-capture.md](../../wiki/entities/paper-mamma-markerless-motion-capture.md)
