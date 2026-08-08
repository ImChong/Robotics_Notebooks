# SLAMFormer-∞ 项目页（tsinghua-mars-lab.github.io/SLAMFormer-Infinity）

> 来源归档（ingest 配套站点）

- **URL：** <https://tsinghua-mars-lab.github.io/SLAMFormer-Infinity>
- **标题：** SLAMFormer-∞ — Infinite SLAM Transformer for Unbounded Frontend and Backend Processing
- **机构：** IIIS / MARS Lab, Tsinghua University（清华大学交叉信息研究院 · MARS Lab）
- **论文：** <https://arxiv.org/abs/2608.03429> — 归档见 [`sources/papers/slamformer_infinity_arxiv_2608_03429.md`](../papers/slamformer_infinity_arxiv_2608_03429.md)
- **配套仓库（占位）：** <https://github.com/Tsinghua-MARS-Lab/SLAMFormer-Infinity> — [`sources/repos/slamformer_infinity.md`](../repos/slamformer_infinity.md)
- **入库日期：** 2026-08-08
- **一句话说明：** 官方落地页：单目 RGB 流式稠密 SLAM；memory-conditioned frontend/local backend + 全局 PGGO；KITTI 在线 demo 与城市场景对比可视化。

## 公开信息要点（截至入库日）

| 项 | 状态 |
|----|------|
| **Paper / BibTeX** | 已链 arXiv:2608.03429 |
| **Demo 视频** | 有（含自采 ~17 km / 45 min 城市驾驶与 KITTI 在线序列） |
| **方法叙事** | 对比 VGGT（有界联合推理）与 VGGT-Long（位姿对齐 + 局部几何拼接）；强调有界局部计算 + 无界全局精炼 |
| **代码按钮** | **未列** 可运行训练/推理入口（页面以 Paper / Explore / BibTeX 为主） |
| **结论** | 项目页可用于方法理解与定性结果；**复现代码待发布** |

## 页面结构速记

1. **Long-range SLAM 定位** — 去掉「有界联合」vs「只拼位姿」的结构取舍。
2. **三种运行节奏** — Conditional frontend（流式）/ Local backend（周期）/ Global PGGO（回环或序列末）。
3. **定量摘要（页面对齐论文）** — KITTI Avg ATE 26.358→23.011 m；Waymo 1.996→1.813 m；7-Scenes 0.068→0.046 m（相对 VGGT-Long / VGGT-SLAM）。

## 关联资料

- 论文摘录：[`sources/papers/slamformer_infinity_arxiv_2608_03429.md`](../papers/slamformer_infinity_arxiv_2608_03429.md)
- 占位仓：[`sources/repos/slamformer_infinity.md`](../repos/slamformer_infinity.md)
- Wiki 实体：[`wiki/entities/paper-slamformer-infinity.md`](../../wiki/entities/paper-slamformer-infinity.md)
