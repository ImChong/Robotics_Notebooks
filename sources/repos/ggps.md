# GGPS / PanoLOG（Insta360-Research-Team/GGPS）

- **标题**: Geometry and Gradient-based Partitioning for Panoramic Outdoor Reconstruction（PanoLOG + G²PS）
- **链接**: [https://github.com/Insta360-Research-Team/GGPS](https://github.com/Insta360-Research-Team/GGPS)
- **类型**: repo / training-code
- **作者**: Chen, Weijian, et al. (2026)；Insta360 Research × SYSU × SCUT × UCAS × HEU × WHU
- **项目页**: [https://insta360-research-team.github.io/GGPS-Website/](https://insta360-research-team.github.io/GGPS-Website/)
- **Hugging Face**: [https://huggingface.co/Insta360-Research/GGPS](https://huggingface.co/Insta360-Research/GGPS)
- **论文**: arXiv:2607.08769 — [`sources/papers/ggps_panolog_arxiv_2607_08769.md`](../papers/ggps_panolog_arxiv_2607_08769.md)
- **许可**: CC BY-NC 4.0
- **入库日期**: 2026-07-26
- **摘要**: 官方仓托管 **PanoLOG** 全景 ERP 3DGS 训练与评测；含粗训、G²PS 划分、块精炼、合并、渲染与指标脚本；参考环境 PyTorch 2.8 / CUDA 12.8（含 RTX 5090 sm_120 说明）。

## 开源状态（截至 2026-07-26）

| 项 | 状态 |
|----|------|
| **仓库** | `Insta360-Research-Team/GGPS`（项目页 Code 按钮指向此处） |
| **可运行入口** | **有** — `scripts_new/prepare_data.sh` → `scripts_new/train.sh`（6 阶段）；亦可直接 `train_large.py` / `data_partition.py` / `merge.py` / `render_large.py` / `metrics_large.py` |
| **Roadmap（README）** | [x] 2026-07-09 完整训练代码；[x] 2026-07-15 Pano360 数据集；[ ] 下旬两景 `.ply`；[ ] UE 渲染插件 |
| **HF 数据** | `datasets/FTP.zip`、`NSC.zip`、`NSK.zip`；`ply/` 仍为占位说明 |
| **结论** | **已开源（训练管线可跑）**；预训练模型与 UE 插件待发布；数据相对论文四景仍缺 BAX/NSN |

## README 要点

1. **管线差异（相对针孔 3DGS）：** `camera_type=3`（ERP）；openMVG → COLMAP txt；可选 DAP 深度/天空掩码；`make_depth_scale.py` 对齐后才能开 `use_depth`。
2. **训练六阶段：** 粗训 30k → `data_partition.py` → 按 `block_id` 精炼 → `merge.py` → `render_large.py` → `metrics_large.py`。
3. **依赖编译：** `diff-gaussian-rasterization` 与 `simple-knn` 须 `--no-build-isolation` 对应当前 torch/nvcc；Blackwell 需 CUDA 12.8 + PyTorch ≥ 2.7 cu128。
4. **致谢底座：** CityGaussian、原版 3DGS、OmniGS（ERP 光栅，见 `NOTICE_OMNIGS.md`）、Mip-Splatting、AbsGS。

## 为什么值得保留

- 锁定可复现入口与 **NC 许可** 边界；读者勿假设可商用部署。
- 节点名对齐 wiki「源码运行时序图」：`prepare_data.sh` / `train_large.py` / `data_partition.py` / `merge.py`。
- HF 权重放出后可在此档补 `.ply` 路径与 `viewer.py` 用法。

## 对 wiki 的映射

- [wiki/entities/paper-panolog-ggps.md](../../wiki/entities/paper-panolog-ggps.md)
- [wiki/entities/paper-panoworld-real-world-panoramic-generation.md](../../wiki/entities/paper-panoworld-real-world-panoramic-generation.md) — 同机构全景生成对照
- [wiki/entities/paper-glob3r.md](../../wiki/entities/paper-glob3r.md) — 离线高精度几何/位姿上游对照
