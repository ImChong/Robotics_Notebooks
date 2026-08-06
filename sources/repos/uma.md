# UMA（kv2000/UMA）

- **标题**: UMA: Ultra-detailed Human Avatars via Multi-level Surface Alignment
- **链接**: [https://github.com/kv2000/UMA](https://github.com/kv2000/UMA)
- **类型**: repo / inference + interactive-demo
- **作者**: Heming Zhu, Guoxing Sun, Christian Theobalt, Marc Habermann（MPI-INF / VIA）
- **项目页**: [https://vcai.mpi-inf.mpg.de/projects/UMA/](https://vcai.mpi-inf.mpg.de/projects/UMA/)
- **论文**: arXiv:2506.01802 — [`sources/papers/uma_arxiv_2506_01802.md`](../papers/uma_arxiv_2506_01802.md)
- **数据集**: [https://gvv-assets.mpi-inf.mpg.de/uma/](https://gvv-assets.mpi-inf.mpg.de/uma/)
- **Demo**: [https://uma4.umaumau.xyz/](https://uma4.umaumau.xyz/)
- **许可**: README 未声明 SPDX（入库日仓库 `license: null`）
- **入库日期**: 2026-08-06
- **摘要**: 官方仓托管 **UMA 数据集下载脚本**、**推理**（几何 `.ply` + Analytic Splatting 渲染）与 **UMA-Viewer** 交互 demo；自定义 CUDA 扩展含 DDC 骨架 FK、Analytic-Splatting 光栅与 `simple_knn`。

## 开源状态（截至 2026-08-06）

| 项 | 状态 |
|----|------|
| **仓库** | `kv2000/UMA`（项目页 GitHub 按钮） |
| **数据集** | **已发布**（注册后下载；`helping_script/dataset_downloader.py`） |
| **推理入口** | **有** — `UMA_inference/testing_script_full_geometry.py`、`testing_script_full_res.py` + per-subject shell |
| **交互 demo** | **有** — `UMA_viewer_clean/scripts/run_viewer.sh`（需先 `clean_dof.py`） |
| **Checkpoint** | Google Drive 链（README）；按被试解压到 `UMA_Checkpoints/` |
| **训练工具** | README TODO：**未发布**（仅有多分辨率 crop 生成辅助脚本） |
| **结论** | **部分开源（可推理 / 可 demo）**；完整训练管线待发 |

## README 时间线

- **2026-05-15**：建仓；**UMA Dataset** 发布
- **2026-07-28**：**Demo + inference code** 发布

## 可运行入口（对齐 wiki 时序图）

1. `bash setup_env.sh` → `conda activate nvdiffrast`（PyTorch 2.1.0 + CUDA 12.1；可选 `WITH_COTRACKER=1`）
2. 下载 metadata → `UMA_MetaData/`；checkpoint → `UMA_Checkpoints/`
3. **几何推理：** `cd UMA_inference && python testing_script_full_geometry.py --conf confs/Subject_0/inference.conf --split train|test`
4. **渲染推理：** `python testing_script_full_res.py --conf ... --split ... --camera_type 0|1 [--save_video 1]`
5. **Viewer：** `python helping_script/clean_dof.py --data-root UMA_MetaData` → `bash UMA_viewer_clean/scripts/run_viewer.sh [train|test] [port]`

## 目录要点

| 路径 | 作用 |
|------|------|
| `UMA_inference/` | 几何 / 全分辨率渲染推理与 conf |
| `UMA_viewer_clean/` | 交互 Web viewer（Viser 前端） |
| `helping_script/` | 数据集下载、DOF 清洗、多分辨率 crop |
| `diff_feat_gaussian_rasterization_fin_color/` | Analytic Splatting + depth/mask |
| `woot_cuda_skeleton_fin/` | DDC 骨架前向运动学（训练亦须用此） |
| `simple_knn/` | 高斯尺度初始化 |

## 对 Wiki 的映射

- **wiki/entities/paper-uma.md**：升格实体；`## 源码运行时序图` 对齐本页入口
- **交叉：** Face Anything / SHELLS（面部与人头上游对照）、SMPL-X（元数据含 SMPL-X 角色）、遥操作 / telepresence
