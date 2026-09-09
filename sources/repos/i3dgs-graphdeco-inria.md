# i3dGS（graphdeco-inria/i3dgs）

> 来源归档

- **标题：** i3dGS — Immediate 3D Gaussian Splat Reconstruction of Unordered Input with Global Consistency
- **类型：** repo
- **链接：** <https://github.com/graphdeco-inria/i3dgs>
- **许可证：** Inria Immediate3DGS license（研究/评测；商业联系 OnTheFly）
- **入库日期：** 2026-09-09
- **一句话说明：** SIGGRAPH 2026 官方实现：乱序 RGB 即时 3DGS 重建；`train.py` 主流程 + 数据集下载脚本 + live/network viewer。
- **沉淀到 wiki：** [paper-i3dgs-immediate-3dgs-unordered](../../wiki/entities/paper-i3dgs-immediate-3dgs-unordered.md)

---

## 核心定位

论文 *Immediate 3D Gaussian Splat Reconstruction of Unordered Input with Global Consistency*（arXiv:2607.14481，SIGGRAPH 2026）的官方代码。面向 **无序图像捕获** 的 **在线 3D Gaussian Splatting 重建**，通过 VPR 匹配、共视性图、聚类回环与渐进层级，在优化过程中提供 **即时可视化反馈**，并扩展至 **大场景**。

- **项目页：** <https://repo-sam.inria.fr/nerphys/i3dgs/>
- **论文：** <https://arxiv.org/abs/2607.14481>
- **Viewer 组件：** [graphdecoviewer](https://github.com/graphdeco-inria/graphdecoviewer)（独立仓库，可复用于其他项目）

---

## 运行入口（README 摘要）

| 组件 | 入口 |
|------|------|
| 环境 | Ubuntu 24.04 / Windows 11；Python 3.12；PyTorch 2.7.1；CUDA 12.8 |
| 安装 | `git clone --recursive` → `pip install -r requirements.txt --no-build-isolation` + `cupy-cuda12x` |
| 数据 | `python scripts/download_datasets.py --out_dir data/`（MipNeRF360 / TUM / StaticHikes / tandt_db 等） |
| 训练/重建 | `python train.py -s ${SOURCE_PATH} -m ${MODEL_PATH}` |
| 评测 | `python train.py ... --test_hold ${N}`；`python scripts/train_eval_all.py` 复现论文表 |
| Live viewer | `python train.py -s ${SOURCE_PATH} --viewer_mode local` |
| 离线浏览 | `python gaussianviewer.py local ${MODEL_PATH}` |
| 路径渲染 | `python scripts/render_path.py -m ${MODEL_PATH} --render_path ...` |

**输入约定：** 默认 `${SOURCE_PATH}/images` 下按字母序排列的 `.png/.jpg/.jpeg/.webp`；可选 `${SOURCE_PATH}/sparse/0` COLMAP 真值位姿用于可视化。

---

## 对 wiki 的映射

- 实体页：[paper-i3dgs-immediate-3dgs-unordered](../../wiki/entities/paper-i3dgs-immediate-3dgs-unordered.md)
- 论文 source：[i3dgs_arxiv_2607_14481.md](../papers/i3dgs_arxiv_2607_14481.md)
- 项目页：[i3dgs-inria.md](../sites/i3dgs-inria.md)
