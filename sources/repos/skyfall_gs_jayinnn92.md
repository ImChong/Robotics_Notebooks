# skyfall_gs_jayinnn92

> 来源归档（repo）

- **名称：** Skyfall-GS
- **类型：** repo
- **原始链接：** <https://github.com/jayin92/Skyfall-GS>
- **入库日期：** 2026-09-20
- **许可：** Apache 2.0
- **一句话说明：** ECCV 2026 Skyfall-GS 官方实现：卫星影像 Stage 1 3DGS 重建 + Stage 2 IDU 扩散精炼；含 JAX/NYC 训练脚本、eval、融合 PLY 与渲染管线。

## 仓库要点

| 项 | 内容 |
|----|------|
| **主入口** | `train.py`（Stage 1 / Stage 2 `--iterative_datasets_update`） |
| **自动化** | `scripts/run_jax.py`、`run_jax_idu.py`、`run_nyc.py`、`run_nyc_idu.py` |
| **可视化** | `create_fused_ply.py` → Mip-Splatting viewer / SuperSplat；`render_video.py` |
| **评测** | `eval.py`（PSNR/SSIM/LPIPS/CLIP-FID/CMMD） |
| **自定义数据** | [SatelliteSfM](https://github.com/jayin92/SatelliteSfM) 预处理；或 COLMAP 格式 |
| **依赖** | Python 3.10、CUDA 12.8、Mip-Splatting 系子模块、FlowEdit、MoGe 等（见 Acknowledgement） |

## 数据资源（README）

- 训练集：[HF Skyfall-GS-datasets](https://huggingface.co/datasets/jayinnn/Skyfall-GS-datasets)
- 评测包：[HF Skyfall-GS-eval](https://huggingface.co/datasets/jayinnn/Skyfall-GS-eval)
- 预融合 PLY：[HF Skyfall-GS-ply](https://huggingface.co/jayinnn/Skyfall-GS-ply)

## 对 wiki 的映射

- [paper-skyfall-gs](../../wiki/entities/paper-skyfall-gs.md)
- [skyfall_gs_jayinnn](../sites/skyfall_gs_jayinnn.md)
