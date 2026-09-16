# LSRM（facebookresearch/Large-Sparse-Reconstruction-Model）

- **URL：** <https://github.com/facebookresearch/Large-Sparse-Reconstruction-Model>
- **项目页：** <https://lzqsd.github.io/LSRM.github.io/>
- **Hugging Face：** <https://huggingface.co/facebook/Large-Sparse-Reconstruction-Model>
- **许可：** CC BY-NC 4.0（非商业）
- **关联论文：** [lsrm_object_reconstruction_arxiv_2604_05182.md](../papers/lsrm_object_reconstruction_arxiv_2604_05182.md)
- **收录日期：** 2026-09-12
- **最后更新：** 2026-09-16
- **开源结论：** **已开源**（训练/推理脚本 + HF 权重 + 示例数据）

## 一句话摘要

LSRM 官方实现：从 **posed sparse multi-view** 图像前馈重建高保真 3D 资产；支持 **NVS（GSO）** 与 **逆渲染（ORB/DTC）** 两条评测脚本；依赖 DINOv3 + Blender headless。

## 技术要点（编译自 README）

| 项 | 内容 |
|----|------|
| 环境 | conda `lsrm`；Python 3.10；torch 2.4.0 cu121 |
| 硬件 | 在 **NVIDIA H200** 验证；推理 **<40 GB** VRAM |
| 依赖 | `../dinov3`（gated DINOv3 权重）+ `../blender` 4.5.3 |
| Checkpoints | `checkpoints/rgb/`（dense+sparse）与 `checkpoints/brdf/` |
| NVS 测试 | `bash test_rgb.sh` → GSO 示例 |
| IR 测试 | `bash test_brdf.sh`（ORB/DTC）；`test_brdf_video.sh` 重渲染视频 |
| 输出 | `mesh.obj`、UV 纹理、novel views、albedo/roughness/metallic、Blender relight |

## 对 Wiki 的映射

- [paper-sa-2604-05182-lsrm 论文实体](../../wiki/entities/paper-sa-2604-05182-lsrm.md)
- [项目页归档](../sites/lsrm-project.md)
