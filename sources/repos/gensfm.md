# GenSfM 代码

> 来源归档

- **仓库：** <https://github.com/dotb1ue/GenSfm>
- **论文：** <https://arxiv.org/abs/2610.35522>
- **项目页：** <https://dotb1ue.github.io/GenSfm/>
- **权重：** <https://huggingface.co/dotblue/GenSfm>（README 指向）
- **许可：** MIT；第三方依赖另见 THIRD_PARTY.md
- **入库日期：** 2026-10-31
- **映射：** [GenSfM](../../wiki/entities/paper-gensfm.md)

README 提供训练、评估及推理入口；`example_multi_image.py` 调用 `trellis/pipelines/gensfm.py` 的 GenSfMPipeline，导出 GLB、Gaussian PLY、相机与重投影图。依赖 CUDA 和 NVIDIA GPU；首次加载权重约 5.8 GB，不含 DINOv2。
