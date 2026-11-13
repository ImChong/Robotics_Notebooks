# GenSfM 代码

> 来源归档

- **仓库：** <https://github.com/dotb1ue/GenSfm>
- **论文：** <https://arxiv.org/abs/2610.35522>
- **项目页：** <https://dotb1ue.github.io/GenSfm/>
- **权重：** <https://huggingface.co/dotblue/GenSfm>（README 指向）
- **许可：** MIT；第三方依赖另见 THIRD_PARTY.md
- **入库日期：** 2026-10-31
- **本次复核：** 2026-11-13
- **核对范围：** 官方 README 的 Quick start、Python API、Outputs、Training/Evaluation；`example_multi_image.py` 预训练推理及导出路径，`trellis/pipelines/gensfm.py` 的 `InputViews`、`reconstruct`、`GenSfMPipeline.run`
- **通读：** 已完成
- **映射：** [GenSfM](../../wiki/entities/paper-gensfm.md)

README 提供训练、评估及推理入口；`example_multi_image.py` 在预训练路径调用 `GenSfMPipeline.from_pretrained(...).cuda().run(...)`。`reconstruct` 先执行 `_sample_ssuv` 与 `_solve_pose_from_ssuv`，再经 `_make_sparse_uvs_from_pose`、`_sample_slat` 细化；`export_result` 导出 GLB、Gaussian PLY、预测相机与重投影图。依赖 CUDA/NVIDIA GPU，首次下载权重约 5.8 GB，不含 DINOv2。
