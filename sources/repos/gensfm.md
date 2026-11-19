# GenSfM 代码

> 来源归档

- **仓库：** <https://github.com/dotb1ue/GenSfm>
- **论文：** <https://arxiv.org/abs/2610.35522>
- **项目页：** <https://dotb1ue.github.io/GenSfm/>
- **权重：** <https://huggingface.co/dotblue/GenSfm>（README 指向）
- **许可：** MIT；第三方依赖另见 THIRD_PARTY.md
- **入库日期：** 2026-10-31
- **本次复核：** 2026-11-19
- **核对范围：** 官方 README 的 Quick start、Python API、Outputs、Training/Evaluation；`example_multi_image.py` 预训练推理及导出路径，`trellis/pipelines/gensfm.py` 的 `InputViews`、`reconstruct`、`GenSfMPipeline.run`
- **通读：** 已完成
- **映射：** [GenSfM](../../wiki/entities/paper-gensfm.md)

README 提供训练、评估及推理入口；`example_multi_image.py` 在预训练路径调用 `GenSfMPipeline.from_pretrained(...).cuda().run(...)`。`reconstruct` 先执行 `_sample_ssuv` 与 `_solve_pose_from_ssuv`，再经 `_make_sparse_uvs_from_pose`、`_sample_slat` 细化；`export_result` 导出 GLB、Gaussian PLY、预测相机与重投影图。依赖 CUDA/NVIDIA GPU，首次下载权重约 5.8 GB，不含 DINOv2。

## 本次复核补充（2026-11-19）

- 官方 `trellis/pipelines/gensfm.py` 中，`InputViews` 对输入前景 alpha 做检查；没有前景透明度且未启用背景移除时会直接报错。
- `reconstruct` 先执行 `_sample_ssuv` / `_solve_pose_from_ssuv`，并验证相机参数有限、焦距为正、存在活跃体素后，才调用 `_make_sparse_uvs_from_pose` 和 `_sample_slat`；官方入口 `example_multi_image.py` 的 `export_result` 负责导出 GLB、PLY、`cameras.json`、重投影图。
- 因仓库输出未提供物性、关节和经物理验证的碰撞表示，将其用于 Real2Sim 时仍需下游物理资产加工与验证；对应 [Real2Sim Stage 1–2](../../roadmap/depth-real2sim.md) 与 [论文节点](../../wiki/entities/paper-gensfm.md)。
