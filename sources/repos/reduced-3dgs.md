# Reduced 3DGS 官方代码

- **代码：** <https://github.com/graphdeco-inria/reduced-3dgs>
- **项目页：** <https://repo-sam.inria.fr/fungraph/reduced_3dgs/>
- **论文：** [I3D 2024 论文归档](../papers/reduced_3dgs.md)
- **状态：** 代码、训练配置与评估脚本公开；README 的数据集链接标为 TODO。
- **入库日期：** 2026-10-27

仓库要求递归克隆子模块；入口包括 `train.py`、`compress.py`、`render.py` 和 `metrics.py`。其 PLY 格式与原始 3DGS 不兼容，仓库提供 `update_old_ply_format.py` 转换旧格式。环境与许可证以仓库为准。
