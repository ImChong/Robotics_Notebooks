# LIBERO-PoseShift 基准配套资源 / LIBERO-PoseShift Benchmark Assets

- **类型：** dataset
- **数据卡：** [Bluexxx/LIBERO-PoseShift](https://huggingface.co/datasets/Bluexxx/LIBERO-PoseShift)
- **论文：** [arXiv:2611.24360v1](https://arxiv.org/abs/2611.24360v1)
- **代码：** [WACO](https://github.com/peacelwh/WACO)
- **许可：** 资源卡标 Apache-2.0；派生校准帧和深度也应核对上游 LIBERO / LIBERO-Plus / LeRobot LIBERO 的许可
- **核对日期：** 2026-11-28

数据仓自身不是一套新的示范轨迹。它分发确定性扰动清单、40 个 LIBERO 任务的校准包、LIBERO-Plus 的空工作区多视图扫描、1,599 个相机轨变体索引、10/20 条示范的子集索引，以及基于公共 `lerobot/libero` 273,465 帧估算并对齐的逐帧深度。深度数组约 8.4 GB，主要供误差感知微调使用；默认评测下载不包含该数组。主要网格的 2,000 episodes 是每模型的评测设计，不是此数据仓新增的演示数。

Hub 网页 Dataset Viewer 的预览报列结构不一致，不影响数据卡和独立资产入口；复现按官方脚本取相应文件，不把混合的元数据当成统一表格分割。

- **对 wiki 的映射：** [WACO / LIBERO-PoseShift](../../wiki/entities/paper-waco-pose-robust-vla.md)
