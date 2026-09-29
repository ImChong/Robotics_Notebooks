# closed-open-eyes（Hugging Face）

> sources/datasets 归档

- **名称：** Open and Closed Eyes Dataset
- **类型：** dataset
- **发布者：** Michal Mlodawski（Hugging Face：`MichalMlodawski/closed-open-eyes`）
- **链接：** <https://huggingface.co/datasets/MichalMlodawski/closed-open-eyes>
- **DOI：** 10.57967/hf/2745（HF 数据集卡片）
- **格式：** Parquet；任务标签含 image-classification、object-detection
- **规模：** 约 10⁵–10⁶ 级（HF `size_categories: 100K<n<1M`）
- **许可：** ODC-By
- **入库日期：** 2026-09-29
- **一句话说明：** 开/闭眼图像分类与检测任务的 **公开参考集**；[OCEC](../repos/ocec.md) README 的数据准备与 `01_dataset_viewer.py` 可与之对照，但 OCEC 训练主路径还包含 **WholeBody34 自动裁剪** 与 `real_data` 自建样本。

## 关联

- [OCEC 仓库归档](../repos/ocec.md)
- [OCEC wiki 实体](../../wiki/entities/ocec.md)
