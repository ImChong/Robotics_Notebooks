# PartLLM 官方仓库

> 来源归档

- **类型：** repo
- **链接：** <https://github.com/czvvd/PartLLM>
- **项目页：** <https://czvvd.github.io/PartLLMPage/>（[归档](../sites/partllm.md)）
- **论文：** <https://arxiv.org/abs/2609.25832>（[归档](../papers/partllm.md)）
- **权重：** <https://huggingface.co/Czvvd/PartLLM>
- **许可：** Apache-2.0（代码；数据集许可另行核对）
- **入库日期：** 2026-10-28

## 复现入口

- `scripts/install.sh` 安装环境，README 指定 Python 3.11；`scripts/install_vllm.sh` 为可选加速。
- `scripts/infer.sh` 分派 `full_shape`、`text_guided`、`interactive`、`refine`，入口位于 `recipe/part_seg/tools/inference/`。具体参数见 [INFERENCE.md](https://github.com/czvvd/PartLLM/blob/main/docs/INFERENCE.md)。
- `scripts/train.sh` 与数据转换脚本见 [TRAINING.md](https://github.com/czvvd/PartLLM/blob/main/docs/TRAINING.md)；原数据集需独立下载，Parquet 中的几何路径须保持有效。
- 页面源码列出代码和模型；本次未实际下载权重或执行模型。

## 对 wiki 的映射

- [PartLLM 实体](../../wiki/entities/paper-partllm.md)
