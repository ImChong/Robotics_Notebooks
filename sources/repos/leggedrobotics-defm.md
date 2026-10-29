# DeFM 官方代码仓库

- **类型：** repo
- **代码：** <https://github.com/leggedrobotics/defm>
- **项目页：** <https://de-fm.github.io/>；[项目页归档](../sites/defm.md)
- **论文：** <https://arxiv.org/abs/2601.18923>；[论文归档](../papers/defm_arxiv_2601_18923.md)
- **权重：** <https://huggingface.co/leggedrobotics/defm>
- **入库日期：** 2026-10-28
- **开放状态：** Apache-2.0，公开模型加载、深度预处理、推理与分类/分割 benchmark 代码及权重；未据 README 推断完整预训练或机器人策略训练代码已发布。

## 运行入口

`pip install defm`；`defm.model_factory.create_defm_model` 加载权重，`defm.preprocess_depth_image` 接收以米为单位的深度，ViT 使用 `get_intermediate_layers`，CNN 使用 `model(...)`。`scripts/notebooks/` 提供示例，`benchmarks/` 提供分类与分割评测。CNN 输入高宽须为 32 的倍数；ViT 示例指定 patch size 14。
