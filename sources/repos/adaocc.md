# AdaOcc 官方代码

- **仓库：** <https://github.com/wangjl-nb/AdaOcc>
- **项目页：** <https://wangjl-nb.github.io/AdaOcc_web/>
- **论文：** <https://arxiv.org/abs/2609.38864>
- **权重：** <https://huggingface.co/wjldragon/AdaOcc>
- **许可证：** MIT（仓库）；外部数据及模型遵循各自许可。
- **核查日期：** 2026-10-29
- **范围：** OccScanNet-mini 训练、评测、数据准备及 full 划分 checkpoint 评测；数据集和外部预训练权重需另行获取。项目页面的真实机器人演示不等于已发布完整部署栈。

## README 复现入口

1. 根据 README 准备 OccScanNet、RADIO 与 AdaOcc checkpoint，生成数据索引和标签。
2. 使用 `configs/occscannet/radio_occscannet_mini.py`；`./dist_train.sh 8 <config>` 训练，`./dist_val.sh 8 <config> <checkpoint>` 评测。
3. mini 发布权重采用 `ADAOCC_ONLINE_DEPTH=1` 评测；full 配置默认使用生成的预计算深度。默认 mini 配置使用本地准备的深度。可选 MSMV CUDA 扩展未构建时按 README 设置 PyTorch fallback。
4. 本次仅核查文档，未下载数据或运行训练评测。

[论文归档](../papers/adaocc_arxiv_2609_38864.md) · [项目页归档](../sites/adaocc.md) · [实体](../../wiki/entities/paper-adaocc.md)
