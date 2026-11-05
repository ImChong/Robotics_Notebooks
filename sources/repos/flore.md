# FLoRe 官方源码

> 来源归档（repo），核查日期 2026-11-05

- **仓库：** <https://github.com/amazon-far/FLoRe>
- **核查版本：** `cb99fd05ac3e43eefdd1269647a29b98be2b6ea3`
- **论文：** <https://arxiv.org/abs/2610.35862>
- **项目页：** <https://louis-leee.github.io/FLoRe/>
- **实体：** [paper-flore](../../wiki/entities/paper-flore.md)
- **开放范围：** 底层控制器训练、ONNX 权重、G1 部署、采集软件及高层扩散策略源码；高层训练和部署指南仍待发布。许可与第三方组件以仓库 LICENSE 和 THIRD_PARTY_NOTICES 为准。

## 已核查入口

- `scripts/collect_data.sh`：调用 `third_party/UMI_collection/run_collection_viz.sh` 采集数据；支持无硬件 mock。
- `third_party/UMI_collection/processing/convert_to_umi.py`：将录制数据转换为 Zarr。
- `scripts/train_s1.py`：解析配置后调用 `Trainer.train()`；`flore/s1_policy/training/trainer.py` 构建 `HLDataset` 及数据加载器。
- `scripts/run_train_diffusion_hl.sh` 是指定本地数据与统计文件的训练示例，需按实际环境配置。
