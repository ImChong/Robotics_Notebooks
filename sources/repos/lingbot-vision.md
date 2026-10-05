# LingBot-Vision

> 来源归档（国内具身开源全景）

- **标题：** LingBot-Vision
- **类型：** repo
- **机构：** 蚂蚁灵波
- **链接：** https://github.com/Robbyant/lingbot-vision
- **分类：** 评测
- **入库日期：** 2026-09-06
- **一句话说明：** 蚂蚁灵波 开源项目 LingBot-Vision（评测），见 [国内具身开源全景](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)。
- **沉淀到 wiki：** [`wiki/entities/cn-os-lingbot-vision.md`](../../wiki/entities/cn-os-lingbot-vision.md)

## 开源状态

- **已开源**：公开仓库（以 README 与 release 为准）。

## 对 wiki 的映射

- [wiki/entities/cn-os-lingbot-vision.md](../../wiki/entities/cn-os-lingbot-vision.md)

## 官方资源补核（2026-10-05）

- **项目页：** <https://technology.robbyant.com/lingbot-vision/>；代码 <https://github.com/robbyant/lingbot-vision>。
- **机制：** masked boundary modeling 视觉表征；G 级教师约 1.1B 参数，提供 S/B/L/G 骨干。
- **入口：** `load_pretrained_backbone`、`extract_patch_tokens`、`load_image`、`scripts/run_pca_demo.sh`。
- **范围：** 发布 `backboneonly.pt` 排除 optimizer、投影与 boundary heads；不能由骨干权重推定完整训练资产开放。代码 Apache-2.0，权重许可另查模型卡。
