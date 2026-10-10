# TouchScale 数据集发布

> 来源归档（ingest · Hugging Face 数据集卡片）

- **类型：** dataset
- **数据集仓库：** <https://huggingface.co/datasets/2077AIDataFoundation/TouchScale>
- **关联论文：** <https://arxiv.org/abs/2610.10288>
- **项目页：** <https://touch-scale.github.io/>
- **许可证：** CC-BY-NC-4.0（数据集卡片）
- **核查日期：** 2026-10-10
- **沉淀到 wiki：** [TouchScale](../../wiki/entities/touchscale.md)

## 实际发布状态（核查日）

论文描述完整 TouchScale 数据集约 500 小时；当前 Hugging Face 卡片注明该仓库暂时托管 100 小时，共 15,324 episodes、929 个任务和 22 种场景类型。全量 500 小时在卡片中标注为计划于 2026 年 11 月前发布，属于计划状态，不是当前可下载事实。

访问数据文件前需要登录并同意分享联系信息。数据集按 WebDataset shards 发布，另有预览子集和 episode metadata。数据卡片列出的传感流包括头戴 RGB-D、左右腕部 RGB、每手 880 taxels 的触觉手套和 IMU，附时间戳、任务说明、标定文件与质量检查元数据。

## 许可和读取注意事项

CC-BY-NC-4.0 限制商业用途；使用者应阅读数据卡片完整条件并遵守署名、非商业及其他适用条款。数据访问的联系信息条件也需按发布页面要求处理。WebDataset 全量文件体量较大；README 提供 streaming 用法，建议先用预览 split 和 metadata 验证处理管线。

## 一手入口

- [TouchScale dataset card](https://huggingface.co/datasets/2077AIDataFoundation/TouchScale)
- [Dataset Viewer](https://huggingface.co/datasets/2077AIDataFoundation/TouchScale/viewer/preview/train)
- [论文归档](../papers/touchscale_arxiv_2610_10288.md)
