# vision-demos

> 来源归档

- **类型：** repo / 计算机视觉示例
- **作者：** Jeremy Park
- **链接：** https://github.com/jeremyipark/vision-demos
- **俯卧撑实现说明：** https://github.com/jeremyipark/vision-demos/blob/main/push_ups/how-it-works.md
- **入库日期：** 2026-10-28
- **开源状态：** 示例代码公开，Apache-2.0；模型通过 VLM Run Gateway 调用。3D 攀岩使用带深度图和相机标定的采集数据；README 称采集应用细节可按请求分享，不应据此推断数据或采集应用已公开。
- **沉淀到 wiki：** [vision-demos](../../wiki/entities/vision-demos.md)

## 内容与运行入口

仓库包含俯卧撑、正反握引体向上、硬拉、跑步、舞蹈同步及 2D/3D 攀岩示例。俯卧撑使用 ViTPose+ Large 姿态关键点，输出标注视频、图表、逐帧及逐次 CSV、JSON 等；其拍摄要求与阈值参见 [README](https://github.com/jeremyipark/vision-demos/blob/main/push_ups/README.md)。

根目录 `app.py` 提供浏览器入口，支持匿名限速调用或配置 API key，依赖 ffmpeg；README 明确 `push_ups` 与 `rock_climbing_3d` 不在该应用中。俯卧撑独立运行的 README 要求配置 API key。
