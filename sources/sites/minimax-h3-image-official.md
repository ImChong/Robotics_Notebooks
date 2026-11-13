# MiniMax H3-Image 官方发布与 API 文档

> 一手来源归档（官方公告与官方服务使用指南；非转载）

- **标题：** MiniMax H3-Image: Bringing 4K Image Generation into the 10-Second Era
- **类型：** blog / 官方发布
- **作者/机构：** MiniMax Research（稀宇科技）
- **公告：** <https://www.minimax.io/blog/minimax-h3-image>
- **官方 API 指南：** <https://platform.minimax.io/docs/guides/image-generation>
- **官方网页体验：** <https://hailuoai.video/tools/minimax-h3-image>
- **公告日期：** 2026-11-11
- **核查日期：** 2026-11-13
- **核对范围：** 通读 2026-11-11 官方公告全部正文（定位、速度、效果、指令跟随、开权重、可用性和价格）；对照官方 API 文档的模型规格、生成/编辑入口与参数限制；性能数字仅记录官方宣称，不当作第三方实测。
- **通读：** 已完成
- **一句话说明：** MiniMax 在 H3 多模态预训练基础上发布 H3-Image 开权重图像生成/编辑模型；官方主张原生约 16MP、4K 单图低于 10 秒。

## 主要事实与证据边界

| 主题 | 官方公布内容 | 注意 |
|---|---|---|
| 任务 | 文生图、参考图条件生成、多图编辑 | 不是视频生成或机器人控制策略 |
| 速度 | 4K（4096×4096）单图 **低于 10 秒** | 官方案例；没有复现实测硬件/并发条件 |
| 画质 | 更丰富的纹理和真实感 | 质量与领先程度是厂商描述，未独立验证 |
| 提示词 | 最多 16 张参考图 | 编辑服务接口的限制，非所有推理框架同一参数 |
| 训练 | 训练数据、强化学习、采样步数、PE 和部署联合优化 | 公告没有给出完整可复现训练配方 |
| 服务 | MiniMax API / MiniMax Code / Hailuo | 服务能力和开权重部署应区别对待 |

## 官方 API 的可核查接口

- 文生图：`POST /v1/images/generations`，`model: MiniMax-H3-Image`。
- 编辑：`POST /v1/images/edits`，提示词与 **1–16** 张参考图。
- 高速服务：`MiniMax-H3-Image-Highspeed`，参数边界与标准版本相同。
- 输出面积：`1K`/`2K`/`4K` 档或自定义宽高；自定义输出 **655,360–20,000,000** 像素，比例 **1:3–3:1**。
- 当前服务不支持蒙版局部编辑和透明背景；批量每次 1–10 张。API 对提示增强关闭的输入有额外参数要求，见指南。
- **不复述促销价格为稳定价格：** 官方公告含限时 4K 优惠；结算价格以平台当时的报价为准。

## 入库映射

- [MiniMax H3-Image 实体](../../wiki/entities/minimax-h3-image.md)
- [官方权重/代码归档](../repos/minimax-h3-image.md)
- [上游视频模型 H3](../../wiki/entities/minimax-h3.md)
