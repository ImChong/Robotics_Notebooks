# Swing（Cartwheel）产品页

> 来源归档（ingest / Cartwheel 官方模型与产品规格）

- **项目：** Swing 人体动作生成模型家族；本页侧重 Swing Choreo
- **官方模型页：** <https://getcartwheel.com/model/swing>
- **Choreo 发布文章：** <https://getcartwheel.com/blog/swing-choreo>
- **Performance Generation：** <https://getcartwheel.com/product/performance-generation>
- **API 文档入口：** <https://api-docs.getcartwheel.com/>
- **开源状态：** 模型权重、训练/推理代码及数据集未见公开链接；产品通过 Cartwheel 云端应用/API 提供
- **核查日期：** 2026-10-18
- **详情节点：** [Swing Choreo](../../wiki/entities/cartwheel-swing-choreo.md)

## 模型定位

Cartwheel 官方将 Swing 描述为从文本或姿态生成连续人体动作的模型家族，覆盖 Swing Sketch、Swing Edit、Swing Choreo 等版本。官方模型页将 Choreo 的特点概括为更精确的表演指令控制、面部和手指动作，以及多角色动作生成。复杂长提示可用 Choreo；需要快速生成和预演时，官方建议考虑 Sketch。

## 产品接口与 DCC 工作流

Performance Generation 产品页列出的输出包括骨架动画、网格与平移数据，格式包括 FBX、GLTF、USD、BVH、MB 与 MA，支持 24、30、60、120 帧率。页面列出的可配参数包括时长、循环、曲线平滑、帧率和变体数量；工具工作流覆盖 Autodesk Maya、Blender、Cinema 4D、Unity、Unreal Engine 等。

这些是产品页面公布的生成与交换能力；不应据此推断 Swing Choreo 可离线运行，也不等于模型权重或训练数据开放。官方另提供 Cartwheel API 和 MCP 客户端接入；现有 MCP 源码仅用于连接服务，不包含 Choreo 模型实现。

## 可核查来源

- [Swing 官方模型家族页](https://getcartwheel.com/model/swing)
- [Swing Choreo 官方发布文章](https://getcartwheel.com/blog/swing-choreo)
- [Performance Generation 产品说明与规格](https://getcartwheel.com/product/performance-generation)
- [Cartwheel API 文档入口](https://api-docs.getcartwheel.com/)
- [Cartwheel MCP 源码资料](../repos/cartwheel-mcp.md)
