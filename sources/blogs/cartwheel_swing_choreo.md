# Swing Choreo：Cartwheel 复合文本到人体动作生成模型

> 来源归档（ingest / Cartwheel 官方发布文章）

- **标题：** Introducing Swing Choreo
- **类型：** 官方博客 / 商业模型发布说明
- **作者 / 组织：** Cartwheel Team
- **原始链接：** <https://getcartwheel.com/blog/swing-choreo>
- **产品页：** <https://getcartwheel.com/model/swing>
- **产品入口：** <https://getcartwheel.com/product/performance-generation>
- **发表时间：** 2026-10（原文标注 October 2026，未列具体日期）
- **入库日期：** 2026-10-18
- **项目节点：** [Swing Choreo](../../wiki/entities/cartwheel-swing-choreo.md)
- **一句话说明：** Cartwheel 发布的商用人体动作生成模型，可把分步、复合文本指令转为连续表演，并生成多人互动、手指姿势和面部动作；模型的权重与推理代码未公开。

## 官方发布要点

Cartwheel 将 Swing Choreo 定位为 Swing 人体运动模型家族中面向复杂表演的版本。与每个短片段对应一个短描述的传统文本到动作设定不同，Choreo 将提示理解为按顺序发生的动作序列，并生成动作间的过渡。发布文章举例包括“跑动、把球砸向地面、庆祝”等连续动作。

文章列出的能力包括：

- **复合动作**：按提示中的先后次序执行多个动作，并保持动作间的衔接和节奏。
- **手指与面部**：从文本生成手型、手势和面部表情。
- **多人互动**：一次生成多名角色，各自执行不同动作并响应其他角色；示例为交谈、猜拳和群舞。
- **空间指令**：控制角色相对世界或镜头的移动方向。
- **生成耗时**：官方称 Choreo 通常需要几分钟；快速预演可用 Swing Sketch。

## 发布方自报评测

Cartwheel 称，在与 Swing Sketch 的盲测比较中，专业动画师整体有 88% 的比较偏好 Choreo；其分项偏好为复合动作 92%、手与手指 94%、多人 100%、空间方向 86%、单动作提示 71%。这是 Cartwheel 自行开展、对比其前代模型的成对偏好测试，不能当作独立第三方基准。

同一文章称其预备版内部基准中 Choreo 得分 61/100，已测试的其他系统最高 38/100；对照系统被匿名化，基准、数据与评测代码当时仍标为即将发布。因此这些数字应视为发布方的初步结果，而非可独立复现的公开评测。

## 开放状态与边界（2026-10-18 核查）

- **模型/推理代码：** 官方模型页、发布文章和 Performance Generation 规格未列出 Swing Choreo 的代码仓库或可下载权重；按商业云端服务记录。
- **产品入口：** 可在 Cartwheel Performance Generation 中使用；产品规格也列出 API 接入。
- **数据：** 发布文章描述 Cartwheel 自建的人体动作捕捉和标注管线，并称数据规模为数万小时；没有提供可下载数据集链接。
- **评测材料：** 发布文章称专用基准、数据与评测代码将后续发布；本次核查未找到相应公开仓库或数据入口。
- **相关开源代码：** Cartwheel MCP 是 API 接入客户端，不是 Swing 模型本身；见[已有 MCP 源码归档](../repos/cartwheel-mcp.md)。

## 来源

- [Cartwheel 官方发布文章](https://getcartwheel.com/blog/swing-choreo)
- [Swing 官方模型家族页](https://getcartwheel.com/model/swing)
- [Performance Generation 产品规格](https://getcartwheel.com/product/performance-generation)
