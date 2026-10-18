---
type: entity
project_id: cartwheel-swing-choreo
project: "https://getcartwheel.com/model/swing"
tags: [human-motion, generative-animation, text-to-motion, dcc, animation, motion-data]
status: complete
updated: 2026-10-18
related:
  - ../concepts/character-animation-vs-robotics.md
  - ../methods/diffusion-motion-generation.md
  - ../concepts/motion-retargeting.md
sources:
  - ../../sources/blogs/cartwheel_swing_choreo.md
  - ../../sources/sites/cartwheel-swing.md
summary: "Swing Choreo 是 Cartwheel 的商用文本到人体动作生成模型，擅长按顺序生成复合动作、手指和面部表现及多人互动；可经 Performance Generation/API 用于动画工作流，模型权重与推理实现未公开。"
---

# Swing Choreo：Cartwheel 复合文本到人体动作生成

**一句话定义：** Swing Choreo 是 Cartwheel 面向复杂表演的人体动作生成模型，可从一段分步文本提示生成连续动作，并同时处理手指、面部、多角色互动与空间方向。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DCC | Digital Content Creation | Maya、Blender 等数字内容创作工具 |
| FBX | Filmbox | 角色骨架与动画交换格式 |
| GLTF | GL Transmission Format | 常用于实时渲染和三维内容交换的格式 |
| USD | Universal Scene Description | 三维场景与动画交换格式 |
| BVH | Biovision Hierarchy | 骨架层级与动作数据格式 |

## 为什么重要

传统文本到动作模型常将文本对应到短单动作，复合描述容易遗漏子动作或把不同动作平均化。Choreo 的产品定位是读取按顺序描述的表演，让角色依次完成多个动作并生成连贯过渡。对动画师而言，这可以把自然语言作为动作草稿与分镜输入；多人关系、手势和表情也能随同身体动作一并生成。

它属于角色动画工具链，输出是人体/角色动作资产，**不是机器人原生控制策略**。把生成结果用于机器人学习仍需骨架映射、动作重定向与物理可行性验证。

## 核心能力与接口

| 维度 | 官方发布资料 |
|------|--------------|
| 复合动作 | 将多段动作按提示中的顺序组织，并生成动作间过渡 |
| 多角色 | 一次生成多名角色的不同动作与相互响应 |
| 手部与面部 | 根据提示生成手型、手势与表情 |
| 空间指令 | 指定角色相对世界或镜头的运动方向 |
| 输入方式 | Performance Generation 中以文本提示生成人体表演；Swing 家族还提供姿态到动作等能力 |
| 输出 | 骨架动画、网格与平移数据；规格列出 FBX、GLTF、USD、BVH、MB、MA |
| 帧率 | 产品规格列出 24、30、60、120 fps |
| 工具链 | 产品页列出 Maya、Blender、Cinema 4D、Unity、Unreal Engine 等工作流 |
| 生成时间 | 发布文章称 Choreo 一次生成通常需数分钟；Swing Sketch 面向更快速的草稿与迭代 |

官方产品页将 Swing 分为 Sketch、Edit、Choreo 等模型。Choreo 面向复杂指令、手部/面部细节和多角色表现；Sketch 更适合快速生成与预演。两者按质量、速度和任务复杂度选用。

## 发布方评测的读法

Cartwheel 发布文章报告专业动画师在与前代 Swing Sketch 的盲测成对比较中，整体有 88% 的比较偏好 Choreo；复合动作、手指和多人任务分别报告 92%、94% 和 100%。这些是**发布方自测**，主要对照是其前代模型，不能直接理解为独立第三方排行榜或对所有商用/开源模型的普遍胜率。

文章还报告其预备版内部基准得分为 61/100，对照系统中最高为 38/100；对照匿名，基准、数据和评测代码当时尚未公开。因此现阶段应将它们作为产品方的早期评测线索，不能据此做可复现实验结论。

## 工程实践

- 动画工作流可先用短提示检查动作顺序、角色间关系与方向，再补充时序、手型和表情约束；官方建议快速 blocking 使用 Sketch，复杂表现使用 Choreo。
- 根据目标 DCC 或引擎选择 FBX、GLTF、USD、BVH、MB 或 MA 等输出，并在导入后检查骨架映射、帧率、根位移、曲线连续性和手指关节映射。
- 若把角色动作作为机器人模仿学习的参考，先经过 [Motion Retargeting](../concepts/motion-retargeting.md)，再在仿真中检查关节限位、接触、动力学与跟踪误差。生成动画外观自然不代表机器人可稳定执行。

## 开源与局限

- Swing Choreo 是商业云端模型。官方模型页与产品规格提供应用/API 使用入口，但未提供模型权重、训练代码、推理实现或可下载数据集。
- Cartwheel 发布文章称模型训练使用其自建的动作捕捉与时间对齐描述数据，并称规模达到数万小时；具体数据清单与数据集没有公开下载入口。
- 官方称专用基准、数据与评测代码将随后发布；本次核查未发现公开的评测代码入口。
- 文章称 Choreo 比 Sketch 慢，生成一次需要数分钟；不适合作为实时机器人控制器或即时动作闭环。
- 多人互动、手指和面部结果需要在目标角色 rig 上人工检查；产品规格支持导出动画，不代表任何角色骨架都无需重定向。
- 已有 [Cartwheel MCP](../../sources/repos/cartwheel-mcp.md) 是云端服务的 API 接入代码，并非 Swing 模型本身的开源实现。

## 关联页面

- [Character Animation vs Robotics](../concepts/character-animation-vs-robotics.md) — 区分角色表演生成与机器人可控动作
- [Diffusion-based Motion Generation](../methods/diffusion-motion-generation.md) — 条件运动生成方法与相关工作
- [Motion Retargeting](../concepts/motion-retargeting.md) — 将人体动作映射到机器人或目标角色骨架

## 参考来源

- [Swing Choreo 官方发布文章归档](../../sources/blogs/cartwheel_swing_choreo.md)
- [Swing 官方模型与产品规格归档](../../sources/sites/cartwheel-swing.md)
- [Cartwheel Swing 模型家族页](https://getcartwheel.com/model/swing)
- [Performance Generation 产品规格](https://getcartwheel.com/product/performance-generation)

## 推荐继续阅读

- [Cartwheel：Introducing Swing Choreo](https://getcartwheel.com/blog/swing-choreo)
- [Cartwheel：Swing 模型家族](https://getcartwheel.com/model/swing)
