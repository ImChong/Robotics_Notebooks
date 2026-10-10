# Meet Odyssey-3: Our Most Powerful Foundation World Model

> 来源归档（ingest · Odyssey 官方发布文章）

- **类型：** official announcement / project page
- **URL：** <https://odyssey.systems/meet-odyssey-3>
- **发布：** 2026-10-08
- **作者：** Oliver Cameron、Jeff Hawke（页面署名）
- **核查日期：** 2026-10-10
- **一句话说明：** Odyssey-3 是 Odyssey 发布的自回归扩散 Transformer 世界模型，预测视觉场景随动作和事件如何演化，并可经动作解码器或策略适配到不同物理系统。
- **沉淀到 wiki：** [Odyssey-3](../../wiki/entities/odyssey-3.md)

## 官方文章要点

Odyssey 将 Odyssey-3 描述为学习动力学系统：从视觉观察中学习物体运动、交互、物理规律和因果变化；模型可用于生成可交互环境，也可作为训练其他物理系统策略的表示/世界知识基础。官方展示环境实时生成、机器人操作、人形、车辆驾驶、多摄像头传感器生成，以及在模型生成世界内训练和评估 agent。

对机器人适配，文章明确提到需要针对目标机器训练 action decoder 或 policy，把模型表示转换成具体机器所需的控制。官方展示的人形结果来自 Flexion 构建的控制策略。应区分“世界模型能够预测/生成环境”和“机器人控制器已能直接输出可部署动作”：后者依赖额外适配与对应本体评测。

## 发布与获取状态

截至核查日，官方称研究预览开放，并邀请 Physical AI 开发者联系团队；文章提供在线体验和 API access 入口。该页面没有公开模型权重、训练代码或完整复现实验资料，因此本条作为官方项目/发布信息记录，不把它标记为开源模型。

## 一手入口

- [官方发布文章](https://odyssey.systems/meet-odyssey-3)
- [Odyssey 官网](https://odyssey.systems/)
- [在线体验入口](https://experience.odyssey.systems/)
- [开发者 API 入口](https://developer.odyssey.ml/)
