# 1X Redwood AI 控制策略官方发布

- **类型：** 官方研究博客
- **标题：** Redwood AI
- **链接：** <https://www.1x.tech/discover/redwood-ai>
- **发布日期：** 2025-06-10
- **核查日期：** 2026-10-05
- **相关世界模型：** [1X World Model 归档](1x-world-model-redwood.md)
- **一句话说明：** Redwood 是 EVE / NEO 的控制策略，官方介绍跨本体 transformer 表征、diffusion 动作解码、全身与移动操作。

## 官方技术摘录

1. **输入与输出：** 预训练语言和视觉编码，加关节位置/施力历史的本体嵌入；transformer latent 经 diffusion policy 解码到 EVE 或 NEO 动作。
2. **训练：** EVE / NEO 遥操作和自主轨迹；动作预测之外使用手与目标物体图像位置等认知预测目标增强空间 grounding。
3. **部署：** 官方称约 160M 参数、机载约 5 Hz，同时预测手臂/手、步行和骨盆指令。展示移动双臂操作与靠墙支撑等多接触行为。

## 开放范围与命名边界

- **未确认公开模型实现：** 发布页未列 Redwood 策略代码、权重或训练数据下载链接。
- Redwood AI **策略**与 Redwood AI World Model **动作条件预测模型**是不同发布物，不能把世界模型 Challenge 数据视作控制策略训练资产。
- 另一个 ArchitectLabs 同名 Redwood 为训练加速项目，亦与 1X 控制策略不同。

## 对 wiki 的映射

- [1X Redwood 控制策略](../../wiki/entities/1x-redwood-policy.md)
- [1X 公司与整机](../../wiki/entities/1x-technologies.md)
