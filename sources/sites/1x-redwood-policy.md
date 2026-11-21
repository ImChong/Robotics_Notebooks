# 1X Redwood AI 控制策略官方发布

- **类型：** 官方研究博客
- **标题：** Redwood AI
- **链接：** <https://www.1x.tech/discover/redwood-ai>
- **发布日期：** 2025-06-10
- **核查日期：** 2026-11-22
- **核对范围：** 通读 1X 官方博客全文 `<https://www.1x.tech/discover/redwood-ai>`（2025-06-10，作者 AI Team），核查跨本体架构、160M 参数与 5 Hz 机载推理、认知辅助目标、语音向量接口、成功/失败 rollout 共训机制及开源/下载链接缺失情况；模型代码、权重与训练数据均未公开，未复现或真机测试。
- **通读：** 已完成
- **相关世界模型：** [1X World Model 归档](1x-world-model-redwood.md)
- **一句话说明：** Redwood 是 EVE / NEO 的控制策略，官方介绍跨本体 transformer 表征、diffusion 动作解码、全身与移动操作。

## 官方技术摘录

1. **输入与输出：** 预训练语言嵌入（语音转文字 LLM 提取目标后经 offboard sentence encoder 编码）、预训练 ViT 视觉 token，加关节位置与施力历史的本体嵌入；经过多层 transformer 提取共享 latent，再由 diffusion policy 解码到 EVE 或 NEO 动作（手臂/灵巧手、步行与骨盆姿态联合预测）。
2. **认知辅助与失败数据利用：** 训练融合 EVE / NEO 在 1X 办公室及员工家庭采集的遥操作与自主轨迹；除动作扩散头外，还加入手部与目标物体图像位置等「认知（cognitive）」预测目标增强空间 grounding；**成功 rollout 同时监督动作扩散与认知预测，失败 rollout 则监督认知预测头**，减轻仅模仿成功样本带来的过拟合。
3. **部署：** 官方称约 160M 参数、完全运行在 NEO 机载嵌入式 GPU 上（约 5 Hz），支持无网络或弱网环境（地下室、户外）自主执行，可自主决定左手、右手或双手抓握，并展示靠墙支撑开门等多接触与全身控制行为。

## 开放范围与命名边界

- **未确认公开模型实现：** 发布页未列 Redwood 策略代码、权重或训练数据下载链接。
- Redwood AI **策略**与 Redwood AI World Model **动作条件预测模型**是不同发布物，不能把世界模型 Challenge 数据视作控制策略训练资产。
- 另一个 ArchitectLabs 同名 Redwood 为训练加速项目，亦与 1X 控制策略不同。

## 对 wiki 的映射

- [1X Redwood 控制策略](../../wiki/entities/1x-redwood-policy.md)
- [1X 公司与整机](../../wiki/entities/1x-technologies.md)
