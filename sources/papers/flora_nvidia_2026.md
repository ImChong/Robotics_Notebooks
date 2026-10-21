# Flora: An Open Model Suite for Scaling Robotics Data Engines

> 来源归档（paper）

- **标题：** Flora: An Open Model Suite for Scaling Robotics Data Engines
- **机构：** NVIDIA
- **类型：** paper / robotics data engine / video annotation / reward models
- **发布日期：** 2026-10-20（论文 PDF）
- **论文：** <https://research.nvidia.com/labs/gear/flora/assets/paper/flora.pdf>
- **项目页：** <https://research.nvidia.com/labs/gear/flora/>
- **模型：** <https://huggingface.co/collections/nvidia/flora>
- **一句话说明：** Flora 将视频分段、时间戳动作描述、结果奖励、过程奖励和 rubric 评分组合成机器人数据标注模型套件，论文称该套件用于训练 GR00T 模型。

## 摘要级要点

- Flora 包含 captioning、temporal segmentation 与 reward 模型，可输出时间戳字幕、过程奖励、结果奖励，并支持测试时 rubric 奖励。
- NVIDIA 报告其系统每天可标注 200,000 小时数据，使用 512 张 GB300 GPU；该规模数据是论文披露的内部生产运行指标。
- 论文在 WGO-Bench、RoboReward、Robometer 等评测中报告结果，并称相较 API 服务有显著成本优势；不同任务的指标不可混为单一质量分数。
- 项目页列出论文与 Hugging Face 模型入口；页面未提供可核验的 GitHub 源码仓库链接。Hugging Face collection 收录 Flora-Captioner、Flora-TempSeg、Flora-Rewarder 权重。

## 对 wiki 的映射

- [Flora 数据引擎实体页](../../wiki/entities/flora-robotics-data-engine.md)
- [World Action Models](../../wiki/concepts/world-action-models.md)
- [灵巧操作数据管线 Query](../../wiki/queries/dexterous-manipulation-data-pipeline.md)
