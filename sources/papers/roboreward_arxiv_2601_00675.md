# RoboReward: General-Purpose Vision-Language Reward Models for Robotics（arXiv:2601.00675）

> 来源归档（paper）

- **论文：** <https://arxiv.org/abs/2601.00675>
- **项目 / 基准页：** <https://crfm.stanford.edu/helm/robo-reward-bench/>
- **数据：** <https://huggingface.co/datasets/teetone/RoboReward>
- **模型：** <https://huggingface.co/teetone/RoboReward-8B>
- **作者 / 机构：** Tony Lee、Andrew Wagenmaker、Karl Pertsch、Percy Liang、Sergey Levine、Chelsea Finn；Stanford University、UC Berkeley
- **一句话说明：** 从成功占多数的机器人轨迹中构造失败与部分进度样本，建立机器人奖励数据集/基准并训练 4B/8B 视觉语言奖励模型。
- **入库日期：** 2026-10-07

## 核心摘录

1. **数据问题：** OXE 等真实机器人语料偏成功轨迹，缺少可靠失败标签。
2. **数据构造：** 对成功轨迹做反事实重标注生成 negatives / near-misses，并按时间裁剪得到部分进度结果。
3. **模型与评测：** RoboReward 数据集和基准覆盖多任务、多本体；4B 与 8B 模型在短时程任务奖励判断上胜过更大的通用 VLM，但不同模型没有在所有任务上都领先。
4. **下游使用：** 8B 模型接入真机 RL，论文报告其比 Gemini Robotics-ER 1.5 更能改善策略学习，并缩小与人类奖励 RL 的差距。
5. **开放状态：** 项目网站发布数据、模型和评测套件；目前归档的是模型/数据入口，未核实到官方训练代码仓库链接。

## 对 wiki 的映射

- [paper-roboreward](../../wiki/entities/paper-roboreward.md)
- [progress-reward-modeling](../../wiki/concepts/progress-reward-modeling.md)
