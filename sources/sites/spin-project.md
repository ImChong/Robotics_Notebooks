# SPIN 官方项目页：Self-Play Fine-Tuning

> 项目站点一手资料归档

- **类型：** site / paper project
- **项目页：** https://uclaml.github.io/SPIN/
- **论文：** https://arxiv.org/abs/2401.01335v3
- **代码：** https://github.com/uclaml/SPIN （[源码归档](../repos/spin-uclaml.md)）
- **机构：** UCLA
- **入库与核查：** 2026-11-17
- **开源状态：** 已开源（训练/数据生成代码、数据和模型资源有公开入口）
- **对应节点：** [SPIN 自博弈微调](../../wiki/entities/paper-spin-self-play-fine-tuning.md)

## 项目定位与一手主张

- 基于上一轮语言模型生成答案，与既有 SFT 参考答案组成可学习的比较对。
- 论文把这种迭代更新解释为 self-play fine-tuning，重点是语言模型训练后的自我改进，而非机器人实体之间对抗。
- 官网与官方 GitHub 提供原始论文、代码，以及模型/数据资源链接；具体复现入口以代码仓库 README 为准。

## 可用性与边界

- **已开源：** https://github.com/uclaml/SPIN；Hugging Face 模型、数据链接详见官方 README。
- **注意：** README 明确修正 2024-04-04 曾错误发布的数据，并提示 SFT checkpoint 应与历史实验版本对齐。
- **未核验：** 未进行本地模型下载、多卡训练或数值复现；开源状态不等于复现已通过。