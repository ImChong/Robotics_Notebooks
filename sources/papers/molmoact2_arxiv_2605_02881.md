# MolmoAct2（arXiv:2605.02881）

- **标题：** MolmoAct2: Action Reasoning Models for Real-world Deployment
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2605.02881>
- **项目页/博客：** <https://allenai.org/blog/molmoact2>
- **代码：** <https://github.com/allenai/molmoact2>
- **机构：** Allen Institute for AI（Ai2）
- **入库日期：** 2026-09-21
- **一句话说明：** 开源 action reasoning VLA 家族：Molmo2-ER 骨干 + flow-matching 连续动作专家；含 base/finetuned 权重、ER 数据集与 LeRobot 集成。

## 开源状态

- **已开源**（Apache-2.0）：`allenai/molmoact2` 训练/评测/部署代码；HF 模型与 MolmoAct2-BimanualYAM 等数据集。

## 核心摘录

1. **Real-world 部署导向：** 相对 frontier 闭源 VLA，强调可复现 fine-tune、真机 Franka/SO-100/YAM 与 ManiSkill 零样本评测。
2. **Molmo2-ER 骨干：** 具身推理 VLM 与动作头解耦；ER 数据集单独发布。
3. **生态：** HuggingFace LeRobot 官方 MolmoAct2 policy；MolmoSpace leaderboard 第一 VLA。

## 对 wiki 的映射

- [paper-molmoact2](../../wiki/entities/paper-molmoact2.md)
- [molmo-er](../../wiki/entities/molmo-er.md)
- [VLA](../../wiki/methods/vla.md)
