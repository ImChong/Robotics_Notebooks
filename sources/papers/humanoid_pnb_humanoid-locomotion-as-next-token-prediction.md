# Humanoid Locomotion as Next Token Prediction

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** Humanoid Locomotion as Next Token Prediction
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/Humanoid_Locomotion_as_Next_Token_Prediction/Humanoid_Locomotion_as_Next_Token_Prediction.html>
- **分类：** 03_High_Impact_Selection
- **子分类：** 行走经典
- **arXiv：** <https://arxiv.org/abs/2402.19469>
- **入库日期：** 2026-06-07
- **一句话说明：** 把真实人形 locomotion 写成「下一词预测」：用 因果 Transformer 对 传感–动作 token 序列 做自回归拟合，模态对齐 地预测下一 token；对缺动作的轨迹用 可学习 mask token 统一格式，从而吃进 RL 策略轨迹、MPC 观测、动捕与 YouTube 人体视频。仅用约 27 小时量级行走数据 训练即可 零样本 在旧金山多路面部署，并能泛化到如 后退行走 等训练外指令。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-humanoid-locomotion-as-next-token-prediction](../../wiki/entities/paper-notebook-humanoid-locomotion-as-next-token-prediction.md).

## 对 wiki 的映射

- [paper-notebook-humanoid-locomotion-as-next-token-prediction](../../wiki/entities/paper-notebook-humanoid-locomotion-as-next-token-prediction.md)
- 分类父节点：[paper-notebook-category-03-high-impact-selection](../../wiki/overview/paper-notebook-category-03-high-impact-selection.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/Humanoid_Locomotion_as_Next_Token_Prediction/Humanoid_Locomotion_as_Next_Token_Prediction.html>
- 论文：<https://arxiv.org/abs/2402.19469>
