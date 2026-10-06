# Semantic Tube Prediction: Beating LLM Data Efficiency with JEPA（arXiv:2602.22617）

> 来源归档（paper）

- **标题：** Semantic Tube Prediction: Beating LLM Data Efficiency with JEPA
- **arXiv：** <https://arxiv.org/abs/2602.22617>
- **HTML：** <https://arxiv.org/html/2602.22617v1>
- **代码：** <https://github.com/galilai-group/llm-jepa#stp>
- **作者：** Hai Huang、Yann LeCun、Randall Balestriero
- **机构：** Atlassian；纽约大学；布朗大学（以论文 HTML 署名为准）
- **提交日期：** 2026-02-26（v1）
- **许可：** arXiv 页面列 CC BY 4.0
- **入库日期：** 2026-10-06
- **一句话说明：** 提出 Semantic Tube Prediction（STP），以语义轨迹的局部几何先验正则化语言模型隐藏状态，报告在 NL-RX-SYNTH 上用约 1/16 训练数据达到基线准确率。

## 论文要点

论文提出“Geodesic Hypothesis”，假设理想 token 序列对应的隐藏状态轨迹在语义流形上局部近似线性；STP 把训练轨迹约束在该路径附近的管状区域。该方法将 JEPA 式 latent prediction 思路用于语言建模，不需要显式的多视图增强。

论文报告 STP 改善训练信号噪声与生成多样性，并在 NL-RX-SYNTH 实验中展示数据效率结果。该结果是论文作者在指定数据集和评测设置下的报告，不等于普遍打破语言模型 scaling law，也不是机器人世界模型实验。

## 对本文的关系

VideoDB 长文用 STP 说明“下一个正确 token”与“隐藏状态保持在有用轨迹上”是不同目标。STP 研究对象是语言模型表征轨迹，不能直接作为动作条件控制器或机器人动力学模型。

## 对 wiki 的映射

- [Semantic Tube Prediction 论文实体页](../../wiki/entities/paper-semantic-tube-prediction.md)
- [VideoDB JEPA 长文实体页](../../wiki/entities/article-videodb-jepa-world-models.md)
