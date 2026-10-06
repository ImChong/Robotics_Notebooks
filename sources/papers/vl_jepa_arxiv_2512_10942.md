# VL-JEPA: Joint Embedding Predictive Architecture for Vision-language（arXiv:2512.10942）

> 来源归档（paper）

- **标题：** VL-JEPA: Joint Embedding Predictive Architecture for Vision-language
- **arXiv：** <https://arxiv.org/abs/2512.10942>
- **HTML：** <https://arxiv.org/html/2512.10942v2>
- **作者：** Delong Chen、Mustafa Shukor、Théo Moutakanni、Willy Chung、Jade Yu、Tejaswi Kasarla、Yejin Bang、Allen Bolourchi、Yann LeCun、Pascale Fung
- **机构：** Meta FAIR；香港科技大学；Sorbonne Université；纽约大学（以论文 HTML 署名为准）
- **版本：** v2，2026-02-02
- **许可：** arXiv 页面列 CC BY 4.0
- **入库日期：** 2026-10-06
- **代码：** arXiv 页面未列出代码链接（核查日期：2026-10-06）
- **一句话说明：** 预测视觉上下文和文本 query 所对应的目标文本 embedding，需要时再用轻量 decoder 把 embedding 解码为文字。

## 论文要点

VL-JEPA 用视觉 encoder 得到视觉表征，用文本 encoder 得到目标文字表征，再训练 predictor 根据视觉表征和 query 预测目标文本 embedding。与直接自回归生成 token 的 VLM 相比，这个目标允许模型在语义向量空间学习，并支持按需解码。

论文在控制变量比较中报告约 50% 更少的可训练参数；选择性解码在维持相近性能时报告约 2.85 倍更少的解码操作。论文也评估分类、检索和 VQA。此工作是视觉-语言表示模型，不是输出机器人动作的 VLA，也不能单凭 latent prediction 视为动作条件世界模型。

## 对 wiki 的映射

- [VL-JEPA 实体页](../../wiki/entities/paper-vl-jepa.md)
- [VideoDB JEPA 长文实体页](../../wiki/entities/article-videodb-jepa-world-models.md)
