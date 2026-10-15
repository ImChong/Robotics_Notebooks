# PAMI 项目页（coral79.github.io/pami）

> 来源归档（ingest · 项目页）

- **标题：** PAMI: Part Anchored Motion for Text to Human-Object Interaction Generation
- **类型：** site（项目页）
- **URL：** <https://coral79.github.io/pami/>
- **论文：** <https://arxiv.org/abs/2609.38466>（归档见 [pami_arxiv_2609_38466.md](../papers/pami_arxiv_2609_38466.md)）
- **代码：** <https://github.com/Coral79/PAMI-Code>（见 [pami-code.md](../repos/pami-code.md)）
- **数据集：** InterAct；项目页与论文用于评测，数据准备脚本尚未发布
- **入库日期：** 2026-10-15

## 页面要点

- PAMI（Part-Anchored Motion for Interaction）面向文本条件的人体-物体交互动作生成。
- 人体 root、手腕、脚踝等部位锚点分别提供物体相对运动候选，并由学习到的逐帧权重融合。
- 页面介绍 PamiVAE、PamiGen 与 PamiRefiner：从动作表示与文本条件生成，到借助表面线索改进接触。
- 作者报告 InterAct 上接触召回比先前最佳方法高 14.5%。

## 源码与数据开放核查

截至 2026-10-15，项目页链接到官方 GitHub 仓库；仓库 README 有数据预处理、权重、推理和训练代码的待发布清单，页面内容注明代码与模型仍在准备发布。因此目前属于**官方仓库已建、可运行实现待发布**，不能据此声称已可复现。InterAct 数据集需遵循其自身获取方式；PAMI 仓库尚未提供转换脚本。

## 对 wiki 的映射

- [PAMI 实体页](../../wiki/entities/paper-pami-part-anchored-motion.md)
- [论文来源归档](../papers/pami_arxiv_2609_38466.md)
- [官方代码归档](../repos/pami-code.md)
