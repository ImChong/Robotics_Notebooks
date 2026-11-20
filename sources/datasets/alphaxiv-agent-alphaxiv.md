# AgentAlphaXiv：A Society of AI Scientists 开放研究归档

- **类型：** 官方数据集 / Agent 科研产物、评审和沟通归档
- **数据集：** <https://huggingface.co/datasets/alphaXiv/AgentAlphaXiv>
- **论文：** <https://www.alphaxiv.org/abs/2611.society-ai-scientists>
- **许可：** 数据集卡显示 CC-BY-4.0
- **入库日期：** 2026-11-20
- **核对范围：** 官方 Dataset Card、数据浏览器字段 `in_paper_cohort`、子集数和版本/时间范围；尚未下载完整数据。
- **通读：** 未完成

## 一句话

官方公开的 AI Agent 科研论文、实验与社区活动归档，是复查论文定量陈述与原始 artifact 的主要入口。

## 数据使用注意

- Hugging Face 浏览器显示 **8 个 subset**；`studies` 子集在 2026-11-20 公开页面呈现 **961 行**，并包含 `in_paper_cohort` 布尔字段。
- 论文研究截至 2026-11-12 的锁定样本是 **936 篇研究**；数据集继续维护后的总行数不能当成论文 cohort。
- 复现研究必须先筛选论文 cohort、保留版本时间戳与 lab 分组；某一实验结果能执行，不代表论文的全部推论正确。
- 本次仅核对公开数据目录与字段，未完成全量统计复算。

## 对 wiki 的映射

- [A Society of AI Scientists 论文详情](../../wiki/entities/paper-society-of-ai-scientists.md)。
- [论文原始材料](../papers/society_ai_scientists_alphaxiv_2611.md)。

## 参考来源

- [alphaXiv/AgentAlphaXiv](https://huggingface.co/datasets/alphaXiv/AgentAlphaXiv)
