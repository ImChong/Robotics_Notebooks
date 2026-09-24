# VLMs Can Describe, But Not Measure: Object-Centric Scene Understanding for Robotic Manipulation

> 来源：[具身智能小站 · 13 篇盘点](../../sources/blogs/wechat_embodied_13_papers_forgetmimic_2026-09-24.md)（2026-09-24）

## 元数据

- **arXiv：** [2609.28184](https://arxiv.org/abs/2609.28184)
- **PDF：** https://arxiv.org/pdf/2609.28184
- **代码：** https://github.com/idra-lab/plantorv
- **项目页：** https://www.github.com/idra-lab/plantorv
- **开源结论（2026-09-24）：** **已开源**

## 核心摘录

- **一句话：** VLM 擅长语义描述但不等于可靠几何；框架把 VLM 标注与 RGB-D 几何拆开再合成对象级表示。
- **机制：** VLM 产语义标签；RGB-D 分支产定位与深度；融合为 object-centric scene graph 供操作栈消费。
- **指标：** **151** 场景验证从描述到可执行感知（具体指标以 PDF 为准）。

## 对 wiki 的映射

- 实体页：[PLANTORV](../../wiki/entities/paper-plantorv.md)
