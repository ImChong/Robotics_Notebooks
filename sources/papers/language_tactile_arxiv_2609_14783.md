# Language-Guided Representation Learning for Robust Cross-Sensor Material Recognition

> 来源归档（ingest）

- **标题：** Language-Guided Representation Learning for Robust Cross-Sensor Material Recognition
- **简称：** Language-Tactile
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.14783>
- **PDF：** <https://arxiv.org/pdf/2609.14783>
- **代码：** <https://github.com/Mashood3624/Language_Tactile>
- **项目页：** <https://mashood3624.github.io/Language_Tactile/>
- **入库日期：** 2026-09-15
- **索引来源：** [具身智能小站 9+EffVLA 盘点](../blogs/wechat_embodied_station_9_papers_resources_effvla_2026-09-15.md)
- **一句话说明：** 语言描述作硬件无关语义监督；~39K 样本、100-shot ~95%，跨传感器平均 +13.3%。

## 开源状态（步骤 2.5，2026-09-15）

**结论：已开源**

## 核心摘录

### 摘录 1

语言描述作硬件无关语义监督；~39K 样本、100-shot ~95%，跨传感器平均 +13.3%。

**对 wiki 的映射：** [paper-language-guided-tactile](../../wiki/entities/paper-language-guided-tactile.md)

### 摘录 2（官方 abstract 要点，2026-09-15 补录）

- **论文题名：** *Language-Guided Representation Learning for Robust Cross-Sensor Material Recognition*。
- **问题：** 视觉式触觉传感器因 **光学、弹性体特性、照明** 差异，对同一材料给出不同观测；单传感器或多传感器直接训练泛化都差。
- **核心主张：** **语言** 编码的触觉高层语义属性（rough / soft / slippery 等）**跨硬件不变**，因此可作天然的 **sensor-agnostic 监督信号**。
- **方法：** **language-guided distillation** 框架 — 训练触觉编码器，把 **传感器特定的触觉图像** 与 **语言 embedding** 对齐到共享语义空间。
- **数据：** 构建 **39K 样本** 的 touch-language 数据集，含 **人工标注的材料标签**。
- **评测设定：** few-shot 学习 + **跨传感器迁移**；并在 **6 个既有触觉数据集** 上做基准对照。
- **结果：** **100-shot 设定 95% 准确率**；跨传感器迁移平均 **+13.3%** 准确率；在 6 个既有数据集上最高 **+19%**。
- **开源：** 代码与数据 <https://mashood3624.github.io/Language_Tactile/>

**对 wiki 的映射：** 同上（补入该页「核心原理（方法）」「实验与评测」「与其他工作对比」三节）

## 当前提炼状态

- [x] 项目页/仓库核查
- [x] wiki 映射
