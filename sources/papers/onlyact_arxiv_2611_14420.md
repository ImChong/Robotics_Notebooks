# OnlyAct: Extracting Action Representations from Foundation Models for World Models｜从视频基础模型提取动作表征

- **类型：** paper
- **原文：** https://arxiv.org/abs/2611.14420
- **PDF：** https://arxiv.org/pdf/2611.14420
- **核对范围：** 核对论文 arXiv v1 摘要、用户 2026-11-12 资讯附件、官方源码 README 与公开入口；未通读全文方法、实验与附录，未执行实验。
- **通读：** 未完成
- **收录日期：** 2026-11-14
- **代码或数据：** https://github.com/runjiali-rl/onlyact

## 核心摘录

1. 从无标签视频帧对读取预训练 V-JEPA 2 视频表征。
   - **对 wiki 的映射：** [方法与评测](../../wiki/entities/paper-onlyact.md)
2. 相同运动跨场景的帧对拉近、同场景不同运动的帧对推远，不依赖动作标签。
   - **对 wiki 的映射：** [方法与评测](../../wiki/entities/paper-onlyact.md)
3. 按动作可表达性、场景泄露与动作驱动视频生成评测，不直接等同真机成功率。
   - **对 wiki 的映射：** [方法与评测](../../wiki/entities/paper-onlyact.md)

## 作者报告的结果与限制

官方项目披露训练 9 个数据集共 168,000 帧对，LoRA 7.1M 可训练参数；13 个数据集的线性探针平均 R²=0.490。

场景解耦依赖样本构造；视频动作转移与动力学控制存在鸿沟。

## 待补

需通读论文图表、正文和附录并逐项校验结果；当前内容为原始摘要及资讯整理，不替代原文。
