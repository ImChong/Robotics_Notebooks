# Ego Guidance: Enhancing Robot Policy Learning with Action-Free Human Egocentric Videos｜无动作标签 Ego 视频辅助机器人策略

- **类型：** paper
- **原文：** https://arxiv.org/abs/2611.13685
- **PDF：** https://arxiv.org/pdf/2611.13685
- **核对范围：** 核对论文 arXiv v1 摘要、用户 2026-11-12 资讯附件、官方源码 README 与公开入口；未通读全文方法、实验与附录，未执行实验。
- **通读：** 未完成
- **收录日期：** 2026-11-14
- **代码或数据：** https://github.com/llong-cs/EgoGuidance

## 核心摘录

1. 预训练视觉语言模型上共享机器人动作监督和人类第一视角视觉未来预测。
   - **对 wiki 的映射：** [方法与评测](../../wiki/entities/paper-ego-guidance.md)
2. 训练期使用双分支表征学习并关注视频筛选和分布差异。
   - **对 wiki 的映射：** [方法与评测](../../wiki/entities/paper-ego-guidance.md)
3. 推理时不必运行辅助未来视觉预测分支，仅部署动作策略。
   - **对 wiki 的映射：** [方法与评测](../../wiki/entities/paper-ego-guidance.md)

## 作者报告的结果与限制

摘要报告分布外任务与视频扩充的收益；新闻附件所列五项真机任务共 50 次测试，模型成功 20 次、基线 22 次，不支持所有真机任务均获提升的说法。

无动作标签视频与机器人执行域存在差距，少量真机结果难支持广泛泛化。

## 待补

需通读论文图表、正文和附录并逐项校验结果；当前内容为原始摘要及资讯整理，不替代原文。
