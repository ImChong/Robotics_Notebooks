# Model-Free Runtime Measurement of Task-Relevant Information｜AST 任务相关信息在线监测

- **类型：** paper
- **原文：** https://arxiv.org/abs/2611.14111
- **PDF：** https://arxiv.org/pdf/2611.14111
- **核对范围：** 核对论文 arXiv v1 摘要、用户 2026-11-12 资讯附件、官方源码 README 与公开入口；未通读全文方法、实验与附录，未执行实验。
- **通读：** 未完成
- **收录日期：** 2026-11-14
- **代码或数据：** https://github.com/dimipan/ast-telemetry

## 核心摘录

1. 领域接口把观测和目标转为可比较的任务解析状态。
   - **对 wiki 的映射：** [方法与评测](../../wiki/entities/paper-ast-task-information.md)
2. 固定因果监测器输出 progress 和 friction 两个连续指标。
   - **对 wiki 的映射：** [方法与评测](../../wiki/entities/paper-ast-task-information.md)
3. 可将 friction 用作运行诊断或 RL 惩罚，但需要领域任务接口。
   - **对 wiki 的映射：** [方法与评测](../../wiki/entities/paper-ast-task-information.md)

## 作者报告的结果与限制

作者在八类领域使用同一监控配置，覆盖网格探索、搜索救援、机械操作与意图估计；论文摘要报告对无效停滞的区分能力。

任务信息坐标需人为定义，假设已获得证据持续有效；无法代替实测成功率。

## 待补

需通读论文图表、正文和附录并逐项校验结果；当前内容为原始摘要及资讯整理，不替代原文。
