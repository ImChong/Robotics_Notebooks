# Elastic-WAM: Adaptive Spatiotemporal Imagination for Efficient Robot Control / 自适应时空想象的高效机器人控制

> 原始论文来源归档；这里只归纳论文的公开方法、数据与评测边界，不替代全文。

- **类型：** paper
- **原文：** https://arxiv.org/abs/2611.14257
- **全文：** https://arxiv.org/html/2611.14257v1
- **版本：** arXiv:2611.14257v1（2026-11）
- **核对范围：** 论文 HTML 的方法、实验与开放资源关键段落；附录、全部表格与逐项引用尚待完整复核。
- **通读：** 未完成
- **收录日期：** 2026-11-12
- **项目页：** https://lihaitian.com/Elastic-WAM/

## 论文概要

Elastic-WAM 是为世界–动作模型动态分配视频未来想象计算的控制方法，决定何时停止、做多少步、更新哪些区域。

## 机制要点

- 学习时空精炼需求场，动态决定未来视频区域刷新与停止时机。
- RoboCasa-GR1 官方报告 57.4% / 388.2ms，相比 Joint-WAM 55.0% / 572.6ms；其他对照并不全部更慢。
- 无已核实的官方实现入口，尚不能重跑实际时延/性能实验。

## 评测边界

论文列出 LIBERO 98.8%、LIBERO-Plus 73.7%、LIBERO-Pro 54.1%；RoboCasa-GR1 成功率 57.4%，相应推理约 388.2ms，所比较 Joint-WAM 为 55.0% / 572.6ms。RoboCasa 上并非比所有优化基线都快，Fast-WAM 在论文对比中约 331ms。真实机器人茶饮等任务展示了不同干扰条件下的表现，仍需结合任务表看泛化。

## 待核实事项

- 完整核对原文附录、基线表、数据划分与失败案例。
- 独立运行复现实验、确认公开权重和训练脚本版本（如适用）。

## 对 wiki 的映射

- [Elastic-WAM](../../wiki/entities/paper-elastic-wam.md)
