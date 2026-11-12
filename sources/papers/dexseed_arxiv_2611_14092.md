# DexSeed: Expanding Human Hand-Object Interactions into Robot Demonstrations for Dexterous Policy Learning / 将人手–物体交互扩展为灵巧操作示范

> 原始论文来源归档；这里只归纳论文的公开方法、数据与评测边界，不替代全文。

- **类型：** paper
- **原文：** https://arxiv.org/abs/2611.14092
- **全文：** https://arxiv.org/html/2611.14092v1
- **版本：** arXiv:2611.14092v1（2026-11）
- **核对范围：** 论文 HTML 的方法、实验与开放资源关键段落；附录、全部表格与逐项引用尚待完整复核。
- **通读：** 未完成
- **收录日期：** 2026-11-12
- **项目页：** https://robbyant-research.github.io/DexSeed/
- **官方代码：** https://github.com/robbyant-research/DexSeed

## 论文概要

DexSeed 是把少量人手–物体交互映射、扩增并物理验证为机器人灵巧抓取示范的数据合成框架。

## 机制要点

- 从少量人手交互种子出发，在模拟接触中生成机器人可执行示范。
- 核心步骤含抓取关系保持重定向、IK/轨迹扩增、RL 手指控制及 VLA 微调。
- 官方仓库仅发布介绍与素材，源码待发布；真机结果属于论文报告，尚未独立复现。

## 评测边界

论文报告由 19 条种子人手交互生成约 25,270 条机器人示范；仿真已见对象组合成功率 97.5%、未见组合配置约 91.25%；真实机器人所列八种抓取方式中，基线到方法的表现跨度约 20%→80%。实验协议与汇总方式需对照原表，不能误写成所有任务统一 80%。

## 待核实事项

- 完整核对原文附录、基线表、数据划分与失败案例。
- 独立运行复现实验、确认公开权重和训练脚本版本（如适用）。

## 对 wiki 的映射

- [DexSeed](../../wiki/entities/paper-dexseed.md)
