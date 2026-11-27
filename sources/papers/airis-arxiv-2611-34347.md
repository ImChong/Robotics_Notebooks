# AIRIS: Benchmarking Social Embodied Intelligence in Egocentric Streaming Videos

- **类型：** 论文预印本
- **作者：** Yuchen Zhou、Hongsheng Wu、Jiayu Tang、Yunxuan Mao、Hanjing Ye、Shenghao Zhang、Jing Tang、Chun-Kai Fan、Yufei Huang、Yonggen Ling、He Zhang
- **机构：** 腾讯 Robotics X 实验室
- **来源日期：** 2026-11-25
- **原文：** [arXiv:2611.34347v1](https://arxiv.org/abs/2611.34347v1) · [HTML](https://arxiv.org/html/2611.34347v1) · [PDF](https://arxiv.org/pdf/2611.34347v1)
- **项目页：** [AIRIS](https://airis-bench.github.io/)（[项目归档](../sites/airis.md)）
- **代码及标注：** [AIRIS-bench/AIRIS](https://github.com/AIRIS-bench/AIRIS)（[源码归档](../repos/airis.md)）
- **通读：** 已完成
- **核对范围：** arXiv v1 正文 §1–4，图 1–4、主结果表 2；补充材料中能力定义、指标、完整模型结果及题目示例。

## 技术摘录

1. 同一互动片段按 T0–T3 对齐，每题仅给当前时刻的因果视频前缀；提前断言不是有效的意图识别。映射：核心机制。
2. IAS 同时考查正确稳定判断的时机和提前承诺的频率；反应 Joint-EM 不允许各子判断在不同样本上分别正确来掩盖组合失败。映射：核心机制与实验。
3. 评测是第一视角视频封闭选项任务，不是在环机器人控制实验。映射：局限与风险。

对应 [AIRIS 节点](../../wiki/entities/paper-airis.md)。
