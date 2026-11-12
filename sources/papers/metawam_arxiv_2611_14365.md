# From Video Frames to Robot Meta-Frames: A Unified Latent Interface for World Action Models / 从视频帧到机器人元帧：世界–动作模型的统一潜空间接口

> 原始论文来源归档；这里只归纳论文的公开方法、数据与评测边界，不替代全文。

- **类型：** paper
- **原文：** https://arxiv.org/abs/2611.14365
- **全文：** https://arxiv.org/html/2611.14365v1
- **版本：** arXiv:2611.14365v1（2026-11）
- **核对范围：** 论文 HTML 的方法、实验与开放资源关键段落；附录、全部表格与逐项引用尚待完整复核。
- **通读：** 未完成
- **收录日期：** 2026-11-12
- **官方代码：** https://github.com/showlab/MetaWAM

## 论文概要

MetaWAM 将机器人动作、状态和价值信号塞入与视频扩散兼容的频域 meta-frame，形成统一世界–动作潜空间接口。

## 机制要点

- 采用正交 2D-DCT 频域 meta-frame 分槽编码 robot proprio、action、future proprio 和 value。
- 官方开源 README 给出 RoboCasa 24 任务、每任务 50 次、3 个随机种子的评测流程。
- 开放源码、RoboCasa 权重及训练脚本；实际评测尚未在本环境复现。

## 评测边界

论文 RoboCasa 24 任务平均成功率 68.9%（对照 Cosmos Policy 67.1%），推理延迟约 0.479 s → 0.291 s（约 -39%）；Franka 三项真机任务平均成功率 85.0%，OOD 设置 68.3%。这些数值是论文自报；时延不等于机器人低层控制周期。

## 待核实事项

- 完整核对原文附录、基线表、数据划分与失败案例。
- 独立运行复现实验、确认公开权重和训练脚本版本（如适用）。

## 对 wiki 的映射

- [MetaWAM](../../wiki/entities/paper-metawam.md)
