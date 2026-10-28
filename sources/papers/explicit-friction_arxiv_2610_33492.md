# Explicit Frictional Contact Modeling in Neural Garment Simulation

- **类型：** paper
- **论文：** <https://arxiv.org/abs/2610.33492>（2026-10-24 提交）
- **项目页：** <https://trthanhnguyen.github.io/explicit-friction-ngs/>（[核查记录](../sites/explicit-friction-ngs.md)）
- **作者：** Thanh-Nguyen Truong、Vu-Hoang Tran、Ching-Chun Huang
- **入库日期：** 2026-10-28
- **一句话说明：** 神经布料预测器之后加入可调摩擦的接触算子，估计法向载荷、执行库仑修正并传播布料运动。

## 摘录

- 问题与方法：神经布料预测器之后加入可调摩擦的接触算子，估计法向载荷、执行库仑修正并传播布料运动。
- 评测或数据：在 HOOD 和 ContourCraft 上，摩擦系数增大时黏附增加、滑移减少；论文报告相对参考求解器约八分之一的运行时间。
- 边界：结果针对论文所测的布料与人体接触；不能据此认定适用于任意机器人接触。项目页访问失败，代码与数据开放状态未核实。

## 对 wiki 的映射

- [显式摩擦布料仿真](../../wiki/entities/paper-explicit-friction.md)
