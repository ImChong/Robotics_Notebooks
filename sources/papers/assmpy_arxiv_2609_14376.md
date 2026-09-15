# Orientation Control of Soft Robots via Adiabatic Spectral Submanifolds

> 来源归档（ingest）

- **标题：** Orientation Control of Soft Robots via Adiabatic Spectral Submanifolds
- **简称：** aSSMPy
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.14376>
- **PDF：** <https://arxiv.org/pdf/2609.14376>
- **代码：** <https://github.com/karakaron/aSSMPy>

- **入库日期：** 2026-09-15
- **索引来源：** [具身智能小站 9+EffVLA 盘点](../blogs/wechat_embodied_station_9_papers_resources_effvla_2026-09-15.md)
- **一句话说明：** 扩展 aSSM 降阶 + 姿态 MPC；FEM 两项任务相对误差 2.6%/2.4%，100 Hz 离散化。

## 开源状态（步骤 2.5，2026-09-15）

**结论：待核实（文内 GitHub 404）**

## 核心摘录

### 摘录 1

扩展 aSSM 降阶 + 姿态 MPC；FEM 两项任务相对误差 2.6%/2.4%，100 Hz 离散化。

**对 wiki 的映射：** [paper-assmpy-soft-robot-orientation](../../wiki/entities/paper-assmpy-soft-robot-orientation.md)

### 摘录 2（官方 abstract 要点，2026-09-15 补录）

- **论文题名：** *Orientation Control of Soft Robots via Adiabatic Spectral Submanifolds*。
- **动机：** 软体机器人常被用于精细环境中的 **安全关键交互**，因此 **位置与姿态** 都必须控得准。
- **难点：** MPC 需要一个 **既准确又便宜** 的模型，而软体机器人的动力学是 **无穷维非线性** 的。
- **理论基础：** **adiabatic spectral submanifolds（aSSM）** 的近期理论及其在软体机器人上的应用，提供 **数据驱动的模型降阶** 手段。
- **本文贡献：** ① 把 aSSM 辨识扩展到 **更大的可观测量数据集（enlarged observable datasets）**；② **升级** 现有的 aSSM-MPC 方案（本文首次把姿态纳入）。
- **评测：** 在 **压力驱动软体臂** 的 **高保真有限元（FEM）仿真** 上评估。
- **结果：** 相对既有 **数据驱动基线**，**位置与姿态跟踪误差均降低 60% 以上**。
- **补充（索引来源口径）：** 盘点侧另记 FEM 两项任务相对误差 **2.6% / 2.4%**、**100 Hz** 离散化；与 abstract 的「>60% 误差下降」为不同口径，引用时须分开标注。

**对 wiki 的映射：** 同上（补入该页「核心原理（方法）」「实验与评测」「与其他工作对比」三节）

## 当前提炼状态

- [x] 项目页/仓库核查
- [x] wiki 映射
