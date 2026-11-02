# CusADi2（arXiv:2610.25476）

> 来源归档（paper）

- **标题：** CusADi2: Latency-Optimized Batch Computation of Robot Kinematics, Jacobians, and Hessians on the GPU
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.25476>
- **PDF：** <https://arxiv.org/pdf/2610.25476>
- **代码：** <https://github.com/johnviljoen/cusadi2>
- **入库日期：** 2026-11-01
- **线索来源：** [模型还在思考，目标已经移动：机器人评测该把时间算进来吗？](../blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md)（具身智能小站，2026-11-01）
- **一句话说明：** CusADi2在批次内部继续并行计算运动学及其导数，面向在线规划的低延迟需求；覆盖多种机器人和控制示例，优势随批量大小与计算图结构变化。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-11-01）：仓库可达（git ls-remote）。

## 核心摘录（公众号转述，未核验原文表格）

- GPU批量计算，也要降低单批延迟。
- CusADi2在批次内部继续并行计算运动学及其导数，面向在线规划的低延迟需求；覆盖多种机器人和控制示例，优势随批量大小与计算图结构变化。
- 指标：公众号未给具体数字，本页不引用。

## 对 wiki 的映射

- [paper-cusadi2](../../wiki/entities/paper-cusadi2.md)
- [动态评测、技能编译与环境潜变量适配：具身智能小站 11-01 期 56 篇阅读坐标](../../wiki/overview/embodied-station-2026-11-01-56-papers-technology-map.md)
