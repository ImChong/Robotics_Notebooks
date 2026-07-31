# SPI-Active 项目页

> 来源归档（site / project page）

- **标题：** Sampling-Based System Identification with Active Exploration for Legged Robot Sim2Real Learning
- **类型：** project page
- **URL：** <https://lecar-lab.github.io/spi-active_/>
- **论文：** <https://arxiv.org/abs/2505.14266>
- **PMLR：** <https://proceedings.mlr.press/v305/sobanbabu25a.html>
- **代码：** <https://github.com/LeCAR-Lab/SPI-Active>
- **视频：** <https://youtu.be/pxyig4D1ZFs>
- **机构：** CMU / LeCAR Lab
- **会议：** CoRL 2025（ORAL）
- **核查日期：** 2026-07-31
- **一句话说明：** SPI-Active 项目页展示两阶段 SysID（采样辨识 + 主动探索最大化 Fisher 信息）及 Go2 / G1 真机对比：前跳、偏航跳、速度跟踪、姿态跟踪与编织杆导航等，并链接 arXiv 与官方 GitHub。

## 开源状态（项目页核查，2026-07-31）

- 项目页提供论文叙事、方法示意与 **真机对比视频**；代码入口指向 [LeCAR-Lab/SPI-Active](https://github.com/LeCAR-Lab/SPI-Active)。
- **未在项目页另挂** 独立权重库或公开数据集下载页；以 GitHub 仓为准。
- 对照仓库 README TODO：**SPI / Active Exploration / Downstream training 已发布**；**Dataset Replay & Visualize、Sim2real 仍待发布** → 记为 **部分开源**（详见 [`sources/repos/spi-active.md`](../repos/spi-active.md)）。

## 核心摘录（归纳，非全文）

- **问题：** DR 易保守；传统 SysID 常需可微动力学与直接力矩测量——富接触腿足上不成立。
- **方法：** Stage-1 用大规模并行采样最小化仿真–真实轨迹误差，估计关键物理参数；Stage-2 优化多行为策略的输入指令以最大化 FIM，采集高信息量数据后 refinement。
- **任务对比（相对 Vanilla）：** Forward Jump、Yaw Jump、Velocity Tracking、Attitude Tracking；另有 G1 人形速度跟踪与 open-loop weave pole 导航演示。
- **主张数字：** 多类 locomotion 任务上相对基线 **提升 42–63%**（与摘要一致）。

## 对 wiki 的映射

- [SPI-Active 实体页](../../wiki/entities/paper-notebook-sampling-based-system-identification-with-active.md)
- [System Identification](../../wiki/concepts/system-identification.md)
- [Sim2Real](../../wiki/concepts/sim2real.md)

## 参考来源（原始）

- 项目页：<https://lecar-lab.github.io/spi-active_/>
- arXiv：<https://arxiv.org/abs/2505.14266>
- 代码：<https://github.com/LeCAR-Lab/SPI-Active>
