# spi_active_arxiv_2505_14266

> 来源归档（ingest）

- **标题：** Sampling-Based System Identification with Active Exploration for Legged Robot Sim2Real Learning
- **类型：** paper
- **作者：** Nikhil Sobanbabu, Guanqi He, Tairan He, Yuxiang Yang, Guanya Shi（CMU / LeCAR Lab）
- **arXiv：** <https://arxiv.org/abs/2505.14266v1>（v1，2025-05-20）
- **PDF：** <https://arxiv.org/pdf/2505.14266v1>
- **会议：** CoRL 2025 Oral；PMLR v305 pp.578–598 — <https://proceedings.mlr.press/v305/sobanbabu25a.html>
- **代码：** <https://github.com/LeCAR-Lab/SPI-Active>
- **项目页：** <https://lecar-lab.github.io/spi-active_/>
- **视频：** <https://youtu.be/pxyig4D1ZFs>
- **入库日期：** 2026-07-31
- **一句话说明：** SPI-Active 两阶段框架：GPU 并行采样（CMA-ES）最小化仿真–真实轨迹误差以辨识腿足质量/惯量与电机参数；再优化探索策略指令以最大化 Fisher 信息（D-最优），采集高信息量数据后 refinement；Go2 / G1 高精度技能零样本迁移，相对基线提升 42–63%。

## 核心论文摘录（MVP）

### 1) 问题与主张（Abstract）

- **链接：** <https://arxiv.org/abs/2505.14266v1>
- **核心贡献：** DR 依赖启发式、策略易过于保守；标准 SysID 依赖可微动力学和/或直接力矩测量，富接触腿足上常不成立。SPI-Active 用 **大规模并行采样** 做参数辨识，并用 **主动探索最大化 FIM** 提升数据信息量，从而缩小高精度 locomotion 的 sim2real gap。
- **对 wiki 的映射：**
  - [SPI-Active 实体](../../wiki/entities/paper-notebook-sampling-based-system-identification-with-active.md)
  - [Domain Randomization](../../wiki/concepts/domain-randomization.md)
  - [System Identification](../../wiki/concepts/system-identification.md)

### 2) Stage-1 SPI — 采样式参数辨识

- **链接：** 项目页 Method 段；深读笔记归纳
- **核心贡献：** 在 Isaac Gym 等 GPU 并行仿真中采样候选参数（base mass、CoM、惯量、模块化电机扭矩模型），最小化「仿真重放 vs 真机轨迹」状态预测误差；**无需可微动力学、无需力矩传感器**，仅用标准状态轨迹。
- **对 wiki 的映射：**
  - [CMA-ES](../../wiki/methods/cma-es.md)
  - [PACE](../../wiki/entities/paper-pace-sim2real-legged-robots.md)（同为采样/CMA-ES 足式辨识，参数层不同）

### 3) Stage-2 Active — 指令序列优化最大化 FIM

- **链接：** 仓库 `active_sysid.md`；论文主动探索节
- **核心贡献：** 先训多行为 omni locomotion controller；再用 CMA-ES/Optuna 优化指令序列（如 `lin_vel_x`、`ang_vel_yaw`、`gait_phase`）以最大化 Fisher 信息（D-最优实验设计），激发高扭矩、高信息量步态后重新辨识。
- **对 wiki 的映射：**
  - [Sim2Real Gap 缩减](../../wiki/queries/sim2real-gap-reduction.md)
  - [Sim2Real 闭环工程](../../wiki/queries/sim2real-closed-loop-engineering.md)

### 4) 下游任务与开源边界

- **链接：** <https://github.com/LeCAR-Lab/SPI-Active>；`spigym/envs/downstream_tasks.md`
- **核心贡献：** 辨识参数后训练/评估前跳、偏航跳、速度跟踪、姿态跟踪等；摘要报告相对基线 **42–63%** 提升。开源仓已放 SPI / Active / Downstream training；**Dataset Replay & Visualize、Sim2real 部署仍待发布**。
- **对 wiki 的映射：**
  - [SPI-Active 仓库归档](../../sources/repos/spi-active.md)
  - [SPI-Active 项目页归档](../../sources/sites/spi-active.md)
  - [Paper Notebooks 笔记锚点](../../sources/papers/humanoid_pnb_spi-active.md)

## 对 wiki 的映射（汇总）

- 主实体：[paper-notebook-sampling-based-system-identification-with-active](../../wiki/entities/paper-notebook-sampling-based-system-identification-with-active.md)
- 概念：[system-identification](../../wiki/concepts/system-identification.md)、[sim2real](../../wiki/concepts/sim2real.md)、[domain-randomization](../../wiki/concepts/domain-randomization.md)
- 方法：[cma-es](../../wiki/methods/cma-es.md)
- 对照：[paper-pace-sim2real-legged-robots](../../wiki/entities/paper-pace-sim2real-legged-robots.md)、[sage-sim2real-actuator-gap-estimator](../../wiki/entities/sage-sim2real-actuator-gap-estimator.md)、[paper-fada-humanoid](../../wiki/entities/paper-fada-humanoid.md)

## 参考来源（原始）

- arXiv：<https://arxiv.org/abs/2505.14266>
- PMLR：<https://proceedings.mlr.press/v305/sobanbabu25a.html>
- 项目页：<https://lecar-lab.github.io/spi-active_/>
- 代码：<https://github.com/LeCAR-Lab/SPI-Active>
- 深读笔记：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/10_Sim-to-Real/SPI-Active__Sampling-Based_System_Identification_with_Active_Exploration/SPI-Active__Sampling-Based_System_Identification_with_Active_Exploration.html>
