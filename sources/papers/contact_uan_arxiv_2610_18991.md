# Contact-UAN（arXiv:2610.18991）

> 来源归档（ingest）

- **标题：** Contact-UAN: Unsupervised Actuator Networks as a Residual World Model for Humanoid Control
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2610.18991>
- **HTML：** <https://arxiv.org/html/2610.18991>
- **PDF：** <https://arxiv.org/pdf/2610.18991>
- **作者：** Nolan Fey、Khai Nguyen、Yash Prabhu、Gabriel B. Margolis、Pulkit Agrawal
- **项目页：** <https://contact-uan.csail.mit.edu/>
- **代码：** <https://github.com/nolie-rolie/contact-uan>
- **论文状态：** arXiv v1，2026-10-13 提交；项目页标注 CoRL 2026
- **一句话说明：** 面向 Unitree G1 接触丰富运动的执行器残差 Sim2Real：从短时真机本体感知数据学习 UAN，保留仿真接触与刚体物理，只校正执行器命令通道。

## 核心摘录

### 1. 问题与方法
仿真策略可能在真机上复现动作外观，却无法复现支撑力、冲击吸收或力跟踪；只做宽范围域随机化也可能让策略过于保守。Contact-UAN 将仿真器作为物理先验，学习残差修正来匹配硬件 rollout，而非从零拟合完整世界模型。

### 2. 接触一致的数据重放
硬件轨迹在仿真中回放前，先用本体感知估计基座速度、足端接触状态与支撑脚位姿，构造接触力一致的仿真初始状态。否则毫米级高度或角度误差造成的接触力差异，可能被 UAN 错学成执行器误差。

### 3. 残差接口与下游训练
UAN 对基线关节位置目标添加残差校正；论文使用约 6.5 分钟 G1 行走数据训练。冻结 UAN 后，在增强仿真中分别训练跑步、力控制、顺应、阻抗控制和跳跃策略。部署时只运行下游任务策略，UAN 与状态估计器用于离线数据重标和仿真训练。

### 4. 论文报告的结果
项目页报告：跑步速度约 3.13 m/s、运行扭矩使用降低约 20%、力跟踪误差降低约 43%，并展示动态跳跃、柔顺与扰动恢复。数值来自论文定义的测试协议，不代表任意硬件或任务上的保证。

### 5. 开源状态与复现边界（2026-10-16）
- 项目页提供 GitHub 官方代码入口；仓库含 WTW 训练、硬件数据收集、六类本体感知估计器、数据重标、UAN/下游策略训练及部署导出流程。
- 预训练策略与硬件数据集未随仓库提供；复现需要自己的 Unitree G1 和采集数据。
- 仓库说明 UAN 不导出到真机控制器；硬件部署运行导出的下游策略。
- 项目页与仓库 README 分别见 [项目页归档](../sites/contact-uan-csail-mit-edu.md) 与 [代码归档](../repos/nolie-rolie-contact-uan.md)。

## 对 wiki 的映射

- [Contact-UAN 实体页](../../wiki/entities/paper-contact-uan.md)
- [Sim2Real 概念页](../../wiki/concepts/sim2real.md) — 执行器残差与接触一致重放
