# Joga 项目页（MIT CSAIL）

> 来源归档（ingest）

- **标题：** Joga: Learning Agile Humanoid Soccer Skills with Active Vision
- **类型：** site / project-page
- **官方入口：** <https://joga.csail.mit.edu/>
- **论文 PDF：** <https://joga.csail.mit.edu/static/joga.pdf>
- **机构：** MIT Improbable AI Lab
- **作者：** Avinash Narula、Khai Nguyen、Nolan Fey、Venki Parsuram、John Marangola、Martin Peticco、Pulkit Agrawal
- **入库日期：** 2026-10-20
- **一句话说明：** Unitree G1 全身策略控制躯干式主动视觉颈部，支持敏捷运球、接球、传球和射门；用短时真机数据对齐仿真中的感知和执行器误差。
- **开源状态（2026-10-20 核查）：** **部分开源** — 官方页面链接到 [Joga Neck 硬件仓库](https://github.com/mfpeticco/joga-neck)，含打印件、CAD/STEP、URDF、BOM 与装配说明；未列策略训练代码、部署代码或模型权重。

## 页面公开信息

- **任务与平台：** Unitree G1 人形足球；每个策略控制全身及携带 RGB 相机的 2-DoF 主动颈部。
- **感知对齐：** PAM 使用 10 分钟配对机载检测与动捕数据，拟合球检测偏差、噪声及漏检。
- **动力学对齐：** WB-UAN 使用 10 分钟真机运球数据，学习全身关节位置命令修正，再在修正仿真中微调运球。
- **技能衔接：** 从运球访问到的状态分布微调传球与射门，使切换可发生在运动过程中。
- **公开结果：** 持续运球 1.71 ± 0.16 m/s、估计峰值 2.31 m/s、135° 急转、最高 5.2 m/s 来球接球。
- **部署边界：** 项目页称 PAM、WB-UAN 留在仿真训练侧；真机运行技能策略、球检测器和本体感知。

## 对 wiki 的映射

- [Joga 人形足球实体页](../../wiki/entities/paper-joga-humanoid-soccer.md)
- [Humanoid Soccer 任务页](../../wiki/tasks/humanoid-soccer.md)
- [论文摘录](../papers/joga_learning_agile_humanoid_soccer.md)
- [Joga Neck 硬件仓库](../repos/joga-neck.md)
