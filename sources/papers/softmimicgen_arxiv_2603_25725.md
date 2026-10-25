# SoftMimicGen: A Data Generation System for Scalable Robot Learning in Deformable Object Manipulation

> 来源归档（ingest · 2026-10-25）

- **类型：** paper / open-source software
- **论文：** https://arxiv.org/abs/2603.25725
- **项目主页（视频与实验）：** https://softmimicgen.github.io/
- **官方代码：** https://github.com/NVlabs/SoftMimicGen
- **作者：** Masoud Moghani, Mahdi Azizian, Animesh Garg, Yuke Zhu, Sean Huver, Ajay Mandlekar
- **机构：** NVIDIA、University of Toronto、Georgia Institute of Technology
- **会议：** ICRA 2026
- **代码许可：** Apache-2.0（第三方组件许可另见官方仓库）
- **一句话说明：** 从少量遥操作示范出发，经过子任务标注与轨迹生成，在高保真仿真中批量生成软体物体操作数据。

## 核心内容

- 数据管线：**record（遥操作录制）→ annotate（子任务边界标注）→ generate（生成新示范）**。
- 对象与任务：毛绒玩具、绳索、毛巾、软立方体、组织、穿线、装袋，以及 Jenga 等操作任务。
- 机器人：Franka 单臂、YAM 双臂、GR1T2 人形、PSM 手术机器人。
- 工具链：Isaac Sim / Isaac Lab；官方安装脚本 `bash softmimicgen.sh`，预标注数据随安装下载。
- 研究意义：将 MimicGen 式自动示范生成从刚体操作扩展到可变形物体操作，并研究 sim-to-real 与 sim-real co-training。

## 与相关项目的区别

- [MimicGen](https://github.com/NVlabs/mimicgen)：主要面向刚体操作任务的数据生成。
- [HumanoidMimicGen](https://humanoidmimicgen.github.io/)：面向人形机器人全身移动操作（loco-manipulation），并非本项目。

## 一手资料

1. 论文：https://arxiv.org/abs/2603.25725
2. 作者项目主页：https://softmimicgen.github.io/
3. NVIDIA 官方开源仓库：https://github.com/NVlabs/SoftMimicGen
