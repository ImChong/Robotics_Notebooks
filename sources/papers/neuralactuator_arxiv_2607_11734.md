# neuralactuator_arxiv_2607_11734

> 来源归档（ingest）

- **标题：** NeuralActuator: Neural Actuation Modeling for Robot Dynamics and External Force Perception
- **类型：** paper
- **作者：** Zhiyang Dou, John U. Onyemelukwe, Hangxing Zhang, Heng Zhang, Minghao Guo, Yunsheng Tian, Michal Piotr Lipiec, Joshua Jacob, Chao Liu, Peter Yichen Chen, Yuri Ivanov, Wojciech Matusik（MIT CDFG 等；* 共同一作）
- **arXiv：** <https://arxiv.org/abs/2607.11734>
- **PDF：** <https://arxiv.org/pdf/2607.11734>
- **HTML：** <https://arxiv.org/html/2607.11734>
- **项目页：** <https://frank-zy-dou.github.io/projects/NeuralActuator/index.html>
- **代码：** <https://github.com/Frank-ZY-Dou/Dynamics-Modeling/tree/main/NeuralActuator>（`Dynamics-Modeling` 仓库子目录）
- **入库日期：** 2026-07-16
- **一句话说明：** 面向低成本舵机平台的 **Transformer 执行器模型**：联合预测 **可微仿真用广义力矩 surrogate**、**无 F/T 传感器的外力与接触门控** 与 **电机工况分数**；发布 **NAD** 双臂遥操作数据集；在 OpenManipulator-X / SO-101 / Franka 上验证动力学 rollout、力估计与 BC 下游增益。

## 核心论文摘录（MVP）

### 1) 问题与动机（Abstract / Introduction）

- **链接：** <https://arxiv.org/abs/2607.11734>；项目页 Overview
- **核心贡献：** 可微仿真已推进策略学习，但 **执行器动力学** 仍是 sim-to-real 主误差源；低成本 **有刷 DC 舵机** 上 **τ = K_t I** 在目标跟踪下因摩擦、迟滞、背隙、温漂而失效。NeuralActuator 用 **历史相关** 的神经网络替代固定电流–力矩映射，并同时支持 **sensorless 力感知** 与 **电机工况诊断**。
- **对 wiki 的映射：**
  - [NeuralActuator 论文实体](../../wiki/entities/paper-neuralactuator-neural-actuation-modeling.md)
  - [Sim2Real](../../wiki/concepts/sim2real.md)
  - [Actuator Network](../../wiki/methods/actuator-network.md)

### 2) 方法：多任务 Transformer + 可微仿真监督（Section III）

- **链接：** <https://arxiv.org/html/2607.11734>（Section III-D）；项目页 Method
- **核心贡献：**
  - **输入：** 9 步历史（8 帧 + 当前）的命令、本体、跟踪误差与执行器遥测（电流/电压/温度等，平台相关）。
  - **四头输出：** (i) **torque surrogate**（clip 后驱动 DiffSim）；(ii) 3D 外力；(iii) **contact gate** $g$（$\hat f = g \hat f_{raw}$）；(iv) **per-motor condition** 分数。
  - **训练：** surrogate 头 **无直接广义力标签**，通过 **pose rollout + 可微物理** 反传；力/门/工况头 **直接监督**。
  - **架构：** 4 层 Transformer encoder（d=192, 4 heads, ~1.44M 参数）；GPU 推理 **~0.25 ms**（60 Hz 控制）。
- **对 wiki 的映射：**
  - [NeuralActuator 论文实体](../../wiki/entities/paper-neuralactuator-neural-actuation-modeling.md)
  - [Implicit / Explicit 执行器建模](../../wiki/concepts/implicit-explicit-actuator-modeling.md)

### 3) NAD 数据集与采集系统（Section III-C）

- **链接：** 项目页 NAD / Dataset Details
- **核心贡献：** **Leader–follower 双臂遥操作**：leader 示教、follower 关节空间闭环跟踪；同步记录 **关节状态、指令、Dynamixel 电流/电压/温度** 与 **末端外力标签**（已知载荷重力或六轴 F/T _fixture）。OpenManipulator-X 实验子集约 **94.52 min**（自由运动 / 力标注 / 机械受限 Joint 3）；全库 **450** 任务分配（含 SO-101 **100** 条）。
- **对 wiki 的映射：**
  - [NeuralActuator 论文实体](../../wiki/entities/paper-neuralactuator-neural-actuation-modeling.md)
  - [LeRobot](../../wiki/entities/lerobot.md)（SO-101 跨平台评测）

### 4) 实验与下游控制（Section IV；项目页 Experiments / BC）

- **链接：** <https://frank-zy-dou.github.io/projects/NeuralActuator/index.html>
- **核心贡献：**
  - **三平台：** 5-DoF OpenManipulator-X（~$500）、6-DoF SO-101、7-DoF Franka（离线载荷力 benchmark）。
  - **Rollout：** 600 步无载 MAE ~3° 量级；力传感器测试集 Force MAE **0.23 N**；载荷测试 **0.11 N**。
  - **力估计基线：** 相对 ID-Linear（avg **1.41 N**）、GMO（**0.66 N**），NeuralActuator avg **0.12 N**。
  - **Joint 3 工况分类：** Accuracy **91.0%**（vs RF **67.1%**）。
  - **BC 下游：** 冻结预训练 NeuralActuator 提供 $\hat f$ 反馈 — pick-and-place **80%→92.5%**，go up-and-stay **85%→95%**（各 40 trials）。
- **对 wiki 的映射：**
  - [NeuralActuator 论文实体](../../wiki/entities/paper-neuralactuator-neural-actuation-modeling.md)
  - [BAM 扩展摩擦论文](../../wiki/entities/paper-bam-extended-friction-servo-actuators.md)（解析摩擦对照）
  - [Current as Touch](../../wiki/entities/paper-current-as-touch-proprioceptive-contact.md)（电流/遥测力感知姊妹线）

## 当前提炼状态

- [x] 摘要、方法管线与 NAD 结构对齐 arXiv HTML + 项目页
- [x] 关键定量结果（rollout / force / condition / BC）对齐项目页表格
- [x] wiki 页面映射确认
