# ANYmal — A Highly Mobile and Dynamic Quadrupedal Robot（IROS 2016）

> 来源归档：ETH Zurich / RSL 的 ANYmal 原型系统论文，区别于后续商业型号 ANYmal D 和 ANYmal X。

- **标题：** ANYmal - A Highly Mobile and Dynamic Quadrupedal Robot
- **作者：** Marco Hutter, Christian Gehring, Dominic Jud, Andreas Lauber, C. Dario Bellicoso, Vassilios Tsounis, Jemin Hwangbo, Karen Bodie, Péter Fankhauser, Michael Bloesch, Remo Diethelm, Samuel Bachmann, Amir Melzer, Mark Hoepflinger
- **机构：** ETH Zurich, Robotic Systems Lab / Autonomous Systems Lab
- **会议：** IEEE/RSJ IROS 2016
- **DOI：** https://doi.org/10.1109/IROS.2016.7758092
- **官方收录：** https://www.research-collection.ethz.ch/entities/publication/d3770b2f-5aa1-4a92-8c2c-a56d79771820
- **全文 PDF：** https://www.research-collection.ethz.ch/bitstreams/09cfc536-6001-4295-b9ae-4e6b049c151e/download
- **类型：** paper
- **核对日期：** 2026-11-17
- **核对范围：** ETH Research Collection 论文全文的 I–VI 节（引言、平台与软件、ANYdrive 执行器与实验、运动控制、步行/小跑/攀爬实验、结论）及参考文献；结合 ANYbotics 官方产品页区分后续商业型号。
- **通读：** 已完成

## 问题与定位

2016 年论文的目标是把动态运动能力、户外适应性、模块化维修、关节力控整合为可独立运行的四足平台。该文介绍的是**原始 ANYmal 系统硬件与基于模型的控制**，不是后续 ANYmal 学习式运动控制论文。

## 机械与电气（Sec. II–III）

- **结构：** 四腿、每腿 HAA / HFE / KFE 三关节，共 12 个驱动自由度；大范围旋转关节与偏置连杆允许不同腿部构型。
- **原型规格：** 质量略低于 30 kg、站立高度约 0.5 m，约 650 Wh 电池；这组数值不能直接移植到 ANYmal D/X。
- **ANYdrive：** 电机 + 谐波减速器 + 串联转动弹簧构成 SEA；关节侧与减速器侧绝对编码器用于弹簧形变量测量，估计输出关节力矩。论文报告的单关节名义电压 48 V、峰值力矩 40 Nm、最大速度 12 rad/s。
- **关节控制：** 电流环之上是带前馈及摩擦补偿的关节力矩 PID；位置 PID 以力矩控制为内环，CAN 总线采用 CANopen。
- **响应实验：** 小信号力矩带宽最高约 70 Hz，10 Nm 激励时降为约 24 Hz；10 Nm 阶跃的 90% 稳定时间约 13 ms，40 Nm 阶跃约 35 ms。**带宽依负载幅度变化**，不能无条件写成 70 Hz。

## 运动、估计与调度（Sec. II.D、IV–V）

- **软件分层：** 三台 Intel NUC 之间通过内部网络协同；locomotion PC 跑实时全身控制/状态估计，navigation PC 负责定位建图与路径规划，第三台可运行巡检任务。历史栈是 ROS + Ubuntu 14.04。
- **实时频率：** 论文中状态估计与全身控制由 CAN 驱动定时，**400 Hz** 与执行器交换信息；非实时模块通过共享内存与 ROS 通信。
- **运动方法：** IMU + 腿运动学 + 足端接触进行本体状态估计；静态行走使用 ZMP 规划；动态步态使用简化倒立摆落脚规划；WBC 通过约束优化满足机体运动、接触力与关节力矩约束。
- **原型实验：** 静态步行约 0.3 m/s、小跑约 0.8 m/s；演示约 50° 工业梯面的谨慎爬行；续航实验显示动态小跑功耗量级约 290 W、可自主运行超过 2 小时。

## 适用边界与复现

1. 该文验证**原型平台的执行器/系统工程**，并非声明整机已经完成工业防爆认证，也不是以后所有 ANYmal 型号的统一技术参数。
2. 论文给出架构与控制算法引用，并**没有提供可直接重建整套系统的官方训练/部署运行入口**；不可据此绘制未经核查的“源码调用”时序图。
3. ETH RSL 2020 年基于 ANYmal 的纯本体感知复杂地形深度强化学习，是**另一项后续研究**：https://rsl.ethz.ch/the-lab/news/2020/10/learning-locomotion-over-challenging-terrain.html
4. 商业巡检和防爆能力须以 ANYbotics 的后续型号文档分别核验：ANYmal D（https://www.anybotics.com/robotics/anymal/）与 ANYmal X（https://www.anybotics.com/robotics/anymal-x/）。

## 对 wiki 的映射

- [ANYmal 平台](../../wiki/entities/anymal.md)：平台分代、原型 SEA 系统、控制流程与商业型号边界。
- [四足机器人](../../wiki/entities/quadruped-robot.md)：硬件形态、运动与任务。
- [Locomotion](../../wiki/tasks/locomotion.md)：WBC 与后续 RL 实验的研究脉络。
