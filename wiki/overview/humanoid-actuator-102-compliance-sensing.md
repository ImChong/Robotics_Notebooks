---
type: overview
tags: [humanoid, actuator, sea, series-elastic, proprioception, category-hub]
status: complete
updated: 2026-11-21
summary: "Actuator 102 · 05 — SEA 物理退让与储能；双编码器与力矩估计构成「神经系统」；柔顺利安全但增仿真难度。"
related:
  - ./humanoid-actuator-102-technology-map.md
  - ./humanoid-actuator-102-gear-reflected-inertia.md
  - ./humanoid-actuator-102-decision-species.md
  - ../entities/paper-physical-ai-mechanical-hardware.md
sources:
  - ../../sources/blogs/wechat_human_five_humanoid_actuator_102.md
  - ../../sources/raw/wechat_humanoid_actuator_102_2026-06-02.md
  - ../../sources/papers/humanoid_actuator_102_reference_catalog.md
  - ../../sources/papers/physical_ai_mechanical_hardware_scirobotics_aee2921_2026.md
original:
  - ../../sources/raw/wechat_humanoid_actuator_102_2026-06-02.md
---

# Actuator 102 · 05：柔顺与感知反馈

> **分类页**：**VII 柔顺性与串联弹性** + **VIII 感知与反馈**。

## 一句话定义

人形执行器的柔顺性与感知反馈体系通过在驱动链中显式引入物理弹性（SEA）或高透明低惯量传动（QDD），配合电机侧与关节输出侧双编码器、绕组电流与热/力矩实时传感，为全身控制与强化学习提供高保真、抗冲击的接触「神经系统」。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SEA | Series Elastic Actuator | 串联弹性执行器，在驱动电机与输出负载间串入弹性元件 |
| PCB | Printed Circuit Board | 印刷电路板，现代关节多将其贴附于电机本体实现驱控一体 |
| EtherCAT | Ethernet for Control Automation Technology | 高实时性工业以太网现场总线 |
| CAN | Controller Area Network | 电机/关节常用的现场总线通信协议 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习控制策略的范式 |
| Sim2Real | Simulation to Real | 把物理仿真中学到的策略高保真迁移落地真机的主线工程 |
| QDD | Quasi-Direct Drive | 准直驱，依托低减速比与电流估计直接感知外部接触力 |

## 为什么重要

没有感知力矩反馈与物理退让机制的高刚度关节，就像失去痛觉和肌腱保护的骨骼：脚底踏空或碰触硬物时，冲击会在控制周期到来前摧毁齿轮；同时在复杂接触操作中，控制器必须依据真实的关节外力矩而非仅靠位置误差来调节阻抗，否则机器人在人机协同与非结构化地形中极易失去平衡或伤及周边物体。

## 核心原理与机制

### 串联弹性（第 VII 章）

- **SEA 核心思想**：在电机与输出关节间人为串联弹簧/弹性体，将高速冲击脉冲吸收并转换为可测量的弹性形变，同步起到**冲击隔离、弹簧形变测力、高步频储能回收（单腿弹簧效应）**的作用。
- **典型路线**：**Agility Digit** 是产业界商业化最彻底的 SEA 双足代表（硬件观点见 [Physical AI × 机械硬件](../entities/paper-physical-ai-mechanical-hardware.md)）；高工作角度的穿戴式反应力串联弹性实例见 MIT **TF8 MC-RFSEA**（[paper-tf8-reaction-force-sea-prosthesis](../entities/paper-tf8-reaction-force-sea-prosthesis.md)）；原始奠基理论见 Pratt & Williamson (1995)（[参考文献索引](../../sources/papers/humanoid_actuator_102_reference_catalog.md)）。
- **与刚性谐波/滚柱对比**：串联弹簧实质等效于一个物理低通滤波器，**绝对机械控制带宽低于刚性直驱传动**，但在**抗摔落冲击、意外人机碰撞安全与软着地寿命**上占优。

### 感知与智能集成（第 VIII 章）

- **双编码器标准配置**：**电机转子高分辨率编码器**（供内部 FOC 高频电流换相与高速测速）+ **关节输出端绝对应变/磁编码器**（消除减速器背隙与弹性变形后的真实关节位置读取，上电无需回零）。
- **力感知灵敏度门槛**：反向驱动启动力矩须压到 **< 1 Nm**（理想 **< 0.5 Nm**），或搭配关节输出侧扭矩传感器，否则执行器对地面的接触力变化处于“盲区”。
- **驱控一体智能节点**：驱动 PCB 紧贴电机定子背端封装、内置 EtherCAT/CAN-FD 节点与贴片热敏电阻——参见 [Hardware 101 · 集成执行器](./humanoid-hardware-101-integrated-actuators.md)。

### 仿真与 Sim2Real 权衡

- **刚性执行器**：物理行为贴近理想刚体关节模型，更容易构建保真度稳定的训练仿真环境（文内认为这是 Tesla 等偏好低间隙刚性推杆/谐波配强化学习的隐性底层原因）。
- **SEA / 弹性部件 / 大减速湿摩擦**：弹簧迟滞、变刚度、温度阻尼漂移会在动力学建模上引入高维非线性间隙，明显提升 Sim2Real 域随机化难度。

## 工程实践

- **优先保证反向驱动或直接出力矩闭环**：若选用了大减速比不可避免引入摩擦滞后（如部分谐波/滚柱），必须在输出轴串联应变片力矩传感器或精密标定摩擦力前馈，不能直接把 $K_t \cdot I_q$ 当成输出外力矩。
- **双编码器差值即物理故障/负载诊断信号**：实时监控 $(\theta_{\text{motor}} / N) - \theta_{\text{joint}}$ 的偏差，既可用于 SEA 弹簧力估计，也可在线检测谐波柔轮打滑、联轴器松动或传动过载。
- **驱动板防震与共地隔离**：贴片安装驱动 PCB 时，必须加注抗振灌封或柔性垫圈并隔离强电开关噪声对高精编码器磁/光信号的串扰。

## 常见误区或局限

- **误区：串联弹簧越高柔顺就一定越容易做出敏捷动作** — 弹簧刚度太低会大幅拉低固有频率与跟踪带宽，导致高频摆腿发飘甚至激发机械谐振。
- **误区：双编码器只为了防止断电丢零位** — 实际上双编码器的真正高阶价值是隔离齿轮背隙振荡，以及反解传动链真实变形量来校验力矩状态。

## 关联页面

- [Actuator 102 技术地图](./humanoid-actuator-102-technology-map.md)
- [减速与反射惯量](./humanoid-actuator-102-gear-reflected-inertia.md)
- [决策与物种](./humanoid-actuator-102-decision-species.md)
- [Hardware 101 · 集成执行器](./humanoid-hardware-101-integrated-actuators.md)
- [Physical AI × 机械硬件（SciRob 2026）](../entities/paper-physical-ai-mechanical-hardware.md)
- [TF8 / MC-RFSEA 仿生膝踝假肢](../entities/paper-tf8-reaction-force-sea-prosthesis.md)
- [参考文献 · SEA / QDD](../../sources/papers/humanoid_actuator_102_reference_catalog.md)

## 推荐继续阅读

- [Actuator 102 · 04：热学与力矩控制](./humanoid-actuator-102-thermal-and-control.md) — 刚性与带宽指标分解
- [Actuator 102 · 06：工业执行器陷阱](./humanoid-actuator-102-industrial-actuator-trap.md) — 工业无退让高刚伺服的失效原因

## 参考来源

- [wechat_human_five_humanoid_actuator_102.md](../../sources/blogs/wechat_human_five_humanoid_actuator_102.md)
- [wechat_humanoid_actuator_102_2026-06-02.md](../../sources/raw/wechat_humanoid_actuator_102_2026-06-02.md)（第七、八章）
