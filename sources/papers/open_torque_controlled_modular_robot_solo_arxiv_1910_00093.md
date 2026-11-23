# An Open Torque-Controlled Modular Robot Architecture for Legged Locomotion Research

> 来源归档

- **标题：** An Open Torque-Controlled Modular Robot Architecture for Legged Locomotion Research
- **类型：** paper
- **作者：** Felix Grimminger, Avadesh Meduri, Majid Khadiv, Julian Viereck, Manuel Wüthrich, Maximilien Naveau, Vincent Berenz, Steve Heim, Felix Widmaier, Thomas Flayols, Jonathan Fiene, Alexander Badri-Spröwitz, Ludovic Righetti
- **机构：** Max Planck Institute for Intelligent Systems；New York University；LAAS / CNRS
- **链接：** https://arxiv.org/abs/1910.00093
- **PDF：** https://arxiv.org/pdf/1910.00093
- **DOI：** https://doi.org/10.1109/LRA.2020.2976639（IEEE RA-L 2020）
- **项目页：** https://open-dynamic-robot-initiative.github.io
- **组织 GitHub：** https://github.com/open-dynamic-robot-initiative
- **代码：** https://github.com/open-dynamic-robot-initiative/open_robot_actuator_hardware（硬件总仓）；https://github.com/open-dynamic-robot-initiative/solo（低层控制）；https://github.com/open-dynamic-robot-initiative/master-board（主控板固件/SDK）；https://github.com/open-dynamic-robot-initiative/odri_control_interface（统一控制接口）
- **许可：** BSD-3-Clause
- **入库日期：** 2026-07-25（2026-11-21 完成 `arXiv:1910.00093v2` 全文通读与 `open_robot_actuator_hardware` 目录核对）
- **通读：** 已完成
- **一句话说明：** ODRI / Solo 开源力矩控制模块化腿足架构：9:1 双级同步带 QDD 执行器模块 + 轻量足底接触开关 + 2.2 kg 八关节四足；系统表征足端阻抗，并用 kino-dynamic 优化轨迹 + CoM/基座阻抗 QP 在真机上跟踪跳跃与行走。
- **开源状态：** **已开源**（项目页核查：机械图纸、电子、固件与软件均在 open-dynamic-robot-initiative 组织下公开，BSD-3-Clause）
- **沉淀到 wiki：** [paper-open-torque-controlled-modular-robot-solo](../../wiki/entities/paper-open-torque-controlled-modular-robot-solo.md)、[odri-solo-and-bolt](../../wiki/entities/odri-solo-and-bolt.md)

---

## 核心贡献与实测数据（原文 PDF 精读）

1. **执行器模块（150 g, 160 mm）**：T-Motor Antigravity 4004（300KV）+ **9:1** 双级 Conti Synchroflex AT3 GEN III 同步带 + Avago AEDM 5810 5000 CPR 电机轴光学编码器；3D 打印塑料壳体（仅电机轴与带轮机加）；关节力矩通过相电流估计 $\tau_{\text{joint}} = k_i\,i\,N = 0.025 \times 9\,i = 0.225\,i$，12 A 下峰值 **2.7 N·m**。
2. **电子、通信与控制频率（§II-A2 & II-D）**：定制 6 层 **MPI Micro-Driver**（TI TMS320F28069M + 2× DRV8305，最高 40 V，10 kHz 双电机 FOC）；**ESP32 (240 MHz Xtensa LX6) Master Board** 经 35 µs 单次双向 SPI 轮询多达 8 块双驱动板，并通过 raw MAC 帧（无 ACK 广播）与 RT-Preempt Ubuntu PC 通信——100 Mbps 以太网往返 **200 µs**、WiFi 往返 **1100 µs**（丢包约 4%），可实现 **12-DoF @ 1 kHz** 闭环。
3. **足底接触开关（10 g, §II-B & III-A4）**：LED–光敏–弹簧（1.5 mm 孔径硅胶调节）构成 **270°** 触发机械开关，3 N 下约 **3 ms** 触发（对比仅凭电机电流估算接触需阈值化避免误报，实测延迟高达 **~31 ms**）。
4. **Solo 整机与阻抗表征（§II-C & §III）**：8 自由度全对称四足 **2.2 kg**（站立髋高 24 cm、最大 34 cm、折叠 5 cm，约 €4,000 材料成本）；准静态 Cartesian 阻抗 $20\text{–}360\,\text{N/m}$ 指令下，无主动阻尼最高测得 **$266\,\text{N/m}$**（对应无量纲腿刚度 $\tilde{k}=k l_0 / (mg) = 10.8$，进入人类跑步 $10\text{–}22$ 区间；0.24 m 跌落测试 $K=150\,\text{N/m}, D=0.5\,\text{N\cdot s/m}$）；最高跳跃 **0.65 m**（静止腿长 2.7×）。
5. **全身运动跟踪（§III-B）**：离线 kino-dynamic 交替优化质心动力学与全身运动学（~2 轮收敛），在线 QP 分配足端接触力 + 低阻抗腿长 PD $\tau_i = J_{i,a}^\top F_i + K(l_i^{\text{ref}}-l_i) + D(\dot{l}_i^{\text{ref}}-\dot{l}_i)$，盲走跨过跷跷板扰动并支持 ±3 圈连续旋转自起身与多种膝关节构型。

## 对 wiki 的映射

- 论文实体页：[paper-open-torque-controlled-modular-robot-solo](../../wiki/entities/paper-open-torque-controlled-modular-robot-solo.md)
- 平台实体：[odri-solo-and-bolt](../../wiki/entities/odri-solo-and-bolt.md)
- QDD 对比：[open-source-qdd-actuator-projects](../../wiki/comparisons/open-source-qdd-actuator-projects.md)
- 对照开源四足：[stanford-doggo-and-pupper](../../wiki/entities/stanford-doggo-and-pupper.md)
- 仓库/站点归档：[open_robot_actuator_hardware](../repos/open_robot_actuator_hardware.md)、[open_dynamic_robot_initiative](../sites/open_dynamic_robot_initiative.md)
