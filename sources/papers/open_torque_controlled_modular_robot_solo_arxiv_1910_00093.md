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
- **入库日期：** 2026-07-25
- **一句话说明：** ODRI / Solo 开源力矩控制模块化腿足架构：9:1 双级同步带 QDD 执行器模块 + 轻量足底接触开关 + 2.2 kg 八关节四足；系统表征足端阻抗，并用 kino-dynamic 优化轨迹 + CoM/基座阻抗 QP 在真机上跟踪跳跃与行走。
- **开源状态：** **已开源**（项目页核查：机械图纸、电子、固件与软件均在 open-dynamic-robot-initiative 组织下公开，BSD-3-Clause）
- **沉淀到 wiki：** [paper-open-torque-controlled-modular-robot-solo](../../wiki/entities/paper-open-torque-controlled-modular-robot-solo.md)、[odri-solo-and-bolt](../../wiki/entities/odri-solo-and-bolt.md)

---

## 核心贡献（摘录）

1. **执行器模块（~150 g）**：T-Motor Antigravity 4004（300KV）+ **9:1** 双级 Conti 同步带 + 电机轴高分辨率光学编码器（Avago AEDM 5810）；外壳可 3D 打印；关节力矩由电流估计 \(\tau_{joint}=0.225\,i\)（\(k_i=0.025\) N·m/A，\(N=9\)），峰值约 **2.7 N·m @ 12 A**。
2. **足底接触开关（~10 g）**：LED–光敏–弹簧孔径结构，约 **270°** 触发范围，阈值约 **3 N / 3 ms**；冲击场景下明显优于仅靠电流估力的接触判定（论文报告约 3 ms vs ~31 ms）。
3. **Solo 四足（2.2 kg）**：八个相同执行器 + 四条带接触传感器的小腿；站立髋高约 **24 cm**（最大约 34 cm），可折到约 **5 cm**；材料成本约 **4000 €**。
4. **阻抗表征**：准静态足端刚度可调约 **20–360 N/m**（无阻尼时测得最大约 **266 N/m**）；无量纲腿刚度最高约 **10.8**，与跑步人体量级可比。
5. **控制演示**：kino-dynamic 优化器生成参考轨迹；在线用 CoM/基座阻抗 + 足端力分配 QP + 低阻抗腿长跟踪；真机完成跷跷板扰动慢走、跳跃（基座高度约 **65 cm**）等，无需环境模型重规划。

## 对 wiki 的映射

- 论文实体页：[paper-open-torque-controlled-modular-robot-solo](../../wiki/entities/paper-open-torque-controlled-modular-robot-solo.md)
- 平台实体：[odri-solo-and-bolt](../../wiki/entities/odri-solo-and-bolt.md)
- QDD 对比：[open-source-qdd-actuator-projects](../../wiki/comparisons/open-source-qdd-actuator-projects.md)
- 对照开源四足：[stanford-doggo-and-pupper](../../wiki/entities/stanford-doggo-and-pupper.md)
- 仓库/站点归档：[open_robot_actuator_hardware](../repos/open_robot_actuator_hardware.md)、[open_dynamic_robot_initiative](../sites/open_dynamic_robot_initiative.md)
