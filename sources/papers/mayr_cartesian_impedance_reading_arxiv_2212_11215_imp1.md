# Cartesian Impedance Controller 论文通读（阻抗控制实现视角）

> 来源归档（原文通读 · 支撑 `wiki/concepts/impedance-control.md`）

- **标题：** A C++ Implementation of a Cartesian Impedance Controller for Robotic Manipulators
- **作者：** Matthias Mayr, Julian M. Salt-Ducaju（隆德大学 / WASP）
- **类型：** paper / impedance / null-space
- **原文：** <https://arxiv.org/abs/2212.11215>（arXiv v1 PDF，通过 pdftotext 获取全文，共 7 页）
- **一句话说明：** 面向力矩控制机械臂的笛卡尔阻抗 C++ 实现：任务空间柔顺 + 零空间关节阻抗 + 期望末端 wrench，并带滤波、饱和、力矩限速三层安全措施。
- **核对范围：** 通读全文 7 页：Summary、Statement of Need（含与 KUKA FRI / franka_ros / libfranka 对比表）、Control Implementation（式 1–6）、Safety Measures（式 7–9、图 1 框图）与参考文献列表。该文无实验章节；Hogan (1985)、Ott (2008)、Lawrence (1988) 等被引文献未阅读，仅按本文转述使用。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **阻抗定义（Summary）：** 阻抗控制在外力与机器人相对参考坐标的偏差之间建立质量-弹簧-阻尼关系；附带好处是人可直接手把手引导。
- **动力学（式 1）：** 重力已补偿的刚体动力学 $M(q)\ddot q + C\dot q = \tau_c + \tau^{ext}$；对非自带重力补偿的机器人需在指令力矩中加重力补偿项。
- **力矩叠加（式 2–6）：** $\tau_c=\tau_c^{ca}+\tau_c^{ns}+\tau_c^{ext}$。笛卡尔阻抗 $\tau_c^{ca}=J^T[-K^{ca}\Delta\xi-D^{ca}J\dot q]$（平移误差为差，旋转误差为 $\xi_{ro}\,\xi^{D\,-1}_{ro}$）；零空间 $\tau_c^{ns}=(I-J^T(J^T)^\dagger)\tau_0$，$\tau_0=-K^{ns}(q-q^D)-D^{ns}\dot q$；期望 wrench $\tau_c^{ext}=J^T F_c^{ext}$。
- **脚注 4：** Moore–Penrose 投影便宜但不用动力学；关节不在静平衡（$\dot q=\ddot q\neq0$）时，任意 $\tau_0$ 可能在笛卡尔空间产生干扰力。
- **为何用力矩控制（Statement of Need）：** 位置 / 速度指令机器人上可能难以得到稳定的柔顺行为（引 Lawrence 1988），故力矩臂（iiwa、Panda）上笛卡尔阻抗通常是首选。
- **安全措施（式 7–9）：** 参考位姿、刚度、阻尼、wrench 在线修改时做一阶低通；刚度 / 阻尼 / wrench 可饱和到上下限；指令力矩相邻周期变化量受 $\Delta\tau_{max}$ 限制。
- **需求清单：** 在线改参考点、在线改阻抗、施加 wrench、零空间关节构型、关节轨迹执行。

## 对 wiki 的映射

- [`wiki/concepts/impedance-control.md`](../../wiki/concepts/impedance-control.md)
- [`wiki/entities/paper-cartesian-impedance-controller.md`](../../wiki/entities/paper-cartesian-impedance-controller.md)

## 参考来源

- 论文：<https://arxiv.org/abs/2212.11215>
