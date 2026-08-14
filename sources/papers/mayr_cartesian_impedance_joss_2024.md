# A C++ Implementation of a Cartesian Impedance Controller for Robotic Manipulators（Mayr & Salt-Ducaju）

> 来源归档（ingest）

- **标题：** A C++ Implementation of a Cartesian Impedance Controller for Robotic Manipulators
- **类型：** paper / software / impedance / null-space / 7-DOF
- **期刊：** *Journal of Open Source Software* 9(93):5194，2024
- **DOI：** <https://doi.org/10.21105/joss.05194>
- **预印本：** <https://arxiv.org/abs/2212.11215>（Submitted 2022-12-21；PDF：<https://arxiv.org/pdf/2212.11215>）
- **代码：** <https://github.com/matthias-mayr/Cartesian-Impedance-Controller>（BSD-3-Clause）
- **项目页：** <https://matthias-mayr.github.io/Cartesian-Impedance-Controller/>
- **作者：** Matthias Mayr、Julian M. Salt-Ducaju
- **机构：** 隆德大学（Lund University）计算机系 / 自动控制系；WASP
- **入库日期：** 2026-08-13
- **一句话说明：** 面向力矩控制机械臂的笛卡尔阻抗 C++ 实现：主任务柔顺 + **零空间关节阻抗** + 期望末端 wrench；已在 **Franka Panda / FR3** 与 **KUKA iiwa7** 真机与仿真部署。
- **沉淀到 wiki：** [`wiki/entities/paper-cartesian-impedance-controller.md`](../../wiki/entities/paper-cartesian-impedance-controller.md)

## 开源状态（步骤 2.5）

- **项目页核查（2026-08-13）：** <https://matthias-mayr.github.io/Cartesian-Impedance-Controller/> Footer / 文档明确指向 GitHub 主仓；Doxygen 由仓库生成。
- **仓库核查：** [matthias-mayr/Cartesian-Impedance-Controller](https://github.com/matthias-mayr/Cartesian-Impedance-Controller) — GitHub API **BSD-3-Clause**、C++、README 含 `colcon build` / Docker / `scripts/install_dependencies.sh`、`src/cartesian_impedance_controller.cpp`（基库）+ `src/cartesian_impedance_controller_ros.cpp`（ros2_control 插件）+ `test/`（`base_tests` 无需仿真）。
- **真机绑定：** README 写明 ROS 2 已部署 **Franka Research 3**（[franka_ros](https://github.com/frankarobotics/franka_ros)）与 **KUKA iiwa7**（[lbr_fri_ros2_stack](https://github.com/lbr-stack/lbr_fri_ros2_stack)）；ROS 1 亦支持 Panda + `iiwa_ros`。
- **结论：** **已开源**（基库可脱离 ROS 嵌入 DART 等；ROS 2 控制器可运行）。重力由机体内补偿时，控制器不再加 $g(q)$；工具重力需用 wrench 命令补。

## 摘录 1：力矩叠加（Control Implementation）

重力已补偿的刚体动力学 $M\ddot q+C\dot q=\tau_c+\tau^{\mathrm{ext}}$。指令

$$
\tau_c=\tau_c^{\mathrm{ca}}+\tau_c^{\mathrm{ns}}+\tau_c^{\mathrm{ext}}
$$

- **笛卡尔阻抗** $\tau_c^{\mathrm{ca}}=J^\top(-K^{\mathrm{ca}}\Delta\xi-D^{\mathrm{ca}}J\dot q)$
- **零空间关节阻抗** $\tau_c^{\mathrm{ns}}=(I-J^\top(J^\top)^\dagger)\tau_0$，$\tau_0=-K^{\mathrm{ns}}(q-q^D)-D^{\mathrm{ns}}\dot q$
- **期望 wrench** $\tau_c^{\mathrm{ext}}=J^\top F_c^{\mathrm{ext}}$

脚注强调：Moore–Penrose 便宜，但**不动力学解耦**；非静平衡时任意 $\tau_0$ 可能在笛卡尔方向产生干扰力（Ott, *Cartesian Impedance Control of Redundant and Flexible-Joint Robots*, 2008）。

**对 wiki 的映射：** 论文实体画运行时序图；概念页把该投影标成「静力学一致 / $W=I$」工程默认。

## 摘录 2：相对厂商控制器的缺口（Statement of Need）

对比 KUKA FRI、franka_ros、libfranka：本包补齐 **参考位姿/刚度/wrench 在线更新、零空间构型、示教、关节轨迹、多机型**。`cartesian_controllers`（Scherzinger）面向位置/速度指令臂；力矩臂上笛卡尔阻抗通常更稳（Lawrence 1988）。

MoveIt 关节轨迹要生效，必须设 **非零 `nullspace_stiffness`**，否则只跟踪末端位姿、忽略规划器给出的 7 轴构型。

**对 wiki 的映射：** 与 [libfranka](../repos/libfranka.md) / [Franka Research 3](../../wiki/entities/franka-research-3.md) 对照表。

## 摘录 3：安全与滤波

在线改 $\xi^D$、$K$、$D$、$F^{\mathrm{ext}}$ 时做低通；刚度/阻尼/wrench 可饱和；指令力矩做 $\Delta\tau_{\max}$ 限速（默认示例 1 Nm/周期）。作者称平移刚度用到 1000 N/m 仍稳定，且**在奇异附近仍稳定**。建议在 URDF 把任务力矩上限压到约 20 Nm，便于人手拖动。

## 建议 wiki 动作

- 升格 [`wiki/entities/paper-cartesian-impedance-controller.md`](../../wiki/entities/paper-cartesian-impedance-controller.md)（含源码运行时序图）
- 归档 [`sources/repos/cartesian-impedance-controller.md`](../repos/cartesian-impedance-controller.md)、[`sources/sites/cartesian-impedance-controller-github-io.md`](../sites/cartesian-impedance-controller-github-io.md)
