# 关节转子惯量与摩擦参数辨识算法一手论文簇

> 来源归档（ingest）

- **标题：** Joint armature / rotor inertia and friction parameter identification
- **类型：** paper（经典线性回归簇 + 开源实现论文）
- **来源：** IEEE T-RA / RAAD / ICRA / IJRR / arXiv
- **入库日期：** 2026-08-13
- **最后更新：** 2026-08-13
- **一句话说明：** 把关节侧待辨识量收成 $I_a$（转子反射惯量）+$b$（粘滞）+$\tau_c$（库仑）等；算法从 Fourier 激励 + 线性回归，到无力矩传感器的 CMA-ES 仿真对齐。
- **沉淀到 wiki：** 是 → [`wiki/methods/joint-actuator-parameter-identification.md`](../../wiki/methods/joint-actuator-parameter-identification.md)

## 开源状态（步骤 2.5）

| 资料 | 代码 | 结论 |
|------|------|------|
| Swevers et al. 1997 | 无官方仓；方法被后续工具箱复现 | **确认未开源** |
| Ayusawa / Venture / Nakamura 2014 | 无原作者仓；FloBaRoID 实现两步摩擦 | 算法论文 **确认未开源** |
| Bethge et al. RAAD 2017（FloBaRoID） | [kjyv/FloBaRoID](https://github.com/kjyv/FloBaRoID) | **已开源**（LGPL-3.0） |
| BAM / Duclusaud et al. ICRA 2025 | [Rhoban/bam](https://github.com/Rhoban/bam) + [文档站](https://bam.readthedocs.io/) | **已开源**（Apache-2.0；既有实体） |
| PACE / Bjelonic et al. 2025 | [leggedrobotics/pace-sim2real](https://github.com/leggedrobotics/pace-sim2real) | **已开源**（Apache-2.0；既有实体） |
| Pinocchio 回归矩阵 | `computeJointTorqueRegressor` | **已开源**（线性惯性核，摩擦列需自拼） |

## 核心论文摘录（MVP）

### 1) Optimal robot excitation and identification（Swevers, Ganseman, De Schutter, Van Brussel, 1997）

- **链接：** <https://doi.org/10.1109/70.631234>
- **核心贡献：** 每关节激励用 **有限 Fourier 级数**（周期、可解析求 $\dot q,\ddot q$、可指定带宽）。辨识用 **最大似然** 而不是只看回归矩阵条件数。刚体惯性参数与 **粘滞 / 库仑摩擦** 对力矩线性，可写进同一 $\tau=Y\pi$。KUKA IR 361 前三轴实验；文中也指出纯 Coulomb–Viscous 对工业减速箱往往偏简。
- **对 wiki 的映射：**
  - [关节执行器参数辨识](../../wiki/methods/joint-actuator-parameter-identification.md)
  - [System Identification](../../wiki/concepts/system-identification.md)
  - [Joint Friction Models](../../wiki/concepts/joint-friction-models.md)

### 2) Identifiability and identification of inertial parameters using the underactuated base-link dynamics（Ayusawa, Venture, Nakamura, IJRR 2014）

- **链接：** <https://doi.org/10.1177/0278364913495934>
- **核心贡献：** 浮动基 **基座六维动力学不含关节摩擦**（摩擦做功在驱动关节）。因此可先用基座 wrench 辨识惯性参数，再把关节力矩残差拟合为摩擦——这是 [FloBaRoID](../../wiki/entities/flobaroid.md) README 里 **two-step friction identification** 的理论来源。
- **对 wiki 的映射：**
  - [关节执行器参数辨识](../../wiki/methods/joint-actuator-parameter-identification.md)
  - [FloBaRoID](../../wiki/entities/flobaroid.md)

### 3) FloBaRoID — A Software Package for the Identification of Robot Dynamics Parameters（Bethge, Malzahn, Tsagarakis, Caldwell, RAAD 2017）

- **链接：** <https://doi.org/10.1007/978-3-319-61276-8_18>
- **代码：** <https://github.com/kjyv/FloBaRoID>
- **核心贡献：** 把 Fourier 激励、预处理、base 参数、物理一致 SDP、以及 **先无摩擦惯性、再逐关节摩擦** 收成可运行工具箱；示例为 **KUKA LWR 4+ 七轴**。动力学核是 iDynTree，不是 Pinocchio。
- **对 wiki 的映射：**
  - [FloBaRoID 实体](../../wiki/entities/flobaroid.md)
  - [关节执行器参数辨识](../../wiki/methods/joint-actuator-parameter-identification.md)

### 4) Extended Friction Models for the Physics Simulation of Servo Actuators（Duclusaud et al., ICRA 2025）

- **链接：** <https://arxiv.org/abs/2410.08650>
- **归档：** [bam_extended_friction_servos_arxiv_2410_08650.md](./bam_extended_friction_servos_arxiv_2410_08650.md)
- **核心贡献：** 摆锤台架 + CMA-ES 同时辨识 **表观惯量 $J_m$（手册有 $J_r$ 则 $J_m=N^2 J_r$，否则当自由参数）**、电气 $k_t,R$ 与 M1–M6 摩擦。不需要六维力传感器。
- **对 wiki 的映射：**
  - [关节执行器参数辨识](../../wiki/methods/joint-actuator-parameter-identification.md)
  - [BAM](../../wiki/entities/bam-better-actuator-models.md)

### 5) Towards Bridging the Gap: Systematic Sim-to-Real Transfer for Diverse Legged Robots（Bjelonic et al., arXiv:2509.06342）

- **链接：** <https://arxiv.org/abs/2509.06342>
- **归档：** [pace_sim2real_arxiv_2509_06342.md](./pace_sim2real_arxiv_2509_06342.md)
- **核心贡献：** 悬空 chirp + 仅编码器；CMA-ES 拟合每关节 **armature / 粘滞 / Coulomb / 偏置** 与全局延迟（$4n+1$ 维）。足式上把「转子惯量 + 摩擦」当成可部署的最小参数集。
- **对 wiki 的映射：**
  - [关节执行器参数辨识](../../wiki/methods/joint-actuator-parameter-identification.md)
  - [PACE](../../wiki/entities/paper-pace-sim2real-legged-robots.md)
