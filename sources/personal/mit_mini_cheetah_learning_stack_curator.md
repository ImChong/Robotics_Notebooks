# MIT Mini Cheetah 学习资料栈（策展整理）

> 来源归档

- **标题：** MIT Mini Cheetah（Sangbae Kim 实验室 / Ben Katz 主导执行器）— 非 100% 完整开源的学习栈
- **类型：** personal（维护者策展 / 学习路线整理）
- **入库日期：** 2026-07-25
- **一句话说明：** 澄清 Mini Cheetah 是「论文 + 软件 + 部分硬件 + 学位论文」形态；按执行器 → FOC 驱动 → Cheetah-Software → Convex MPC → RL → ROS/CHAMP 给出优先学习清单，并标明整机 CAD 未公开。
- **沉淀到 wiki：** [wiki/entities/mit-mini-cheetah.md](../../wiki/entities/mit-mini-cheetah.md)

---

## 开源边界（策展总判）

| 内容 | 是否公开 |
|------|----------|
| 控制软件（MPC / 状态估计 / WBC / LCM / Gazebo） | ✅ [`Cheetah-Software`](https://github.com/mit-biomimetics/Cheetah-Software) |
| 电机驱动 PCB / FOC / CAN / BOM | ✅ [`3phase_integrated`](https://github.com/bgkatz/3phase_integrated) |
| 电机控制固件 | ✅（见 Katz 附录与配套仓） |
| Ben Katz MSc thesis（执行器设计） | ✅（DSpace / PDF） |
| 整机 SolidWorks / Fusion / STEP / 装配图 | ❌ |
| 电机绕线数据 / 完整电磁设计 / 加工图纸 | ❌ |

**结论：** 不是「下载即可复刻整机」的 100% 开源项目；对做人形力矩电机与四足控制，**执行器 thesis + 驱动板 + Cheetah-Software + Convex MPC** 仍是高价值教材栈。

---

## 资料分层（策展摘录）

### 1. 官方软件（最重要）

- **仓库：** https://github.com/mit-biomimetics/Cheetah-Software
- **真机文档：** https://github.com/mit-biomimetics/Cheetah-Software/blob/master/documentation/running_mini_cheetah.md
- **覆盖：** robot controller、locomotion、MPC、state estimator、WBC、LCM、仿真、硬件接口
- **归档：** [cheetah-software](../repos/cheetah-software.md)

### 2. 硬件 / 执行器（Ben Katz）

- **论文：** *A Low Cost Modular Actuator for Dynamic Robots*（MIT S.M. 2018）
- **内容：** 电机、行星减速、编码器、电流环、PCB、CAN、FOC、热、CAD 叙事
- **已有 wiki：** [paper-low-cost-modular-actuator-katz](../../wiki/entities/paper-low-cost-modular-actuator-katz.md)

### 3. 开源电机控制器

- **仓库：** https://github.com/bgkatz/3phase_integrated
- **覆盖：** PCB、STM32/mbed 固件、FOC、CAN、BOM
- **归档：** [bgkatz_3phase_integrated](../repos/bgkatz_3phase_integrated.md)

### 4. 官方论文（建议时序）

1. Bosworth et al. — *The MIT Super Mini Cheetah*（SSR 2015）— 早期小尺度力/阻抗腿
2. Katz et al. — *Mini Cheetah: A Platform…*（ICRA 2019）— 整机 + 后空翻 + cMPC 步态
3. Bledt et al. — *MIT Cheetah 3: Design and Control…*（IROS 2018）— 机械与 WBC 框架
4. Di Carlo et al. — *Dynamic Locomotion … Through Convex MPC*（IROS 2018）— Convex MPC 祖师爷
5. Margolis et al. — *Rapid Locomotion via Reinforcement Learning*（arXiv:2205.02824）— 3.9 m/s Sim2Real RL
6. Kurtz et al. — *Mini Cheetah, the Falling Cat*（arXiv:2109.04424）— 空中姿态 + landing
7. Jeon et al. — *Real-time Optimal Landing Control…*（arXiv:2110.02799）— 落地接触优化 / MPC

论文集合归档：[mit_mini_cheetah_control_papers](../papers/mit_mini_cheetah_control_papers.md)

### 5. ROS / 教学移植

| 项目 | 链接 | 特点 |
|------|------|------|
| mini_cheetah_ROS | https://github.com/Gleboss1/mini_cheetah_ROS | ROS + PyBullet，LCM 替代向学习 |
| quadruped_ctrl | https://github.com/Derek-TH-Wang/quadruped_ctrl | PyBullet + ROS，Mini Cheetah 仿真控制 |
| CHAMP | https://github.com/chvmp/champ | ROS 四足框架，借鉴 Cheetah 思想、更易上手 |

### 6. 面向人形力矩电机的优先序（策展）

| 优先级 | 资料 | 学什么 |
|--------|------|--------|
| ⭐⭐⭐⭐⭐ | Katz MSc thesis | 执行器机械/电气/热全流程 |
| ⭐⭐⭐⭐⭐ | `3phase_integrated` | FOC 驱动 PCB / CAN |
| ⭐⭐⭐⭐⭐ | Cheetah-Software | 整机控制框架 |
| ⭐⭐⭐⭐ | Convex MPC 论文 | 足力优化 / 步态规划祖师爷 |
| ⭐⭐⭐⭐ | Rapid Locomotion | RL + Sim2Real + curriculum |
| ⭐⭐⭐ | mini_cheetah_ROS / quadruped_ctrl | ROS 学习入口 |
| ⭐⭐⭐ | CHAMP | 快速理解四足控制骨架 |

覆盖链：**执行器设计 → 电机驱动 → 底层控制 → MPC → 强化学习 → Sim2Real**。

## 对 wiki 的映射

- 平台主页：[mit-mini-cheetah](../../wiki/entities/mit-mini-cheetah.md)
- 执行器深读：[paper-low-cost-modular-actuator-katz](../../wiki/entities/paper-low-cost-modular-actuator-katz.md)
- Convex MPC 概念：[srbd-convex-mpc-wbc](../../wiki/concepts/srbd-convex-mpc-wbc.md)、[mpc-wbc-integration](../../wiki/concepts/mpc-wbc-integration.md)
- 开源 QDD 对比：[open-source-qdd-actuator-projects](../../wiki/comparisons/open-source-qdd-actuator-projects.md)
- 力矩电机纵深：[depth-torque-motor-design](../../roadmap/depth-torque-motor-design.md)
- 下游 RL 影响：[paper-rma-rapid-motor-adaptation](../../wiki/entities/paper-rma-rapid-motor-adaptation.md)、[paper-walk-these-ways-quadruped-mob](../../wiki/entities/paper-walk-these-ways-quadruped-mob.md)、[extreme-parkour](../../wiki/entities/extreme-parkour.md)
