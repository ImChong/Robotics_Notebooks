# bam_extended_friction_servos_arxiv_2410_08650

> 来源归档（ingest）

- **标题：** Extended Friction Models for the Physics Simulation of Servo Actuators
- **类型：** paper
- **作者：** Marc Duclusaud, Grégoire Passault, Vincent Padois, Olivier Ly（Bordeaux / Inria Auctus）
- **arXiv：** <https://arxiv.org/abs/2410.08650v1>（v1，2024-10）
- **PDF：** <https://arxiv.org/pdf/2410.08650v1>
- **HTML：** <https://arxiv.org/html/2410.08650v1>
- **会议：** ICRA 2025（IEEE，pp. 12091–12097）
- **代码：** <https://github.com/Rhoban/bam>
- **演示视频：** <https://youtu.be/P5-Ked8EoWk>（论文配套）；README 另链 <https://youtu.be/5XPEEKDnQEM>
- **数据：** <https://drive.google.com/drive/folders/1SwVCcpJko7ZBsmSTuu3G_ZipVQFGZ11N?usp=drive_link>（MX-64 / MX-106 / eRob80:50 / eRob80:100 摆锤日志）
- **入库日期：** 2026-05-28
- **一句话说明：** 为舵机/伺服执行器提出 M1–M6 扩展摩擦上界模型（Stribeck、负载相关、方向性与二次项），在摆锤台架用 CMA-ES 辨识参数，并在 MuJoCo 2R 臂上相对经典 Coulomb–Viscous 将轨迹 MAE 降低约 50% 以上，面向 RL 低增益控制下的 sim2real 执行器建模。

## 核心论文摘录（MVP）

### 1) 问题与动机（Abstract / Introduction）

- **链接：** <https://arxiv.org/abs/2410.08650v1>
- **核心贡献：** MuJoCo、Isaac Gym 等主流仿真器默认 **Coulomb–Viscous（M1）** 摩擦过于简化，难以刻画舵机减速箱中的 **静摩擦、Stribeck、负载相关、谐波减速器二次效应**；在 **RL 常用的低 PD 增益** 下，仿真轨迹与真机偏差显著（Fig.1：2R 臂三角波跟踪）。
- **对 wiki 的映射：**
  - [BAM 扩展摩擦论文实体](../../wiki/entities/paper-bam-extended-friction-servo-actuators.md)
  - [Sim2Real](../../wiki/concepts/sim2real.md)
  - [System Identification](../../wiki/concepts/system-identification.md)

### 2) 摩擦模型族 M1–M6（Section III）

- **链接：** <https://arxiv.org/html/2410.08650v1>（Section III）
- **核心贡献：** 将摩擦表述为可阻止运动的 **力矩上界** $\tau_f^m$，仿真步内用 $\tau_{f,stop}$ 裁剪得到实际 $\tau_f$，避免 $\dot\theta=0$ 时不连续。模型递进：
  - **M1** Coulomb–Viscous：$K_v|\dot\theta|+K_c$
  - **M2** Stribeck：加指数衰减静摩擦项 $K_c^s e^{-|\dot\theta/\dot\theta^s|^\alpha}$
  - **M3** 负载相关：$+K_l|\tau_m-\tau_e|$
  - **M4** Stribeck + 负载相关（**Dynamixel 2R 验证最优**）
  - **M5** 方向性负载（$K_m\tau_m$ vs $K_e\tau_e$）
  - **M6** 二次负载项（**eRob80 谐波减速 2R 最优**）
- **对 wiki 的映射：**
  - [BAM 论文实体](../../wiki/entities/paper-bam-extended-friction-servo-actuators.md)
  - [Actuator Network](../../wiki/methods/actuator-network.md)（数据驱动对照）

### 3) 仿真集成与摆锤辨识（Section IV–V）

- **链接：** <https://arxiv.org/html/2410.08650v1>（Algorithm 1；Section IV-E MuJoCo 在线更新 $K_c,K_v$）
- **核心贡献：** 伺服模型 = **控制律（电压/电流 PID）+ 电机方程** + 扩展摩擦；辨识用四类摆锤轨迹（加速振荡、双频 sin、慢抬放、lift-and-drop），**CMA-ES（optuna）** 最小化仿真–实测 MAE；75% 训练 / 25% 验证。
- **对 wiki 的映射：**
  - [BAM（Better Actuator Models）仓库实体](../../wiki/entities/bam-better-actuator-models.md)

### 4) 实验与 2R 验证（Section VI）

- **链接：** <https://arxiv.org/abs/2410.08650v1>；数据见 Google Drive
- **核心贡献：** 四款舵机 **MX-64、MX-106、eRob80:50、eRob80:100**；摆锤辨识 MAE 相对 M1 约 **1.5×–2.9×** 改善；2R 臂（Dynamixel / eRob）跟踪圆/方/方波/三角波，**M4（Dynamixel）与 M6（eRob）** 在高低增益下 MAE 均约为 M1 的 **一半以下**。
- **对 wiki 的映射：**
  - [Sim2Real Gap 缩减实战](../../wiki/queries/sim2real-gap-reduction.md)

## BibTeX（仓库 README 提供）

```bibtex
@inproceedings{duclusaud2025extended,
  title={Extended Friction Models for the Physics Simulation of Servo Actuators},
  author={Duclusaud, Marc and Passault, Gr{\'e}goire and Padois, Vincent and Ly, Olivier},
  booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={12091--12097},
  year={2025},
  organization={IEEE}
}
```

## 当前提炼状态

- [x] 摘要与 M1–M6 模型族对齐 arXiv HTML
- [x] 与 Rhoban/bam README 流程（record / fit / 2R.sim）交叉索引
- [x] wiki 页面映射确认
