# A Unified Fractal Impedance Control Framework（arXiv:2611.08009）

> 来源归档（paper）

- **标题：** A Unified Fractal Impedance Control Framework
- **类型：** paper
- **作者与机构（论文首页）：** Adnan Takriti、Mahmoud Sadeq、Carlo Tiseo（英国 University of Sussex 工程与信息学院）；Quentin Rouxel（香港中文大学机械与自动化工程系）；Daniele Cafolla（英国 Swansea University 计算机科学系）；Michael Mistry（英国 University of Edinburgh 信息学院）
- **arXiv：** <https://arxiv.org/abs/2611.08009>（v1，cs.RO，2026-11-05）
- **PDF：** <https://arxiv.org/pdf/2611.08009v1>
- **代码：** https://github.com/TRL-Sussex/FIC_Interface
- **入库日期：** 2026-11-09
- **一句话说明：** 把分形阻抗控制（FIC）的各条架构整理为统一软件套件（FIC Suite / FIC Studio），并首次在机器人上实现完整的 H-PMP；在 Kinova Gen3 上以三个实验（轨迹跟踪、墙面按压与突然脱离、拖动示教）与另外三种 FIC 架构比较。
- **核对范围：** 下载 arXiv v1 PDF，通读正文 §I–§VI 全文（摘要、引言、相关工作、FIC Suite、控制器公式 (1)–(9)、三个实验、结论）、图 1–9 的图注与图中标注数值、表 I–IV 与参考文献列表。论文无附录或补充材料。代码仓库只读了 README（静态阅读，未运行、未逐文件核对源码，也未核对 `fic_docs/` 文档）；引用的 [3]、[4]、[18] 等先前 FIC / H-PMP 论文未打开，相关表述仅来自本文转述。
- **通读：** 已完成

## 开源状态

- **已开源**：仓库 `TRL-Sussex/FIC_Interface`（论文 [33]），README 说明含 `fic_library/`（C++ 控制库）、`fic_ws/`（ROS 2 工作区与 FIC Studio）、`robot_library/`（Franka libfranka、Kinova Kortex 驱动）、`docker/`（仅仿真镜像）、`fic_docs/`。README 列出 Franka FR3 与 Kinova Gen3 7-DOF 有硬件支持，KUKA iiwa 14 与 Pendulum 仅配置。本次只读 README，未核对许可证与代码质量。

## 核心摘录

1. **定位（摘要、§I）：** 论文提供 FIC 的机器人实现与实验验证，重点是 Harmonic-Passive Motion Paradigm（H-PMP），并给出统一的 FIC 部署框架（ROS2、MuJoCo、实体硬件）。作者自述 FIC 已被先前工作「证明」具有设计上的有界交互安全特性（引用 [3]），本文不重新证明。
2. **FIC 机制（§IV-A/B）：** 力（effort）函数在期望状态附近线性，两侧各有 n、m 个 tanh 型饱和段（式 (1)(2)）；FIC 是临界阻尼 Liénard 方程的算法实现，分发散与收敛两阶段（式 (3)），收敛阶段回程用上一次发散阶段的最大误差 x̃_M 构造。
3. **HapFIC 与 H-PMP（§IV-E/F，图 4）：** 纯导纳累积位置偏移直到关节限位；HapFIC 用 FIC 的饱和点限制该偏移，因此在导纳与阻抗之间平滑过渡。H-PMP 把任务空间导纳或 HapFIC 嵌套在关节空间 HapFIC 内，由估计的外力矩经 τ_des = Jᵀw（式 (8)）下发。
4. **实验设置（§V）：** 7-DOF Kinova Gen3 + Robotiq 2F-85，1 kHz 力矩接口，无外部力传感器，外力由关节力矩估计。对照为三种 FIC 架构：带规划器的笛卡尔阻抗、任务空间 HapFIC、笛卡尔阻抗加关节空间 HapFIC。
5. **实验 A（李萨如 8 字，A=0.10 m，周期 20 s，每个控制器 10 次，表 II）：** RMSE 依次为 H-PMP 7.80 mm、笛卡尔阻抗+关节 HapFIC 12.56 mm、任务空间 HapFIC 21.11 mm、带规划器笛卡尔阻抗 43.19 mm；峰值误差 16.9 / 20.9 / 32.2 / 69.4 mm；x 轴增益 0.99 / 0.94 / 0.97 / 0.69。姿态误差反而是 H-PMP 3.33°、任务空间 HapFIC 3.65°，高于另两者的 1.41° 与 1.94°。作者承认 H-PMP 轨迹比纯笛卡尔控制器「更不平滑」，归因于关节空间 HapFIC 以位置精度换力矩跟踪。
6. **实验 B（墙面按压 20 N 预期最大力后墙板突然脱落，表 III、图 7）：** 释放前接触力：笛卡尔阻抗+关节 HapFIC 16.86 N、H-PMP 12.59 N、任务空间 HapFIC 4.59 N；释放后越过墙面最远距离 68.7 / 23.4 / 36.3 mm；H-PMP 与任务空间 HapFIC 的设定点平移 50.0 mm，另一个为 0。每个控制器按压 2 次，任务空间 HapFIC 只有 1 次；H-PMP 首次按压中从墙面反弹过一次。
7. **实验 C（拖动示教加重放，表 IV、图 8–9，每个设置只示教一次）：** 重放 RMSE H-PMP 40.7 mm、任务空间 HapFIC 55.7 mm（示教平均手速 144 与 150 mm/s）；施加手推扰动时 H-PMP 峰值力 11.3 N、最大偏离 67 mm、柔顺度 6.0 mm/N，任务空间 HapFIC 27.8 N、132 mm、4.8 mm/N。H-PMP 关节层在 MCU（Arduino Ventuno Q 的 STM32H5）上 1 kHz 运行，任务层仍在板载 Linux 以约 95 Hz 通信，重放 RMSE 42.4 mm，推力 51.3 N、最大偏离 167 mm（3.3 mm/N）、2 s 后仍偏 85 mm。不同设置的推力大小不同。
8. **作者自述的局限与后续（§V-A/B、§VI）：** 可用机器人动力学模型改进期望力矩估计，或加入在线调参；后续工作聚焦自适应调参。

## 对 wiki 的映射

- [paper-fractal-impedance-control-framework](../../wiki/entities/paper-fractal-impedance-control-framework.md)
