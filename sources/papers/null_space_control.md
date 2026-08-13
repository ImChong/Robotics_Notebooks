# 机器人零空间控制（冗余解析）一手论文簇

> 来源归档（ingest）

- **标题：** Null-space / task-priority redundancy control
- **类型：** paper（经典簇 + 综述 + 7 轴开源实现论文）
- **来源：** IJRR / IEEE / JOSS / DLR elib
- **入库日期：** 2026-08-13
- **最后更新：** 2026-08-13
- **一句话说明：** 把「主任务占用任务空间、剩余关节自由度做次级姿态」形式化为零空间投影；7 轴臂是最常见工程实例（末端 6D + 肘部 1D 自运动）。
- **沉淀到 wiki：** 是 → [`wiki/concepts/null-space-control.md`](../../wiki/concepts/null-space-control.md)

## 开源状态（步骤 2.5）

本簇多数经典论文**无官方代码**。可运行实现以项目页/仓库为准：

| 资料 | 代码 | 结论 |
|------|------|------|
| Dietrich et al. 2015 综述 | 无官方仓；实验在 DLR LWR-III | **确认未开源**（方法综述） |
| Nakamura / Khatib / Siciliano 经典 | 公式被广泛复现，无原作者仓 | **确认未开源** |
| Mayr & Salt-Ducaju JOSS 2024 | [Cartesian-Impedance-Controller](https://github.com/matthias-mayr/Cartesian-Impedance-Controller) + [项目页](https://matthias-mayr.github.io/Cartesian-Impedance-Controller/) | **已开源**（BSD-3-Clause） |
| Franka 官方 7 轴示例 | [libfranka](https://github.com/frankarobotics/libfranka) `examples/cartesian_impedance_control.cpp` | **已开源**（Apache-2.0） |
| stack-of-tasks TSID | [tsid](https://github.com/stack-of-tasks/tsid) | **已开源**（BSD-2-Clause）；HQP 替代显式投影器 |

## 核心论文摘录（MVP）

### 1) Task-priority based redundancy control of robot manipulators（Nakamura, Hanafusa, Yoshikawa, 1987）

- **链接：** <https://doi.org/10.1177/027836498700600201>
- **核心贡献：** 把冗余臂逆运动学写成**任务优先级**：低优先级子任务只能利用高优先级任务的零空间。标准速度层公式

$$
\dot q = J_1^+ \dot x_1 + (I - J_1^+ J_1) z
$$

其中 $z$ 常取次级任务的伪逆解或标量函数梯度（关节居中、可操作度、避障）。
- **对 wiki 的映射：**
  - [零空间控制](../../wiki/concepts/null-space-control.md)
  - [逆运动学](../../wiki/formalizations/inverse-kinematics.md)
  - [HQP](../../wiki/concepts/hqp.md)

### 2) A unified approach for motion and force control of robot manipulators: The operational space formulation（Khatib, 1987）

- **链接：** <https://doi.org/10.1109/JRA.1987.1087068>
- **核心贡献：** 操作空间（operational space）力控 + **动力学一致**零空间投影。次级力矩经 $N = I - J^\top \bar J^\top$ 投影，使主任务加速度不被次级任务惯性耦合；加权伪逆用惯量 $M(q)$。
- **对 wiki 的映射：**
  - [零空间控制](../../wiki/concepts/null-space-control.md)
  - [阻抗控制](../../wiki/concepts/impedance-control.md)
  - [TSID](../../wiki/concepts/tsid.md)

### 3) An overview of null space projections for redundant, torque-controlled robots（Dietrich, Ott, Albu-Schäffer, IJRR 2015）

- **链接：** <https://doi.org/10.1177/0278364914566516>；开放 PDF：<https://elib.dlr.de/101443/2/NullspaceSurvey.pdf>
- **核心贡献：** 力矩控制零空间投影的统一综述：**(i) successive vs augmented 层次**；**(ii) static / dynamic / stiffness consistency**；**(iii) Khatib 动力学一致加权矩阵可推广为无穷多族**。真机实验在 **DLR/KUKA LWR-III 七轴**上做三层阻抗（位置 > 姿态 > 关节构型）。关键工程结论：理论最优的动力学一致投影在真机上因惯量/摩擦建模误差，优势明显缩小。
- **对 wiki 的映射：**
  - [零空间投影综述（论文实体）](../../wiki/entities/paper-null-space-projections-survey.md)
  - [零空间控制](../../wiki/concepts/null-space-control.md)

### 4) A general framework for managing multiple tasks in highly redundant robotic systems（Siciliano & Slotine, IROS 1991）

- **链接：** <https://doi.org/10.1109/IROS.1991.174710>
- **核心贡献：** **Augmented Jacobian** 任务堆叠：把已满足的高层任务并进增广雅可比，再对下层求零空间。这是 Dietrich 文中 “augmented” 分支的运动学源头，也是后续 HQP「上层最优值作下层等式约束」的连续时间对应物。
- **对 wiki 的映射：**
  - [零空间控制](../../wiki/concepts/null-space-control.md)
  - [HQP](../../wiki/concepts/hqp.md)

### 5) Cartesian Impedance Control of Redundant Robots（Albu-Schäffer, Ott, Frese, Hirzinger, ICRA 2003）

- **链接：** DLR 技术报告/会议稿常引为 DLR Light-Weight Arms 笛卡尔阻抗；相关开放稿：<http://www.informatik.uni-bremen.de/agebv2/downloads/published/albuschaeffericra03.pdf>
- **核心贡献：** 在 **7 轴轻型臂**上把笛卡尔阻抗与关节零空间阻抗叠在同一力矩指令里；工程上常用 **静力学一致、不显式乘 $M(q)$** 的投影（Dietrich 表中 $W=I$ 静力学一致列）。这是 Franka / iiwa 笛卡尔阻抗示例的直接祖先。
- **对 wiki 的映射：**
  - [零空间控制](../../wiki/concepts/null-space-control.md)
  - [阻抗控制](../../wiki/concepts/impedance-control.md)
  - [Franka Research 3](../../wiki/entities/franka-research-3.md)

### 6) A C++ Implementation of a Cartesian Impedance Controller for Robotic Manipulators（Mayr & Salt-Ducaju, JOSS 2024 / arXiv:2212.11215）

- **链接：** JOSS <https://doi.org/10.21105/joss.05194>；预印本 <https://arxiv.org/abs/2212.11215>
- **代码：** <https://github.com/matthias-mayr/Cartesian-Impedance-Controller>
- **项目页：** <https://matthias-mayr.github.io/Cartesian-Impedance-Controller/>
- **核心贡献：** 力矩指令 $\tau_c=\tau^{\mathrm{ca}}+\tau^{\mathrm{ns}}+\tau^{\mathrm{ext}}$：笛卡尔阻抗 + **雅可比零空间关节阻抗** + 期望末端 wrench。相对官方 `franka_ros` / `libfranka` / KUKA FRI：**多机型、可在线改刚度/零空间构型、可跟 MoveIt 关节轨迹**。论文明确：Moore–Penrose 投影计算便宜但不动力学解耦，非静平衡时 $\tau_0$ 可能漏到笛卡尔方向（Ott 2008）。
- **对 wiki 的映射：**
  - [Cartesian Impedance Controller（论文实体）](../../wiki/entities/paper-cartesian-impedance-controller.md)
  - [零空间控制](../../wiki/concepts/null-space-control.md)

## 当前提炼状态

- [x] 经典速度层 / 操作空间 / 综述 / 7 轴阻抗开源 六条主线摘要
- [x] 步骤 2.5：开源边界写入对应 `sources/repos/` 与 wiki
- [ ] 后续可补：Flacco SNS（限位饱和）、Kanoun 2011（不等式任务 → HQP）已在 HQP 页引用，不必重复造页
