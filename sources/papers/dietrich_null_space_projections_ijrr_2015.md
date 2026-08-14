# An overview of null space projections for redundant, torque-controlled robots（Dietrich / Ott / Albu-Schäffer，IJRR 2015）

> 来源归档（ingest）

- **标题：** An overview of null space projections for redundant, torque-controlled robots
- **类型：** paper / survey / torque-control / redundancy
- **期刊：** *The International Journal of Robotics Research* 34(11):1385–1400，2015
- **DOI：** <https://doi.org/10.1177/0278364914566516>
- **开放 PDF：** <https://elib.dlr.de/101443/2/NullspaceSurvey.pdf>（DLR elib；入库日可访问）
- **作者：** Alexander Dietrich、Christian Ott、Alin Albu-Schäffer
- **机构：** 德国航空航天中心（DLR）机器人与机电一体化研究所；慕尼黑工业大学（TUM）
- **入库日期：** 2026-08-13
- **一句话说明：** 力矩控制冗余解析的权威综述：比较 successive/augmented 层次与 static/dynamic/stiffness 一致性，并在 **7 轴 LWR-III** 上做真机对照。
- **沉淀到 wiki：** [`wiki/entities/paper-null-space-projections-survey.md`](../../wiki/entities/paper-null-space-projections-survey.md)

## 开源状态（步骤 2.5）

- **项目页：** 无独立 `*.github.io` 项目页；入口为期刊 DOI 与 DLR elib PDF。
- **代码：** 论文未附 GitHub / 仿真包。实验平台为实验室 **DLR KUKA lightweight robot III**（7 DoF）。
- **结论：** **确认未开源**（方法综述 + 封闭硬件实验）。工程复现应转开源实现：[Cartesian Impedance Controller](../repos/cartesian-impedance-controller.md)、[libfranka](../repos/libfranka.md)、[TSID](../repos/tsid.md)。

## 摘录 1：问题与层次结构（§1–§2）

- 冗余解析的主流工具仍是 1980 年代零空间投影（Khatib 1987；Nakamura et al. 1987；Siciliano & Slotine 1991）：高层任务用满自由度，下层任务投进上层零空间。
- **Successive：** $N_i^{\mathrm{suc}}=N_{i-1}^{\mathrm{suc}}(I-J_{i-1}^\top(J_{i-1}^\#)^\top)$。实现简单，但多层后**不能严格保证**对所有更高层的静力学解耦。
- **Augmented：** 从第三层起用堆叠雅可比 $J_{1:i-1}$ 一次投影，层次更严；代价是增广矩阵与算法奇异。
- **对 wiki 的映射：** [零空间控制](../../wiki/concepts/null-space-control.md) 的「层次结构」节；对照 [HQP](../../wiki/concepts/hqp.md)（不等式时代替显式 $N$）。

## 摘录 2：投影器一致性（§3）

- **静力学一致：** 稳态 $\dot q=\ddot q=0$ 时，投影后的次级力矩不在主任务方向产生力。常用 $W=I$ 的 Moore–Penrose，**不需要惯量模型**。
- **动力学一致：** 次级力矩不产生主任务加速度。Khatib 操作空间取 $W=M(q)$；本文指出满足同一判据的加权矩阵有**无穷多**，$M$ 只是直观特例。
- **刚度一致（本文新概念）：** 当高层任务由机械弹簧被动维持时，用刚度信息代替惯量做伪逆加权，避免主动控制器「对抗」弹簧零空间。
- **真机读法（§5）：** 仿真里动力学一致明显更好；**LWR-III 实验中差距显著缩小**，因为惯量/运动学/摩擦模型不准。选投影器要按「有没有可信 $M(q)$」而不是按论文排名。

**对 wiki 的映射：** 论文实体页写「理论最优 ≠ 真机最优」；概念页给选型表。

## 摘录 3：7 轴真机实验（§4.2）

三层阻抗、**7 DoF LWR-III**：

| 优先级 | 任务 |
|--------|------|
| 1 | 笛卡尔平移阻抗，保持初始 TCP 位置 |
| 2 | 笛卡尔姿态阻抗，跟踪大范围旋转轨迹 |
| 3 | 完整关节阻抗，维持初始构型（零空间姿态） |

对照还包括「完全不加投影、把三层力矩直接相加」——下层会明显干扰上层。稳定实验中，§3.3.2 那种「加速度层投影再补惯量」的变体出现**稳定性问题**，不宜当默认实现。

**对 wiki 的映射：** 7 轴工程实践以「位置 > 姿态 > 关节居中」为最小可跑层次；开源对照 Mayr 控制器与 libfranka 示例。

## 建议 wiki 动作

- 升格 [`wiki/entities/paper-null-space-projections-survey.md`](../../wiki/entities/paper-null-space-projections-survey.md)
- 概念页 [`wiki/concepts/null-space-control.md`](../../wiki/concepts/null-space-control.md) 编译投影公式与 7 轴选型
- 交叉 [HQP](../../wiki/concepts/hqp.md) / [阻抗控制](../../wiki/concepts/impedance-control.md) / [TSID](../../wiki/concepts/tsid.md)
