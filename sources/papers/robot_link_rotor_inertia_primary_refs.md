# 机器人连杆惯量与转子惯量（一手资料索引）

> 来源归档（ingest）

- **标题：** 连杆刚体惯量 + 电机转子反射惯量：规范、教材与经典论文一手摘录
- **类型：** paper / textbook / specification（合集）
- **入库日期：** 2026-05-20
- **一句话说明：** 汇总 URDF 连杆惯性张量、开放链刚体动力学教材表述、Gautier–Khalil 最小惯性参数集、以及 MuJoCo `armature`（转子反射惯量）的官方定义，作为「连杆惯量 vs 转子惯量」分工的原始依据。
- **沉淀到 wiki：** 是 → [`wiki/concepts/robot-link-and-rotor-inertia.md`](../../wiki/concepts/robot-link-and-rotor-inertia.md)

## 为什么值得保留

- 仿真与控制里常把「URDF 里的 link 惯量」与「关节侧的 armature / 反射惯量」混为一谈，导致 **PD 增益、SysID、Sim2Real** 在错误自由度上修正参数。
- 本索引只收录 **规范文档、教材章节、期刊论文、仿真器官方文档** 四类一手来源，避免二手博客口径漂移。

## 核心摘录

### 1) URDF：连杆 `<inertial>`（连杆刚体惯量）

- **来源：** ROS Wiki — [urdf/XML/link — inertial](http://wiki.ros.org/urdf/XML/link#inertial)（规范级描述）
- **要点：**
  - 每个 `link` 可选 `<inertial>`：定义 **质量**、**质心相对 link 坐标系 L 的位姿**（`<origin xyz rpy>`）、以及 **相对质心坐标系 C 的惯性张量**（`ixx, ixy, ixz, iyy, iyz, izz`）。
  - 张量分量定义在 **质心系 C**；`Ĉx,Ĉy,Ĉz` 不必与主轴对齐，但文档建议对齐主轴使 **惯量积为零**，以减少 CAD→URDF 符号约定问题（URDF 对惯量积采用 **负号约定**，与部分 CAD 导出工具不同）。
  - 未指定时默认为 **零质量、零惯量**——这是许多「动力学仿真发飘」的隐蔽根因之一。
- **对 wiki 的映射：** [robot-link-and-rotor-inertia](../../wiki/concepts/robot-link-and-rotor-inertia.md)

### 2) Modern Robotics Ch.8：开放链动力学中的连杆惯量

- **来源：** Lynch & Park, *Modern Robotics* (CUP 2017)，[官方 PDF](https://hades.mech.northwestern.edu/images/7/7f/MR.pdf)，[课程站](https://hades.mech.northwestern.edu/index.php/Modern_Robotics)
- **要点：**
  - 第 8 章 *Dynamics of Open Chains* 用拉格朗日 / 牛顿–欧拉建立 $M(q)\ddot q + C(q,\dot q)\dot q + g(q) = \tau$；$M(q)$ 由 **各连杆空间惯量（spatial inertia）** 经运动学递归组合得到。
  - 教材强调：连杆惯量进入 **广义惯性矩阵**，与关节力矩在同一 $n$ 维广义坐标上耦合；这是 Pinocchio / Drake / Crocoddyl 等库的数学底座。
  - **不包含** 电机转子经减速器反射的额外惯量——该部分需在仿真器或执行器模型中 **单独参数化**（见下条 MuJoCo `armature`）。
- **对 wiki 的映射：** [robot-link-and-rotor-inertia](../../wiki/concepts/robot-link-and-rotor-inertia.md)、[modern-robotics-book](../../wiki/entities/modern-robotics-book.md)

### 3) Gautier & Khalil (1990)：串联机器人最小惯性参数集

- **来源：** M. Gautier, W. Khalil, *Direct Calculation of Minimum Set of Inertial Parameters of Serial Robots*, IEEE Trans. Robotics and Automation, 6(3):368–373, 1990. DOI: [10.1109/70.56657](https://doi.org/10.1109/70.56657)；HAL: [hal-05223021](https://hal.science/hal-05223021/)
- **要点：**
  - 动力学模型中大量 **连杆惯性参数线性相关或不可辨识**；论文给出 **闭式关系** 直接计算 **最小参数集**，降低建模与辨识成本。
  - 工程含义：SysID 时不必盲目拟合 URDF 中全部 10 个惯性分量；应区分 **可辨识组合** 与 **对力矩无贡献的参数**。
  - 后续激励轨迹设计见 Gautier & Khalil, *Exciting Trajectories for Identification* (1992)，DOI: [10.1109/70.163864](https://doi.org/10.1109/70.163864)（已收录于 `sources/papers/system_identification.md`）。
- **对 wiki 的映射：** [robot-link-and-rotor-inertia](../../wiki/concepts/robot-link-and-rotor-inertia.md)、[system-identification](../../wiki/concepts/system-identification.md)

### 4) MuJoCo MJCF：关节 `armature`（转子反射惯量）

- **来源：** DeepMind MuJoCo — [XML Reference — body/joint — armature](https://mujoco.readthedocs.io/en/latest/XMLreference.html)（源码 `doc/XMLreference.rst`，与 [armature_equivalence 测试模型](https://github.com/google-deepmind/mujoco/blob/main/test/engine/testdata/armature_equivalence.xml) 一致）
- **要点：**
  - `armature`：**不来自连杆质量的附加关节惯量**，通常由 **经减速器放大的转子惯量** 引起。
  - 官方说明：齿轮比 $G$ 在力与长度上各出现一次，等效为 **反射惯量（reflected inertia）** $J_{\mathrm{ref}} = J_{\mathrm{rotor}} \cdot G^2$（示例中 $G=3 \Rightarrow$ 放大 $9$ 倍）。
  - 文档建议：适当正的 `armature` 可 **显著改善数值稳定性**；高减速比腿足/人形关节在 Sim2Real 中常不可忽略。
  - 与 URDF 分工：URDF **无标准字段** 表达 armature；迁移到 MJCF 时常在 **joint** 上补 `armature`，或等价建模为带传动比的附加惯量体。
- **对 wiki 的映射：** [robot-link-and-rotor-inertia](../../wiki/concepts/robot-link-and-rotor-inertia.md)、[armature-modeling](../../wiki/concepts/armature-modeling.md)

## 推荐继续阅读（外部）

- Featherstone, *Rigid Body Dynamics Algorithms* — 空间惯量与递推动力学（研究级补全）
- Nguyen & Park (IJRR 2011) — 物理一致性约束下的惯性参数辨识（见 `sources/papers/system_identification.md`）

## 当前提炼状态

- [x] 四类一手来源摘录与 wiki 映射
- [x] 沉淀统一概念页 `wiki/concepts/robot-link-and-rotor-inertia.md`
- [ ] 后续可补：SolidWorks/Fusion → URDF 惯量积符号勘误 checklist（工程向）
