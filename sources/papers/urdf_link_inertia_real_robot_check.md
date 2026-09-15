# URDF 连杆惯量与真机对照检查（一手资料索引）

> 来源归档（ingest）

- **标题：** URDF `<inertial>` 规范 + CAD 惯量积约定 + 物理一致性判据 + 真机对照实验：一手摘录
- **类型：** specification / textbook / paper / official docs（合集）
- **入库日期：** 2026-09-15
- **一句话说明：** 汇总把 URDF 连杆惯量与真机对齐所需的**规范字段、符号约定、书桌物理一致性检查、称重/静力学/动力学对照**一手来源；服务「怎么查 URDF 惯量对不对」而不是再讲一遍连杆 vs 转子分工。
- **沉淀到 wiki：** 是 → [`wiki/queries/urdf-link-inertia-real-robot-check.md`](../../wiki/queries/urdf-link-inertia-real-robot-check.md)
- **姊妹索引：** [`robot_link_rotor_inertia_primary_refs.md`](./robot_link_rotor_inertia_primary_refs.md)（连杆 vs `armature` 分工）

## 为什么值得保留

- 厂商 URDF、CAD 导出、网上公开模型经常 **质量对、惯量积符号错、主惯量相对几何过大、缺 `<inertial>` 默认为零**。
- 真机对照不是「把 CAD 数字抄进 XML」：Atkeson 1986 已证明 **辨识参数对力矩的预测可以优于 CAD**；Gautier–Khalil 证明 **10 个惯性分量不能全部独立辨识**。
- 本索引只收录 **规范、官方仿真器文档、期刊/会议论文、库源码**，避免二手博客把「称一下整机」和「SysID 写回 URDF」混成一步。

## 核心摘录

### 1) URDF：`<inertial>` 字段语义（规范）

- **来源：** ROS Wiki — [urdf/XML/link](http://wiki.ros.org/urdf/XML/link)（镜像：[osuosl](http://wiki.ros.osuosl.org/urdf/XML/link)）
- **要点：**
  - `<inertial>` 可选；**未指定则默认零质量、零惯量**。
  - `<origin xyz rpy>`：质心系 **C** 相对 link 系 **L** 的位姿。`xyz` 是 $L_o \to C_o$；`rpy` 是 $\hat{C}$ 相对 $\hat{L}$ 的欧拉角。**$\hat{C}$ 不必与主惯量轴对齐**。
  - `<mass>`：连杆质量（千克）。
  - `<inertia ixx ixy ixz iyy iyz izz>`：相对 **质心 $C_o$**、沿 $\hat{C}$ 的惯性张量（含惯量积）。
  - **URDF 采用负号惯量积约定**（与部分 CAD 工具相反）。官方建议：把 $\hat{C}$ 对齐主轴使 **惯量积为零**，避开符号坑。
- **对 wiki 的映射：** [urdf-link-inertia-real-robot-check](../../wiki/queries/urdf-link-inertia-real-robot-check.md)、[urdf-robot-description](../../wiki/concepts/urdf-robot-description.md)、[robot-link-and-rotor-inertia](../../wiki/concepts/robot-link-and-rotor-inertia.md)

### 2) ROS 教程：默认值、单位、禁止用单位阵

- **来源：** [Adding Physical and Collision Properties to a URDF Model](http://wiki.ros.org/urdf/Tutorials/Adding%20Physical%20and%20Collision%20Properties%20to%20a%20URDF%20Model)（ROS 2 同文：[Humble 教程](https://docs.ros.org/en/humble/Tutorials/Intermediate/URDF/Adding-Physical-and-Collision-Properties.html)）
- **要点：**
  - 质量单位 **kg**；惯性矩阵对称，故只写 6 个数。
  - 初值可来自 MeshLab / CAD，或按 Wikipedia 几何体惯量公式（均匀密度近似）。
  - **不要填单位阵**：对 0.1 m 边长盒子，单位阵对应约 **600 kg**；中等连杆更合理的占位是 $I_{xx}=I_{yy}=I_{zz}\approx 10^{-3}$ 或更小。
  - 实时控制器里 **接近零的惯量** 会让模型在无警告下塌缩，所有 link 原点叠到世界原点。
- **对 wiki 的映射：** [urdf-link-inertia-real-robot-check](../../wiki/queries/urdf-link-inertia-real-robot-check.md)

### 3) CAD 惯量积符号：MathWorks / SolidWorks ↔ URDF

- **来源：** MathWorks Simscape — [Specifying Custom Inertias](https://www.mathworks.com/help/sm/ug/specify-custom-inertia.html)；工程复现讨论：[ros/solidworks_urdf_exporter#117](https://github.com/ros/solidworks_urdf_exporter/issues/117)
- **要点：**
  - 负号约定：$I_{xy}=-\int xy\,\rho\,dv$（以及 $I_{xz}$、$I_{yz}$）。这是 URDF / Simscape 写入的对称矩阵非对角元。
  - SolidWorks Mass Properties 常用 **去掉负号** 的交替约定。从 SW 拷数字进 URDF 时，**必须把非对角元取反**，或在 SW 切到 Negative Tensor Notation。
  - 张量必须表达在 **URDF 的质心系**；若 CAD 给的是零件原点惯量，先用平行轴定理搬到 CoM，再旋转到 `<origin rpy>`。
- **对 wiki 的映射：** [urdf-link-inertia-real-robot-check](../../wiki/queries/urdf-link-inertia-real-robot-check.md)

### 4) Gazebo：质量/主惯量为零会炸；用 CoM 可视化抽检

- **来源：** OSRF — [Using a URDF in Gazebo](https://github.com/osrf/gazebo_tutorials/blob/master/ros_urdf/tutorial.md)
- **要点：**
  - 每个 link **必须**有配置正确的 `<inertial>`。
  - **质量必须 > 0**，否则 Gazebo 会忽略该 link。
  - 主惯量 $I_{xx},I_{yy},I_{zz}=0$ 会在有限力矩下产生 **无限加速度**。
  - 可视化抽检：Gazebo `View → Wireframe` + `Center of Mass`，看质心是否落在连杆几何内。
- **对 wiki 的映射：** [urdf-link-inertia-real-robot-check](../../wiki/queries/urdf-link-inertia-real-robot-check.md)

### 5) MuJoCo：公开 URDF 惯量经常「相对质量过大」；看等效惯量盒

- **来源：** DeepMind MuJoCo — [XML Reference — compiler `inertiafromgeom`](https://mujoco.readthedocs.io/en/latest/XMLreference.html#compiler-inertiafromgeom)
- **要点：**
  - `inertiafromgeom`：`false` 必须手写 inertial；`true` **用 geom 覆盖** XML 里的 inertial；`auto`（默认）仅在缺 inertial 时从 geom 推断。
  - 官方点名：不少公开 URDF **惯量相对质量任意偏大**，等效惯量盒远超出几何边界。内置可视化可画 **equivalent inertia box**。
  - 把「看起来能仿真」当验收是不够的：几何对、惯量盒飞出外壳，说明 CAD/URDF 数字未过物理尺度检查。
- **对 wiki 的映射：** [urdf-link-inertia-real-robot-check](../../wiki/queries/urdf-link-inertia-real-robot-check.md)、[robot-link-and-rotor-inertia](../../wiki/concepts/robot-link-and-rotor-inertia.md)

### 6) Pinocchio：URDF → 空间惯量；总质量 / 重力 / 回归矩阵

- **来源：** [stack-of-tasks/pinocchio `src/parsers/urdf/model.cpp`](https://github.com/stack-of-tasks/pinocchio/blob/master/src/parsers/urdf/model.cpp) 的 `convertFromUrdf`；算法文档 [`computeTotalMass`](https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/devel/doxygen-html/group__pinocchio__algorithm.html)
- **要点：**
  - 解析：把 URDF 的 $I_C$ 按 `<origin>` 旋转到 link，$Y = \mathrm{Inertia}(m, c, R I_C R^\top)$。缺 inertial 则 **零惯量**。
  - 对照真机的三个官方入口：`computeTotalMass`（Σ 质量 vs 台秤）、`centerOfMass`（整机 CoM vs 悬挂/测力板）、`computeGeneralizedGravity`（$g(q)$ vs 静止力矩）。
  - `computeJointTorqueRegressor` 给出 $\tau = Y_{\mathrm{rb}}\pi_{\mathrm{rb}}$ 的连杆 10 参数回归；**不含** armature / 粘滞 / 库仑。
- **对 wiki 的映射：** [urdf-link-inertia-real-robot-check](../../wiki/queries/urdf-link-inertia-real-robot-check.md)、[pinocchio](../../wiki/entities/pinocchio.md)、[gravity-compensation](../../wiki/concepts/gravity-compensation.md)

### 7) Atkeson, An, Hollerbach (IJRR 1986)：CAD 不是力矩真值

- **来源：** C. G. Atkeson, C. H. An, J. M. Hollerbach, *Estimation of Inertial Parameters of Manipulator Loads and Links*, IJRR 5(3):101–119, 1986. DOI: [10.1177/027836498600500306](https://doi.org/10.1177/027836498600500306)
- **要点：**
  - 把牛顿–欧拉改写成对惯性参数 **线性** 的回归，用最小二乘从一般运动估计载荷与连杆参数。
  - 载荷：**质量与质心**估计较好；**转动惯量更难**。
  - 在 MIT 直接驱动臂上，用辨识参数预测的关节力矩 **优于 CAD 建模参数**。
  - 靠近基座、传感不足时部分连杆参数 **不可辨识**。
- **对 wiki 的映射：** [urdf-link-inertia-real-robot-check](../../wiki/queries/urdf-link-inertia-real-robot-check.md)、[system-identification](../../wiki/concepts/system-identification.md)

### 8) Traversaro et al. (IROS 2016)：完全物理一致性 = 正定 + 三角不等式

- **来源：** S. Traversaro, S. Brossette, A. Escande, F. Nori, *Identification of Fully Physical Consistent Inertial Parameters using Optimization on Manifolds*, IROS 2016. arXiv: [1610.08703](https://arxiv.org/abs/1610.08703)
- **要点：**
  - 仅要求 $m>0$、$I_C \succ 0$ **不够**：还须主惯量满足三角不等式 $J_1+J_2\ge J_3$ 等，否则 **不存在** 能生成该 10 参数的物理刚体。
  - 在 iCub 手臂上做了单刚体辨识实验验证。
- **对 wiki 的映射：** [urdf-link-inertia-real-robot-check](../../wiki/queries/urdf-link-inertia-real-robot-check.md)

### 9) Wensing, Kim, Slotine (RA-L 2018)：物理一致性 LMI；辨识时连带传动件

- **来源：** P. M. Wensing, S. Kim, J.-J. E. Slotine, *Linear Matrix Inequalities for Physically Consistent Inertial Parameter Identification*, IEEE RA-L 3(1):60–67, 2018. arXiv: [1701.04395](https://arxiv.org/abs/1701.04395)
- **要点：**
  - 把 Traversaro 的完全物理一致性写成 **LMI**，可与最小二乘一起做成全局最优 SDP。
  - 物理一致性看的是质量分布的 **二阶矩 / 协方差**，不是随便一组 $I_{xx}$。
  - 可加包围盒约束：质量必须落在 CAD 几何内。
  - 在 **MIT Cheetah 3 单腿** 上同时辨识连杆惯量与传动件；不能把减速器惯量偷偷写进 URDF 质量。
- **对 wiki 的映射：** [urdf-link-inertia-real-robot-check](../../wiki/queries/urdf-link-inertia-real-robot-check.md)、[robot-link-and-rotor-inertia](../../wiki/concepts/robot-link-and-rotor-inertia.md)、[system-identification](../../wiki/concepts/system-identification.md)

### 10) Ayusawa, Venture, Nakamura (T-RO 2014)：浮动基可用接触力对照全身惯量

- **来源：** K. Ayusawa, G. Venture, Y. Nakamura, *Identifiability and Full Body Estimation of Inertial Parameters for Legged Robots*, IEEE T-RO 30(2), 2014（基座方程线；FloBaRoID 亦引用）。综述式入口见 Wensing 2018 §I 对 [11, 23] 的引用。
- **要点：**
  - 浮动基牛顿–欧拉嵌在未驱动基座行里：用 **接触力 + 运动学** 即可估计全身惯性参数，**不必**每关节都有力矩计。
  - 关节摩擦不进入基座 wrench 方程（FloBaRoID 两步法的理论依据）。
- **对 wiki 的映射：** [urdf-link-inertia-real-robot-check](../../wiki/queries/urdf-link-inertia-real-robot-check.md)、[flobaroid](../../wiki/entities/flobaroid.md)、[humanoid-closed-loop-inertia-calibration](../../wiki/concepts/humanoid-closed-loop-inertia-calibration.md)

## 推荐继续阅读（外部）

- Gautier & Khalil 1990 — 最小惯性参数集（已收录 `robot_link_rotor_inertia_primary_refs.md` / `system_identification.md`）
- Sousa & Cortesão, *Physical feasibility of robot base inertial parameter identification* (2014) — 半一致性 LMI（$I \succ 0$）
- Featherstone, *Rigid Body Dynamics Algorithms* — 空间惯量与等效惯量盒直觉
- [FloBaRoID](https://github.com/kjyv/FloBaRoID) — 激励 → 测量 → 物理一致辨识 → 写回 URDF

## 当前提炼状态

- [x] 规范 / CAD 符号 / 仿真器可视化 / 经典辨识 / 物理一致性 一手摘录
- [x] 沉淀操作指南 `wiki/queries/urdf-link-inertia-real-robot-check.md`
- [ ] 后续可补：单连杆三线摆 / 双线摆实验室测 $I$ 的实验规程（工程向，非本轮范围）
