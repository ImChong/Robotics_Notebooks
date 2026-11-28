# NASA NTRS：ADAMS 多体动力学一手工程与航天仿真报告通读

> 来源归档（原文通读 · 支撑 `wiki/entities/adams.md` 与 `wiki/comparisons/adams-vs-mujoco-vs-isaac-sim-three-problem-types.md`）

- **标题：** Application of the ADAMS Program to Deployable Space Truss Structures（NASA-CR-177927）& Extraction and Separation Modeling of Orion Test Vehicles with ADAMS Simulation（NASA/JSC AIAA CPAS）
- **作者 / 来源：** R. E. Calleson（Kentron International / NASA Langley, 1985）；Usbaldo Fraire Jr., Keith Anderson, P. A. Cuthbert（NASA Johnson Space Center / Jacobs / ATK, 2013）
- **类型：** paper / technical report（NASA NTRS 公开全文）
- **原文：**
  - NASA-CR-177927：<https://ntrs.nasa.gov/api/citations/19850022054/downloads/19850022054.pdf>
  - NASA CPAS Orion ADAMS：<https://ntrs.nasa.gov/api/citations/20130011078/downloads/20130011078.pdf>
- **入库日期：** 2026-11-28
- **一句话说明：** 经由 NASA NTRS 公开全文通读两份 ADAMS 一手工程技术报告，核实 MDI ADAMS 的 5 种分析模式、每部件 15 个一阶微分的 Lagrange 乘子 DAE 列式、Gear 刚性 vs Adams-Moulton 非刚性预测-校正积分、Grübler 自由度判据、刚体锁定（latch-up）极限以及现代 COTS ADAMS 在 Orion 载人飞船伞降分离 6-DOF 接触、航电延迟与蒙特卡洛标定中的工程应用。
- **沉淀到 wiki：** 是 → [`wiki/entities/adams.md`](../../wiki/entities/adams.md)、[`wiki/comparisons/adams-vs-mujoco-vs-isaac-sim-three-problem-types.md`](../../wiki/comparisons/adams-vs-mujoco-vs-isaac-sim-three-problem-types.md)
- **通读：** 已完成
- **核对范围：** 通读 `19850022054.pdf` 全文 42 页提取文本（Summary、Introduction 与 DISCOS/DADS/IMP 对比、Section 3 ADAMS 描述与公式/分析模式、Section 4–6 箱形/四面体天线桁架算例与锁紧弯矩分析）及 `20130011078.pdf` 全文 12 页（Orion PTV/CPSS 提取分离 6-DOF 模型、Ball-and-Socket 接触机制、90 ms 航电延迟、Monte Carlo 与飞行试验视频叠加标定）。注：Orlandea 1973 学位论文（U-M Deep Blue 校园限制）、1977 ASME 双篇（订阅限制）与 2026-11-28 Cadence/Hexagon 产品网页（Cloudflare 验证屏障）未直接抓取全文，均保持如实标注。

## 核心摘录（均出自 NASA NTRS 公开原文）

### 1) NASA-CR-177927 (1985)：MDI ADAMS 程序结构与数值机制

- **定位与多软件横评：** NASA Langley 评估了 IMP、DADS、DISCOS、SNAP 与 MDI 开发的 **ADAMS**（当时运行在 VAX 11/780 分时服务上），最终因用户文档、培训与复杂多件系统可工程化能力选用 ADAMS（相较之下 DISCOS 虽支持柔性体但多部件使用繁琐）。
- **5 种分析模式：**
  1. **Kineto-static（运动静力学 / 零自由度运动学）：** 仅求解代数方程，通过关节运动生成器反推位移、速度、加速度与反力，比动力学积分更快更准；
  2. **Static Equilibrium（静态平衡）：** 求解零速度/零加速度平衡构型；
  3. **Dynamic (Stiff Integration)：** 默认面向刚性大特征值跨度系统，使用变阶变步长 **Gear 算法** 预测-校正积分；
  4. **Dynamic (Non-Stiff Integration)：** 使用 **Adams-Moulton** 算法进行非刚性预测-校正积分；
  5. **Quasistatic（准静态）：** 按时间点串联一系列静态平衡求解（例如倾翻临界角分析）。
- **DAE 列式与建模原语：** 每个刚体部件（Part）通过二阶动力学方程与代数约束方程联立并引入 **Lagrange 乘子**（从而同步求得铰链约束反力），生成 **15 个一阶微分方程**；模型由 Parts、Markers（体固连点+正交局部坐标系）、Joints（支持 8 类：revolute/translational/screw/rack-and-pinion 1-DOF、cylindrical/universal 2-DOF、spherical/planar 3-DOF）、Forces、Generators 构成。
- **Grübler 自由度校核：** 采用 `DOF = 6*(PARTS-1) - 5*(TRA+REV) - 4*(CYL+UNI) - 3*(SPH+PLA) - 1*(GEN+RACK+SCREW+UMCON)`：`< 0` 为超静定结构报错停止，`= 0` 自动切换 kineto-static，`> 0` 进行力驱动瞬态仿真。
- **历史局限实录：** 1985 年版 ADAMS 仍限于刚体装配，展开末期杆件锁紧（latch-up）冲击弯矩无法直接由刚体求解得出，报告第 4 节不得不另行以一阶简支梁弯曲模态与动能—应变能守恒估算，并期待 MDI 于 1986 年补齐柔性体模块（即后续 ADAMS 刚柔耦合路线的前夜）。

### 2) NASA/JSC CPAS Orion ADAMS 报告 (2013)：COTS ADAMS 的高保真 6-DOF 分离标定

- **多体 6-DOF 与接触建模：** 将商业现货（COTS）ADAMS 用于 30,000 磅级 Orion PTV/CPSS 空投分离模型（AESM），引入球窝（ball-and-socket）分离机构接触几何、实时气动查表（迎角 $\alpha$ / 侧滑角 $\beta$）、YPG 大气表以及 **90 ms 航电延迟**。
- **工程 V&V 闭环：** 将 ADAMS 作为前端分离状态向量生成器，经由与另一套仿真 DSSA 的轨迹对比与飞行实测重构（引入尾流下洗力 downdraft、拖拽面积折减与 C-12 伴飞视频叠加）反复校准模型，再利用 **Monte Carlo** 分散确定 Smart Separation Window（MinPR 上调至 $10^\circ/\mathrm{s}$），显著提升试飞预测保真度。

## 对 wiki 的映射

- [`wiki/entities/adams.md`](../../wiki/entities/adams.md)
- [`wiki/comparisons/adams-vs-mujoco-vs-isaac-sim-three-problem-types.md`](../../wiki/comparisons/adams-vs-mujoco-vs-isaac-sim-three-problem-types.md)
