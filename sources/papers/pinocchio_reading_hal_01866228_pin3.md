# Pinocchio 论文通读（SII 2019）

> 来源归档（原文通读 · 支撑 `wiki/entities/pinocchio.md`）

- **标题：** The Pinocchio C++ library: A fast and flexible implementation of rigid body dynamics algorithms and their analytical derivatives
- **作者：** Justin Carpentier, Guilhem Saurel, Gabriele Buondonno, Joseph Mirabel, Florent Lamiraux, Olivier Stasse, Nicolas Mansard（LAAS-CNRS / INRIA）
- **类型：** paper / software / rigid-body-dynamics
- **原文：** <https://laas.hal.science/hal-01866228>（2026-11 通过 HAL PDF 获取全文，7 页）
- **一句话说明：** 介绍 Pinocchio 的功能集（空间代数、模型/数据分离、Lie 群几何、RNEA/CRBA/ABA、解析导数、自动微分与代码生成）、静态多态加 variant 的实现模式，以及对 RBDL 的性能对比。
- **核对范围：** 通读全文 §I–§VII（含 §II 功能、§III 性能设计、§IV 上手示例、§V 性能实验、§VI 应用、§VII 结论）及参考文献表；图 1–4 只读到文字描述与图注，未核对原始数值曲线。未读其引用的解析导数论文（RSS 2018）等延伸文献，也未读当前 GitHub 仓库的新版文档。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **定位（§I）：** 同时兼顾两种范式：代码生成（针对单机器人生成专用源码，快但改模型要重生成）与运行时加载描述文件（灵活但较慢）。Pinocchio 可运行时加载任意模型，也能在运行时生成机器人专用源码。
- **空间代数与模型/数据分离（§II-A/B）：** 提供 SE3、Motion、Force、Inertia 类；model 创建后不被算法修改，data 存放计算结果与中间量以避免动态内存分配；算法统一签名 `algorithm(model, data, arg1, ...)`，多进程可共享 model、各持一份 data。模型可由 C++ API 或 URDF / Lua / Python 载入。
- **关节与 Lie 群（§II-C/D）：** 支持 revolute、prismatic、spherical、translation、planar、free-floating，以及复合关节；各关节类型有自己的配置空间与切空间，实现对应的积分（exp）与差分（log）算子。
- **几何模型（§II-E）：** 另有 GeometricModel/GeometricData，碰撞与距离计算基于 FCL（用的是 0.3.1 版分支）。
- **主要算法（§II-F）：** 递归实现，利用运动树稀疏性；FK 到二阶、运动学雅可比、RNEA（逆动力学）、CRBA（关节空间惯量矩阵）、ABA（正动力学）；另有约束正动力学、冲击动力学、惯量矩阵逆、质心动力学。
- **解析导数与自动微分（§II-G/H）：** 声称是首个原生实现解析导数的刚体框架；全标量模板化，可接 ADOL-C、CasADi、CppAD；经 CppADCodeGen 可生成 C 等代码；原文指出自动微分通常比解析导数慢得多。
- **速度来源（§III）：** 每种关节有自己的稀疏表示；用 CRTP 静态多态替代虚函数，避免间接调用并便于编译器优化；用 variant（JointModelVariant）在运行时选择每个关节的步骤，恢复灵活性。代价是不同关节类不能转成同一基类、不能直接放进同一个基类容器。
- **上手（§IV）：** BSD-2-Clause；GitHub stack-of-tasks/pinocchio；Python 示例 `buildModelFromUrdf` + `rnea` 对 Atlas 计算逆动力学；教程在 pinocchio-tutorials 仓库。
- **性能（§V）：** 8 个机器人模型（LWR、HyQ、TIAGo、Nao、Poppy、Atlas、TALOS 等），每例 100000 次随机输入，Intel Core i7 2.4 GHz 笔记本。动态加载下，机械臂约 1 微秒、腿足约 3 微秒；RNEA 与 RBDL 相近，ABA 与 CRBA 优于 RBDL；代码生成使耗时降为约 1/3（复杂机器人）到 1/8（LWR 机械臂）；生成代码后 RNEA / ABA / CRBA 成本接近。解析导数相对有限差分更快（图 4）。
- **应用（§VI）：** HRP-2 多接触爬楼梯（约 100 次实机，成功率约 80%）；TALOS-01 运动学/动力学任务；150 次 RNEA 在 1 ms 内用于动力学滤波；全身最优控制与规划（HPP）。
- **结论与计划（§VII）：** 后续计划加入传动（齿轮、滑轮、肌腱）与驱动（电机、肌肉）模型。

## 对 wiki 的映射

- [`wiki/entities/pinocchio.md`](../../wiki/entities/pinocchio.md)

## 参考来源

- 论文：<https://laas.hal.science/hal-01866228>
