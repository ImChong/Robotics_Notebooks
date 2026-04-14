# robot_kinematics_tools

> 来源归档（ingest）

- **标题：** 机器人运动学/动力学工具核心论文
- **类型：** paper
- **来源：** ICRA / RA-L / Humanoids / arXiv
- **入库日期：** 2026-04-14
- **最后更新：** 2026-04-14
- **一句话说明：** 覆盖 Pinocchio、RBDL、Crocoddyl 等运动学/动力学计算库的原始论文

## 核心论文摘录

### 1) PINOCCHIO: Fast Forward and Inverse Dynamics for Poly-articulated Systems（Carpentier et al., 2019）
- **链接：** <https://hal.inria.fr/hal-01866228>
- **核心贡献：** 基于空间向量代数实现 RNEA / CRBA / ABA；支持任意关节类型（球/旋转/棱柱/自由浮动）；Python/C++ 双接口；与 casADi / OSQP / qpOASES 无缝集成；成为 WBC / TO 领域事实标准
- **对 wiki 的映射：**
  - [pinocchio](../../wiki/entities/pinocchio.md)
  - [tsid](../../wiki/concepts/tsid.md)

### 2) Crocoddyl: An Efficient and Versatile Framework for Multi-Contact Optimal Control（Mastalli et al., ICRA 2020）
- **链接：** <https://arxiv.org/abs/1909.07286>
- **核心贡献：** 在 Pinocchio 基础上实现 DDP/FDDP 最优控制求解；多接触动力学约束；差分动力学模型（ActionModel）支持任意成本函数；在 ANYmal 上验证了实时 MPC（50Hz）
- **对 wiki 的映射：**
  - [crocoddyl](../../wiki/entities/crocoddyl.md)
  - [trajectory-optimization](../../wiki/methods/trajectory-optimization.md)

### 3) RBDL: An Efficient Rigid-Body Dynamics Library（Felis, 2017）
- **链接：** <https://joss.theoj.org/papers/10.21105/joss.00500>
- **核心贡献：** 纯 C++ 实现的轻量级刚体动力学库；支持 URDF 加载；提供 RNEA/CRBA/ABA 和雅可比计算；适合嵌入式或对依赖要求严格的场景（无 Eigen 大块依赖）
- **对 wiki 的映射：**
  - [pinocchio](../../wiki/entities/pinocchio.md)

### 4) A Survey of Motion Planning and Control Techniques for Humanoid Robots（Wieber et al., 2016）
- **链接：** <https://link.springer.com/chapter/10.1007/978-3-319-29803-8_1>
- **核心贡献：** 综述人形机器人运动规划全链路：步态生成/ZMP/MPC/全身控制；系统梳理运动学/动力学工具在各层的应用；是学习该领域工具链的权威入门综述
- **对 wiki 的映射：**
  - [pinocchio](../../wiki/entities/pinocchio.md)
  - [whole-body-control](../../wiki/concepts/whole-body-control.md)

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [ ] 关联 wiki 页面的参考来源段落已添加 ingest 链接
