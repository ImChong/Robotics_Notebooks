# ControlIt! - A Software Framework for Whole-Body Operational Space Control（arXiv:1506.01075）

> 来源归档（ingest）

- **标题：** ControlIt! - A Software Framework for Whole-Body Operational Space Control
- **类型：** paper / software framework
- **作者：** C.-L. Fok, G. Johnson, J. D. Yamokoski, A. Mok, Luis Sentis
- **arXiv：** <https://arxiv.org/abs/1506.01075>
- **PDF：** <https://arxiv.org/pdf/1506.01075>
- **发表：** 2015-06-02
- **代码：** <https://github.com/liangfok/controlit>（**已开源**，LGPL-2.1；ROS Indigo / Catkin 时代）
- **入库日期：** 2026-09-15
- **步骤 2.5：** 项目页即 arXiv + GitHub；仓库含 README、插件架构与 Dreamer 仿真演示配置 → **已开源**（依赖较老 ROS 栈，维护活跃度有限）。

## 摘要级要点

- **算法：** **Whole Body Operational Space Control（WBOSC）** — 浮基高冗余机器人在物理约束下统一做操作空间运动/力控制。
- **软件缺口：** 此前 **UTA-WBC** 等实现与特定应用/平台绑定，伺服延迟约 **5 ms**；缺通用架构与 API 研究。
- **ControlIt! 贡献：**
  - **多线程** 提升标准 PC 上伺服频率；
  - **参数绑定机制** 经可扩展传输协议与外部进程紧耦合；
  - **插件化：** 新机器人仅需 **两个插件 + URDF**；新 WBC 原语通过 Task/Constraint 插件扩展。
- **真机验证：** **Dreamer** 16-DoF 力控人形上半身（串联弹性与 co-actuated 关节）；产品拆解任务；笛卡尔位置目标在线更新以应对物体位姿变化。
- **性能：** 两笛卡尔位置 + 两姿态 + 低优先级姿态任务配置下，平均伺服延迟约 **0.5 ms**（显著低于 UTA-WBC 5 ms）。
- **许可：** LGPL；作者希望成为 WBC 社区长期开发与集成平台。

## 对 wiki 的映射

- 新建实体：[controlit](../../wiki/entities/controlit.md)
- 理论线：[paper-khatib-sentis-ijhr-2004-whole-body-dynamic-behavior](../../wiki/entities/paper-khatib-sentis-ijhr-2004-whole-body-dynamic-behavior.md)、[paper-operational-space-formulation](../../wiki/entities/paper-operational-space-formulation.md)
- 仓库归档：[sources/repos/controlit.md](../repos/controlit.md)
