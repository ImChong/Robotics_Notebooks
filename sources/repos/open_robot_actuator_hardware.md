# open_robot_actuator_hardware（ODRI）

> 来源归档

- **标题：** Open Robot Actuator Hardware
- **类型：** repo
- **组织：** open-dynamic-robot-initiative
- **链接：** https://github.com/open-dynamic-robot-initiative/open_robot_actuator_hardware
- **项目页：** https://open-dynamic-robot-initiative.github.io
- **许可：** BSD-3-Clause
- **星标（截至 2026-07-25）：** ~1432
- **入库日期：** 2026-07-25
- **一句话说明：** ODRI 开源力控关节硬件总仓：机械结构、行星/皮带减速、驱动 PCB、编码器、电流/力矩控制与装配测试资料。
- **开源状态：** **已开源**（硬件设计与文档；电机本体通常采购现成无刷外转子）
- **架构论文：** https://arxiv.org/abs/1910.00093（归档：[open_torque_controlled_modular_robot_solo_arxiv_1910_00093.md](../papers/open_torque_controlled_modular_robot_solo_arxiv_1910_00093.md)）
- **沉淀到 wiki：** [odri-solo-and-bolt](../../wiki/entities/odri-solo-and-bolt.md)、[paper-open-torque-controlled-modular-robot-solo](../../wiki/entities/paper-open-torque-controlled-modular-robot-solo.md)、[open-source-qdd-actuator-projects](../../wiki/comparisons/open-source-qdd-actuator-projects.md)

---

## 为什么值得保留

- 完整开源 QDD 关节体系的学术基线：结构—减速—驱动—传感—通信—测试闭环，适合作为力控执行器「第一深读仓」。
- 最初服务四足（Solo 等），架构可迁移到小型/中型人形髋膝踝原型。

## 覆盖范围（策展归纳）

| 模块 | 内容 |
|------|------|
| 机械 | 关节结构、行星或皮带减速 |
| 电气 | 电机驱动 PCB、编码器接口 |
| 控制 | 电流环与关节力矩控制 |
| 通信 | CAN / 以太网 |
| 工程 | 装配步骤、测试方法、热管理与测试台思路 |

## 局限

- **不含完整电机电磁设计**；典型做法是采购高扭矩密度外转子无刷电机 + 低减速比。

## 对 wiki 的映射

- [ODRI Solo / Bolt](../../wiki/entities/odri-solo-and-bolt.md)
- [Solo 架构论文](../../wiki/entities/paper-open-torque-controlled-modular-robot-solo.md)
- [开源 QDD 执行器项目对比](../../wiki/comparisons/open-source-qdd-actuator-projects.md)
