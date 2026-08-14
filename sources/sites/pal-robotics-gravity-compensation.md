# PAL OS Gravity Compensation controller 文档

- **类型：** 网站 / 硬件控制器文档
- **入口：** <https://docs.pal-robotics.com/25.01/hardware/controllers/gravity-compensation.html>
- **主体：** PAL Robotics（TIAGo / TIAGo PRO）
- **代码：** 教程仓 <https://github.com/pal-robotics/gravity_compensation_controller_tutorial>（**部分开源**，见 [仓库归档](../repos/gravity-compensation-controller-tutorial.md)）
- **收录日期：** 2026-08-13
- **抓取说明：** 以 2026-08-13 对 PAL OS 25.01 文档页公开文案为准。

## 一句话

文档描述的是 **PAL OS 发行版**里的 `pal_controllers/GravityCompensationController`：effort 接口、kinesthetic teaching、TIAGo PRO **7 轴** YAML。它**不是**教程仓里的那份 C++ 源码。

## 开源与项目页核查（2026-08-13）

| 项 | 结论 |
|----|------|
| **生产代码** | **未开源** — 类型名为 `pal_controllers/GravityCompensationController`，文档未给 GitHub |
| **教程代码** | **已公开** — 见上表教程仓；许可未声明 |
| **数据 / 权重** | 不适用 |
| **范围** | 仅手臂；头/躯干不可用此模式 |
| **参数** | `root_link` / `tip_links` 定运动学链；`torque_gain`；每关节 `static_friction` / `viscous_friction`；SEA 臂的 $K_t$ 与减速比在底层已计入 |

## 与仓库内实体的关系

- [教程仓归档](../repos/gravity-compensation-controller-tutorial.md)
- [重力补偿论文簇](../papers/gravity_compensation.md)
- [重力补偿概念页](../../wiki/concepts/gravity-compensation.md)
