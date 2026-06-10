# Ansys Motor-CAD 电机设计工作流（官方资料归档）

- **标题：** Ansys Motor-CAD — Electric Machine Design
- **类型：** site（厂商官方产品文档与工作流说明）
- **链接：** <https://www.ansys.com/products/electronics/ansys-motor-cad>
- **补充：** [Motor-CAD 技术白皮书入口](https://www.ansys.com/resource-center/white-paper/motor-cad-electric-machine-design)（Ansys 资源中心）
- **入库日期：** 2026-06-10
- **一句话说明：** Motor-CAD 将电机设计拆为 **EMag（电磁）/ Therm（热）/ Lab（测试对标）/ Mech（机械）** 等模块，支持从规格到效率地图、TN 曲线与温升的迭代设计流程。

## 为什么值得保留

- 产业界（含新能源车电驱与机器人关节模组厂）常用 Motor-CAD 快速出 **TN 曲线、效率地图、温升**；与 Maxwell 电磁 FEA 常组合使用。
- 本条目为 **设计流程** 的一手厂商叙事，与对话整理 [`motor_curves_and_em_simulation_faq.md`](../personal/motor_curves_and_em_simulation_faq.md) 及 wiki [电机电磁仿真软件选型](../../wiki/comparisons/motor-em-simulation-software.md) 互补。

## 官方工作流要点（编译摘要）

| 模块 | 典型任务 |
|------|----------|
| **EMag** | 槽极配合、绕组、磁路；转矩、反电势、Ld/Lq、齿槽转矩 |
| **Therm** | 稳态/瞬态温升、冷却方式、连续功率边界 |
| **Lab** | 将仿真与台架测试数据对标，校准模型 |
| **Mech** | 转子应力、轴承载荷（与电磁结果耦合） |

典型迭代：**指标输入 → 电磁初设 → 热/效率评估 → 不满足则改槽型/磁钢/绕组 → Lab 对标 → 出图（TN、效率地图）**。

## 对 wiki 的映射

- [电机设计流程（规格到台架）](../../wiki/overview/motor-design-workflow.md)
- [电机转矩-转速曲线（TN 曲线）](../../wiki/concepts/motor-torque-speed-curve.md)
- [电机电磁仿真软件选型](../../wiki/comparisons/motor-em-simulation-software.md)
