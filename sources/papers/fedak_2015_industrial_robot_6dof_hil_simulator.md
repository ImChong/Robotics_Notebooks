# HIL Simulator of Drives of an Industrial Robot with 6 DOF

> 来源归档

- **标题：** HIL Simulator of Drives of an Industrial Robot with 6 DOF
- **类型：** paper（期刊）
- **作者：** Viliam Fedák, František Durovsky, Robert Uveges, Karol Kyslan, Milan Lacko
- **机构：** Technical University of Košice
- **期刊：** Elektronika ir Elektrotechnika（KTU）
- **年份：** 2015
- **DOI：** https://doi.org/10.5755/j01.eee.21.2.11506
- **文章页：** https://eejournal.ktu.lt/index.php/elt/article/view/11506
- **入库日期：** 2026-09-06
- **一句话说明：** 六自由度工业机械臂 **驱动级 HIL**：SINAMICS S120 变频器经 CAN 与 **RT-LAB** 实时主控闭环，在台架上验证驱动与控制算法，实验响应与预期一致。
- **开源状态：** **不适用** — 商用 RT-LAB + SINAMICS 硬件台架；论文无代码发布。

---

## 核心贡献（摘录）

1. **HIL 对象：** 工业六轴机械臂的 **电机驱动链**（非仅运动学层），需融合机械、电力驱动、控制理论、机器人学与变频器内部结构知识。
2. **硬件栈：** Siemens **SINAMICS S120** 变频器 + **CAN** 总线 + **RT-LAB**（Opal-RT 系实时仿真平台）执行实时控制算法。
3. **验证方式：** 所提算法经实验验证，时域响应与期望结果 **吻合良好**。
4. **关键词：** Robotics, robot control, motion analysis, hardware-in-the-loop.

## 对 wiki 的映射

- 概念页：[Hardware-in-the-Loop](../../wiki/concepts/hardware-in-the-loop.md)
- 厂商 HIL 栈：[OPAL-RT HIL 产品页](../sites/opal-rt-hardware-in-the-loop.md)
