# Design and Characterization of 3D Printed, Open-Source Actuators for Legged Locomotion

> 来源归档

- **标题：** Design and Characterization of 3D Printed, Open-Source Actuators for Legged Locomotion
- **类型：** paper
- **作者：** Karthik Urs, Challen Enninful Adu, Elliott J. Rouse, Talia Y. Moore（University of Michigan）
- **链接：** https://arxiv.org/abs/2202.12395
- **入库日期：** 2026-07-25
- **一句话说明：** 面向 8–15 kg 腿式机器人的两种 3D 打印 QDD 执行器：成品电机 + 打印件 + 低减速比；系统测热、连续/峰值力矩、效率、背隙与 42 万步态循环寿命。
- **开源状态：** 论文宣称机械/电气/软件完全开源；**截至 2026-07-25 在 arXiv HTML 中未定位到稳定公开 GitHub 链接**，复现入口待作者实验室站点跟进。
- **沉淀到 wiki：** [paper-3d-printed-open-source-actuators-legged](../../wiki/entities/paper-3d-printed-open-source-actuators-legged.md)

---

## 核心贡献（摘录）

1. 两种 QDD：行星 **7.5:1** 与 bilateral drive **~15:1**；电机 T-Motor **RI50**（\(K_T\approx0.105\) N·m/A）；单执行器材料成本 &lt;$200；驱动用 **moteus r4.5**。
2. 塑料执行器热方案：热限制下可用力矩接近 **提升一倍**。
3. **420k** 步态循环后：效率仅降约 **2%**，背隙增长约 **26 mrad**。
4. 教学价值：不能只看电机峰值力矩，须从热、效率、寿命、背隙评价整关节。

## 对 wiki 的映射

- 论文实体页：[paper-3d-printed-open-source-actuators-legged](../../wiki/entities/paper-3d-printed-open-source-actuators-legged.md)
- 对比页：[open-source-qdd-actuator-projects](../../wiki/comparisons/open-source-qdd-actuator-projects.md)
- 驱动：[moteus](../../wiki/entities/moteus.md)
