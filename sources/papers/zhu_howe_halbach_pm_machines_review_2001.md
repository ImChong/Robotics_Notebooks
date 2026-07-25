# Halbach permanent magnet machines and applications: a review

> 来源归档

- **标题：** Halbach permanent magnet machines and applications: a review
- **类型：** paper（review）
- **作者：** Z. Q. Zhu, D. Howe
- **机构：** University of Sheffield, Department of Electronic & Electrical Engineering
- **年份：** 2001
- **DOI：** https://doi.org/10.1049/ip-epa:20010479
- **期刊：** IEE Proceedings - Electric Power Applications 148(4):299–308
- **入库日期：** 2026-07-25
- **一句话说明：** 电机工程侧一手综述：多极 Halbach 转子拓扑、烧结分段 vs 粘结环冲磁实现、径向/轴向/直线/球形机，以及飞轮、伺服、被动磁轴承等应用。
- **开源状态：** **不适用**（综述论文；**非 OA**）。本库不入库 PDF。
- **沉淀到 wiki：** [halbach-array](../../wiki/concepts/halbach-array.md)、[paper-zhu-howe-halbach-pm-machines-review](../../wiki/entities/paper-zhu-howe-halbach-pm-machines-review.md)

---

## 核心贡献（摘录）

1. **动机：** 多极 Halbach 磁化转子无刷机具有气隙场正弦度好、可弱化/去掉转子背铁、转矩密度与伺服性能等吸引点。
2. **实现路径对比：**
   - **预磁化烧结稀土分段：** 用离散磁钢逼近理想 Halbach 磁化分布 → **必然妥协性能**（谐波/制造公差）。
   - **粘结各向同性/各向异性 NdFeB 环：** 冲磁成形 Halbach 场分布，更接近连续磁化。
3. **拓扑覆盖：** 径向场与轴向场；有槽/无槽；旋转、直线（管状/平面）、球形。
4. **应用例：** 高速飞轮峰值功率缓冲电机/发电机、高性能直线与旋转伺服、被动磁轴承。
5. **对机器人关节的读法：** DIY QDD（如 Ironless）多用 **分段烧结磁钢 + 90° 步进**——正是文中「compromise」路径；读综述时勿把理想连续 Halbach 指标直接当分段样机预期。

## 对 wiki 的映射

- 概念主页：[halbach-array](../../wiki/concepts/halbach-array.md)
- 论文实体：[paper-zhu-howe-halbach-pm-machines-review](../../wiki/entities/paper-zhu-howe-halbach-pm-machines-review.md)
- 奠基几何：[paper-halbach-permanent-multipole-magnets](../../wiki/entities/paper-halbach-permanent-multipole-magnets.md)
- 电磁完整度对比：[open-source-torque-motor-em-design](../../wiki/comparisons/open-source-torque-motor-em-design.md)
- 样机：[ironless-qdd-actuator](../../wiki/entities/ironless-qdd-actuator.md)、[pcb-motor](../../wiki/entities/pcb-motor.md)
