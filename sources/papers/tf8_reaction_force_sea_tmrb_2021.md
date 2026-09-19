# Design and Preliminary Results of a Reaction Force Series Elastic Actuator for Bionic Knee and Ankle Prostheses

> 来源归档

- **标题：** Design and Preliminary Results of a Reaction Force Series Elastic Actuator for Bionic Knee and Ankle Prostheses
- **类型：** paper（journal）
- **作者：** Matthew E. Carney, Tony Shu, Roman Stolyarov, Jean-Francois Duval, Hugh M. Herr
- **机构：** MIT Media Lab · Biomechatronics Group
- **期刊：** IEEE Transactions on Medical Robotics and Bionics, vol. 3, no. 3, pp. 542–553, Aug. 2021
- **DOI：** https://doi.org/10.1109/TMRB.2021.3098921
- **预印本（engrXiv）：** https://engrxiv.org/3wt5j/
- **Media Lab 页：** https://www.media.mit.edu/publications/design-and-preliminary-results-of-a-reaction-force-series-elastic-actuator-for-bionic-knee-and-ankle-prostheses/
- **项目页：** [mit_tf8_personalized_bionics](../sites/mit_tf8_personalized_bionics.md)
- **入库日期：** 2026-09-19
- **一句话说明：** 提出 **MC-RFSEA（TF8）** 无缆下肢 powered prosthesis 执行器：用 gait 数据 kinematically clamp 到 SEA 动力学做 **电机/减速比/弹簧** 电能耗最优搜索；台架 + 膝下截肢者穿戴验证；踝配置平地 1.5 m/s **eCOT 0.053 J/kg**。
- **开源状态：** **确认未开源**（见项目页核查）
- **沉淀到 wiki：** [paper-tf8-reaction-force-sea-prosthesis](../../wiki/entities/paper-tf8-reaction-force-sea-prosthesis.md)

---

## 核心贡献（摘录）

1. **Reaction-force SEA 架构（MC-RFSEA）**：moment-coupled cantilever-beam 串联弹性 + 反力传感路径，面向 untethered 膝/踝 powered prostheses，复刻生物 kinematics/kinetics。
2. **Co-design 优化**：将步行 gait 数据 kinematically clamp 到 SEA 动态模型，在电机、减速比、弹簧刚度空间搜索 **电能耗最优** 硬件规格；覆盖平地与变地形力矩/角度/速度需求。
3. **关键规格（硬件本体）**：标称力矩 **85 N·m**；重复峰值 **175 N·m**；**105°** 关节 RoM；硬件质量 **1.6 kg**。
4. **人体实验（踝配置）**：单侧膝下截肢者穿戴；平地 1.5 m/s 步行 preliminary eCOT **0.053 J/kg**。
5. **传感（工程叙事）**：FUTEK **LCM300** load cell 用于 prosthetic force measurement（见 FUTEK 案例页与 USPS Forever stamp 宣传）。

## 对 wiki 的映射

- 论文实体页：[paper-tf8-reaction-force-sea-prosthesis](../../wiki/entities/paper-tf8-reaction-force-sea-prosthesis.md)
- 博士论文扩展：[tf8_carney_phd_thesis_2020.md](./tf8_carney_phd_thesis_2020.md)
- 弹簧 energetics 分析：[tf8_springs_terrain_biorob_2020.md](./tf8_springs_terrain_biorob_2020.md)
- 项目页：[mit_tf8_personalized_bionics](../sites/mit_tf8_personalized_bionics.md)
