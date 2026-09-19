# Design and Evaluation of a Reaction-Force Series Elastic Actuator Configurable as Biomimetic Powered Ankle and Knee Prostheses

> 来源归档

- **标题：** Design and Evaluation of a Reaction-Force Series Elastic Actuator Configurable as Biomimetic Powered Ankle and Knee Prostheses
- **类型：** paper（PhD thesis）
- **作者：** Matthew E. Carney
- **机构：** Massachusetts Institute of Technology, 2020
- **答辩：** https://www.media.mit.edu/events/matthew-carney-dissertation-defense/
- **Media Lab 页：** https://www.media.mit.edu/publications/design-and-evaluation-of-a-reaction-force-series-elastic-actuator-configurable-as-biomimetic-powered-ankle-and-knee-prostheses/
- **项目页：** [mit_tf8_personalized_bionics](../sites/mit_tf8_personalized_bionics.md)
- **期刊精简版：** [tf8_reaction_force_sea_tmrb_2021.md](./tf8_reaction_force_sea_tmrb_2021.md)
- **入库日期：** 2026-09-19
- **一句话说明：** TF8 **RFSEA** 平台完整博士论文：可换 flat-plate composite spring 做个性化惯量/动态匹配；高极对数 drone motor + ball screw + 可调低减速比；仿真 co-design + 台架 + N=3 踝穿戴与楼梯 preliminary。
- **开源状态：** **确认未开源**
- **沉淀到 wiki：** [paper-tf8-reaction-force-sea-prosthesis](../../wiki/entities/paper-tf8-reaction-force-sea-prosthesis.md)

---

## 核心贡献（摘录）

1. **可配置 prosthesis 动机**：商业 powered leg 多为 one-size-fits-all；本文平台可通过 **换弹簧板** 匹配不同体重/步态/运动模式（sport vs economy）。
2. **RFSEA 机械链**：高扭矩高极对 drone motor **直驱 ball screw** + 可调低减速 lead；**flat-plate composite spring** 易制造、可 swap 调动态。
3. **两级优化框架**：
   - 将关节输出 clamp 到 subject-specific 生物 gait，搜索 motor / gear / spring 的 **最小电能** 组合；
   - 再优化 linkage + spring 几何，在离散 COTS 传动件约束下逼近设计目标。
4. **性能（设计点：90 kg、2.0 m/s 非截肢参考 gait）**：
   - 最小可行执行器质量 **1.4 kg**；
   - 标称力矩控制带宽 **6 Hz @ 82 N·m**；
   - 重复峰值 **175 N·m**；演示峰值功率 **>400 W**（理论 **>1 kW**）；
   - **110°** RoM；力矩密度 **125 N·m/kg**；功率密度 **286 W/kg**；
   - 踝：背屈 **35°** + 跖屈 **75°**；膝：**110°** 屈伸（楼梯/坡道）。
5. **系统质量**：踝足系统（含电池与电子）**2.2 kg**；膝配置 **1.6 kg** — 论文称当时最轻、最 adaptable、最 biomimetic 的 published leg system。
6. **人体实验**：有限状态机踝控制器；**N=3** 膝下截肢者 treadmill 1.5 m/s + 1 人楼梯；净正功 **0.2 J/kg**、峰值关节力矩 **1.5 N·m/kg**、峰值机械功率 **4.3 W/kg** 均落在 intact-limb 生物均值 **1 SD** 内。

## 对 wiki 的映射

- 论文实体页：[paper-tf8-reaction-force-sea-prosthesis](../../wiki/entities/paper-tf8-reaction-force-sea-prosthesis.md)
- TMRB 期刊版：[tf8_reaction_force_sea_tmrb_2021.md](./tf8_reaction_force_sea_tmrb_2021.md)
- 膝部工程升级叙事：[elchun TF8 knee 页](https://elchun.github.io/project_pages/tf8_knee.html)（非官方，仅硬件集成记录）
