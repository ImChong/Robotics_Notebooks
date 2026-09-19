# Electric-Energetic Consequences of Springs in Lower-Extremity Powered Prostheses on Varied Terrain

> 来源归档

- **标题：** Electric-Energetic Consequences of Springs in Lower-Extremity Powered Prostheses on Varied Terrain
- **类型：** paper（conference）
- **作者：** Matthew E. Carney, Hugh M. Herr
- **机构：** MIT Media Lab · Biomechatronics Group
- **会议：** 2020 8th IEEE RAS/EMBS International Conference on Biomedical Robotics and Biomechatronics (BioRob), New York City, NY, USA, pp. 989–996
- **DOI：** https://doi.org/10.1109/BioRob49111.2020.9224458
- **Media Lab 页：** https://www.media.mit.edu/publications/electric-energetic-consequences-of-springs-in-lower-extremity-powered-prostheses-on-varied-terrain/
- **项目页：** [mit_tf8_personalized_bionics](../sites/mit_tf8_personalized_bionics.md)
- **入库日期：** 2026-09-19
- **一句话说明：** 用 kinematically clamped 分析比较 powered 膝/踝假肢中 **电机、减速比 N、串联弹簧 Ks、并联弹簧 Kp、RoM** 对 **eCOT** 的影响，覆盖平地与楼梯上下；并联弹簧平地省能但楼梯可能更费电。
- **开源状态：** **确认未开源**（分析论文，无代码链）
- **沉淀到 wiki：** [paper-tf8-reaction-force-sea-prosthesis](../../wiki/entities/paper-tf8-reaction-force-sea-prosthesis.md)

---

## 核心贡献（摘录）

1. **五参数设计空间**：motor、减速比 **N**、串联弹簧刚度 **Ks**、并联弹簧刚度 **Kp**、允许关节 **RoM** — 后两者 strongly affect 电能消耗。
2. **地形扩展**：除平地步行/跑步外，首次系统考察 **楼梯上下** 等大 RoM 任务对设计参数的影响。
3. **主要结论**：
   - **并联弹簧（PS）** 改善平地 energetics，但在楼梯上可能 **增加** 电能代价；
   - 可变传动 + PS 的复杂组合 **不优于** 简单限制 RoM（尤其对膝）；
   - 膝关节从 PS/可变传动复杂度中获益有限。
4. **与 TF8 关系**：为 RFSEA/MC-RFSEA 选择 **串联反力弹簧** 与 RoM 权衡提供理论依据；TF8 强调可 swap composite spring 做 personalization。

## 对 wiki 的映射

- 论文实体页：[paper-tf8-reaction-force-sea-prosthesis](../../wiki/entities/paper-tf8-reaction-force-sea-prosthesis.md)
- TMRB / Thesis 硬件实现：[tf8_reaction_force_sea_tmrb_2021.md](./tf8_reaction_force_sea_tmrb_2021.md)
