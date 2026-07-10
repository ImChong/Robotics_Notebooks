# TEDi: Temporally-Entangled Diffusion for Long-Term Motion Synthesis

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** TEDi: Temporally-Entangled Diffusion for Long-Term Motion Synthesis
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/TEDi__Temporally-Entangled_Diffusion_for_Long-Term_Motion_Synthesis/TEDi__Temporally-Entangled_Diffusion_for_Long-Term_Motion_Synthesis.html>
- **分类：** 14_Human_Motion
- **arXiv：** <https://arxiv.org/abs/2307.15042>
- **入库日期：** 2026-07-10
- **一句话说明：** 去噪扩散概率模型（DDPM）逐步、小增量地合成样本——这种渐进性是其关键。TEDi 把这种渐进性应用到运动序列，并扩展 DDPM 以实现随时间变化的去噪，从而把"扩散时间轴"与"运动时间轴"两条轴纠缠（entangle）起来。具体：维护一个运动缓冲区（motion buffer），里面是越靠后越噪的姿态序列，对其迭代去噪，自回归地生成任意长的帧序列。每个扩散步只推进运动时间轴、而扩散时间轴保持静止，于是干净帧从缓冲区滑出、末端追加新噪声向量，实现长时程动作合成，适用于角色动画等。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-tedi-temporally-entangled-diffusion-for-long-ter](../../wiki/entities/paper-notebook-tedi-temporally-entangled-diffusion-for-long-ter.md).

## 对 wiki 的映射

- [paper-notebook-tedi-temporally-entangled-diffusion-for-long-ter](../../wiki/entities/paper-notebook-tedi-temporally-entangled-diffusion-for-long-ter.md)
- 分类父节点：[paper-notebook-category-14-human-motion](../../wiki/overview/paper-notebook-category-14-human-motion.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/TEDi__Temporally-Entangled_Diffusion_for_Long-Term_Motion_Synthesis/TEDi__Temporally-Entangled_Diffusion_for_Long-Term_Motion_Synthesis.html>
- 论文：<https://arxiv.org/abs/2307.15042>
