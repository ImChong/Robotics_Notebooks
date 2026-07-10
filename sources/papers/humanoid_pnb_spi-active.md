# Sampling-Based System Identification with Active Exploration for Legged Robot Sim2Real Learning

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** Sampling-Based System Identification with Active Exploration for Legged Robot Sim2Real Learning
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/10_Sim-to-Real/SPI-Active__Sampling-Based_System_Identification_with_Active_Exploration/SPI-Active__Sampling-Based_System_Identification_with_Active_Exploration.html>
- **分类：** 10_Sim-to-Real
- **arXiv：** <https://arxiv.org/abs/2505.14266>
- **入库日期：** 2026-07-10
- **一句话说明：** 高精度腿足技能（如精准落点的跳跃）对 sim-real gap 极其敏感——差一点动力学参数，跳跃就偏几十厘米。主流做法域随机化（DR）靠"把未知量全随机化"求鲁棒，但会让策略偏保守、且难以精确。传统系统辨识（SysID）又常假设动力学可微、能直接测扭矩，这些在富接触腿足系统里根本不成立。SPI-Active 给出两阶段方案：① SPI——用 GPU 上的大规模并行采样（CMA-ES）最小化"仿真 vs 真实"轨迹误差，反推质量-惯量与电机扭矩参数；② Active——不再被动采数据，而是优化探索策略的指令序列去最大化 Fisher 信息（等价 D-最优实验设计），专门激发"最能暴露参数"的高扭矩步态，再回炉重新辨识。最终高精度技能零样本迁移，较基线提升 42–63%。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-sampling-based-system-identification-with-active](../../wiki/entities/paper-notebook-sampling-based-system-identification-with-active.md).

## 对 wiki 的映射

- [paper-notebook-sampling-based-system-identification-with-active](../../wiki/entities/paper-notebook-sampling-based-system-identification-with-active.md)
- 分类父节点：[paper-notebook-category-10-sim-to-real](../../wiki/overview/paper-notebook-category-10-sim-to-real.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/10_Sim-to-Real/SPI-Active__Sampling-Based_System_Identification_with_Active_Exploration/SPI-Active__Sampling-Based_System_Identification_with_Active_Exploration.html>
- 论文：<https://arxiv.org/abs/2505.14266>
