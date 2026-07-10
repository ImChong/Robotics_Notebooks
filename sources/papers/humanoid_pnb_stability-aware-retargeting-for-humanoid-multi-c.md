# Stability-Aware Retargeting for Humanoid Multi-Contact Teleoperation

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** Stability-Aware Retargeting for Humanoid Multi-Contact Teleoperation
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/Stability-Aware_Retargeting_for_Humanoid_Multi-Contact_Teleoperation/Stability-Aware_Retargeting_for_Humanoid_Multi-Contact_Teleoperation.html>
- **分类：** 07_Teleoperation
- **arXiv：** <https://arxiv.org/abs/2510.04353>
- **入库日期：** 2026-07-10
- **一句话说明：** 当人形机器人需要用手去推墙、撑天花板、按在斜面/不平表面上完成作业时，多了几个接触点反而让稳定性更难算、更易崩——操作员一个看似合理的指令就可能让某个关节力矩饱和或让手打滑。本文提出稳定性感知的重定向（stability-aware retargeting）：用 actuation-aware 的质心稳定区域度量「现在离失稳还有多远」，再解析地算出稳定裕度对接触点/关节位形的梯度，从而在遥操作回路里实时、低开销地微调接触位置与上身姿态，把机器人往更稳的位形拉，同时仍尊重操作员的高层意图。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-stability-aware-retargeting-for-humanoid-multi-c](../../wiki/entities/paper-notebook-stability-aware-retargeting-for-humanoid-multi-c.md).

## 对 wiki 的映射

- [paper-notebook-stability-aware-retargeting-for-humanoid-multi-c](../../wiki/entities/paper-notebook-stability-aware-retargeting-for-humanoid-multi-c.md)
- 分类父节点：[paper-notebook-category-07-teleoperation](../../wiki/overview/paper-notebook-category-07-teleoperation.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/Stability-Aware_Retargeting_for_Humanoid_Multi-Contact_Teleoperation/Stability-Aware_Retargeting_for_Humanoid_Multi-Contact_Teleoperation.html>
- 论文：<https://arxiv.org/abs/2510.04353>
