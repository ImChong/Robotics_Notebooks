# EgoNav: Learning Humanoid Navigation from Human Data

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** EgoNav: Learning Humanoid Navigation from Human Data
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/EgoNav__Learning_Humanoid_Navigation_from_Human_Data/EgoNav__Learning_Humanoid_Navigation_from_Human_Data.html>
- **分类：** 08_Navigation
- **arXiv：** <https://arxiv.org/abs/2604.00416>
- **入库日期：** 2026-07-10
- **一句话说明：** EgoNav 把「导航」拆成一个与机器人本体无关的轨迹分布预测问题：用一个 46M 参数的扩散 UNet，在仅 5 小时人类行走数据上学会「给定过去轨迹 + 360° 第一视角视觉记忆，未来该往哪些方向走」的多模态轨迹分布；推理时用 DDIM+DDPM 混合采样做到实时（Jetson Thor 上 110 traj/s），再由 receding-horizon 控制器从分布里挑路；最终零样本迁移到 Unitree G1，在没见过的室内外环境里连续走 37.5 分钟 / 1137 米 / 96%+ 自主率，还自发涌现出「等门开、绕人群、避玻璃墙」等行为。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-egonav](../../wiki/entities/paper-notebook-egonav.md).

## 对 wiki 的映射

- [paper-notebook-egonav](../../wiki/entities/paper-notebook-egonav.md)
- 分类父节点：[paper-notebook-category-08-navigation](../../wiki/overview/paper-notebook-category-08-navigation.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/EgoNav__Learning_Humanoid_Navigation_from_Human_Data/EgoNav__Learning_Humanoid_Navigation_from_Human_Data.html>
- 论文：<https://arxiv.org/abs/2604.00416>
