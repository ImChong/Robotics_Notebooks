# FocusNav: Spatial Selective Attention with Waypoint Guidance for Humanoid Local Navigation

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** FocusNav: Spatial Selective Attention with Waypoint Guidance for Humanoid Local Navigation
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/FocusNav__Spatial_Selective_Attention_with_Waypoint_Guidance_for_Humanoid_Local/FocusNav__Spatial_Selective_Attention_with_Waypoint_Guidance_for_Humanoid_Local.html>
- **分类：** 08_Navigation
- **arXiv：** <https://arxiv.org/abs/2601.12790>
- **机构：** Shanghai Jiao Tong University；Shanghai Innovation Institute
- **项目 / 代码：** **未开源**（2026-07-28 核查：论文与 arXiv 未列项目页、仓库、权重或数据）
- **入库日期：** 2026-06-07
- **一句话说明：** FocusNav 把人形局部导航做成"路径点先告诉我往哪走，注意力再去看那条路上的细节"：用 WGSCA（路径点引导的空间交叉注意力）把感知聚焦到未来轨迹附近，用 SASG（稳定性感知选择门控）在打滑/失稳时主动屏蔽远端信息、把策略压回到脚下安全，在 Unitree G1 上显著提升复杂场景下的导航成功率。

## 核心摘录（策展，非全文）

- **方法：** LiDAR + depth 点云编码 BEV，目标条件 decoder 从目标向机器人反向生成无碰撞 waypoint；WGSCA 沿 waypoint 聚焦。
- **稳定性门控：** SASG 用 roll/pitch 与角速度构造稳定度，经 Gumbel-Softmax 在失稳时截断远端特征，仅保留脚下 terrain。
- **评测：** 最难动态非结构化仿真 SR 87.02%，Gallant / PGCA 为 50.32% / 63.67%；G1 真机覆盖 16 cm 台阶、22° 坡面与动态行人。
- **复现边界：** 无官方代码；不能从论文结构推断出可运行 Isaac Gym / Fast-LIO 工程。
- 知识归纳见 wiki 实体页：[paper-notebook-focusnav](../../wiki/entities/paper-notebook-focusnav.md).

## 对 wiki 的映射

- [paper-notebook-focusnav](../../wiki/entities/paper-notebook-focusnav.md)
- 分类父节点：[paper-notebook-category-08-navigation](../../wiki/overview/paper-notebook-category-08-navigation.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/FocusNav__Spatial_Selective_Attention_with_Waypoint_Guidance_for_Humanoid_Local/FocusNav__Spatial_Selective_Attention_with_Waypoint_Guidance_for_Humanoid_Local.html>
- 论文：<https://arxiv.org/abs/2601.12790>
