# LookOut: Real-World Humanoid Egocentric Navigation

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** LookOut: Real-World Humanoid Egocentric Navigation
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/LookOut__Real-World_Humanoid_Egocentric_Navigation/LookOut__Real-World_Humanoid_Egocentric_Navigation.html>
- **分类：** 08_Navigation
- **arXiv：** <https://arxiv.org/abs/2508.14466>
- **入库日期：** 2026-07-10
- **一句话说明：** LookOut 把「人形导航」重新表述成一个第一视角预测问题：给定一段以头为中心的 egocentric 视频，预测未来一串 6-DoF 头部位姿（平移 + 旋转）。平移对应「走哪条无碰撞路」，旋转对应「往哪看」——后者正是人在拐弯、过马路前转头主动收集信息的行为。模型把每帧的 2D DINO 特征反投影到 3D 并按时间聚合，从而同时建模静态结构与动态障碍，再回归出未来轨迹；配套发布 Aria Navigation Dataset（AND），4 小时真实世界导航录制。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-lookout](../../wiki/entities/paper-notebook-lookout.md).

## 对 wiki 的映射

- [paper-notebook-lookout](../../wiki/entities/paper-notebook-lookout.md)
- 分类父节点：[paper-notebook-category-08-navigation](../../wiki/overview/paper-notebook-category-08-navigation.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/LookOut__Real-World_Humanoid_Egocentric_Navigation/LookOut__Real-World_Humanoid_Egocentric_Navigation.html>
- 论文：<https://arxiv.org/abs/2508.14466>
