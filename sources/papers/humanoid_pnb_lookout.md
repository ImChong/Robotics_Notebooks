# LookOut: Real-World Humanoid Egocentric Navigation

> 来源归档（ingest · Robot Learning Paper Notebooks 深读笔记）

- **标题：** LookOut: Real-World Humanoid Egocentric Navigation
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/08_Navigation/LookOut__Real-World_Humanoid_Egocentric_Navigation/LookOut__Real-World_Humanoid_Egocentric_Navigation.html>
- **分类：** 08_Navigation
- **arXiv：** <https://arxiv.org/abs/2508.14466>
- **项目页：** <https://sites.google.com/stanford.edu/lookout>
- **开放状态：** **部分开放**（2026-07-28：项目页提供 AND Data 申请；未列代码仓库或权重）
- **入库日期：** 2026-07-10
- **一句话说明：** LookOut 把「人形导航」重新表述成一个第一视角预测问题：给定一段以头为中心的 egocentric 视频，预测未来一串 6-DoF 头部位姿（平移 + 旋转）。平移对应「走哪条无碰撞路」，旋转对应「往哪看」——后者正是人在拐弯、过马路前转头主动收集信息的行为。模型把每帧的 2D DINO 特征反投影到 3D 并按时间聚合，从而同时建模静态结构与动态障碍，再回归出未来轨迹；配套发布 Aria Navigation Dataset（AND），4 小时真实世界导航录制。

## 核心摘录（策展，非全文）

- **任务：** posed egocentric video → 未来 6-DoF 头部轨迹；平移监督无碰撞路径，旋转监督主动转头。
- **方法：** DINO 2D feature 通过相机位姿无参数反投影到 3D canonical frame，并跨时间聚合后回归轨迹。
- **AND：** Project Aria 采集 4 h / 274k RGB frames / 36k clips / 18 个密集室内外地点。
- **评测与边界：** translation / rotation L1 为 0.17 / 0.16；结果是 held-out 离线 forecasting，不是人形机器人闭环部署；Data 可申请但代码未开源。
- 知识归纳见 wiki 实体页：[paper-notebook-lookout](../../wiki/entities/paper-notebook-lookout.md).

## 对 wiki 的映射

- [paper-notebook-lookout](../../wiki/entities/paper-notebook-lookout.md)
- [LookOut 项目页归档](../sites/lookout.md)
- 分类父节点：[paper-notebook-category-08-navigation](../../wiki/overview/paper-notebook-category-08-navigation.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/08_Navigation/LookOut__Real-World_Humanoid_Egocentric_Navigation/LookOut__Real-World_Humanoid_Egocentric_Navigation.html>
- 论文：<https://arxiv.org/abs/2508.14466>
